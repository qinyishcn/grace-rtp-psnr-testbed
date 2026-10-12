import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from dataclasses import asdict
import av
from PIL import Image
import numpy as np
from .blocks import compress_features, restore_features, packetize
from .codec import GraceCodec, iframe, psnr, complexity,latent_telemetry
from .network import Profile, Network, capture_stats


def save_rows(path, rows):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def prepare(args):
    """Sender phase: encode once using independent, clean transmitter references."""
    root=Path(args.cache);root.mkdir(parents=True,exist_ok=True)
    codec=GraceCodec(args.grace,args.checkpoint,args.q,args.scale)
    records=[]
    for video_id in range(1,args.videos+1):
        path=Path(args.videos_dir)/f"{video_id:03d}.mp4"
        cap=av.open(str(path));stream=cap.streams.video[0]
        source_fps=float(stream.average_rate)
        frames=iter(enumerate(cap.decode(video=0)))
        current_index=-1;native=None
        previous=None
        for frame_id in range(args.frames):
            target=round((args.start_seconds+frame_id/args.fps)*source_fps)
            while current_index<target:
                current_index,native=next(frames)
            image=np.asarray(native.to_image().resize((args.width,args.height),Image.Resampling.BILINEAR),dtype=np.float32)/255
            c=complexity(image,image if previous is None else previous)
            if frame_id%args.gop==0:
                bs,txref=iframe(image,args.iframe_quality)
                np.savez_compressed(root/f"v{video_id}_f{frame_id}.npz",source=image,iframe=np.frombuffer(bs,dtype=np.uint8),clean=txref)
                shape=None;length=0;latent={}
            else:
                code,shape=codec.encode(image,txref)
                clean=codec.decode(code,txref,shape)
                blocks=compress_features(code)
                restored=restore_features(blocks,len(code))
                if not np.array_equal(code,restored):raise AssertionError("Entropy roundtrip mismatch")
                # Verify the bytes-decoded tensor yields identical clean reconstruction.
                np.savez_compressed(root/f"v{video_id}_f{frame_id}.npz",source=image,code=code,clean=clean)
                txref=clean;length=len(code)
                latent=latent_telemetry(code,shape)
            records.append(dict(video_id=video_id,frame_id=frame_id,gop_id=frame_id//args.gop,
                                frame_pos=frame_id%args.gop,shape=shape,length=length,latent=latent,**c))
            previous=image
        cap.close()
        print("encoded",video_id,flush=True)
    manifest=dict(records=records,checkpoint_sha256=codec.checkpoint_hash,q=args.q,scale=args.scale,
                  gop=args.gop,width=args.width,height=args.height,fps=args.fps,source_fps=source_fps,
                  iframe_quality=args.iframe_quality,start_seconds=args.start_seconds,
                  sampling="nearest native frame to target FPS at start_seconds",
                  grace_commit="14f1721663acd5d16ca776ce61b4a5860167d24d")
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2))


def run(args):
    """Receiver phase: source is used only for PSNR, never for reconstruction."""
    cache=Path(args.cache);manifest=json.loads((cache/"manifest.json").read_text())
    codec=GraceCodec(args.grace,args.checkpoint,manifest["q"],manifest["scale"])
    import yaml
    profiles=yaml.safe_load(Path(args.profiles).read_text())["profiles"]
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    rows=[];t=time.time()
    traces=(output/"packet_trace.jsonl").open("w",encoding="utf-8")
    for name,config in profiles.items():
        profile=Profile(**config)
        for seed in range(args.seeds):
            network=None;seq=0;rxref=None;history=0.;last_loss=0.
            for record in manifest["records"]:
                v,f=record["video_id"],record["frame_id"]
                if f==0:
                    network=Network(profile,seed*100003+v);seq=0
                data=np.load(cache/f"v{v}_f{f}.npz")
                if record["frame_pos"]==0:
                    # Reliable control path: decode actual JPEG bytes, never clean/source bypass.
                    import io
                    from PIL import Image
                    rxref=np.asarray(Image.open(io.BytesIO(data["iframe"].tobytes())),dtype=np.float32)/255
                    history=0.;last_loss=0.
                    continue
                blocks=compress_features(data["code"])
                packets=packetize(blocks,f,seq,fps=manifest["fps"])
                seq=(seq+len(packets))%65536
                received,trace=network.transmit(packets,f*1000/manifest["fps"],f)
                if args.fixed_loss is not None:
                    # Controlled application labels are separate from packet-network experiments.
                    import random
                    rng=random.Random(seed*100003+v*1009+record["gop_id"])
                    lost=set(rng.sample(range(10),round(args.fixed_loss*10)))
                    received=[None if b in lost else blocks[b] for b in range(10)]
                mask=[int(b is None) for b in received];loss=sum(mask)/10
                recovered=restore_features(received,record["length"])
                decoded=codec.decode(recovered,rxref,record["shape"])
                stats=capture_stats(trace,f*1000/manifest["fps"]+profile.deadline_ms)
                row=dict(video_id=v,gop_id=record["gop_id"],frame_id=f,frame_pos=record["frame_pos"],
                         profile=name,seed=seed,mode="controlled_mask" if args.fixed_loss is not None else "packet_network",
                         psnr=psnr(data["source"],decoded),clean_psnr=psnr(data["source"],data["clean"]),
                         previous_loss=last_loss,cumulative_loss=history,block_loss=loss,mask="".join(map(str,mask)),
                         q=manifest["q"],si=record["si"],ti=record["ti"],deadline_ms=profile.deadline_ms,**stats,
                         **record.get('latent',{}),
                         final_capture_loss=capture_stats(trace)['capture_loss'],
                         random_drops=sum(x["reason"]=="random_drop" for x in trace),
                         queue_drops=sum(x["reason"]=="queue_drop" for x in trace),
                         late_packets=sum(x["reason"]=="late" for x in trace))
                rows.append(row)
                if args.trace:
                    for x in trace:traces.write(json.dumps(dict(video_id=v,profile=name,seed=seed,**x))+"\n")
                rxref=decoded;history+=loss;last_loss=loss
        print(name,"rows",len(rows),"seconds",round(time.time()-t,1),flush=True)
        save_rows(output/"frames.csv",rows)
    traces.close()
    evidence=dict(manifest={k:v for k,v in manifest.items() if k!="records"},profiles=profiles,seeds=args.seeds,
                  rows=len(rows),transport="aiortc RTP serialized bytes, virtual one-way FIFO; no retransmission",
                  fixed_loss=args.fixed_loss,seconds=time.time()-t)
    evidence['feature_observation_boundary']='Only packet arrivals at or before frame generation + deadline; final_capture_loss is analysis-only'
    (output/"provenance.json").write_text(json.dumps(evidence,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["prepare","run"])
    p.add_argument("--grace",required=True);p.add_argument("--checkpoint",required=True)
    p.add_argument("--cache",default="cache/q1");p.add_argument("--videos-dir")
    p.add_argument("--videos",type=int,default=8);p.add_argument("--frames",type=int,default=16)
    p.add_argument("--gop",type=int,default=8);p.add_argument("--fps",type=float,default=30)
    p.add_argument("--start-seconds",type=float,default=10)
    p.add_argument("--width",type=int,default=256);p.add_argument("--height",type=int,default=192)
    p.add_argument("--q",type=float,default=1);p.add_argument("--scale",type=float,default=.5)
    p.add_argument("--iframe-quality",type=int,default=90);p.add_argument("--seeds",type=int,default=3)
    p.add_argument("--profiles",default="configs/profiles.yaml");p.add_argument("--output",default="results/pilot")
    p.add_argument("--fixed-loss",type=float);p.add_argument("--trace",action="store_true")
    args=p.parse_args()
    if args.width%64 or args.height%64: p.error("Dimensions must be multiples of 64")
    (prepare if args.phase=="prepare" else run)(args)


if __name__=="__main__":main()
