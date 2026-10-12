"""Real aiortc ICE/DTLS/SRTP path, independent block receiver, no NACK/RTX.

The narrow DTLS hooks are private aiortc APIs; versions are recorded in evidence.
Control manifest and I-frame bytes travel on a reliable ordered DataChannel.
"""
import asyncio
import argparse
import base64
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from aiortc import RTCPeerConnection, RTCConfiguration
from aiortc.rtcrtpparameters import RTCRtpReceiveParameters, RTCRtpCodecParameters, RTCRtpDecodingParameters
from .blocks import BlockAssembler, compress_features, restore_features, packetize, SSRC, PT
from .network import Network, Profile


class Sink:
    def __init__(self):
        self.assemblers={};self.events=[];self.origin=None;self.deadlines={}
    async def _handle_rtp_packet(self,packet,arrival_time_ms):
        from .blocks import HEADER
        frame=HEADER.unpack_from(packet.payload)[1]
        now=(time.perf_counter()-self.origin)*1000
        self.events.append(dict(frame_id=frame,sequence=packet.sequence_number,arrival_ms=now,size=len(packet.serialize())))
        if frame in self.deadlines and now<=self.deadlines[frame]:
            self.assemblers.setdefault(frame,BlockAssembler(frame)).add(packet.serialize())
    async def _handle_rtcp_packet(self,packet):pass
    def _handle_disconnect(self):pass


async def transfer(frame_blocks, iframe_bytes, profile, fps=30, seed=0,testbed=None,profile_name=None,frame_layout=None):
    emulator=None;kernel_state=None
    left=RTCPeerConnection(RTCConfiguration(iceServers=[]));right=RTCPeerConnection(RTCConfiguration(iceServers=[]))
    channel=left.createDataChannel("manifest",ordered=True);opened=asyncio.Event();control={};control_done=asyncio.Event()
    @channel.on("open")
    def onopen():opened.set()
    @right.on("datachannel")
    def onchannel(c):
        @c.on("message")
        def onmessage(data):
            obj=json.loads(data);control[obj['key']]=obj['value']
            if obj['key']=='end':control_done.set()
    sink=Sink()
    try:
        await left.setLocalDescription(await left.createOffer())
        await right.setRemoteDescription(left.localDescription)
        await right.setLocalDescription(await right.createAnswer())
        await left.setRemoteDescription(right.localDescription)
        await asyncio.wait_for(opened.wait(),15)
        # The application receiver accepts block payloads directly, bypassing stock frame jitter buffering.
        params=RTCRtpReceiveParameters(codecs=[RTCRtpCodecParameters(mimeType="video/GRACE",clockRate=90000,payloadType=PT)],
                                      encodings=[RTCRtpDecodingParameters(ssrc=SSRC,payloadType=PT)])
        right.sctp.transport._register_rtp_receiver(sink,params)
        for key,value in [('layout',frame_layout or {f:{} for f in frame_blocks}),('iframe',base64.b64encode(iframe_bytes).decode()),('end',True)]:
            channel.send(json.dumps(dict(key=key,value=value)))
        await asyncio.wait_for(control_done.wait(),10)
        assert base64.b64decode(control['iframe'])==iframe_bytes
        if testbed:
            import subprocess
            sys_path=str(Path(testbed).resolve()/"netemu"/"src")
            import sys
            sys.path.insert(0,sys_path)
            from netemu import NetworkEmulator
            emulator=NetworkEmulator(interface="lo",profiles_path=str(Path(testbed).resolve()/"aitestbed/configs/profiles.yaml"),bidirectional=False)
            if profile_name!="no_emulation" and not emulator.apply_profile(profile_name):
                raise RuntimeError("Testbed netemu rejected profile")
            kernel_state=subprocess.check_output(["tc","-s","-j","qdisc","show","dev","lo"],text=True)
        sink.origin=time.perf_counter();network=Network(profile,seed);seq=0;pending=[];traces=[]
        async def delayed_send(when,packet):
            await asyncio.sleep(max(0,when/1000-(time.perf_counter()-sink.origin)))
            await left.sctp.transport._send_rtp(packet)
        for frame,blocks in frame_blocks.items():
            packets=packetize(blocks,frame,seq,fps=fps);seq=(seq+len(packets))%65536
            generated=frame*1000/fps
            received_meta=control['layout'][str(frame)]
            sink.deadlines[frame]=received_meta.get('generated_ms',generated)+received_meta.get('deadline_ms',profile.deadline_ms)
            arrivals,trace=network.schedule(packets,generated,frame);traces.extend(trace)
            pending.extend(asyncio.create_task(delayed_send(t,p)) for t,p in arrivals)
        await asyncio.gather(*pending)
        final=max(sink.deadlines.values())
        await asyncio.sleep(max(0,final/1000-(time.perf_counter()-sink.origin))+.05)
        received={frame:sink.assemblers.get(frame,BlockAssembler(frame)).finish() for frame in frame_blocks}
        if emulator:
            kernel_state=subprocess.check_output(["tc","-s","-j","qdisc","show","dev","lo"],text=True)
        return received,dict(events=sink.events,scheduled_packets=len(traces),rtp_received=len(sink.events),
                             iframe_sha256=hashlib.sha256(iframe_bytes).hexdigest(),iframe_verified=True,
                             received_layout=control['layout'],
                             left_state=left.connectionState,right_state=right.connectionState,
                             nack=False,rtx=False,transport="aiortc ICE/DTLS/SRTP, application virtual-schedule impairment before encryption",
                             trace=traces,kernel_qdisc=kernel_state,testbed_profile=profile_name)
    finally:
        if emulator:emulator.clear()
        await left.close();await right.close()


async def main_async(args):
    from .codec import GraceCodec,psnr
    from PIL import Image
    import io
    cache=Path(args.cache);m=json.loads((cache/'manifest.json').read_text())
    records=[r for r in m['records'] if r['video_id']==1 and r['gop_id']==0]
    blocks={r['frame_id']:compress_features(np.load(cache/f"v1_f{r['frame_id']}.npz")['code']) for r in records if r['frame_pos']}
    ib=np.load(cache/'v1_f0.npz')['iframe'].tobytes()
    p=Profile(loss_pct=args.loss,delay_ms=args.delay,deadline_ms=200)
    if args.isolated_loopback:
        import aioice.ice
        aioice.ice.get_host_addresses=lambda use_ipv4,use_ipv6:["127.0.0.1"]
    layout={r['frame_id']:dict(shape=r['shape'],length=r['length'],generated_ms=r['frame_id']*1000/m['fps'],deadline_ms=p.deadline_ms) for r in records if r['frame_pos']}
    got,evidence=await transfer(blocks,ib,p,m['fps'],args.seed,args.testbed,args.profile,layout)
    # Closing the transport can allow a final, already-late callback to run.
    evidence['rtp_received_at_drain']=evidence['rtp_received']
    evidence['rtp_received']=len(evidence['events'])
    codec=GraceCodec(args.grace,args.checkpoint,m['q'],m['scale']);rx=np.asarray(Image.open(io.BytesIO(ib)),dtype=np.float32)/255
    rows=[]
    for r in records:
        if not r['frame_pos']:continue
        f=r['frame_id'];data=np.load(cache/f'v1_f{f}.npz');received_layout=evidence['received_layout'][str(f)]
        code=restore_features(got[f],received_layout['length'])
        rx=codec.decode(code,rx,received_layout['shape'])
        rows.append(dict(frame_id=f,mask=''.join(str(int(b is None)) for b in got[f]),psnr=psnr(data['source'],rx),
                         original_sha256=hashlib.sha256(data['code'].tobytes()).hexdigest(),
                         received_sha256=hashlib.sha256(code.tobytes()).hexdigest()))
    import aiortc
    evidence.update(aiortc_version=aiortc.__version__,profile=vars(p),frames=rows)
    if args.testbed:evidence['transport']='aiortc ICE/DTLS/SRTP through original Testbed netemu in isolated Linux network namespace'
    Path(args.output).write_text(json.dumps(evidence,indent=2));print(json.dumps(dict(frames=rows,rtp_received=evidence['rtp_received'],iframe_verified=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',default='cache/q1');p.add_argument('--grace',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--loss',type=float,default=0)
    p.add_argument('--delay',type=float,default=0);p.add_argument('--seed',type=int,default=0);p.add_argument('--output',default='results/rtc.json')
    p.add_argument('--testbed');p.add_argument('--profile',default='no_emulation');p.add_argument('--isolated-loopback',action='store_true')
    asyncio.run(main_async(p.parse_args()))
