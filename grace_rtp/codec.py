"""Adapter uses an independently obtained GRACE installation, never copies it."""
import hashlib
import io
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def psnr(original, decoded):
    mse = float(np.mean((np.asarray(original,dtype=np.float64)-decoded)**2))
    return float("inf") if mse == 0 else -10*np.log10(mse)


def iframe(image, quality=90):
    stream = io.BytesIO()
    Image.fromarray(np.uint8(np.clip(image,0,1)*255)).save(stream,format="JPEG",quality=quality,subsampling=0)
    data = stream.getvalue()
    decoded = np.asarray(Image.open(io.BytesIO(data)),dtype=np.float32)/255
    return data, decoded


class GraceCodec:
    def __init__(self, root, checkpoint, q=1, scale=.5):
        import torch
        self.torch = torch
        root, checkpoint = Path(root).resolve(), Path(checkpoint).resolve()
        sys.path.insert(0,str(root))
        previous = Path.cwd()
        try:
            # Upstream optical-flow model resolves models/other relative to cwd.
            os.chdir(root)
            from grace.grace_model import GraceModel
            self.model = GraceModel({"path":str(checkpoint)},use_half=False,scale_factor=scale)
        finally:
            os.chdir(previous)
        self.model.model.set_quantization_param(q)
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        actual=self.model.model.state_dict()
        if set(actual)-set(state):
            raise ValueError("Checkpoint lacks model parameters")
        for key in actual:
            if not torch.equal(actual[key].cpu(),state[key].cpu()):
                raise ValueError(f"Checkpoint parameter mismatch: {key}")
        self.checkpoint_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()

    def tensor(self, frame):
        return self.torch.from_numpy(np.asarray(frame,dtype=np.float32).transpose(2,0,1).copy())

    def encode(self, image, reference):
        code,mvshape,resshape = self.model.encode(self.tensor(image),self.tensor(reference))
        return code.detach().cpu().numpy().astype("<f4"),dict(mv=list(mvshape),res=list(resshape))

    def decode(self, code, reference, shape):
        out = self.model.decode(self.torch.from_numpy(code.copy()),self.tensor(reference),shape["mv"],shape["res"])
        return out.detach().float().clamp(0,1).cpu().numpy().transpose(1,2,0)


def complexity(image, previous):
    """Source telemetry, explicitly forbidden in the strict capture-only tier."""
    gray = np.asarray(image).mean(axis=2)
    gx=gray[:-2,2:]+2*gray[1:-1,2:]+gray[2:,2:]-gray[:-2,:-2]-2*gray[1:-1,:-2]-gray[2:,:-2]
    gy=gray[2:,:-2]+2*gray[2:,1:-1]+gray[2:,2:]-gray[:-2,:-2]-2*gray[:-2,1:-1]-gray[:-2,2:]
    spatial = np.std(gx)+np.std(gy)
    temporal = np.std(image-previous)
    return dict(si=float(spatial),ti=float(temporal))


def latent_telemetry(code,shape):
    mvsize=int(np.prod(shape['mv']));mv=code[:mvsize];res=code[mvsize:]
    return dict(mv_abs=float(np.mean(abs(mv))),mv_rms=float(np.sqrt(np.mean(mv**2))),
                res_abs=float(np.mean(abs(res))),res_rms=float(np.sqrt(np.mean(res**2))),
                mv_nonzero=float(np.mean(mv!=0)),res_nonzero=float(np.mean(res!=0)))
