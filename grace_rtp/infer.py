"""Stateful online inference from deadline-bounded capture features and protocol context."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np


class Predictor:
    def __init__(self,path):
        self.bundle=joblib.load(path);self.history={}

    def predict(self,row,source_feedback=False):
        b=self.bundle
        x=np.array([[float(row[k]) for k in b['capture']+b['context']]])
        mask=b['stage1'].predict(x)[0]
        key=row.get('stream_id','default');gop=row['gop_id']
        previous=self.history.get(key)
        last,total=previous[1:] if previous and previous[0]==gop else (0.,0.)
        loss=float(mask.mean())
        features=[*[float(row[k]) for k in b['context']],loss,last,total,*mask]
        model=b['stage2']
        if source_feedback:
            features.extend([float(row['si']),float(row['ti'])]);model=b['source_feedback']
        value=float(model.predict(np.array([features]))[0]);self.history[key]=(gop,loss,total+loss)
        return dict(psnr_estimate=value,expected_missing_block_fraction=loss,block_missing_probabilities=mask.tolist())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    predictor=Predictor(a.model)
    with Path(a.output).open('w') as out:
        for line in Path(a.input).open():out.write(json.dumps(predictor.predict(json.loads(line)))+'\n')
