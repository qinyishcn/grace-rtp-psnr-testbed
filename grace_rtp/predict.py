"""Source-disjoint PSNR predictors; application labels excluded from network inputs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

CAPTURE=["packet_count","wire_bytes","capture_loss","delay_mean","delay_p95","delay_std","iat_std","max_size","deadline_ms"]
CONTEXT=["frame_pos","q"]
APP=["block_loss","previous_loss","cumulative_loss"]
LATENT=['mv_abs','mv_rms','res_abs','res_rms','mv_nonzero','res_nonzero']


def matrix(rows,keys):return np.array([[float(r[k]) for k in keys] for r in rows])


def histories(rows,masks):
    result=[];state={}
    for row,mask in zip(rows,masks):
        key=(row['video_id'],row['q'],row['profile'],row['seed'],row['gop_id'])
        last,total=state.get(key,(0.,0.));loss=float(np.mean(mask))
        result.append([loss,last,total,*mask])
        state[key]=(loss,total+loss)
    return np.array(result)


def evaluate(paths,output):
    rows=[]
    for p in paths:
        with open(p,newline='') as f:rows.extend(csv.DictReader(f))
    ids=sorted(set(int(r['video_id']) for r in rows));n=len(ids)
    trainids=ids[:max(1,n-4)];validids=ids[-4:-2];testids=ids[-2:]
    if n<8:raise ValueError("Need >= 8 independent videos")
    groups={name:[r for r in rows if int(r['video_id']) in vids] for name,vids in [('train',trainids),('validation',validids),('test',testids)]}
    train=groups['train'];x=matrix(train,CAPTURE+CONTEXT);y=matrix(train,['psnr']).ravel()
    masks=np.array([[int(b) for b in r['mask']] for r in train])
    stage1=RandomForestRegressor(n_estimators=160,min_samples_leaf=5,max_depth=16,random_state=2026,n_jobs=-1).fit(x,masks)
    # stage 2 is an application response model; true masks are legitimate training labels.
    appx=np.c_[matrix(train,CONTEXT),histories(train,masks)]
    stage2=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,random_state=2026,n_jobs=-1).fit(appx,y)
    feedback=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,random_state=2026,n_jobs=-1).fit(np.c_[appx,matrix(train,['si','ti'])],y)
    latent_feedback=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,random_state=2026,n_jobs=-1).fit(np.c_[appx,matrix(train,['si','ti']+LATENT)],y)
    direct=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,random_state=2026,n_jobs=-1).fit(x,y)
    loss_only=RandomForestRegressor(n_estimators=100,min_samples_leaf=5,random_state=2026,n_jobs=-1).fit(matrix(train,['capture_loss']),y)
    upstream=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,random_state=2026,n_jobs=-1).fit(matrix(train,['packet_count','wire_bytes','frame_pos','q']),y)
    metrics=[];predictions=[];stage1scores=[]
    for split in ['validation','test']:
        target=groups[split];z=matrix(target,CAPTURE+CONTEXT);truth=matrix(target,['psnr']).ravel()
        actual=np.array([[int(b) for b in r['mask']] for r in target]);prob=stage1.predict(z)
        estapp=np.c_[matrix(target,CONTEXT),histories(target,prob)]
        oracleapp=np.c_[matrix(target,CONTEXT),histories(target,actual)]
        pred={
            'mean':np.full(len(target),np.mean(y)),
            'upstream_capture_only':upstream.predict(matrix(target,['packet_count','wire_bytes','frame_pos','q'])),
            'packet_loss_only':loss_only.predict(matrix(target,['capture_loss'])),
            'direct_paired_capture':direct.predict(z),
            'two_stage_paired_capture':stage2.predict(estapp),
            'two_stage_source_feedback':feedback.predict(np.c_[estapp,matrix(target,['si','ti'])]),
            'two_stage_latent_feedback':latent_feedback.predict(np.c_[estapp,matrix(target,['si','ti']+LATENT)]),
            'oracle_mask':stage2.predict(oracleapp),
            'oracle_mask_source_feedback':feedback.predict(np.c_[oracleapp,matrix(target,['si','ti'])])}
        stage1scores.append(dict(split=split,loss_fraction_mae=float(np.mean(abs(actual.mean(axis=1)-prob.mean(axis=1)))),
                                 bit_accuracy=float(np.mean((prob>=.5)==actual)),
                                 bit_brier=float(np.mean((prob-actual)**2)),
                                 all_blocks_good_bit_accuracy=float(np.mean(actual==0)),
                                 pattern_accuracy=float(np.mean(np.all((prob>=.5)==actual,axis=1))),
                                 note="Fractional block probabilities are fed into stage 2; oracle uses exact masks"))
        for name,values in pred.items():
            error=abs(values-truth);rng=np.random.default_rng(2026);videoids=np.array([int(r['video_id']) for r in target]);vids=np.unique(videoids)
            boots=[np.mean(np.concatenate([error[videoids==v] for v in rng.choice(vids,len(vids),replace=True)])) for _ in range(1000)]
            metrics.append(dict(split=split,model=name,n=len(target),mae=float(mean_absolute_error(truth,values)),
                                rmse=float(np.sqrt(mean_squared_error(truth,values))),r2=float(r2_score(truth,values)),
                                video_bootstrap_mae_lo=float(np.percentile(boots,2.5)),video_bootstrap_mae_hi=float(np.percentile(boots,97.5))))
        for i,row in enumerate(target):predictions.append(dict(split=split,video_id=row['video_id'],q=row['q'],profile=row['profile'],seed=row['seed'],frame_id=row['frame_id'],psnr=row['psnr'],actual_block_loss=float(actual[i].mean()),predicted_block_loss=float(prob[i].mean()),**{k:float(v[i]) for k,v in pred.items()}))
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    def save_rows(path,rows):
        with Path(path).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save_rows(out/'prediction_metrics.csv',metrics);save_rows(out/'predictions.csv',predictions)
    byprofile=[]
    for split in ['validation','test']:
        for profile in sorted(set(r['profile'] for r in predictions)):
            subset=[r for r in predictions if r['split']==split and r['profile']==profile]
            for model in pred:
                byprofile.append(dict(split=split,profile=profile,model=model,n=len(subset),
                                      mae=float(np.mean([abs(float(r['psnr'])-r[model]) for r in subset]))))
    save_rows(out/'prediction_by_profile.csv',byprofile)
    evidence=dict(train_videos=trainids,validation_videos=validids,test_videos=testids,
                  capture_features=CAPTURE,protocol_context=CONTEXT,stage1=stage1scores,
                  split_counts={k:len(v) for k,v in groups.items()},
                  warning="Paired synchronized endpoint captures and known playout deadline required. Upstream capture alone cannot identify downstream erasure. Test set has only two videos.",
                  tta="Not claimed: no runtime PSNR labels; minimizing own predictions does not establish quality improvement")
    evidence.update(latent_source_feedback=LATENT,study_status='Exploratory pilot; held-out sources, no confirmatory generalization claim',observation_boundary='frame deadline')
    (out/'prediction_provenance.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(dict(metrics=metrics,stage1=stage1scores),indent=2))
    import joblib
    joblib.dump(dict(stage1=stage1,stage2=stage2,source_feedback=feedback,latent_feedback=latent_feedback,
                     capture=CAPTURE,context=CONTEXT,latent=LATENT),out/'model.joblib',compress=3)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv',nargs='+');p.add_argument('--output',default='results/prediction');a=p.parse_args();evaluate(a.csv,a.output)
