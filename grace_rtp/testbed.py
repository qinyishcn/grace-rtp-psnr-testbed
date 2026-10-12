"""AITestbed extension runner using the external TrafficLogger and netemu.

No LLM/API client is required. Run inside an isolated network namespace for lo shaping.
"""
import argparse,asyncio,json,sys,time,uuid
from pathlib import Path
from types import SimpleNamespace


class GraceScenario:
    scenario_type="grace_feature_video"

    def __init__(self,testbed,config,logger):
        self.testbed,self.config,self.logger=Path(testbed).resolve(),config,logger

    def run(self,network_profile,run_index=0):
        from .rtc import main_async
        from analysis.logger import LogRecord
        args=SimpleNamespace(**self.config,testbed=str(self.testbed),profile=network_profile)
        started=time.time();asyncio.run(main_async(args));elapsed=time.time()-started
        evidence=json.loads(Path(args.output).read_text())
        session=str(uuid.uuid4());records=[]
        for row in evidence['frames']:
            sent=[t for t in evidence['trace'] if t['frame_id']==row['frame_id']]
            metadata=dict(row,transport=evidence['transport'],iframe_verified=evidence['iframe_verified'],
                          timing_scope="latency unavailable; offline neural decode after scheduled network drain",
                          fully_delivered='1' not in row['mask'],kernel_qdisc=evidence['kernel_qdisc'])
            record=LogRecord(timestamp=started,scenario_id=self.scenario_type,session_id=session,
                             turn_index=row['frame_id'],run_index=run_index,provider='grace-local',model='GRACE-4096',
                             request_bytes=sum(t['size']-28 for t in sent),response_bytes=0,
                             network_profile=network_profile,success=True,is_streaming=True,metadata=json.dumps(metadata))
            self.logger.log(record);records.append(record)
        return dict(session_id=session,frame_count=len(records),network_profile=network_profile,
                    runner_wall_seconds=elapsed,success=True,all_blocks_delivered=all('1' not in r['mask'] for r in evidence['frames']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--testbed',required=True);p.add_argument('--grace',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--cache',default='cache/q1')
    p.add_argument('--profile',default='no_emulation');p.add_argument('--output',default='results/testbed_scenario.json')
    p.add_argument('--db',default='results/traffic.db');p.add_argument('--isolated-loopback',action='store_true');a=p.parse_args()
    sys.path.insert(0,str(Path(a.testbed).resolve()/'aitestbed'))
    from analysis.logger import TrafficLogger
    config=dict(cache=a.cache,grace=a.grace,checkpoint=a.checkpoint,loss=0,delay=0,seed=0,output=a.output,isolated_loopback=a.isolated_loopback)
    result=GraceScenario(a.testbed,config,TrafficLogger(a.db)).run(a.profile)
    Path(a.output+'.scenario.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
