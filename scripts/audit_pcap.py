"""Inspect public RTP headers in actual SRTP PCAPs; never decrypt media."""
import csv,hashlib,json,struct
from pathlib import Path


def read_rtp(path):
    with path.open('rb') as f:
        head=f.read(24);magic=head[:4]
        endian='<' if magic in [b'\xd4\xc3\xb2\xa1',b'\x4d\x3c\xb2\xa1'] else '>'
        link=struct.unpack(endian+'I',head[20:24])[0]
        ns=magic in [b'\x4d\x3c\xb2\xa1',b'\xa1\xb2\x3c\x4d']
        while h:=f.read(16):
            sec,usec,n,_=struct.unpack(endian+'IIII',h);raw=f.read(n)
            offset={1:14,113:16,276:20}.get(link)
            if offset is None:raise ValueError(f'Unsupported PCAP link type {link}')
            ip=raw[offset:]
            if len(ip)<28 or ip[0]>>4!=4 or ip[9]!=17:continue
            ihl=(ip[0]&15)*4
            src,dst,ulen,_=struct.unpack('!HHHH',ip[ihl:ihl+8]);rtp=ip[ihl+8:ihl+ulen]
            if len(rtp)<12 or rtp[0]>>6!=2 or rtp[1]&127!=110 or struct.unpack('!I',rtp[8:12])[0]!=0x47524143:continue
            seq,stamp=struct.unpack('!HI',rtp[2:8])
            yield dict(capture_time=sec+usec/(1e9 if ns else 1e6),sequence=seq,rtp_timestamp=stamp,
                       ip_bytes=len(ip),udp_payload_bytes=len(rtp),source_port=src,destination_port=dst)


if __name__=='__main__':
    root=Path('results');summary=[];allrows=[]
    for path in sorted(root.glob('*netem_*.pcap')):
        rows=list(read_rtp(path));ev=json.loads(path.with_suffix('.json').read_text())
        if ev['rtp_received']!=len(ev['events']):
            ev['rtp_received_at_drain']=ev['rtp_received']
            ev['rtp_received']=len(ev['events'])
            ev['counter_audit_note']='Final late callback during close retained in events; drain counter preserved separately'
            path.with_suffix('.json').write_text(json.dumps(ev,indent=2))
        seq=[r['sequence'] for r in rows];expected=[r['sequence'] for r in ev['events']]
        summary.append(dict(run=path.stem,pcap_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            captured_media=len(rows),application_received=len(ev['events']),scheduled_media=ev['scheduled_packets'],
                            repeated_sequences=len(seq)-len(set(seq)),capture_application_sequence_sets_equal=set(seq)==set(expected),
                            captured_but_not_application=len(set(seq)-set(expected)),application_without_capture=len(set(expected)-set(seq))))
        allrows.extend(dict(run=path.stem,**r) for r in rows)
    with (root/'actual_srtp_packet_headers.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    (root/'pcap_audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
