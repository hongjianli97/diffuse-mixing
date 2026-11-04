"""One pure point job, unchanged exact builder; serialize packet creation under a lock."""
from pathlib import Path
import argparse,fcntl,gc,json,os,resource,time
import numpy as np
from audit_route_decision import ROOT,save,digest
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from shear_voronoi_fv_study import verify_probe_cells


def main(frame,output,receipt,lock_file,K,step):
    started=time.monotonic();p=Path(frame)
    with np.load(p,allow_pickle=False) as z:
        assert set(z.files)=={'ids','positions','rho'}
        ids=z['ids'];positions=z['positions'];rho=z['rho']
    probes=[0,127,255,32768,65535]
    def progress(stage,k,total):
        if k%16384==0 or k==total:print(f'{stage}:{k}/{total};{time.monotonic()-started:.3f}s',flush=True)
    graph=build_periodic_faces(ids,positions,rho,.02,(1,1),1,progress)
    checked=verify_probe_cells(graph,probes);build_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    # Hold through process exit: a second worker cannot retain a packet concurrently.
    handle=open(lock_file,'a');fcntl.flock(handle,fcntl.LOCK_EX)
    packet=coefficient_packet(graph);save(output,packet)
    row={'N':256,'K':K,'step':step,'frame':2*step+1,'file':str(Path(output).relative_to(Path(receipt).parent.parent)),
      'sha256':digest(output),'frame_input_sha256':digest(frame),'geometry_diagnostics':graph.diagnostics,'independent_fixed_probes':checked,
      'geometry_builder_sha256':digest(ROOT/'research/reconnect/periodic_voronoi_fv.py'),
      'worker_source_sha256':digest(Path(__file__)),'build_and_probes_peak_RSS_MiB':build_peak,
      'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'elapsed_seconds':time.monotonic()-started,
      'only_pure_point_inputs_received':True,'coefficient_packet_creation_serialized':True}
    save(receipt,row);print(json.dumps(row,indent=2),flush=True)
    os._exit(0)  # Keep the packet lock until OS process teardown releases its memory.


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frame',required=True);p.add_argument('--output',required=True);p.add_argument('--receipt',required=True)
    p.add_argument('--lock-file',required=True);p.add_argument('--K',required=True,type=int);p.add_argument('--step',required=True,type=int)
    a=p.parse_args();main(a.frame,a.output,a.receipt,a.lock_file,a.K,a.step)
