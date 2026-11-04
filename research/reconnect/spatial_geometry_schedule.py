"""Resource-bounded ordinary subprocess jobs; no agents or scientific adaptation."""
from pathlib import Path
import gc,math,os,resource,subprocess,sys,time
from audit_route_decision import ROOT,read,save,digest


def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])/1024
    except FileNotFoundError:pass
    return 0.


def construct(out,jobs,guard,cap=2048):
    out=Path(out);worker=ROOT/'research/reconnect/spatial_geometry_worker.py';active=[];completed=[];index={};peak=0.
    build_max=final_max=0.;limit=1;queue=list(jobs);calibration=False
    def launch(job):
        K,step=job;folder=out/f'k{K}';log=(folder/f'geometry-build-step{step:02d}.log').open('x')
        cmd=[sys.executable,str(worker),'--frame',str(folder/f'frame-input-step{step:02d}.npz'),'--output',str(folder/f'geometry-step{step:02d}.json'),
          '--receipt',str(folder/f'geometry-build-step{step:02d}.json'),'--lock-file',str(out/'packet-serialization.lock'),'--K',str(K),'--step',str(step)]
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);active.append((job,child,log))
    try:
        while queue or active:
            while queue and len(active)<limit:launch(queue.pop(0))
            current=rss(os.getpid())+sum(rss(p.pid) for _,p,_ in active);peak=max(peak,current)
            if current>cap:raise RuntimeError(f'aggregate main+geometry-worker RSS exceeded cap:{current}MiB')
            guard('geometry-scheduler',1,2)
            for item in list(active):
                (K,step),child,log=item;code=child.poll()
                if code is None:continue
                log.close();active.remove(item)
                if code:raise RuntimeError(f'pure geometry worker failed:k{K} step{step},exit{code}; preserve log')
                row=read(out/f'k{K}/geometry-build-step{step:02d}.json');assert row['worker_source_sha256']==digest(worker)
                build_max=max(build_max,row['build_and_probes_peak_RSS_MiB']);final_max=max(final_max,row['peak_RSS_MiB'])
                bound=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024+final_max+max(0,limit-1)*build_max
                if bound>cap:raise RuntimeError(f'conservative main/worker peak envelope exceeded:{bound}MiB')
                row['midpoint_time_saved']=float(jobs[K,step]);index[f'k{K}-step{step:02d}']=row;completed.append((K,step))
                guard(f'sealed-geometry-k{K}-step{step:02d}')
                if not calibration:
                    candidates=[w for w in (1,2,3) if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024+1.2*(final_max+(w-1)*build_max)+128<=cap]
                    limit=max(candidates) if candidates else 1;calibration=True
                    save(out/'geometry-scheduling-lock.json',{'first_job':[K,step],'first_receipt_sha256':digest(out/f'k{K}/geometry-build-step{step:02d}.json'),
                      'workers_selected':limit,'maximum_workers':3,'packet_creation_serialized':True,
                      'selection_rule':'largest w in1,2,3 with parent_peak+1.2*(first_final_peak+(w-1)*first_build_peak)+128<=2048MiB',
                      'main_and_worker_RSS_cap_MiB':cap,'adaptation_uses_scalar_mass_or_reference':False})
                    print(f'pure geometry resource selection:workers={limit},build_peak={build_max:.2f},final_peak={final_max:.2f}MiB',flush=True)
            if active:time.sleep(.2)
        conservative=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024+final_max+(limit-1)*build_max
        return index,{'workers_selected':limit,'sampled_main_and_active_workers_peak_RSS_MiB':peak,
          'conservative_main_and_workers_peak_RSS_MiB':conservative,'maximum_worker_build_peak_RSS_MiB':build_max,
          'maximum_worker_final_peak_RSS_MiB':final_max,'all48_jobs_completed':len(completed)==48,
          'all_packet_creation_serialized':True,'geometry_backend_changed':False,'scalar_mass_or_reference_in_scheduler':False}
    finally:
        for _,child,log in active:
            if child.poll() is None:child.terminate()
            child.wait();log.close()
