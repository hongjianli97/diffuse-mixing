"""All56 sealed new-world geometries; no material or scalar loads."""
from pathlib import Path
import argparse,gc,resource,time
from audit_route_decision import ROOT,read,save,digest
from audit_sliding_world_geometry import audit_sliding_world_geometry


def main(directory,main_log):
    out=Path(directory).resolve();started=time.monotonic();count=0
    for N in (16,64):
        folder=out/f'n{N}'
        while not (folder/'input-matching.json').exists():
            if (out/'failed-execution.json').exists():raise RuntimeError('producer failed; partial audit retained')
            if time.monotonic()-started>1650:raise TimeoutError('pure inputs unavailable within main horizon')
            time.sleep(1)
        match=read(folder/'input-matching.json');assert digest(folder/'pure-inputs.json')==match['pure_inputs_sha256'];world=read(folder/'pure-inputs.json')
        for K in (4,8,16):
            for step in range(K):
                marker=f'sealed-spatial-sliding-n{N}-k{K}-step{step:02d}:0/0;'
                while marker not in Path(main_log).read_text():
                    if (out/'failed-execution.json').exists():raise RuntimeError('producer failed; partial audit retained')
                    if time.monotonic()-started>1650:raise TimeoutError('closed cache unavailable within main horizon')
                    time.sleep(1)
                p=folder/f'k{K}/world-geometry-step{step:02d}.json';h=digest(p);before=time.monotonic();frame=world['cases'][str(K)]['frames'][2*step+1]
                probes=list(dict.fromkeys([0,N//2,N-1,N*N//2,N*N-1]))
                edges,g,degree,cert=audit_sliding_world_geometry(read(p),frame,world['ids'],frame['rho'],frame['D_float'],probes)
                assert digest(p)==h;del edges,g,degree;gc.collect();peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert peak<1024
                save(folder/f'k{K}/world-geometry-audit-step{step:02d}.json',{'N':N,'K':K,'step':step,'world_cache_sha256':h,'pure_inputs_sha256':match['pure_inputs_sha256'],
                  'audit_source_sha256':digest(ROOT/'research/reconnect/audit_sliding_world_geometry.py'),'driver_source_sha256':digest(Path(__file__)),
                  'mass_c0_reference_or_oldstates_loaded':False,'original_or_sliding_geometry_builder_called_by_auditor':False,
                  'elapsed_graph_seconds':time.monotonic()-before,'peak_auditor_RSS_MiB':peak,'certificate':cert})
                count+=1;print(f'independent sliding spatial world geometry:{count}/56',flush=True)
        del world;gc.collect()
    save(out/'geometry-audit-execution.json',{'graphs_verified':count,'all56_independent_sliding_world_geometry_audits_passed':True,
      'mass_c0_reference_or_oldstates_loaded':False,'original_or_sliding_geometry_builder_called_by_auditor':False,
      'elapsed_including_wait_seconds':time.monotonic()-started,'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--main-log',required=True);a=p.parse_args();main(a.directory,a.main_log)
