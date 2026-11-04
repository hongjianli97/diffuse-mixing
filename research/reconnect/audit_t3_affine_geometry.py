"""Pure direct world-coordinate audits of all56 closed constitutive graph caches."""
from pathlib import Path
import argparse,gc,resource,time
from audit_route_decision import ROOT,read,save,digest
from audit_oriented_world_geometry import audit_world_geometry


def main(directory,main_log):
    out=Path(directory).resolve();started=time.monotonic()
    while not (out/'input-matching.json').exists():
        if (out/'failed-execution.json').exists():raise RuntimeError('pure input producer failed')
        if time.monotonic()-started>120:raise TimeoutError('pureaffineinputsnotready')
        time.sleep(1)
    match=read(out/'input-matching.json');assert digest(out/'pure-inputs.json')==match['pure_inputs_sha256'];world=read(out/'pure-inputs.json');proto=read(out/'implementation-protocol.json');count=0
    for model in ('constant_D','constant_Gamma'):
        for K in (4,8,16):
            for step in range(K):
                marker=f'sealed-affine-{model}-k{K}-step{step:02d}:0/0;'
                while marker not in Path(main_log).read_text():
                    if (out/'failed-execution.json').exists():raise RuntimeError('main failed; partial auditretained')
                    if time.monotonic()-started>600:raise TimeoutError('registeredcoefficientclosurewait')
                    time.sleep(1)
                p=out/model/f'k{K}/world-geometry-step{step:02d}.json';h=digest(p);before=time.monotonic();packet=read(p)
                frame=world['models'][model][str(K)]['frames'][2*step+1]
                edges,g,degree,cert=audit_world_geometry(packet,frame,world['ids'],frame['rho'],frame['D_float'],proto['fixed_geometry_probe_ids'])
                assert digest(p)==h;del packet,edges,g,degree;gc.collect();peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert peak<1024
                save(out/model/f'k{K}/world-geometry-audit-step{step:02d}.json',{'model':model,'K':K,'step':step,'world_cache_sha256':h,
                  'pure_inputs_sha256':match['pure_inputs_sha256'],'audit_source_sha256':digest(ROOT/'research/reconnect/audit_oriented_world_geometry.py'),
                  'driver_source_sha256':digest(Path(__file__)),'mass_c0_reference_or_oldstates_loaded':False,'originalbuildercalledbyauditor':False,
                  'elapsed_graph_seconds':time.monotonic()-before,'peak_auditor_RSS_MiB':peak,'certificate':cert})
                count+=1;print(f'independent affineworldgeometry:{count}/56',flush=True)
    save(out/'geometry-audit-execution.json',{'graphs_verified':count,'all56_independent_world_geometry_audits_passed':True,
      'mass_c0_reference_or_oldstates_loaded':False,'originalbuildercalledbyauditor':False,'elapsed_including_wait_seconds':time.monotonic()-started,
      'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--main-log',required=True);a=p.parse_args();main(a.directory,a.main_log)
