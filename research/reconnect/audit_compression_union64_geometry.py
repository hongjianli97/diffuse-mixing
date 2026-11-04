"""Independent closed-cache exact geometry audit using pure N64 point inputs only."""
from pathlib import Path
import argparse,gc,json,resource,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from audit_shear_voronoi_time import audit_geometry


def main(directory,main_log):
    out=Path(directory).resolve();started=time.monotonic()
    while not (out/'input-matching.json').exists():
        if (out/'failed-execution.json').exists():raise RuntimeError('input producer failed')
        if time.monotonic()-started>120:raise TimeoutError('pure builder packet not ready')
        time.sleep(1)
    proto=read(out/'implementation-protocol.json');matching=read(out/'input-matching.json')
    assert digest(out/'builder-inputs.npz')==matching['builder_packet_sha256']
    with np.load(out/'builder-inputs.npz',allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
    assert set(b)=={'ids'}|{f'k{K}_{k}' for K in (16,32) for k in ('time','positions','rho','boundary_time','midpoint_time')}
    auditor=ROOT/'research/reconnect/audit_shear_voronoi_time.py';driver=Path(__file__).resolve();count=0
    for K in (16,32):
        for step in range(K):
            marker=f'sealed-geometry-k{K}-step{step:02d}:0/0;'
            while marker not in Path(main_log).read_text():
                if (out/'failed-execution.json').exists():raise RuntimeError('main failed; preserve audit evidence')
                if time.monotonic()-started>700:raise TimeoutError('registered geometry closure wait cap')
                time.sleep(1)
            file=out/f'k{K}/geometry-step{step:02d}.json';h=digest(file);before=time.monotonic();packet=read(file);frame=2*step+1
            edges,g,degree,cert=audit_geometry(packet,b[f'k{K}_positions'][frame],b[f'k{K}_rho'][frame],b['ids'],proto['fixed_geometry_probe_ids'])
            assert digest(file)==h;del packet,edges,g,degree;gc.collect();peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            assert peak<1024
            save(out/f'k{K}/geometry-only-audit-step{step:02d}.json',{'K':K,'step':step,'cache_sha256':h,
              'builder_packet_sha256':digest(out/'builder-inputs.npz'),'audit_source_sha256':digest(auditor),'driver_source_sha256':digest(driver),
              'pure_geometry_only_no_mass_c0_reference_loaded':True,'elapsed_graph_seconds':time.monotonic()-before,
              'peak_auditor_RSS_MiB':peak,'certificate':cert})
            count+=1;print(f'independent geometry:k{K} step{step+1}/{K};peak{peak:.2f}MiB',flush=True)
    result={'graphs_verified':count,'all48_pure_geometry_audits_passed':True,'mass_c0_reference_or_PDE_output_loaded':False,
      'geometry_builder_rerun':False,'audit_source_sha256':digest(auditor),'driver_source_sha256':digest(driver),
      'elapsed_including_wait_seconds':time.monotonic()-started,'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    save(out/'geometry-audit-execution.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--main-log',required=True);a=p.parse_args();main(a.directory,a.main_log)
