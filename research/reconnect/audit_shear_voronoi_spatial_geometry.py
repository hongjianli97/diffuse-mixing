"""Audit closed geometry caches concurrently, without loading mass/state/reference."""
from pathlib import Path
import argparse,gc,json,resource,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from audit_shear_voronoi_time import audit_geometry


def main(directory,log):
    out=Path(directory).resolve();spec=read(out/'source-gate-spec.json');protocol=read(out/'implementation-protocol.json')
    assert digest(out/'builder-inputs.npz')==spec['builder_packet_sha256']
    with np.load(out/'builder-inputs.npz',allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
    assert set(b)=={'ids'}|{f'k{K}_{suffix}' for K in (4,8) for suffix in ['time','positions','rho','boundary_time','midpoint_time']}
    audit_path=ROOT/'research/reconnect/audit_shear_voronoi_time.py'
    driver_path=Path(__file__).resolve();started=time.monotonic();records=[]
    for K in (4,8):
        for step in range(K):
            marker=f'sealed-geometry-k{K}-step{step:02d}:0/0;'
            # The locked producer prints this only after save/close/hash/release.
            while marker not in Path(log).read_text():
                if (out/'failed-execution.json').exists():raise RuntimeError('main failed; retain partial geometry audits')
                if time.monotonic()-started>2400:raise TimeoutError('closed geometry not available within fixed wait limit')
                time.sleep(1)
            path=out/f'geometry-k{K}-step{step:02d}.json';sha=digest(path);cache=read(path);frame=2*step+1
            before=time.monotonic()
            edges,g,degree,certificate=audit_geometry(cache,b[f'k{K}_positions'][frame],b[f'k{K}_rho'][frame],b['ids'],protocol['fine_geometry_probe_ids'])
            assert digest(path)==sha
            del cache,edges,g,degree;gc.collect()
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            if peak>2048:raise RuntimeError('independent geometry auditor exceeded2048MiB; no resource relaxation')
            item={'K':K,'step':step,'cache_sha256':sha,'builder_packet_sha256':digest(out/'builder-inputs.npz'),
                'independent_audit_source_sha256':digest(audit_path),'driver_source_sha256':digest(driver_path),
                'pure_geometry_only_no_mass_c0_reference_loaded':True,'elapsed_graph_audit_seconds':time.monotonic()-before,
                'peak_auditor_RSS_MiB':peak,'certificate':certificate}
            save(out/f'geometry-only-audit-k{K}-step{step:02d}.json',item);records.append(item)
            print(f'closed-cache independent geometry audit:k{K} step{step+1}/{K};peak{peak:.2f}MiB',flush=True)
    result={'graphs_verified':len(records),'all12_pure_geometry_audits_passed':True,
        'mass_scalar_reference_or_PDE_outputs_loaded':False,'geometry_rebuilt_by_builder':False,
        'audit_geometry_function_unchanged':True,'elapsed_including_wait_seconds':time.monotonic()-started,
        'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'independent_audit_source_sha256':digest(audit_path),'driver_source_sha256':digest(driver_path)}
    save(out/'geometry-audit-execution.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--main-log',required=True);a=p.parse_args();main(a.directory,a.main_log)
