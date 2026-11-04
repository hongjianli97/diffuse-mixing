"""Audit closed caches on both grids using only pure point inputs."""
from pathlib import Path
import argparse,gc,json,resource,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from audit_shear_voronoi_time import audit_geometry


def main(directory,main_log):
    out=Path(directory).resolve();started=time.monotonic()
    while not (out/'input-matching.json').exists():
        if (out/'failed-execution.json').exists():raise RuntimeError('input producer failed; no numerical repair')
        if time.monotonic()-started>120:raise TimeoutError('pure builder inputs not ready')
        time.sleep(1)
    protocol=read(out/'implementation-protocol.json');matching=read(out/'input-matching.json')
    assert digest(out/'builder-inputs.npz')==matching['builder_packet_sha256']
    with np.load(out/'builder-inputs.npz',allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
    assert set(b)=={f'n{N}_{k}' for N in (128,256) for k in ['ids','time','positions','rho','boundary_time','midpoint_time']}
    rows=[];old_audit=ROOT/'research/reconnect/audit_shear_voronoi_time.py';driver=Path(__file__).resolve()
    for N in (128,256):
        for step in range(16):
            marker=f'sealed-geometry-n{N}-step{step:02d}:0/0;'
            while marker not in Path(main_log).read_text():
                if (out/'failed-execution.json').exists():raise RuntimeError('main failed; preserve partial independent audit')
                if time.monotonic()-started>4000:raise TimeoutError('registered geometry closure wait limit reached')
                time.sleep(1)
            file=out/f'n{N}/geometry-step{step:02d}.json';h=digest(file);cache=read(file);frame=2*step+1;before=time.monotonic()
            edges,g,degree,certificate=audit_geometry(cache,b[f'n{N}_positions'][frame],b[f'n{N}_rho'][frame],b[f'n{N}_ids'],protocol['fixed_probe_ids'][str(N)])
            assert digest(file)==h;del cache,edges,g,degree;gc.collect()
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            if peak>2048:raise RuntimeError('independent geometry auditor exceeded2048MiB')
            row={'N':N,'step':step,'cache_sha256':h,'builder_packet_sha256':digest(out/'builder-inputs.npz'),
                'audit_source_sha256':digest(old_audit),'driver_source_sha256':digest(driver),
                'pure_geometry_only_no_mass_c0_reference_loaded':True,'elapsed_graph_seconds':time.monotonic()-before,
                'peak_auditor_RSS_MiB':peak,'certificate':certificate}
            save(out/f'n{N}/geometry-only-audit-step{step:02d}.json',row);rows.append(row)
            print(f'closed-cache independent geometry audit:n{N} step{step+1}/16;peak{peak:.2f}MiB',flush=True)
    result={'graphs_verified':len(rows),'all32_pure_geometry_audits_passed':True,
        'mass_c0_reference_or_PDE_output_loaded':False,'geometry_builder_rerun':False,
        'audit_source_sha256':digest(old_audit),'driver_source_sha256':digest(driver),
        'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'elapsed_including_wait_seconds':time.monotonic()-started}
    save(out/'geometry-audit-execution.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--main-log',required=True);a=p.parse_args();main(a.directory,a.main_log)
