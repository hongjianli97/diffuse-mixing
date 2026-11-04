"""Independent audits of the three closed pure-point static T1 geometries."""
from pathlib import Path
import argparse,gc,json,resource,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from audit_shear_voronoi_time import audit_geometry


def main(directory,main_log):
    out=Path(directory).resolve();started=time.monotonic()
    while not (out/'input-matching.json').exists():
        if (out/'failed-execution.json').exists():raise RuntimeError('input producer failed')
        if time.monotonic()-started>120:raise TimeoutError('pure point packet unavailable')
        time.sleep(1)
    matching=read(out/'input-matching.json');assert digest(out/'builder-inputs.npz')==matching['builder_packet_sha256']
    with np.load(out/'builder-inputs.npz',allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
    assert set(b)=={f'n{N}_{key}' for N in (16,32,64) for key in ('ids','positions','rho')}
    for N in (16,32,64):
        while f'sealed-geometry-n{N}:0/0;' not in Path(main_log).read_text():
            if (out/'failed-execution.json').exists():raise RuntimeError('geometry producer failed')
            if time.monotonic()-started>600:raise TimeoutError('registered geometry closure wait')
            time.sleep(1)
        p=out/f'n{N}/geometry.json';h=digest(p);before=time.monotonic();packet=read(p)
        edges,g,degree,cert=audit_geometry(packet,b[f'n{N}_positions'],b[f'n{N}_rho'],b[f'n{N}_ids'],[0,N//2,N-1,N*N//2,N*N-1])
        assert digest(p)==h;del packet,edges,g,degree;gc.collect();peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert peak<1024
        save(out/f'n{N}/geometry-only-audit.json',{'N':N,'cache_sha256':h,'builder_packet_sha256':matching['builder_packet_sha256'],
          'audit_source_sha256':digest(ROOT/'research/reconnect/audit_shear_voronoi_time.py'),'driver_source_sha256':digest(Path(__file__)),
          'pure_geometry_only_no_mass_c0_reference_loaded':True,'elapsed_graph_seconds':time.monotonic()-before,'peak_auditor_RSS_MiB':peak,'certificate':cert})
        print(f'independent geometry N={N} passed',flush=True)
    save(out/'geometry-audit-execution.json',{'graphs_verified':3,'all3_pure_geometry_audits_passed':True,
      'mass_c0_reference_or_symbol_loaded':False,'geometry_builder_rerun':False,'elapsed_including_wait_seconds':time.monotonic()-started,
      'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--main-log',required=True);a=p.parse_args();main(a.directory,a.main_log)
