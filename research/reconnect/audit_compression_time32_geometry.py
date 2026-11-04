"""Independent pure-point audits of each closed32-interval compression cache."""
from pathlib import Path
import argparse,gc,json,resource,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from audit_shear_voronoi_time import audit_geometry


def main(directory,main_log):
    out=Path(directory).resolve();started=time.monotonic()
    while not (out/'input-matching.json').exists():
        if (out/'failed-execution.json').exists():raise RuntimeError('pure input producer failed')
        if time.monotonic()-started>120:raise TimeoutError('pure input packet not ready')
        time.sleep(1)
    proto=read(out/'implementation-protocol.json');matching=read(out/'input-matching.json')
    assert digest(out/'builder-inputs.npz')==matching['builder_packet_sha256']
    with np.load(out/'builder-inputs.npz',allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
    assert set(b)=={'ids'}|{f'k32_{k}' for k in ('time','positions','rho','boundary_time','midpoint_time')}
    auditor=ROOT/'research/reconnect/audit_shear_voronoi_time.py';driver=Path(__file__).resolve()
    for step in range(32):
        marker=f'sealed-geometry-k32-step{step:02d}:0/0;'
        while marker not in Path(main_log).read_text():
            if (out/'failed-execution.json').exists():raise RuntimeError('main failed; partial audit preserved')
            if time.monotonic()-started>1400:raise TimeoutError('registered geometry closure cap')
            time.sleep(1)
        file=out/f'k32/geometry-step{step:02d}.json';h=digest(file);before=time.monotonic();packet=read(file);frame=2*step+1
        edges,g,degree,cert=audit_geometry(packet,b['k32_positions'][frame],b['k32_rho'][frame],b['ids'],proto['fixed_geometry_probe_ids'])
        assert digest(file)==h;del packet,edges,g,degree;gc.collect();peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert peak<1024
        save(out/f'k32/geometry-only-audit-step{step:02d}.json',{'K':32,'step':step,'cache_sha256':h,
          'builder_packet_sha256':digest(out/'builder-inputs.npz'),'audit_source_sha256':digest(auditor),'driver_source_sha256':digest(driver),
          'pure_geometry_only_no_mass_c0_reference_loaded':True,'elapsed_graph_seconds':time.monotonic()-before,'peak_auditor_RSS_MiB':peak,'certificate':cert})
        print(f'independent geometry:k32 step{step+1}/32;peak{peak:.2f}MiB',flush=True)
    r={'graphs_verified':32,'all32_pure_geometry_audits_passed':True,'mass_c0_reference_or_PDE_output_loaded':False,
      'geometry_builder_rerun':False,'audit_source_sha256':digest(auditor),'driver_source_sha256':digest(driver),
      'elapsed_including_wait_seconds':time.monotonic()-started,'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    save(out/'geometry-audit-execution.json',r);print(json.dumps(r,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--main-log',required=True);a=p.parse_args();main(a.directory,a.main_log)
