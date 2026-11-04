"""Pure world-coordinate audits of each closed continuous-rotation geometry cache."""
from pathlib import Path
import argparse,gc,json,resource,time
from audit_route_decision import ROOT,read,save,digest
from audit_oriented_world_geometry import audit_world_geometry


def main(directory,main_log):
    out=Path(directory).resolve();started=time.monotonic()
    while not (out/'input-matching.json').exists():
        if (out/'failed-execution.json').exists():raise RuntimeError('pure world input producer failed')
        if time.monotonic()-started>120:raise TimeoutError('world trajectory packet unavailable')
        time.sleep(1)
    matching=read(out/'input-matching.json');assert digest(out/'world-trajectory.json')==matching['world_trajectory_sha256']
    world=read(out/'world-trajectory.json');proto=read(out/'implementation-protocol.json')
    for step in range(16):
        marker=f'sealed-world-geometry-step{step:02d}:0/0;'
        while marker not in Path(main_log).read_text():
            if (out/'failed-execution.json').exists():raise RuntimeError('main failed; partial evidence retained')
            if time.monotonic()-started>300:raise TimeoutError('registered geometry closure wait')
            time.sleep(1)
        p=out/f'k16/world-geometry-step{step:02d}.json';h=digest(p);before=time.monotonic();packet=read(p);frame=world['frames'][2*step+1]
        edges,g,degree,cert=audit_world_geometry(packet,frame,world['ids'],frame['rho'],.02,proto['probe_ids'])
        assert digest(p)==h;del packet,edges,g,degree;gc.collect();peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert peak<1024
        save(out/f'k16/world-geometry-audit-step{step:02d}.json',{'step':step,'frame':2*step+1,'world_cache_sha256':h,
          'world_trajectory_sha256':matching['world_trajectory_sha256'],'audit_source_sha256':digest(ROOT/'research/reconnect/audit_oriented_world_geometry.py'),
          'driver_source_sha256':digest(Path(__file__)),'only_world_points_given_boundary_density_D_and_IDs_loaded':True,
          'mass_c0_reference_staticgeometry_or_localcloud_loaded':False,'elapsed_graph_seconds':time.monotonic()-before,'peak_auditor_RSS_MiB':peak,'certificate':cert})
        print(f'independent world geometry:{step+1}/16;peak{peak:.2f}MiB',flush=True)
    save(out/'geometry-audit-execution.json',{'graphs_verified':16,'all16_independent_world_geometry_audits_passed':True,
      'mass_c0_reference_staticgeometry_or_localcloud_loaded':False,'original_builder_rerun_by_auditor':False,
      'elapsed_including_wait_seconds':time.monotonic()-started,'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--main-log',required=True);a=p.parse_args();main(a.directory,a.main_log)
