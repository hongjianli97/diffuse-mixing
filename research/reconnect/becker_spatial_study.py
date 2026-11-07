"""One fixed moving Becker column-space check. No alterations to frozen backends."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json,math,resource,shutil,signal,sys,time,traceback
import numpy as np
from audit_route_decision import ROOT,read,save as raw_save,digest
from affine_shear_completion import plain_json
from bounded_periodic_voronoi import build_bounded_faces
from audit_bounded_geometry import audit_geometry
from becker_point_core import sample_frames,native_run,Becker
from aligned_mass_reference import build_mesh,Mesh,solve,RawStateError,StepRestrictionError
from aligned_becker_check import projected,arrays
from becker_spatial_core import NS,KS,REFS,analyze
sys.path.insert(0,str(ROOT/'research/rebuild'))
from becker import initial

PARENT=ROOT/'runs/reconnect-aligned-becker';OLD=ROOT/'runs/reconnect-becker-point';PROTOCOL=ROOT/'research/reconnect/becker_spatial_protocol.json'


def save(p,value):raw_save(p,plain_json(value))


def preservation():
    old=read(PARENT/'preservation.json');formal=old['formal_manifests_verified']+[{'manifest':str((PARENT/'sha256.json').relative_to(ROOT)),
      'manifest_sha256':digest(PARENT/'sha256.json'),'entries_verified':len(read(PARENT/'sha256.json'))}]
    assert len(formal)==107
    for item in formal:
        assert digest(ROOT/item['manifest'])==item['manifest_sha256']
        for rel,h in read(ROOT/item['manifest']).items():assert digest(ROOT/rel)==h,rel
    originals=read('/workspace/original-handoff-sha256.json');attachments=read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals)==782 and len(attachments)==15
    for rel,h in originals.items():assert digest(ROOT/rel)==h
    for item in attachments:assert digest(Path('/workspace')/item['path'])==item['sha256']
    return {'formal_manifests_verified':formal,'originals_verified':782,'attachments_verified':15,
      'historical_diagnostic_differences_retained':old['historical_diagnostic_differences_retained'],'all_parent_scientific_files_unchanged':True}


def main(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();graphs=paths_done=refs_done=0
    def guard(stage,i=0,total=0):
        elapsed=time.monotonic()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>2400 or peak>1024:raise RuntimeError(f'registered cap exceeded:{elapsed}s/{peak}MiB')
        if not total or i==total:print(f'{stage}:{i}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(*args):raise TimeoutError('registered2400s main/300s reference limit')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(2400)
    try:
        p=read(PROTOCOL);gate=read(PARENT/'next-gate-spec.json');assert not gate['already_executed'] and p['source_gate_sha256']==digest(PARENT/'next-gate-spec.json')
        for item in p['sources']:assert digest(ROOT/item['path'])==item['sha256']
        for rel,h in read(PARENT/'sha256.json').items():assert digest(ROOT/rel)==h,rel
        shutil.copyfile(PROTOCOL,out/'protocol.json');shutil.copyfile(PARENT/'next-gate-spec.json',out/'source-gate-spec.json')
        save(out/'implementation-lock.json',{'files':{name:digest(ROOT/name) for name in p['implementation_files']},
          'fixed_before_new_scientific_labels_points_graphs_mass_scalar_reference_states':True})
        labels={};worlds={};indices={};copies=[]
        for N in NS:
            folder=out/f'n{N}';folder.mkdir();source=ROOT/f'runs/rebuild-stage3-space/inputs-cells{N}.npz'
            with np.load(source,allow_pickle=False) as z:labels[N]=z['mass_edges'].copy()
            assert labels[N].shape==(N+1,);np.save(folder/'source-mass-labels.npy',labels[N])
        assert np.array_equal(labels[32],labels[64][::2]) and np.array_equal(labels[32],labels[128][::4]) and np.array_equal(labels[64],labels[128][::2])
        save(out/'labels-only-preflight.json',{'nested_original_edges_bitwise':True,'mass_c0_reference_or_oldstates_parsed':False})
        flow=Becker(delta=.08)
        for N in (64,128):
            folder=out/f'n{N}';world=sample_frames(labels[N],flow,rows=8);worlds[N]=world;save(folder/'pure-inputs.json',world);index={}
            for K in KS:
                (folder/f'k{K}').mkdir()
                for i in range(K):
                    f=world['cases'][str(K)]['frames'][2*i+1]
                    packet=build_bounded_faces(world['ids'],f['positions'],f['rho'],f['D'],f['x_bounds'],progress=guard)
                    rel=f'k{K}/geometry-step{i:02d}.json';save(folder/rel,packet)
                    index[f'k{K}-step{i:02d}']={'K':K,'step':i,'file':rel,'sha256':digest(folder/rel),'midpoint_time_saved':float(F(f['time_exact']))}
                    _,_,_,certificate=audit_geometry(packet,f['positions'],f['rho'],world['ids'],f['D'],f['x_bounds'])
                    save(folder/f'k{K}/geometry-audit-step{i:02d}.json',{'cache_sha256':digest(folder/rel),'pure_inputs_sha256':digest(folder/'pure-inputs.json'),
                      'audit_source_sha256':digest(ROOT/'research/reconnect/audit_bounded_geometry.py'),'mass_c0_reference_loaded':False,'certificate':certificate})
                    graphs+=1;guard(f'n{N}-sealed-geometry-{graphs}/56')
            save(folder/'graph-index.json',index);indices[N]=index
        save(out/'coefficient-seal.json',{'graphs':graphs,'all56_sealed_before_mass_scalar_oldstates_or_reference':True,
          'internal_material_faces_given_to_builder':False,'indices':{str(N):digest(out/f'n{N}/graph-index.json') for N in (64,128)}})
        assert graphs==56
        def copy(src,dest):
            shutil.copyfile(src,dest);copies.append({'source':str(src.relative_to(ROOT)),'copy':str(dest.relative_to(out)),'sha256':digest(src)})
        columns={};data={}
        for N in NS:
            folder=out/f'n{N}';copy(ROOT/f'runs/rebuild-stage3-space/inputs-cells{N}.npz',folder/'column.npz')
            columns[N]=arrays(folder/'column.npz');col=columns[N]
            assert np.array_equal(col['mass'],np.diff(labels[N]))
            assert all(F(float(b))-F(float(a))==F(float(m)) for a,b,m in zip(labels[N][:-1],labels[N][1:],col['mass']))
            data[N]={'mass':np.tile(col['mass']/8,8),'c0':np.tile(col['c0'],8),'ids':np.arange(8*N)}
            np.savez_compressed(folder/'material.npz',**data[N])
        for K in KS:
            (out/f'n32/k{K}').mkdir()
            for name in ('native-results.json','trajectory-states.npz'):copy(OLD/f'k{K}'/name,out/f'n32/k{K}'/name)
        copy(OLD/'results.json',out/'reused-original-results.json');copy(PARENT/'results.json',out/'reused-aligned-results.json');save(out/'source-copies.json',copies)
        for N in (64,128):
            for K in KS:
                run,path=native_run(out/f'n{N}',K,data[N],indices[N],guard)
                save(out/f'n{N}/k{K}/native-results.json',run);np.savez_compressed(out/f'n{N}/k{K}/trajectory-states.npz',**path)
                paths_done+=1;guard(f'candidate-n{N}-k{K}-completed')
        ref_start=time.monotonic();remaining=2400-(ref_start-start);signal.alarm(max(1,math.ceil(min(300,remaining))))
        rf=out/'reference';rf.mkdir();parent=labels[128];total=F(float(parent[-1]))-F(float(parent[0]));h=total/16384
        counts=tuple(math.ceil(F(float(v))/h) for v in columns[128]['mass']);assert 16384<=sum(counts)<=16511
        meshes=[build_mesh(parent,counts,level=l) for l in (0,1)];assert np.array_equal(meshes[0].edges,meshes[1].edges[::2])
        save(rf/'mesh-contract.json',{'original_parent_columns':128,'counts':list(counts),'h_exact':str(h),'base_cells':sum(counts),'fine_cells':2*sum(counts),'nested':True})
        receipts=[]
        for level,k in REFS:
            mesh=meshes[level];c0=initial(mesh.edges);name=f'l{level}-k{k}';tic=time.monotonic()
            np.savez_compressed(rf/f'input-{name}.npz',mass_edges=mesh.edges,mass=mesh.mass,c0=c0)
            try:c,diag,_=solve(mesh,c0,flow.rho_mu,k,T=.1,D=.02,positivity='monitor',history=False)
            except (RawStateError,StepRestrictionError) as exc:
                np.savez_compressed(rf/f'rejected-{name}.npz',raw_state=getattr(exc,'raw_state',np.array([])))
                save(rf/f'rejected-{name}.json',{'step':exc.step,'diagnostics':exc.diagnostics});raise
            np.savez_compressed(rf/f'raw-{name}.npz',mass_edges=mesh.edges,mass=mesh.mass,c0=c0,c=c)
            receipt={'level':level,'cells':len(mesh.mass),'steps':k,'solve_seconds':time.monotonic()-tic,**diag};save(rf/f'guard-{name}.json',receipt);receipts.append(receipt);observations={}
            for N in NS:
                starts=mesh.starts[::128//N];view=Mesh(labels[N],tuple(map(int,np.diff(starts))),mesh.edges,mesh.mass,mesh.centres,starts)
                for key,value in projected(view,c0,c,columns[N]).items():observations[f'n{N}_{key}']=value
            np.savez_compressed(rf/f'observations-{name}.npz',**observations);refs_done+=1;guard(f'reference-{name}-complete')
        assert time.monotonic()-ref_start<300;signal.alarm(max(1,math.ceil(2400-(time.monotonic()-start))))
        save(rf/'reference-runs.json',receipts);result,common=analyze(out);np.savez_compressed(out/'common-observations.npz',**common);save(out/'results.json',result)
        save(out/'preservation.json',preservation());guard('comparison-complete')
        save(out/'execution.json',{'first_attempt_passed':True,'exit_code':0,'elapsed_seconds':time.monotonic()-start,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'reference_seconds':time.monotonic()-ref_start,'new_graphs':graphs,'new_candidate_paths':paths_done,'new_independent_candidate_paths':paths_done,
          'new_reference_PDE_solves':refs_done,'old_candidate_reruns':0,'new_software_checks':0,'published_or_pushed':False})
        print(json.dumps(plain_json(result),indent=2),flush=True)
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,
          'completed_geometries':graphs,'completed_candidate_paths':paths_done,'completed_references':refs_done});raise
    finally:signal.alarm(0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
