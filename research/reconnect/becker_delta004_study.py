"""One frozen fixed-material thickness experiment; no scan or automatic retry."""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,math,resource,shutil,signal,time,traceback
import numpy as np
import becker_point_core
from audit_route_decision import ROOT,read,digest
from audit_becker_spatial import save
from becker_point_core import Becker,native_run
from bounded_periodic_voronoi import build_bounded_faces
from audit_bounded_geometry import audit_geometry
from aligned_becker_check import arrays,projected
from aligned_mass_reference import Mesh,build_mesh,solve,RawStateError,StepRestrictionError
from becker_delta004_sampling import sample_frames_delta004,verify_sampling_rule
from becker_delta004_core import analyze,KS
from becker_delta_definition import preservation

PARENT=ROOT/'runs/reconnect-becker-delta-definition'
PROTOCOL=ROOT/'research/reconnect/becker_delta004_protocol.json'
REFS=((0,8192),(1,8192),(1,16384))


def main(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();graphs=paths=references=0;reference_seconds=0.
    def guard(stage,i=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>900 or peak>1024:raise RuntimeError(f'registered cap exceeded: {elapsed}s/{peak}MiB')
        if not total or i==total:print(f'{stage}: {i}/{total}; {elapsed:.3f}s; {peak:.2f}MiB',flush=True)
    def alarm(*_):raise TimeoutError('registered main900s or reference300s limit')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(900)
    try:
        p=read(PROTOCOL);gate=read(PARENT/'next-gate-spec.json')
        assert gate['type']=='single_delta004_fixed_material_N32_K8_16_32_diagnostic' and not gate['already_executed']
        assert digest(PARENT/'next-gate-spec.json')==p['source_gate_sha256']
        for name,h in p['source_files'].items():assert digest(ROOT/name)==h,name
        for name,h in read(PARENT/'sha256.json').items():assert digest(ROOT/name)==h,name
        shutil.copyfile(PROTOCOL,out/'protocol.json');shutil.copyfile(PARENT/'next-gate-spec.json',out/'source-gate-spec.json')
        save(out/'implementation-lock.json',{'files':{name:digest(ROOT/name) for name in p['implementation_files']},
            'fixed_before_new_labels_points_graphs_mass_scalar_states':True})
        rule=verify_sampling_rule();assert becker_point_core.COUNTS==(4,8,16)
        folder=out/'n32';folder.mkdir()
        with np.load(ROOT/'runs/reconnect-becker-spatial/n32/column.npz',allow_pickle=False) as z:labels=z['mass_edges'].copy()
        assert labels.shape==(33,) and labels[0]==0 and float(labels[-1]).hex()==gate['M_star_hex']
        np.save(folder/'source-mass-labels.npy',labels)
        flow=Becker(delta=.04);world=sample_frames_delta004(labels,flow,rows=8);save(folder/'pure-inputs.json',world)
        for K in KS:
            assert all(f==world['cases']['32']['frames'][i*(32//K)] for i,f in enumerate(world['cases'][str(K)]['frames']))
        index={}
        for K in KS:
            (folder/f'k{K}').mkdir()
            frames=world['cases'][str(K)]['frames']
            for i in range(K):
                tic=time.monotonic();f=frames[2*i+1]
                packet=build_bounded_faces(world['ids'],f['positions'],f['rho'],f['D'],f['x_bounds'],progress=guard)
                rel=f'k{K}/geometry-step{i:02d}.json';save(folder/rel,packet)
                _,_,_,certificate=audit_geometry(packet,f['positions'],f['rho'],world['ids'],f['D'],f['x_bounds'])
                save(folder/f'k{K}/geometry-audit-step{i:02d}.json',{'cache_sha256':digest(folder/rel),
                    'pure_inputs_sha256':digest(folder/'pure-inputs.json'),'audit_source_sha256':digest(ROOT/'research/reconnect/audit_bounded_geometry.py'),
                    'mass_c0_reference_loaded':False,'certificate':certificate})
                index[f'k{K}-step{i:02d}']={'K':K,'step':i,'file':rel,'sha256':digest(folder/rel),
                    'midpoint_time_saved':float(F(f['time_exact'])),'construction_and_audit_seconds':time.monotonic()-tic}
                graphs+=1;guard(f'sealed-geometry-{graphs}/56')
        assert graphs==56
        save(folder/'graph-index.json',index)
        save(out/'labels-only-preflight.json',{'sampling_rule':rule,'all_new_shared_frames_bitwise_equal':True,
            'fixed_M_star_hex':float(labels[-1]).hex(),'new_flow_total_used':False,'mass_c0_reference_oldstates_loaded':False})
        save(out/'coefficient-seal.json',{'graphs':56,'all56_sealed_before_mass_scalar_oldstates_reference':True,
            'internal_material_faces_given_to_builder':False,'index_sha256':digest(folder/'graph-index.json')})
        copies=[]
        for item in p['saved_data_files']:
            target=out/item['copy'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/item['source'],target);copies.append({**item,'sha256':digest(target)})
        save(out/'source-copies.json',copies)
        col=arrays(folder/'column.npz');data=arrays(folder/'material.npz')
        np.testing.assert_array_equal(col['mass_edges'],labels)
        np.testing.assert_array_equal(data['mass'],np.tile(col['mass']/8,8))
        np.testing.assert_array_equal(data['c0'],np.tile(col['c0'],8))
        xx=flow.x_mu(labels,0.);widths=np.diff(xx)
        save(out/'physical-geometry.json',{'delta':.04,'fixed_M_star':float(labels[-1]),'initial_x_endpoints':xx[[0,-1]].tolist(),
            'initial_material_cell_width_range':[float(widths.min()),float(widths.max())],
            'initial_hx_over_delta_range':[float(widths.min()/.04),float(widths.max()/.04)],
            'u1':flow.u1,'u2':flow.u2,'j':flow.j,'fluid_stagnation_enthalpy':flow.H,'B':flow.B,'eta':flow.eta,
            'thermal_conductivity':flow.eta*flow.cp/.75,'scalar_D':.02,
            'true_material_faces_computed_only_after_graph_seal':True,'true_internal_faces_supplied_to_graph_builder':False})
        for K in KS:
            run,path=native_run(folder,K,data,index,guard)
            save(folder/f'k{K}/native-results.json',run);np.savez_compressed(folder/f'k{K}/trajectory-states.npz',**path)
            paths+=1;guard(f'candidate-K{K}-complete')
        refstart=time.monotonic();signal.setitimer(signal.ITIMER_REAL,min(300.,900.-(refstart-started)))
        parent=arrays(out/'n128/column.npz')['mass_edges'];counts=read(out/'reference/mesh-contract.json')['counts']
        meshes={l:build_mesh(parent,counts,level=l) for l in (0,1)};completed=[]
        for level,k in REFS:
            name=f'l{level}-k{k}';mesh=meshes[level];inp=arrays(out/f'reference/input-{name}.npz')
            np.testing.assert_array_equal(mesh.edges,inp['mass_edges']);np.testing.assert_array_equal(mesh.mass,inp['mass'])
            calls=0;samples=[]
            def rho(mu,t):
                nonlocal calls
                assert t==(calls+.5)*(.1/k)
                value=flow.rho_mu(mu,t)
                if calls in (0,k//2,k-1):samples.append({'step':calls,'time':t,'rho_float64_sha256':hashlib.sha256(value.tobytes()).hexdigest()})
                calls+=1;return value
            tic=time.monotonic()
            try:c,diag,_=solve(mesh,inp['c0'],rho,k,T=.1,D=.02,positivity='monitor',history=False)
            except (RawStateError,StepRestrictionError) as exc:
                np.savez_compressed(out/f'reference/rejected-{name}.npz',raw_state=getattr(exc,'raw_state',np.array([])),margin=getattr(exc,'margin',np.array([])))
                save(out/f'reference/rejected-{name}.json',{'step':exc.step,'diagnostics':exc.diagnostics});raise
            assert calls==diag['steps_completed']==diag['linear_solves']==k
            np.savez_compressed(out/f'reference/raw-{name}.npz',**inp,c=c)
            receipt={'level':level,'cells':len(mesh.mass),'steps':k,'solve_seconds':time.monotonic()-tic,'delta':.04,
                'rho_callback_calls':calls,'rho_callback_sample_fingerprints':samples,**diag}
            save(out/f'reference/guard-{name}.json',receipt)
            starts=mesh.starts[::4];view=Mesh(labels,tuple(np.diff(starts)),mesh.edges,mesh.mass,mesh.centres,starts)
            obs=projected(view,inp['c0'],c,col)
            np.savez_compressed(out/f'reference/observations-{name}.npz',**{f'n32_{key}':v for key,v in obs.items()})
            completed.append(receipt);references+=1;guard(f'reference-{name}-complete')
        save(out/'reference/reference-runs.json',completed);reference_seconds=time.monotonic()-refstart
        assert reference_seconds<300
        signal.setitimer(signal.ITIMER_REAL,max(.001,900.-(time.monotonic()-started)))
        result,common=analyze(out);save(out/'results.json',result);np.savez_compressed(out/'common-observations.npz',**common)
        kept=preservation();kept['formal_manifests_verified'].append({'manifest':str((PARENT/'sha256.json').relative_to(ROOT)),
            'manifest_sha256':digest(PARENT/'sha256.json'),'entries_verified':len(read(PARENT/'sha256.json'))})
        assert len(kept['formal_manifests_verified'])==113;save(out/'preservation.json',kept);guard('comparison-complete')
        save(out/'execution.json',{'completed':True,'exit_code':0,'elapsed_seconds':time.monotonic()-started,'reference_seconds':reference_seconds,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'new_graphs':graphs,'new_primary_paths':paths,
            'new_independent_paths':paths,'new_reference_PDE_solves':references,'old_candidate_reruns':0,'new_software_checks':0,'published_or_pushed':False})
        print(result['comparison_decision'],flush=True)
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
            'elapsed_seconds':time.monotonic()-started,'completed_geometries':graphs,'completed_candidate_paths':paths,'completed_reference_solves':references})
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);main(parser.parse_args().output)
