"""Exactly one new128/K32 path; all old states and references reused."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json,resource,shutil,signal,time,traceback
import numpy as np
import becker_point_core
from audit_route_decision import ROOT,read,save as raw_save,digest
from affine_shear_completion import plain_json
from becker_point_core import Becker,native_run
from bounded_periodic_voronoi import build_bounded_faces
from audit_bounded_geometry import audit_geometry
from aligned_becker_check import arrays
from becker_fine_time_core import sample_frames_k32,verify_sampling_rule,analyze

PARENT=ROOT/'runs/reconnect-becker-spatial-budget';OLD=ROOT/'runs/reconnect-becker-spatial'
PROTOCOL=ROOT/'research/reconnect/becker_fine_time_protocol.json'


def save(path,value):raw_save(path,plain_json(value))


def preserve():
    prior=read(PARENT/'preservation.json')
    formal=prior['formal_manifests_verified']+[{'manifest':str((PARENT/'sha256.json').relative_to(ROOT)),
        'manifest_sha256':digest(PARENT/'sha256.json'),'entries_verified':len(read(PARENT/'sha256.json'))}]
    assert len(formal)==109
    for item in formal:
        assert digest(ROOT/item['manifest'])==item['manifest_sha256']
        for name,h in read(ROOT/item['manifest']).items():assert digest(ROOT/name)==h,name
    originals=read('/workspace/original-handoff-sha256.json');attachments=read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals)==782 and len(attachments)==15
    for name,h in originals.items():assert digest(ROOT/name)==h
    for item in attachments:assert digest(Path('/workspace')/item['path'])==item['sha256']
    return {'formal_manifests_verified':formal,'originals_verified':782,'attachments_verified':15,
        'historical_diagnostic_differences_retained':prior['historical_diagnostic_differences_retained'],'all_prior_scientific_files_unchanged':True}


def main(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();graphs=paths=0
    def guard(stage,i=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>2400 or peak>1024:raise RuntimeError(f'registered cap exceeded:{elapsed}s/{peak}MiB')
        if not total or i==total:print(f'{stage}:{i}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(*args):raise TimeoutError('registered2400s main limit')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(2400)
    try:
        p=read(PROTOCOL);gate=read(PARENT/'next-gate-spec.json')
        assert gate['type']=='single_finest_Becker_K32_concentration_time_diagnostic' and not gate['already_executed']
        for item in p['sources']+gate['sources']:assert digest(ROOT/item['path'])==item['sha256'],item['path']
        for name,h in read(PARENT/'sha256.json').items():assert digest(ROOT/name)==h,name
        shutil.copyfile(PROTOCOL,out/'protocol.json');shutil.copyfile(PARENT/'next-gate-spec.json',out/'source-gate-spec.json')
        save(out/'implementation-lock.json',{'files':{name:digest(ROOT/name) for name in p['implementation_files']},'fixed_before_new_labels_points_graphs_mass_scalar_states':True})
        rule=verify_sampling_rule();assert becker_point_core.COUNTS==(4,8,16)
        folder=out/'n128';(folder/'k32').mkdir(parents=True)
        with np.load(OLD/'n128/column.npz',allow_pickle=False) as z:labels=z['mass_edges'].copy()
        assert labels.shape==(129,);np.save(folder/'source-mass-labels.npy',labels)
        world=sample_frames_k32(labels,Becker(delta=.08),rows=8);save(folder/'pure-inputs.json',world)
        old_world=read(OLD/'n128/pure-inputs.json');old_frames=old_world['cases']['16']['frames'];frames=world['cases']['32']['frames']
        assert len(frames)==65 and all(frames[2*i]==frame for i,frame in enumerate(old_frames))
        assert becker_point_core.COUNTS==(4,8,16)
        save(out/'labels-only-preflight.json',{**rule,'old_K16_shared33_frames_bitwise_equal':True,'only_K32_sampled':True,'mass_c0_reference_oldstates_loaded':False})
        del old_world,old_frames
        index={}
        for i in range(32):
            tic=time.monotonic();f=frames[2*i+1]
            packet=build_bounded_faces(world['ids'],f['positions'],f['rho'],f['D'],f['x_bounds'],progress=guard)
            rel=f'k32/geometry-step{i:02d}.json';save(folder/rel,packet)
            _,_,_,certificate=audit_geometry(packet,f['positions'],f['rho'],world['ids'],f['D'],f['x_bounds'])
            save(folder/f'k32/geometry-audit-step{i:02d}.json',{'cache_sha256':digest(folder/rel),'pure_inputs_sha256':digest(folder/'pure-inputs.json'),
                'audit_source_sha256':digest(ROOT/'research/reconnect/audit_bounded_geometry.py'),'mass_c0_reference_loaded':False,'certificate':certificate})
            index[f'k32-step{i:02d}']={'K':32,'step':i,'file':rel,'sha256':digest(folder/rel),
                'midpoint_time_saved':float(F(f['time_exact'])),'construction_and_audit_seconds':time.monotonic()-tic}
            graphs+=1;guard(f'sealed-geometry-{graphs}/32')
        save(folder/'graph-index.json',index)
        save(out/'coefficient-seal.json',{'graphs':32,'all32_sealed_before_mass_scalar_oldstates_reference':True,
            'internal_material_faces_given_to_builder':False,'index_sha256':digest(folder/'graph-index.json')})
        copies=[]
        for rel in p['saved_data_files']:
            target=out/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OLD/rel,target)
            copies.append({'source':str((OLD/rel).relative_to(ROOT)),'copy':str(target.relative_to(out)),'sha256':digest(target)})
        shutil.copyfile(OLD/'n128/material.npz',folder/'material.npz')
        copies.append({'source':str((OLD/'n128/material.npz').relative_to(ROOT)),'copy':'n128/material.npz','sha256':digest(folder/'material.npz')})
        save(out/'source-copies.json',copies)
        data=arrays(folder/'material.npz');col=arrays(out/'source/n128/column.npz')
        np.testing.assert_array_equal(data['mass'],np.tile(col['mass']/8,8));np.testing.assert_array_equal(data['c0'],np.tile(col['c0'],8))
        run,path=native_run(folder,32,data,index,guard);save(folder/'k32/native-results.json',run);np.savez_compressed(folder/'k32/trajectory-states.npz',**path)
        paths=1;guard('candidate-K32-complete')
        result,common=analyze(out);save(out/'results.json',result);np.savez_compressed(out/'common-observations.npz',**common)
        save(out/'preservation.json',preserve());guard('comparison-complete')
        save(out/'execution.json',{'first_attempt_passed':True,'exit_code':0,'elapsed_seconds':time.monotonic()-started,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'new_graphs':graphs,'new_primary_paths':paths,
            'new_independent_paths':paths,'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_software_checks':0,'published_or_pushed':False})
        print(json.dumps(plain_json(result),indent=2),flush=True)
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
            'elapsed_seconds':time.monotonic()-started,'completed_geometries':graphs,'completed_candidate_paths':paths});raise
    finally:signal.alarm(0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
