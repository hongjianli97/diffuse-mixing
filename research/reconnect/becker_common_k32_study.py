"""Only32/K32 and64/K32 are new; every graph sealed before scalar loading."""
from pathlib import Path
from fractions import Fraction as F
import argparse, json, resource, shutil, signal, time, traceback
import numpy as np
import becker_point_core
from audit_route_decision import ROOT, read, digest
from audit_becker_spatial import save
from becker_point_core import Becker, native_run
from bounded_periodic_voronoi import build_bounded_faces
from audit_bounded_geometry import audit_geometry
from aligned_becker_check import arrays
from becker_fine_time_core import sample_frames_k32, verify_sampling_rule
from becker_common_k32_core import analyze
from stage3_evidence_review import preserve as preserve_older

PARENT = ROOT/'runs/reconnect-stage3-evidence'
OLD = ROOT/'runs/reconnect-becker-spatial'
PROTOCOL = ROOT/'research/reconnect/becker_common_k32_protocol.json'


def main(output):
    out = Path(output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    graphs = paths = 0
    def guard(stage,i=0,total=0):
        elapsed = time.monotonic()-started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed > 900 or peak > 1024:
            raise RuntimeError(f'registered cap exceeded: {elapsed}s/{peak}MiB')
        if not total or i == total:
            print(f'{stage}: {i}/{total}; {elapsed:.3f}s; peak {peak:.2f}MiB',flush=True)
    def alarm(*_):
        raise TimeoutError('registered900s main limit')
    signal.signal(signal.SIGALRM,alarm)
    signal.alarm(900)
    try:
        p = read(PROTOCOL)
        gate = read(PARENT/'next-gate-spec.json')
        assert gate['type'] == 'new_common_K32_three_column_spatial_diagnostic_with_all_H_outcomes_retained'
        assert not gate['already_executed'] and gate['main_wall_seconds_cap'] == 900
        assert digest(PARENT/'next-gate-spec.json') == p['source_gate_sha256']
        for name,h in p['source_files'].items():
            assert digest(ROOT/name) == h,name
        for name,h in read(PARENT/'sha256.json').items():
            assert digest(ROOT/name) == h,name
        shutil.copyfile(PROTOCOL,out/'protocol.json')
        shutil.copyfile(PARENT/'next-gate-spec.json',out/'source-gate-spec.json')
        save(out/'implementation-lock.json',{'files': {name:digest(ROOT/name) for name in p['implementation_files']},
            'fixed_before_new_labels_points_graphs_mass_scalar_states': True})
        rule = verify_sampling_rule()
        assert becker_point_core.COUNTS == (4,8,16)
        preflight = []
        for N in (32,64):
            folder = out/f'n{N}'
            (folder/'k32').mkdir(parents=True)
            with np.load(OLD/f'n{N}/column.npz',allow_pickle=False) as z:
                labels = z['mass_edges'].copy()
            assert labels.shape == (N+1,)
            np.save(folder/'source-mass-labels.npy',labels)
            world = sample_frames_k32(labels,Becker(delta=.08),rows=8)
            save(folder/'pure-inputs.json',world)
            old_world = read(ROOT/p['old_pure_input_files'][str(N)])
            old_frames = old_world['cases']['16']['frames']
            frames = world['cases']['32']['frames']
            assert len(frames) == 65 and all(frames[2*i] == f for i,f in enumerate(old_frames))
            preflight.append({'columns':N, 'old_K16_shared33_frames_bitwise_equal':True,
                              'mass_c0_reference_oldstates_loaded':False})
            del old_world,old_frames
            index = {}
            for i in range(32):
                tic = time.monotonic()
                f = frames[2*i+1]
                packet = build_bounded_faces(world['ids'],f['positions'],f['rho'],f['D'],f['x_bounds'],progress=guard)
                rel = f'k32/geometry-step{i:02d}.json'
                save(folder/rel,packet)
                _,_,_,certificate = audit_geometry(packet,f['positions'],f['rho'],world['ids'],f['D'],f['x_bounds'])
                save(folder/f'k32/geometry-audit-step{i:02d}.json',{'cache_sha256':digest(folder/rel),
                    'pure_inputs_sha256':digest(folder/'pure-inputs.json'),
                    'audit_source_sha256':digest(ROOT/'research/reconnect/audit_bounded_geometry.py'),
                    'mass_c0_reference_loaded':False,'certificate':certificate})
                index[f'k32-step{i:02d}'] = {'K':32,'step':i,'file':rel,'sha256':digest(folder/rel),
                    'midpoint_time_saved':float(F(f['time_exact'])),
                    'construction_and_audit_seconds':time.monotonic()-tic}
                graphs += 1
                guard(f'sealed-geometry-n{N}-{i+1}/32')
            save(folder/'graph-index.json',index)
        assert graphs == 64 and becker_point_core.COUNTS == (4,8,16)
        save(out/'labels-only-preflight.json',{'sampling_rule':rule,'grids':preflight,'only_K32_sampled':True})
        save(out/'coefficient-seal.json',{'graphs':64,'all64_sealed_before_mass_scalar_oldstates_reference':True,
            'internal_material_faces_given_to_builder':False,
            'index_sha256':{str(N):digest(out/f'n{N}/graph-index.json') for N in (32,64)}})
        copies = []
        for item in p['saved_data_files']:
            target = out/item['copy']
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/item['source'],target)
            copies.append({**item,'sha256':digest(target)})
        save(out/'source-copies.json',copies)
        for N in (32,64):
            folder = out/f'n{N}'
            data = arrays(folder/'material.npz')
            col = arrays(out/f'source/old/n{N}/column.npz')
            np.testing.assert_array_equal(data['mass'],np.tile(col['mass']/8,8))
            np.testing.assert_array_equal(data['c0'],np.tile(col['c0'],8))
            run,path = native_run(folder,32,data,read(folder/'graph-index.json'),guard)
            save(folder/'k32/native-results.json',run)
            np.savez_compressed(folder/'k32/trajectory-states.npz',**path)
            paths += 1
            guard(f'candidate-n{N}-K32-complete')
        result,common = analyze(out)
        save(out/'results.json',result)
        np.savez_compressed(out/'common-observations.npz',**common)
        preservation = preserve_older()
        preservation['formal_manifests_verified'].append({'manifest':str((PARENT/'sha256.json').relative_to(ROOT)),
            'manifest_sha256':digest(PARENT/'sha256.json'),'entries_verified':len(read(PARENT/'sha256.json'))})
        assert len(preservation['formal_manifests_verified']) == 111
        save(out/'preservation.json',preservation)
        guard('comparison-complete')
        save(out/'execution.json',{'completed':True,'exit_code':0,'elapsed_seconds':time.monotonic()-started,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'new_graphs':graphs,'new_primary_paths':paths,'new_independent_paths':paths,
            'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_software_checks':0,'published_or_pushed':False})
        print(result['comparison_decision'],flush=True)
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
            'elapsed_seconds':time.monotonic()-started,'completed_geometries':graphs,'completed_candidate_paths':paths})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
