"""Audit one smooth snapshot by independent half resistances and nodewise sums."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import inspect
import json
import numpy as np

from audit_route_decision import ROOT,read,save,digest,preservation
from smooth_auxiliary_fv_study import SOURCE,record,evaluate
from auxiliary_tensor_fv import build_auxiliary_faces


def from_record(v):return F(int(v['numerator_hex'],16),int(v['denominator_hex'],16))


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    spec=read(SOURCE/'next-gate-spec.json');cfg=read(out/'config.json');execution=read(out/'execution.json')
    assert cfg['protocol']==spec and cfg['protocol_sha256']==digest(SOURCE/'next-gate-spec.json')
    assert (out/'source-gate-spec.json').read_bytes()==(SOURCE/'next-gate-spec.json').read_bytes()
    assert cfg['source_manifest_entries_verified']==1769
    assert read(out/'tests.json')=={'successful':True,'checks_run':5,'failures':0,'errors':0,'skipped':0}
    first=read(out/'tests-first-attempt.json');assert not first['successful'] and first['errors']==1
    assert 'TypeError' in (out/'tests-first-attempt.log').read_text()
    assert execution['test_attempts']==2 and not execution['first_test_attempt_passed']
    assert execution['main_study_first_attempt_passed'] and not execution['backend_modified_after_initial_test_error']
    assert execution['study_exit_code']==execution['tests_exit_code']==0
    assert execution['elapsed_main_seconds']<600 and execution['peak_RSS_MiB']<512
    seal=read(out/'coefficient-seal.json');cache=read(out/'coefficient-cache.json')
    assert seal['coefficient_cache_sha256']==digest(out/'coefficient-cache.json')
    assert seal['builder_source_sha256']==cfg['builder_source_sha256']==digest(ROOT/'research/reconnect/auxiliary_tensor_fv.py')
    assert seal['actual_scalar_and_reference_not_loaded_until_after_this_seal']
    assert not seal['builder_received_mass_scalar_reference_true_geometry']
    assert list(inspect.signature(build_auxiliary_faces).parameters)==['ids','positions','rho','D','periods','thickness']
    with np.load(out/'original-state.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(out/'builder-inputs.npz',allow_pickle=False) as z:
        for k in z.files:assert record(z[k])==record(data[k])==seal['builder_array_records'][k]
    p=read(out/'input-provenance.json');assert p['original_state_npz_sha256']==digest(out/'original-state.npz')
    assert p['source_npz_sha256']==digest(ROOT/spec['source_npz'])==spec['source_npz_sha256']
    with np.load(ROOT/spec['source_npz'],allow_pickle=False) as z:
        for k,v in data.items():
            original=z[k][spec['snapshot_index']] if k in ['positions','rho'] else z[k]
            assert record(v)==record(original)==p['array_records'][k],k
    assert np.array_equal(data['ids'],np.arange(128**2))
    D=F(spec['D_exact_of_binary']);rho=list(map(lambda v:F(float(v)),data['rho']))
    mass=list(map(lambda v:F(float(v)),data['mass']));state=list(map(lambda v:F(float(v)),data['c0']))
    points=[tuple(F(float(v)) for v in row) for row in data['positions']]
    x=[points[i][0] for i in range(128)];y=[points[i*128][1] for i in range(128)]
    xwidth=[];ywidth=[]
    for values,output in [(x,xwidth),(y,ywidth)]:
        for i,v in enumerate(values):
            before=values[i-1];after=values[(i+1)%128]
            if before>=v:before-=1
            if after<=v:after+=1
            output.append((after-before)/2)
    rows=[[] for _ in mass];seen=set()
    assert len(cache['edges'])==len(cache['conductances_exact'])==32768
    for k,(i,j) in enumerate(cache['edges']):
        assert 0<=i<j<len(mass) and (i,j) not in seen;seen.add((i,j))
        assert cache['edge_ids'][k]==[int(data['ids'][i]),int(data['ids'][j])]
        axis=cache['axes'][k]
        if axis==0:
            assert points[i][1]==points[j][1]
            delta=abs(points[i][0]-points[j][0]);d=min(delta,1-delta);length=ywidth[i//128]
        else:
            assert axis==1 and points[i][0]==points[j][0]
            delta=abs(points[i][1]-points[j][1]);d=min(delta,1-delta);length=xwidth[i%128]
        assert 0<d<F(1,2) and length>0
        expected=length/(d/(2*rho[i]*D)+d/(2*rho[j]*D))
        g=F(cache['conductances_exact'][k]);assert g==expected>0
        assert F(cache['center_distances_exact'][k])==d and F(cache['face_lengths_exact'][k])==length
        rows[i].append((j,g));rows[j].append((i,g))
    assert all(len(row)==4 for row in rows)
    q=tuple(sum((g*(state[j]-state[i]) for j,g in row),F(0))/mass[i] for i,row in enumerate(rows))
    assert list(map(str,q))==read(out/'rates-exact.json')['exact_rates']
    result=read(out/'results.json');assert sum((m*v for m,v in zip(mass,q)),F(0))==from_record(result['exact_shared_mass_rate_sum'])==0
    energy=-sum((m*u*v for m,u,v in zip(mass,state,q)),F(0))
    assert energy==from_record(result['exact_graph_dissipation'])>=0
    assert from_record(result['exact_energy_pairing_residual'])==0
    graph=build_auxiliary_faces(data['ids'],data['positions'],data['rho'],spec['D'],spec['periods'],spec['unit_thickness'])
    regenerated,controls,arrays,exact=evaluate(graph,data)
    assert regenerated==result and controls==read(out/'action-controls.json') and exact==q
    with np.load(out/'actions.npz',allow_pickle=False) as z:
        for k,v in arrays.items():assert np.array_equal(z[k],v),k
    assert result['comparison_decision']=='FV_better_than_saved_kernel_resolvably'
    assert result['error_reduction']>result['combined_comparison_indicator_relative']
    prior=read(out/'source-kernel-results.json')
    assert abs(prior['relative_mass_weighted_instantaneous_rate_error']-result['saved_kernel_relative_rate_error_recomputed'])<=64*np.finfo(float).eps
    s=read(out/'stage3-status.json');assert s['recorded_checks_total']==428+5==433
    assert s['new_actual_frozen_state_cases']==s['new_candidate_primary_actions']==s['independent_float_action_controls']==1
    assert s['new_candidate_graphs']==1 and s['new_candidate_conductances']==32768
    for k in ['new_candidate_transient_runs','new_PDE_reference_solves','new_kernel_builds','new_density_path_builds','new_global_matrices','new_exact_algebra_certificate_groups']:
        assert s[k]==0,k
    for obj in [s,result]:
        assert not obj['general_nontensor_smooth_flow_admitted'] and not obj['finite_node_candidate_admitted']
        assert not obj['contact_transient_physically_admitted'] and not obj['fluid_orc_enabled']
        assert obj['stage3_physical_acceptance']=='incomplete'
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior_manifests=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
        {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),
            'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in prior_manifests:
        if item['manifest'] in known:continue
        path=ROOT/item['manifest'];assert digest(path)==item['manifest_sha256']
        entries=read(path);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==78
    save(out/'preservation.json',preserved)
    audit={'original_saved_state_input_and_reference_arrays_verified_by_bytes':True,
        'state_independent_coefficient_seal_and_builder_signature_verified':True,
        'all32768_coefficients_verified_by_independent_half_resistances':True,
        'all_exact_rates_verified_by_independent_nodewise_sums':True,
        'mass_energy_and_raw_float_metrics_replay_verified':True,
        'same_normalization_saved_kernel_comparison_and_controls_verified':True,
        'one_frozen_smooth_case_not_general_flow_or_stage3_acceptance':True,
        'initial_test_exception_expectation_failure_preserved':True,'new_software_checks':5,'recorded_checks_total':433,
        'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':78,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'frozen-audit.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);args=p.parse_args();main(args.directory)
