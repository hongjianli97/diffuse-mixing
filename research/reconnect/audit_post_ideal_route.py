"""Audit interpretation, time-reference matching and unexecuted action plan."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np

from audit_route_decision import ROOT,read,save,digest,preservation
from post_ideal_route_review import SOURCE,GLOBAL,REFERENCE,FV,record,replay


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists();replay(out)
    cfg=read(out/'config.json');p=read(out/'input-provenance.json')
    assert cfg['parent_status_sha256']==digest(SOURCE/'stage3-status.json')
    assert (out/'source-stage3-status.json').read_bytes()==(SOURCE/'stage3-status.json').read_bytes()
    assert p['source_point_inputs_sha256']==digest(SOURCE/'original-inputs.npz')
    assert p['source_PDE_T_reference_sha256']==digest(REFERENCE/'reference-T0.05.npz')
    assert p['source_FV_T_sha256']==digest(FV/'FV-T0.05.npz')
    assert p['source_coefficient_cache_sha256']==digest(GLOBAL/'coefficient-cache.json')
    with np.load(out/'matched-time-reference-inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(SOURCE/'original-inputs.npz',allow_pickle=False) as z:
        for k in ['ids','mass','c0','positions','rho','near_contact','far_from_contact']:assert record(z[k])==record(data[k])
    with np.load(REFERENCE/'reference-T0.05.npz',allow_pickle=False) as z:
        for k in ['ids','mass','c0','positions','rho','reference_cbar_T']:assert record(z[k])==record(data[k])
    with np.load(FV/'FV-T0.05.npz',allow_pickle=False) as z:
        for k in ['ids','mass','c0','positions','rho','reference_cbar_T']:assert record(z[k])==record(data[k])
        assert record(z['semidiscrete'])==record(data['saved_FV_semidiscrete_T'])
    assert sum((F(float(v)) for v in data['mass']),F(0))==F(5,2)
    assert np.isfinite(data['c0']).all() and np.all(data['c0']>=0) and np.all(data['c0']<=1)
    cost=read(out/'saved-operator-cost.json');cache=read(GLOBAL/'coefficient-cache.json')
    # Independent incidence-style degree sum, not the planner's incoming-column lookup.
    g=tuple(map(F,cache['conductances_exact']));weights=[F(0)]*128
    for c in range(128):
        for k,(dx,dy) in enumerate(cache['offsets_positive_half']):
            w=g[cache['coefficient_map'][c][k]];weights[c]+=w;weights[(c+dx)%128]+=w
    rates=[w/F(float(data['mass'][c])) for c,w in enumerate(weights)]
    assert list(map(str,rates))==cost['outgoing_rates_exact']
    maximum=max(rates);Lam=cost['uniformization_lambda_integer']
    assert maximum==F(cost['maximum_outgoing_rate_exact']) and F(Lam)>=maximum>Lam-1
    assert Lam==168
    z=F(Lam,20);M=cost['Poisson_max_degree_predeclared'];tol=F(cost['Poisson_tail_tolerance_absolute_concentration_exact'])
    def tail(n):
        first=F(1)
        for k in range(1,n+2):first*=z/k
        assert z<n+2
        return first/(1-z/F(n+2))
    bound=tail(M)
    assert bound==F(cost['Poisson_tail_bound_exact'])<=tol<tail(M-1)
    assert M==47 and cost['Poisson_terms_0_through_M']==48
    assert F(cost['mass_norm_tail_bound_exact'])==F(8,5)*bound
    decision=read(out/'route-decision.json');ledger=read(out/'route-ledger.json');gate=read(out/'next-gate-spec.json');status=read(out/'stage3-status.json')
    assert len(ledger)==8 and decision['interpretation_completed']
    assert decision['candidate_coefficients_frozen'] and decision['global_combined_quadrature_projection_split'].startswith('still not measured')
    pair=read(SOURCE/'results.json');full=read(GLOBAL/'results.json')
    assert decision['point_vs_ideal_pair_ordering_kept']=={k:pair[k] for k in decision['point_vs_ideal_pair_ordering_kept']}
    assert decision['global_frozen_improvement_kept']==full['reconstructed_global_error']
    assert gate==status['next_finite_time_research_gate_predeclared'] and not gate['already_executed']
    assert gate['T_exact']==read(REFERENCE/'stage3-status.json')['T_exact']=='1/20'
    assert gate['initial_fluctuation_norm_locked']==read(FV/'results.json')['initial_fluctuation_norm']
    assert gate['FV_semidiscrete_error_context_locked']==read(FV/'results.json')['spatial_semidiscrete_relative_error']
    assert gate['all_128_transverse_modes_retained_including_roundoff_size_modes']
    assert gate['raw_eigenvalues_including_roundoff_zero_mode_retained']
    assert gate['no_Poisson_weight_sum_renormalization_no_concentration_clipping']
    assert gate['no_new_PDE_solve_or_spectral_reference_evaluation'] and gate['new_full_propagator_matrices']==0
    assert gate['research_only_explicit_new_gate_not_physical_admission']
    for obj in [gate,status]:
        assert not obj['finite_node_candidate_admitted'] and not obj['fluid_orc_enabled']
        assert not obj['contact_transient_physically_admitted']
    assert status['point_kernel_finite_time_result']=='not_computed' and status['recorded_checks_total']==423
    for key in ['new_software_checks','previous_checks_rerun','new_candidate_weights','new_kernel_or_path_evaluations',
        'new_operator_actions','new_quadrature_evaluations','new_Fourier_transforms_or_eigensolves','new_candidate_graphs',
        'new_numerical_or_flow_cases','new_candidate_transient_runs','new_PDE_reference_solves','new_exact_algebra_certificate_groups']:
        assert status[key]==0,key
    for name in ['scipy','cached_global_exchange','ideal_material_pair_reference','cartesian_fv_modes',
        'oracle_path_reference','contact_spectral_reference','contact_finite_time_study']:
        assert name not in sys.modules,name
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']
    prior=prior+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
        'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in prior:
        if item['manifest'] in known:continue
        path=ROOT/item['manifest'];assert digest(path)==item['manifest_sha256']
        entries=read(path);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==72
    save(out/'preservation.json',preserved)
    audit={'parent_read_only_scope_and_route_decision_verified':True,
        'same_original_state_masses_ids_positions_density_and_T_reference_match_by_bytes':True,
        'saved_FV_time_context_reused_without_resolving_it':True,
        'degree_incidence_check_of_original_cached_coefficients_passed':True,
        'a_priori_Poisson_tail_and_minimal_fixed_degree_verified_without_exponential_action':True,
        'research_time_gate_distinguished_from_physical_admission_and_not_executed':True,
        'all_modes_raw_eigenvalues_and_unrenormalized_tail_contract_verified':True,
        'no_candidate_kernel_graph_FFT_eigensolver_quadrature_proof_or_propagator_executed':True,
        'new_software_checks':0,'recorded_checks_total':423,'previous_checks_rerun':0,
        'new_candidate_weights':0,'new_candidate_transient_runs':0,'new_PDE_reference_solves':0,
        'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':72,
        'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'review-audit.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True)
    main(parser.parse_args().directory)
