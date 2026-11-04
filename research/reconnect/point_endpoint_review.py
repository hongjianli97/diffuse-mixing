"""Interpret saved endpoint variance by exact binary-data algebra; no operators."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import math
import shutil
import sys
import numpy as np

from audit_route_decision import ROOT,read,save,digest

SOURCE=ROOT/'runs/reconnect-point-fixed-time'
BLOCKED=['scipy','saved_exchange_time','point_fixed_time_study','cached_global_exchange',
    'oracle_path_reference','cartesian_fv_modes','contact_spectral_reference','ideal_material_pair_reference']
ZERO_FIELDS=['new_software_checks','previous_checks_rerun','new_candidate_weights','new_kernel_or_path_evaluations',
    'new_operator_actions','new_quadrature_evaluations','new_Fourier_transforms_or_eigensolves',
    'new_candidate_graphs','new_numerical_or_flow_cases','new_candidate_transient_runs',
    'new_PDE_reference_solves','new_exact_algebra_certificate_groups']


def record(a):
    return {'shape':list(a.shape),'dtype':a.dtype.str,'C_order_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def fractions(a):return [F(float(v)) for v in a]
def inner(m,u,v):return sum((a*b*c for a,b,c in zip(m,u,v)),F(0))
def number(q):return {'exact_of_saved_binary_data':str(q),'display':float(q)}


def evaluate(data,saved,controls,fv):
    m=fractions(data['mass']);c0=fractions(data['c0']);w=fractions(data['reference_cbar_T'])
    M=sum(m,F(0));S0=inner(m,c0,[F(1)]*len(m));Sw=inner(m,w,[F(1)]*len(m))
    E0=inner(m,[v-S0/M for v in c0],[v-S0/M for v in c0])/2
    centered_w=[v-Sw/M for v in w];Ew=inner(m,centered_w,centered_w)/2
    assert E0==F(saved['initial_resolved_variance_E_exact'])
    assert Ew==F(saved['outputs']['PDE_reference']['resolved_material_variance_E_exact'])
    assert data['near_contact'].dtype==data['far_from_contact'].dtype==np.dtype(bool)
    assert np.array_equal(~data['near_contact'],data['far_from_contact'])
    methods={}
    ref_eta=controls['reference_indicator_mass_norm']
    for label,key,source_key,eta in [
        ('point','point_cbar_T','point',controls['solver_indicator_mass_norm']),
        ('saved_FV','saved_FV_semidiscrete_T','saved_FV_context',
            fv['solver_indicator_over_initial_fluctuation']*saved['initial_fluctuation_norm_locked'])]:
        v=fractions(data[key]);e=[a-b for a,b in zip(v,w)]
        Sv=inner(m,v,[F(1)]*len(m));delta_mean=(Sv-Sw)/M
        cross=inner(m,centered_w,e);quad=inner(m,e,e)/2;mean_correction=-M*delta_mean**2/2
        Ev=inner(m,[q-Sv/M for q in v],[q-Sv/M for q in v])/2
        assert Ev==F(saved['outputs'][source_key]['resolved_material_variance_E_exact'])
        residual=Ev-Ew-cross-quad-mean_correction
        assert residual==0
        H_bias=-(Ev-Ew)/E0
        sigma=saved['initial_fluctuation_norm_locked']
        error_exact_gram_display=math.sqrt(float(2*quad))/sigma
        H_indicator=((math.sqrt(float(2*Ev))*eta+.5*eta**2)+
            (math.sqrt(float(2*Ew))*ref_eta+.5*ref_eta**2))/float(E0)
        regions={}
        for name in ['near_contact','far_from_contact']:
            inds=np.flatnonzero(data[name]);regional_cross=sum((m[i]*centered_w[i]*e[i] for i in inds),F(0))
            regional_quad=sum((m[i]*e[i]**2/2 for i in inds),F(0))
            regions[name]={'cells':len(inds),'signed_cross_term':number(regional_cross),
                'half_error_squared':number(regional_quad),
                'H_cross_contribution':number(-regional_cross/E0),'H_square_contribution':number(-regional_quad/E0)}
        assert sum((F(v['signed_cross_term']['exact_of_saved_binary_data']) for v in regions.values()),F(0))==cross
        assert sum((F(v['half_error_squared']['exact_of_saved_binary_data']) for v in regions.values()),F(0))==quad
        methods[label]={
            'signed_cross_term':number(cross),'half_error_squared':number(quad),'raw_mean_correction':number(mean_correction),
            'raw_mean_difference':number(delta_mean),'endpoint_variance_difference':number(Ev-Ew),
            'exact_identity_residual':str(residual),'H_bias_from_exact_saved_data':number(H_bias),
            'H_cross_contribution':number(-cross/E0),'H_square_contribution':number(-quad/E0),
            'H_mean_contribution':number(-mean_correction/E0),
            'saved_H_subtraction_display':saved['outputs'][source_key]['H']-saved['outputs']['PDE_reference']['H'],
            'exact_gram_relative_error_display':error_exact_gram_display,
            'saved_relative_error_unchanged':saved['outputs'][source_key]['error_over_original_initial_fluctuation'],
            'norm_display_difference_not_used_to_replace_saved_value':error_exact_gram_display-saved['outputs'][source_key]['error_over_original_initial_fluctuation'],
            'resolved_variance_loss_from_initial':number(E0-Ev),'H_sensitivity_indicator_from_saved_state_indicators':H_indicator,
            'H_sensitivity_indicator_not_certified_physical_bound':True,'original_regions':regions}
    return {'scope':'exact algebra on saved binary material-mean endpoint values; no new evolution',
        'mass_exact':str(M),'initial_variance_exact':str(E0),'reference_variance_exact':str(Ew),
        'formula':'E(v)-E(w)=<w-mean(w),v-w>_m+0.5*||v-w||_m^2-0.5*M*(mean(v)-mean(w))^2',
        'methods':methods,'no_means_or_endpoints_modified':True,'regions_are_observations_not_causal_mechanism_fractions':True,
        'variance_loss_not_claimed_as_continuous_PDE_integrated_dissipation':True,
        'source_field_error_ordering_unchanged':saved['comparison_decision'],
        'reference_bridge_requires_explicit_cellwise_constant_perturbation_lift':True}


def route_decision():
    return {'current_path_model':'freeze as a research benchmark; end this fixed-contact propagation line without retuning',
        'ordinary_central_resistance_FV':'retain limited geometry baseline; no new FV grid/time study',
        'mass_average_resistance_flux':'experimental comparison only',
        'density_path_weak_results':'retain conditional theorems and exact primitives; do not reprove completed work',
        'ideal_material_pair_reference':'closed two-row diagnostic; no order/global expansion or weight replacement',
        'common_cross_phase_scalar_or_negative_offdiagonal_repair':'remain closed within previously stated families',
        'all_positive_exchange_methods':'not ruled out; no feasible new formula obtained here',
        'finite_time_causal_split':'not identifiable from these endpoints and one initial action; Duhamel requires residual history',
        'mixing_bias_interpretation':'signed endpoint variance identity, not mechanism allocation or full subcell dissipation',
        'next_work':'theory-only definition of a distinct material-average closure and its allowed inputs before selecting any numerical gate',
        'new_numerical_gate_selected':False,'no_repeated_read_only_repackaging_as_next_scientific_task':True,
        'finite_node_candidate_admitted':False,'contact_transient_physically_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}


def replay(directory):
    out=Path(directory);p=read(out/'input-provenance.json')
    assert digest(out/'saved-endpoints.npz')==p['copied_packet_sha256']
    with np.load(out/'saved-endpoints.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    for k,v in data.items():assert record(v)==p['array_records'][k],k
    result=evaluate(data,read(out/'source-results.json'),read(out/'source-solver-controls.json'),read(out/'source-FV-results.json'))
    assert result==read(out/'variance-budget.json') and route_decision()==read(out/'route-decision.json')
    for name in BLOCKED:assert name not in sys.modules,name
    print('Exact saved endpoint variance/mean/region identities and route statements replay; no graph, solver or physical case executed.',flush=True)


def main(output):
    parent=read(SOURCE/'sha256.json')
    for rel,h in parent.items():assert digest(ROOT/rel)==h,rel
    status=read(SOURCE/'stage3-status.json')
    assert status['recorded_checks_total']==428 and status['next_work'].startswith('read-only interpretation')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',{'scope':'read-only endpoint variance algebra and final route disposition',
        'source_manifest_entries_verified':len(parent),'source_manifest_sha256':digest(SOURCE/'sha256.json'),
        'old_operators_theorems_tests_and_time_solvers_not_executed':True})
    for target,src in [('source-results.json',SOURCE/'results.json'),('source-solver-controls.json',SOURCE/'solver-controls.json'),
        ('source-FV-results.json',SOURCE/'source-FV-results.json'),('source-stage3-status.json',SOURCE/'stage3-status.json')]:
        shutil.copyfile(src,out/target)
    with np.load(SOURCE/'original-inputs.npz',allow_pickle=False) as z:
        data={k:z[k].copy() for k in ['ids','mass','c0','positions','rho','near_contact','far_from_contact','reference_cbar_T','saved_FV_semidiscrete_T']}
    with np.load(SOURCE/'point-T0.05.npz',allow_pickle=False) as z:data['point_cbar_T']=z['point_semidiscrete_T'].copy()
    np.savez_compressed(out/'saved-endpoints.npz',**data)
    save(out/'input-provenance.json',{'source_original_inputs_sha256':digest(SOURCE/'original-inputs.npz'),
        'source_point_endpoint_sha256':digest(SOURCE/'point-T0.05.npz'),'copied_packet_sha256':digest(out/'saved-endpoints.npz'),
        'array_records':{k:record(v) for k,v in data.items()},'all_endpoint_and_original_input_members_copied_without_modification':True})
    result=evaluate(data,read(out/'source-results.json'),read(out/'source-solver-controls.json'),read(out/'source-FV-results.json'))
    save(out/'variance-budget.json',result);save(out/'route-decision.json',route_decision())
    save(out/'stage3-status.json',{'updated':'2026-11-01','timezone':'Etc/UTC',**{k:0 for k in ZERO_FIELDS},
        'recorded_checks_total':428,'saved_endpoint_algebra_only':True,'fixed_contact_point_propagation_line_paused':True,
        'new_numerical_gate_selected':False,'finite_node_candidate_admitted':False,'contact_transient_physically_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'next_work':route_decision()['next_work'],'pdf_generated':False,'software_release_created':False,'published':False})
    replay(out)
    for label,v in result['methods'].items():
        print(label,{k:v[k] for k in ['H_cross_contribution','H_square_contribution','H_mean_contribution','H_bias_from_exact_saved_data','H_sensitivity_indicator_from_saved_state_indicators']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--replay');args=p.parse_args()
    if args.replay:replay(args.replay)
    elif args.output:main(args.output)
    else:p.error('--output or --replay required')
