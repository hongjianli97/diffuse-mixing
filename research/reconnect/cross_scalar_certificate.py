"""Exact compatibility certificates from two cached rows; no parameter runs.

Fraction.from_float interprets saved binary coefficients exactly. NumPy only
loads the snapshot and records floating reductions. No new weights, diagonal,
matrix, kernel quadrature or scalar-propagation result is computed.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse,json,platform
import numpy as np
from audit_contact_constitutive import digest
from package_compression_midpoint import dump


def exact(v):return F.from_float(float(v))


def rational(v):return {'exact':str(v),'decimal_for_reading_only':float(v)}


def decompose(arrays,side,target):
    get=lambda key:arrays[side+'_'+key]
    cols=get('row_columns');L=get('saved_L_row');u=get('probe_quadratic_mass_means');rho=get('original_rho0_point')
    x=get('normal_centers');m=get('original_mass');at=int(np.flatnonzero(cols==target)[0])
    mass=exact(m[at]);ut=exact(u[at]);ell=F(1,128);origin=F(1,2)
    physical=[(exact(value)-origin)**2/2+ell**2/24 for value in x];physical_target=physical[at]
    off=cols!=target;same=off&(rho==rho[at]);cross=off&(rho!=rho[at])
    assert np.all((same|cross)==off) and not np.any(same&cross)
    A=F(0);C=F(0);Ap=F(0);Cp=F(0);binary_raw=F(0);physical_raw=F(0)
    for k in range(len(cols)):
        q=-exact(L[k])/mass
        binary_raw+=q*exact(u[k]);physical_raw+=q*physical[k]
        if not off[k]:continue
        contribution=q*(exact(u[k])-ut);physical_contribution=q*(physical[k]-physical_target)
        if same[k]:A+=contribution;Ap+=physical_contribution
        else:C+=contribution;Cp+=physical_contribution
    sigma=-sum((exact(value) for value in L),F(0))/mass
    bias=ut*sigma;physical_bias=physical_target*sigma
    assert binary_raw==A+C+bias and physical_raw==Ap+Cp+physical_bias
    terms=get('centered_action_terms_per_s')
    Af=float(terms[same].sum());Cf=float(terms[cross].sum())
    floating={'same_phase_centered_contribution_per_s':Af,'cross_phase_centered_contribution_per_s':Cf,
        'sum_phase_contributions_per_s':Af+Cf,
        'phase_sum_minus_saved_centered_action_per_s':Af+Cf-float(terms.sum()),
        'float_A_minus_exact_A_per_s':Af-float(A),'float_C_minus_exact_C_per_s':Cf-float(C)}
    records={'target_index':target,'original_mass':rational(mass),'target_rho':rational(exact(rho[at])),
        'same_phase_offdiagonal_entries':int(same.sum()),'cross_phase_offdiagonal_entries':int(cross.sum()),
        'A_same_phase':rational(A),'C_cross_phase':rational(C),'exact_centered_original_response':rational(A+C),
        'exact_saved_diagonal_constant_defect_per_s':rational(sigma),'exact_target_times_diagonal_defect_per_s':rational(bias),
        'exact_raw_CSR_on_cached_binary_means':rational(binary_raw),
        'physical_quadratic_A':rational(Ap),'physical_quadratic_C':rational(Cp),
        'physical_quadratic_exact_raw_CSR':rational(physical_raw),
        'cached_mean_minus_exact_physical_mean_max':rational(max(abs(exact(v)-p) for v,p in zip(u,physical))),
        'floating_reductions':floating}
    masks={'same_phase_mask':same,'cross_phase_mask':cross,'offdiagonal_mask':off}
    return records,masks,(A,C,bias),(Ap,Cp,physical_bias)


def compatibility(left,right,D):
    AL,CL,bL=left;AR,CR,bR=right
    assert CL>0 and CR>0
    determinant=(D-AL)*CR-(D-AR)*CL
    # Elimination: CR*eL-CL*eR=-determinant for every symbolic t.
    lower=abs(determinant)/(CL+CR)
    raw_det=(D-AL-bL)*CR-(D-AR-bR)*CL
    raw_lower=abs(raw_det)/(CL+CR)
    return {'centered_constraint':'A_i + t*C_i = D; one shared t >= 0, symbolic only',
        'determinant':rational(determinant),'necessary_left_constraint_ratio':rational((D-AL)/CL),
        'necessary_right_constraint_ratio':rational((D-AR)/CR),
        'one_shared_scalar_compatible':determinant==0,
        'universal_max_absolute_error_lower_per_s':rational(lower),
        'universal_max_relative_error_lower':rational(lower/D),
        'same_saved_diagonal_defects_retained_determinant':rational(raw_det),
        'same_saved_diagonal_defects_retained_universal_max_error_lower_per_s':rational(raw_lower),
        'same_saved_diagonal_defects_retained_relative_lower':rational(raw_lower/D),
        'source_biases_are_not_fitted_or_used_to_fix_rows':True}


def certificates(snapshot):
    with np.load(snapshot) as z:arrays={key:z[key] for key in z.files}
    rows={};tuples=[];physical=[];derived={}
    for side,target in [('left',63),('right',64)]:
        rows[side],masks,t,p=decompose(arrays,side,target);tuples.append(t);physical.append(p)
        for key,value in masks.items():derived[side+'_'+key]=value
    D=F(1,50);AL,CL,bL=tuples[0];AR,CR,bR=tuples[1]
    assert AL==AR and CL==4*CR and 0<AL<D and CR>0
    assert F(rows['right']['original_mass']['exact'])==4*F(rows['left']['original_mass']['exact'])
    proof=compatibility(tuples[0],tuples[1],D)
    physical_proof=compatibility(physical[0],physical[1],D)
    lower=F(proof['universal_max_absolute_error_lower_per_s']['exact'])
    assert lower==F(3,5)*(D-AL)>0
    assert F(proof['determinant']['exact'])!=0 and not proof['one_shared_scalar_compatible']
    assert F(proof['same_saved_diagonal_defects_retained_determinant']['exact'])!=0
    assert physical[0][0]==physical[1][0] and physical[0][1]==4*physical[1][1]
    assert not physical_proof['one_shared_scalar_compatible']
    for side,side_source in [('left',tuples[0]),('right',tuples[1])]:
        raw=float(rows[side]['exact_raw_CSR_on_cached_binary_means']['decimal_for_reading_only'])
        previous_terms=arrays[side+'_raw_CSR_action_terms_per_s']
        rows[side]['floating_reductions']['saved_float_term_sum_minus_exact_raw_per_s']=float(previous_terms.sum())-raw
    groups={
        'original_phase_partition_and_mass_ratio':{'verified':True,'mass_ratio_right_over_left':4,'source_row_partitions_disjoint_complete':True},
        'exact_raw_centered_and_original_diagonal_residual_identity':{'verified':True,'identity':'exact raw CSR = A+C+u_target*sigma; original diagonal unchanged'},
        'mirror_affine_coefficients':{'verified':True,'same_phase_A_left_equals_A_right':True,'cross_phase_C_left_over_C_right':rational(CL/CR),
            'positive_reference_gap_D_minus_A':rational(D-AL)},
        'two_centered_constraints_incompatible':{'verified':True,'nonzero_elimination_determinant':proof['determinant'],
            'necessary_ratios_are_not_executed_or_selected_parameters':True},
        'universal_max_error_lower_bound':{'verified':True,'elimination':'CR*eL-CL*eR=-det; max(abs(eL),abs(eR)) >= abs(det)/(CL+CR)',
            'lower_equals_three_fifths_D_minus_A':rational(lower),'relative_lower':proof['universal_max_relative_error_lower']},
        'retained_original_diagonal_defects_do_not_restore_compatibility':{'verified':True,
            'nonzero_determinant':proof['same_saved_diagonal_defects_retained_determinant'],
            'relative_lower':proof['same_saved_diagonal_defects_retained_relative_lower']},
        'independent_exact_material_polynomial_same_no_go':{'verified':True,'cached_binary_means_not_replaced':True,
            'physical_polynomial_determinant':physical_proof['determinant'],'relative_lower':physical_proof['universal_max_relative_error_lower']}}
    assert len(groups)==7 and all(v['verified'] for v in groups.values())
    result={'rows':rows,'D_reference':rational(D),'compatibility':proof,'physical_polynomial_compatibility':physical_proof,
        'universal_max_relative_error_lower_percent':100*float(lower/D),'certificate_groups':groups,'verified_exact_algebra_groups':7,
        'analytic_proofs_are_in_report':'research/reconnect/CROSS_PHASE_SCALAR_COMPATIBILITY.md',
        'new_numerical_flow_or_unit_tests':0,'previous_recorded_checks':363,'recorded_checks_total':363,
        'parameter_values_evaluated_on_a_graph':0,'new_candidate_graphs':0,'new_edge_weights':0,'new_operator_quadrature_evaluations':0,
        'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'finite_node_candidate_admitted':False,'contact_transient_admitted':False,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'scope_limits':['one common cross-phase multiplier, fixed within-phase exchanges/stencil/masses and specified quadratic probe',
            'not impossibility of edge-dependent coefficients, richer input closure or all positive exchange methods',
            'constraint quotients and universal bound are algebraic certificates, not fitted replacements or a parameter scan']}
    return result,arrays,derived


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-interface-pair'
    entries=json.loads((source/'sha256.json').read_text())
    for rel,h in entries.items():assert digest(root/rel)==h,rel
    protocol=json.loads((source/'stage3-status.json').read_text())['next_cross_phase_scalar_compatibility_review_predeclared']
    assert protocol['target_indices']==[63,64] and protocol['fixed_original_masses']
    assert protocol['no_parameter_fitted_or_used_to_replace_edges'] and protocol['no_diagonal_or_moment_repair']
    assert not protocol['parameter_or_mesh_matrix_scan'] and not protocol['fluid_orc_enabled']
    snapshot=root/protocol['source_snapshot'];assert digest(snapshot)==protocol['source_snapshot_sha256']
    result,arrays,derived=certificates(snapshot)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    dump(out/'config.json',{'scope':'predeclared theory-only shared cross-phase scalar compatibility review',
        'source':str(source),'source_manifest_entries_verified':len(entries),'protocol':protocol,'protocol_source_sha256':digest(source/'stage3-status.json'),
        'source_snapshot_sha256':digest(snapshot),'source_results_sha256':digest(source/'results.json'),
        'arithmetic':'Fraction.from_float exact binary stored L/m/means; independent exact physical quadratic reference; decimal D=1/50',
        'no_source_row_state_or_diagonal_changes':True,'parameter_values_evaluated_on_a_graph':0,'fluid_orc_enabled':False})
    previous=json.loads((source/'results.json').read_text())
    controls={side:{'previous_raw_CSR_response_per_s':previous['rows'][side]['row']['raw_CSR_quadratic_response_per_s'],
        'previous_centered_response_per_s':previous['rows'][side]['row']['centered_difference_response_per_s'],
        'previous_raw_minus_exact_binary_raw_per_s':previous['rows'][side]['row']['raw_CSR_quadratic_response_per_s']-result['rows'][side]['exact_raw_CSR_on_cached_binary_means']['decimal_for_reading_only'],
        'previous_centered_minus_exact_A_plus_C_per_s':previous['rows'][side]['row']['centered_difference_response_per_s']-result['rows'][side]['exact_centered_original_response']['decimal_for_reading_only'],
        'phase_float_sum_minus_previous_centered_per_s':result['rows'][side]['floating_reductions']['sum_phase_contributions_per_s']-previous['rows'][side]['row']['centered_difference_response_per_s']}
        for side in ['left','right']}
    np.savez_compressed(out/'cross-scalar.npz',**arrays,**derived)
    dump(out/'proof-certificates.json',result);dump(out/'raw-controls.json',controls)
    dump(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'arithmetic':'standard-library Fraction; NumPy for NPZ and raw reductions'})
    print(json.dumps(result,indent=2),flush=True);print(json.dumps(controls,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--snapshot');p.add_argument('--certificates-only',action='store_true');a=p.parse_args()
    if a.certificates_only:
        if a.output or not a.snapshot:p.error('certificates-only requires --snapshot, no output directory')
        print(json.dumps(certificates(a.snapshot)[0],indent=2),flush=True)
    else:
        if not a.output or a.snapshot:p.error('provenance-checked run requires --output only')
        main(a.output)
