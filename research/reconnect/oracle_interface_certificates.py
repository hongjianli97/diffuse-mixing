"""Exact algebra for a theory-only review of the unchanged oracle kernel.

The report supplies the analytic limit, sign and averaging proofs. These
rational certificates check their constants; they are not numerical flow
experiments or a complete automated theorem prover. Standard library only.
"""
from pathlib import Path
from fractions import Fraction as F
from math import comb
import argparse,hashlib,json,platform


def multiply(a,b):
    p=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):p[i+j]+=x*y
    return p


def integral(poly,power):
    return sum((v*F(2)**(j+power+1)/F(j+power+1) for j,v in enumerate(poly)),F(0))


def rational(x):
    return {'exact':str(x),'decimal_for_reading_only':float(x)}


def certificates():
    W=multiply([F(comb(6,j))*(-F(1,2))**j for j in range(7)],[F(1),F(3),F(35,12)])
    B=[-F(j+2)*W[j+2] for j in range(len(W)-2)]
    raw=[F(7,3)*v for v in multiply([F(2),F(5)],[F(comb(5,j))*(-F(1,2))**j for j in range(6)])]
    assert B==raw
    C2_times_pi=F(9,4);full_second=C2_times_pi*integral(B,3)
    assert 2*C2_times_pi*integral(W,1)==full_second==1
    half_second=full_second/2
    D=F(1,50);rhoL=F(1);rhoR=F(4);GL=rhoL*D;GR=rhoR*D;average=(GL+GR)/2
    # Coefficients of Q_L, Q_R, T in the common density-weighted trace.
    normal_coefficients=[GL*half_second,GR*half_second]
    transverse_coefficient=(GL+GR)*half_second
    assert sum(normal_coefficients)==transverse_coefficient==average==F(1,20)
    rateL=average/rhoL;rateR=average/rhoR;ratioL=rateL/D;ratioR=rateR/D
    assert (rateL,rateR,ratioL,ratioR)==(F(1,20),F(1,80),F(5,2),F(5,8))
    assert rateL-D==F(3,100) and rateR-D==-F(3,400)
    homogeneous=GL/rhoL==GR/rhoR==D
    assert homogeneous and rateL!=D and rateR!=D
    # Geometry is fixed analytically, not scanned or selected from results.
    Delta=F(1,128);h=F(1,25);lam=Delta/h;alpha_max=2*lam
    assert lam==F(25,128) and alpha_max==F(25,64)<1
    low_K_over_GL=4/(1+3*alpha_max)
    high_K_over_GR=1/(4-3*alpha_max)
    assert low_K_over_GL==F(256,139)>1 and high_K_over_GR==F(64,181)<1
    # u in [1/2,1], |v| in [1/2,1]: q<=sqrt(2)<3/2.
    assert F(2)<F(3,2)**2
    C2_lower=F(9,4)/F(22,7)  # pi < 22/7
    B_lower=C2_lower*F(7,3)*2*F(1,4)**5
    rectangle_area=2*F(1,2)**2;v_squared_lower=F(1,4)
    moment_lower=B_lower*rectangle_area*v_squared_lower
    assert moment_lower==F(147,360448)>0
    gapL=(low_K_over_GL-1)*moment_lower
    gapR=(1-high_K_over_GR)*moment_lower
    assert gapL==F(17199,50102272) and gapR==F(17199,65241088)
    # pi > 3 bounds the quadratic boundary-layer residual uniformly in h.
    normal_uniform_bound=F(1)+rhoR/rhoL*(1+F(64,9))
    transverse_uniform_bound=1+rhoR/rhoL
    assert normal_uniform_bound==F(301,9) and transverse_uniform_bound==5
    groups={
        'raw_kernel_derivative_and_second_moments':{'verified':True,'W_coefficients':[str(v) for v in W],
            'B_over_C2_coefficients':[str(v) for v in B],'full_normal_and_transverse_second_moment':rational(full_second),
            'each_half_space_second_moment':rational(half_second)},
        'quadratic_interface_trace_coefficients':{'verified':True,'GammaL':rational(GL),'GammaR':rational(GR),
            'coefficients_of_QL_QR':[rational(v) for v in normal_coefficients],
            'coefficient_of_shared_T':rational(transverse_coefficient),'common_Q_and_T_coefficient':rational(average)},
        'common_curvature_one_sided_rate_ratios':{'verified':True,'D':rational(D),
            'left_rate_per_unit_Q_or_T':rational(rateL),'right_rate_per_unit_Q_or_T':rational(rateR),
            'left_ratio_to_PDE':rational(ratioL),'right_ratio_to_PDE':rational(ratioR),
            'left_rate_error_per_unit_Q_or_T':rational(rateL-D),'right_rate_error_per_unit_Q_or_T':rational(rateR-D)},
        'iterated_limits_do_not_commute':{'verified':True,
            'h_first_then_interface_left_and_right':rational(D),
            'interface_first_then_h_left':rational(rateL),'interface_first_then_h_right':rational(rateR),
            'scope':'local quadratic; values per unit Q or T; analytic limit proofs in report'},
        'original_cell_to_kernel_ratio_and_path_bounds':{'verified':True,'Delta':rational(Delta),'h':rational(h),'lambda':rational(lam),
            'rectangle_target_fraction_upper_bound':rational(alpha_max),'left_K_over_GammaL_lower':rational(low_K_over_GL),
            'right_K_over_GammaR_upper':rational(high_K_over_GR),'rectangle_z_squared_B_moment_strict_lower':rational(moment_lower)},
        'adjacent_mass_average_strict_defects_at_original_ratio':{'verified':True,
            'left_average_rate_over_D_T_strict_lower':rational(1+gapL),
            'right_average_rate_over_D_T_strict_upper':rational(1-gapR),
            'left_positive_relative_defect_strict_lower':rational(gapL),'right_negative_relative_defect_magnitude_strict_lower':rational(gapR),
            'not_actual_cosine_state_predictions':True,'proof_uses':'pointwise positive integral over a fixed rectangle, then original positive material-mass averaging'},
        'bounded_quadratic_layer_and_limit_scaling':{'verified':True,
            'uniform_error_bound_over_D_times_abs_Q':rational(normal_uniform_bound),
            'uniform_error_bound_over_D_times_abs_T':rational(transverse_uniform_bound),
            'layer_full_normal_width_over_h':4,'fixed_cell_average_bound_order':'h/Delta',
            'fixed_window_mass_weighted_L1_order':'h','fixed_window_mass_weighted_L2_order':'sqrt(h)',
            'finite_lambda_adjacent_cell_bias_may_persist_while_global_norm_vanishes':True}}
    assert len(groups)==7 and all(v['verified'] for v in groups.values())
    return {'certificate_groups':groups,'verified_exact_algebra_groups':7,
        'analytic_proofs_are_in_report':'research/reconnect/ORACLE_INTERFACE_SECOND_ORDER.md',
        'arithmetic':'fractions.Fraction; decimals are display only','new_numerical_flow_or_unit_tests':0,
        'previous_recorded_checks':352,'recorded_checks_total':352,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'new_operator_quadrature_evaluations':0,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'limits':['unchanged scalar path-resistance times radial kernel with extra oracle density',
            'local quadratic transmission states; not quantitative attribution of saved cosine residuals',
            'one-sided trace and adjacent-cell consistency differ from fixed-cell/weak/global convergence',
            'no impossibility theorem for all positive graph methods, no transient or ORC admission']}


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-oracle-directional'
    entries=json.loads((source/'sha256.json').read_text())
    for rel,h in entries.items():
        with (root/rel).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,rel
    protocol=json.loads((source/'stage3-status.json').read_text())['next_second_order_interface_review_predeclared']
    assert protocol['density_contact']==[1,4] and protocol['D_m2_per_s']==.02 and protocol['h_m']==.04
    assert protocol['original_radial_kernel_and_second_moments_unchanged'] and protocol['no_new_kernel_coefficient_or_density_reconstruction']
    assert protocol['new_candidate_graphs']==protocol['new_candidate_transient_runs']==0 and not protocol['parameter_or_mesh_matrix_scan'] and not protocol['fluid_orc_enabled']
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    def dump(name,value):(out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    with (source/'stage3-status.json').open('rb') as f:protocol_sha=hashlib.file_digest(f,'sha256').hexdigest()
    dump('config.json',{'scope':'predeclared theory-only second-order interface consistency review',
        'source':str(source),'source_manifest_entries_verified':len(entries),'protocol':protocol,'protocol_source_sha256':protocol_sha,
        'state_definition':'fixed material-cell rho-weighted mean; continuous quadratic jets are analytic test functions, not replacement data',
        'candidate_saved_total_and_all_original_states_unchanged':True,'arithmetic':'Python standard-library Fraction',
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'fluid_orc_enabled':False})
    result=certificates();dump('proof-certificates.json',result)
    dump('environment.json',{'python':platform.python_version(),'dependencies':'Python standard library only'})
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--certificates-only',action='store_true');a=p.parse_args()
    if a.certificates_only:
        if a.output:p.error('choose certificates-only or provenance-checked output')
        print(json.dumps(certificates(),indent=2,ensure_ascii=False))
    else:
        if not a.output:p.error('--output required unless --certificates-only')
        main(a.output)
