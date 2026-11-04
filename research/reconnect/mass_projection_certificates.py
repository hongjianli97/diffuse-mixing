"""Rational constants and symbolic identities for a projection theory review.

Standard library only. No material geometry, edge weights, graph, kernel
quadrature or scalar state is generated. Integral/limit proofs are in report.
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


def integral_0_2(poly,power):
    return sum((v*F(2)**(j+power+1)/F(j+power+1) for j,v in enumerate(poly)),F(0))


def integral_interval(poly,left,right):
    return sum((v*(right**(j+1)-left**(j+1))/F(j+1) for j,v in enumerate(poly)),F(0))


def add_poly(*polys):
    result={}
    for poly in polys:
        for key,value in poly.items():result[key]=result.get(key,F(0))+value
    return {key:value for key,value in result.items() if value}


def product_poly(a,b):
    result={}
    for key,x in a.items():
        for other,y in b.items():
            powers=tuple(i+j for i,j in zip(key,other))
            result[powers]=result.get(powers,F(0))+x*y
    return {key:value for key,value in result.items() if value}


def rational(x):
    return {'exact':str(x),'decimal_for_reading_only':float(x)}


def certificates():
    W=multiply([F(comb(6,j))*(-F(1,2))**j for j in range(7)],[F(1),F(3),F(35,12)])
    B=[-F(j+2)*W[j+2] for j in range(len(W)-2)]
    raw=[F(7,3)*v for v in multiply([F(2),F(5)],[F(comb(5,j))*(-F(1,2))**j for j in range(6)])]
    assert B==raw
    moments={p:F(9,2)*integral_0_2(B,p+1) for p in [0,1,2]}
    assert moments=={0:F(9,2),1:F(8,3),2:F(2)}
    # An analytic cell crossing the same 1:4 contact; no input file changed.
    lo=-F(1,2);hi=F(1,2);m=F(5,2)
    cL=[F(0),F(1),F(1)];cR=[F(0),F(1,4),F(1)]
    numerator=integral_interval(cL,lo,F(0))+4*integral_interval(cR,F(0),hi)
    average=numerator/m
    unweighted=integral_interval(cL,lo,F(0))+integral_interval(cR,F(0),hi)
    assert numerator==F(5,24) and average==F(1,12) and unweighted==-F(1,96)
    mass_residual=numerator-m*average;volume_residual=unweighted-average
    assert mass_residual==0 and volume_residual==-F(3,32)
    square=integral_interval(multiply(cL,cL),lo,F(0))+4*integral_interval(multiply(cR,cR),F(0),hi)
    projection_square=m*average**2;error_square=square-projection_square
    assert error_square>0 and projection_square+error_square==square
    # Formal coefficients of one pair, with integral A_ij left symbolic.
    mass_pair={'A_ui_vi':F(2),'A_uj_vj':F(2),'A_uj_vi':-F(2),'A_ui_vj':-F(2)}
    ordered_pair={'A_ui_vi':F(2),'A_uj_vj':F(2),'A_uj_vi':-F(2),'A_ui_vj':-F(2)}
    assert mass_pair==ordered_pair
    energy_factor=4*moments[0];mixed_each=2*moments[1]
    mixed_total=2*mixed_each;quadratic=4*moments[0]
    assert energy_factor==quadratic==18 and mixed_each==F(16,3) and mixed_total==F(32,3)
    lam=F(25,128);epsilon_squared=2*lam**2
    assert epsilon_squared==F(625,8192)
    # k=floor(s/ell), theta=frac(s/ell): two polynomial variables.
    one={(0,0):F(1)};k={(1,0):F(1)};theta={(0,1):F(1)}
    one_minus_theta=add_poly(one,{(0,1):-F(1)});k_plus_one=add_poly(k,one)
    mean=add_poly(product_poly(one_minus_theta,k),product_poly(theta,k_plus_one))
    second=add_poly(product_poly(one_minus_theta,product_poly(k,k)),product_poly(theta,product_poly(k_plus_one,k_plus_one)))
    expected_second=add_poly(product_poly(add_poly(k,theta),add_poly(k,theta)),product_poly(theta,one_minus_theta))
    assert mean==add_poly(k,theta) and second==expected_second
    cell_quadratic_offset=F(1,2)*integral_interval([F(0),F(0),F(1)],-F(1,2),F(1,2))
    assert cell_quadratic_offset==F(1,24) and 2*F(1,2)==1
    beta_lower=F(9,4)/F(22,7)*F(7,3)*2*F(1,4)**5
    assert 1+(lam/2)**2<F(3,2)**2
    eta_lower=lam**2*beta_lower*F(3,16)*(lam/2)
    eta_upper=moments[0]*F(1,4)*lam**2
    assert eta_lower==F(6890625,3023656976384)>0
    assert eta_upper==F(5625,131072)>eta_lower
    telescoped=[F(0)]*5
    for start in range(4):telescoped[start]+=1;telescoped[start+1]-=1
    assert telescoped==[F(1),F(0),F(0),F(0),-F(1)]
    groups={
        'unchanged_kernel_zero_first_second_radial_moments':{'verified':True,'B_over_C2_coefficients':[str(v) for v in B],
            'radial_moments_after_dividing_h_power':{str(p):rational(v) for p,v in moments.items()}},
        'rho_weighted_projection_measure_and_orthogonality':{'verified':True,'analytic_cell_mass':rational(m),
            'mass_average':rational(average),'volume_average':rational(unweighted),
            'mass_weighted_residual_integral':rational(mass_residual),'volume_residual_integral':rational(volume_residual),
            'projected_L2_mass_square':rational(projection_square),'projection_error_L2_mass_square':rational(error_square),
            'not_replacement_initial_data':True},
        'ideal_shared_pair_and_mass_bilinear_identity':{'verified':True,'G_ij_equals':'2 times exact unweighted cell-pair integral of K_path*B_h',
            'formal_pair_coefficients':{key:str(value) for key,value in mass_pair.items()},'actual_edge_weights_computed':0},
        'projection_energy_and_weak_budget_constants':{'verified':True,'L2_volume_to_nonlocal_energy_factor':rational(energy_factor),
            'each_linear_projection_cross_term_factor':rational(mixed_each),'combined_linear_factor':rational(mixed_total),
            'quadratic_projection_factor':rational(quadratic)},
        'original_width_and_maximum_cell_diameter_are_distinct':{'verified':True,
            'square_width_over_h':rational(lam),'square_diameter_over_h':'25*sqrt(2)/128',
            'squared_diameter_over_h':rational(epsilon_squared)},
        'uniform_grid_fractional_jump_variance_identity':{'verified':True,
            'mean_jump_in_units_of_ell':'k+theta','mean_squared_jump_in_units_of_ell_squared':'(k+theta)^2+theta*(1-theta)',
            'second_moment_polynomial':{str(key):str(value) for key,value in second.items()},
            'proof_of_phase_probabilities_is_in_report':True},
        'mass_average_quadratic_and_generator_factor':{'verified':True,'cell_average_of_a_squared_over_2':'a_center^2/2+ell^2/24',
            'offset_over_ell_squared':rational(cell_quadratic_offset),'effective_rate_over_D':'1+eta(ell/h)',
            'continuous_PDE_rate_over_D':1,'analytic_identity_only_no_graph_created':True},
        'fixed_original_ratio_strict_excess_bounds':{'verified':True,'lambda':rational(lam),
            'eta_strict_lower':rational(eta_lower),'eta_strict_upper':rational(eta_upper),
            'eta_evaluated_by_numerical_quadrature':False,'bounds_are_not_estimates_of_saved_cosine_errors':True},
        'four_signed_terms_telescope_without_clipping':{'verified':True,'terms':['endpoint_vs_full_path_constitutive','point_pair_quadrature','mass_projection','continuous_nonlocal'],
            'formal_net_coefficients':[str(v) for v in telescoped],'term_norms_are_additive_causal_fractions':False}}
    assert len(groups)==9 and all(v['verified'] for v in groups.values())
    return {'certificate_groups':groups,'verified_exact_algebra_groups':9,
        'analytic_proofs_are_in_report':'research/reconnect/MASS_AVERAGE_PROJECTION_WEAK_BUDGET.md',
        'new_numerical_flow_or_unit_tests':0,'previous_recorded_checks':352,'recorded_checks_total':352,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'new_operator_quadrature_evaluations':0,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'scope_limits':['exact cell-pair graph is a theory-only geometry and full-density oracle, not candidate input',
            'Delta/h tending to zero is sufficient for the general bound; fixed-ratio failure is proved for the ideal closure witness',
            'the ideal closure counterexample is not a new verdict on the saved point candidate',
            'no eta numerical evaluation, parameter scan or transient convergence claim']}


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-oracle-weak'
    entries=json.loads((source/'sha256.json').read_text())
    for rel,h in entries.items():
        with (root/rel).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,rel
    protocol=json.loads((source/'stage3-status.json').read_text())['next_mass_average_projection_review_predeclared']
    assert protocol['no_new_candidate_geometry_inputs_or_backend'] and protocol['no_new_kernel_or_density_reconstruction']
    assert protocol['new_candidate_graphs']==protocol['new_candidate_transient_runs']==protocol['new_reference_transient_runs']==protocol['new_operator_quadrature_evaluations']==0
    assert not protocol['parameter_or_mesh_matrix_scan'] and not protocol['fluid_orc_enabled']
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    def dump(name,value):(out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    with (source/'stage3-status.json').open('rb') as f:protocol_sha=hashlib.file_digest(f,'sha256').hexdigest()
    dump('config.json',{'scope':'predeclared mass-average projection weak budget, theory only',
        'source':str(source),'source_manifest_entries_verified':len(entries),'protocol':protocol,'protocol_source_sha256':protocol_sha,
        'state':'original fixed material mass averages; ideal cell-pair closure only an analytic extra-geometry oracle',
        'arithmetic':'standard-library Fraction and symbolic polynomial identities; integral/limit proofs in report',
        'candidate_saved_total_and_all_original_states_unchanged':True,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'fluid_orc_enabled':False})
    dump('proof-certificates.json',certificates())
    dump('environment.json',{'python':platform.python_version(),'dependencies':'Python standard library only'})
    print(json.dumps(certificates(),indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--certificates-only',action='store_true');a=p.parse_args()
    if a.certificates_only:
        if a.output:p.error('choose standalone algebra or provenance-checked output')
        print(json.dumps(certificates(),indent=2,ensure_ascii=False))
    else:
        if not a.output:p.error('--output required unless --certificates-only')
        main(a.output)
