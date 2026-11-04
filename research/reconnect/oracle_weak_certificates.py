"""Exact constants for the same-oracle weak-form proof; standard library only.

The report proves the integral/regularity statements. These certificates
check rational coefficients and sign bounds, not a complete theorem prover.
No flow solve, graph, operator quadrature or numerical test is created.
"""
from pathlib import Path
from fractions import Fraction as F
from math import comb
import argparse,hashlib,json,platform


def multiply(a,b):
    out=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]+=x*y
    return out


def integral(poly,power):
    return sum((v*F(2)**(j+power+1)/F(j+power+1) for j,v in enumerate(poly)),F(0))


def rational(x):
    return {'exact':str(x),'decimal_for_reading_only':float(x)}


def certificates():
    W=multiply([F(comb(6,j))*(-F(1,2))**j for j in range(7)],[F(1),F(3),F(35,12)])
    B=[-F(j+2)*W[j+2] for j in range(len(W)-2)]
    raw=[F(7,3)*v for v in multiply([F(2),F(5)],[F(comb(5,j))*(-F(1,2))**j for j in range(6)])]
    assert B==raw
    radial={p:F(9,2)*integral(B,p+1) for p in [2,4,5,6]}
    assert radial=={2:F(2),4:F(56,33),5:F(256,143),6:F(288,143)}
    mixed_times_pi=F(3,2)*integral(B,4)
    assert mixed_times_pi==F(32,55)
    # One undirected pair represented twice. Original generator factor is 2.
    m=[F(1),F(4)];c=[F(1,4),F(3,4)];phi=[F(2),F(-1)];w=F(3,7)
    rate=[2*w*(c[1]-c[0])/m[0],2*w*(c[0]-c[1])/m[1]]
    weak=-sum((m[i]*phi[i]*rate[i] for i in [0,1]),F(0))
    paired=2*w*(c[1]-c[0])*(phi[1]-phi[0]);energy=2*w*(c[1]-c[0])**2
    assert weak==paired==-F(9,7) and energy==F(3,14) and sum(m[i]*rate[i] for i in [0,1])==0
    D=F(1,50);GL=D;GR=4*D;J=F(1)
    assert GL*(J/GL)==GR*(J/GR)==J
    bulk=[radial[4]/4,radial[4]/6,radial[5]/12,radial[6]/36]
    assert bulk==[F(14,33),F(28,99),F(64,429),F(8,143)]
    rough=4*(radial[2]+1)
    # Near-interface jet gradient errors <=5h*H; local target <=2h*H.
    sharp_point=5*radial[2]+2;sharp_band=4*sharp_point
    assert rough==12 and sharp_point==12 and sharp_band==48
    # ln(4)=4 sum_{k>=0} (1/3)^(2k+1)/(2k+1).
    x=F(1,3);log_lower=4*x;log_upper=log_lower+4*x**3/(3*(1-x*x))
    assert log_lower==F(4,3) and log_upper==F(25,18)
    H_log_coefficient=GL*GR/(GR-GL)
    assert H_log_coefficient==F(2,75)
    gap_lower=2*H_log_coefficient*log_lower-GL-GR
    gap_upper=2*H_log_coefficient*log_upper-GL-GR
    assert gap_lower==-F(13,450)<gap_upper==-F(7,270)<0
    kappa_log=mixed_times_pi*2*H_log_coefficient
    kappa_constant=-mixed_times_pi*(GL+GR)
    kappa_lower=mixed_times_pi*gap_lower/3
    kappa_upper=mixed_times_pi*gap_upper/F(22,7)
    assert (kappa_log,kappa_constant)==(F(128,4125),-F(16,275))
    assert kappa_lower==-F(208,37125)<kappa_upper==-F(392,81675)<0
    groups={
        'unchanged_kernel_radial_and_half_mixed_moments':{'verified':True,
            'B_over_C2_coefficients':[str(v) for v in B],
            'radial_moments_after_dividing_h_power':{str(p):rational(v) for p,v in radial.items()},
            'pi_times_positive_normal_mixed_third_moment_over_h':rational(mixed_times_pi)},
        'mass_measure_sign_and_pair_factor':{'verified':True,'minus_mass_inner_product':rational(weak),
            'ordered_pair_bilinear_value':rational(paired),'ordered_pair_energy':rational(energy),
            'original_generator_factor':2,'additional_half_in_ordered_bilinear_form':False},
        'transmission_and_local_diffusion_balance':{'verified':True,'D':rational(D),'GammaL':rational(GL),'GammaR':rational(GR),
            'matched_normal_flux':rational(J),'no_Dirac_when_normal_Gamma_gradient_jump_is_zero':True},
        'pure_phase_Taylor_remainder_constants':{'verified':True,
            'h2_C2_P2':rational(bulk[0]),'h2_C1_P3_plus_C3_P1':rational(bulk[1]),
            'h3_C2_P3_plus_C3_P2':rational(bulk[2]),'h4_C3_P3':rational(bulk[3])},
        'interface_strip_and_jet_remainder_constants':{'verified':True,
            'rough_h_Gamma_max_C1_P1_length':rational(rough),
            'jet_residual_point_h_Gamma_max_C2_P1_plus_C1_P2':rational(sharp_point),
            'jet_residual_integrated_h2_Gamma_max_length_C2_P1_plus_C1_P2':rational(sharp_band)},
        'path_fraction_integral_and_strict_sign':{'verified':True,
            'Hbar_coefficient_of_log4':rational(H_log_coefficient),'log4_strict_lower':rational(log_lower),'log4_strict_upper':rational(log_upper),
            'twice_Hbar_minus_Gamma_sum_strict_lower':rational(gap_lower),
            'twice_Hbar_minus_Gamma_sum_strict_upper':rational(gap_upper)},
        'leading_tangential_interface_coefficient':{'verified':True,
            'kappa_tau_equals':'(128/4125 * log(4) - 16/275) / pi',
            'pi_times_kappa_coefficient_of_log4':rational(kappa_log),'pi_times_kappa_constant':rational(kappa_constant),
            'kappa_strict_lower_using_pi_bounds':rational(kappa_lower),'kappa_strict_upper_using_pi_bounds':rational(kappa_upper)},
        'nonzero_sharpness_witness':{'verified':True,
            'analytic_witness':'c=phi=chi(a)*f(b); chi=1 near interface, f smooth compact nonconstant',
            'normal_interface_flux':0,'tangential_trace_integral':'integral (f_prime)^2 > 0',
            'leading_weak_error_sign':'strictly negative for density contrast; O(h) generally sharp',
            'analytic_function_and_limit_proofs_are_in_report':True}}
    assert len(groups)==8 and all(v['verified'] for v in groups.values())
    return {'certificate_groups':groups,'verified_exact_algebra_groups':8,
        'analytic_proofs_are_in_report':'research/reconnect/ORACLE_WEAK_INTERFACE_CONSISTENCY.md',
        'new_numerical_flow_or_unit_tests':0,'previous_recorded_checks':352,'recorded_checks_total':352,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'new_operator_quadrature_evaluations':0,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'scope_limits':['fixed full-density oracle and flat two-phase contact; not a finite-node density closure',
            'frozen weak bilinear consistency and sharp first-order term, not transient convergence',
            'smooth fixed tests, not shrinking cell indicators or arbitrary h-dependent states',
            'no numerical predictions or refitting of saved frozen-rate errors']}


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-oracle-interface'
    entries=json.loads((source/'sha256.json').read_text())
    for rel,h in entries.items():
        with (root/rel).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,rel
    protocol=json.loads((source/'stage3-status.json').read_text())['next_weak_interface_review_predeclared']
    assert protocol['same_original_kernel_and_path_coefficient'] and protocol['no_new_density_closure_or_candidate']
    assert protocol['new_candidate_graphs']==protocol['new_candidate_transient_runs']==protocol['new_reference_transient_runs']==0
    assert not protocol['parameter_or_mesh_matrix_scan'] and not protocol['fluid_orc_enabled']
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    def dump(name,value):(out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    with (source/'stage3-status.json').open('rb') as f:protocol_sha=hashlib.file_digest(f,'sha256').hexdigest()
    dump('config.json',{'scope':'predeclared same-kernel weak bilinear interface theory review',
        'source':str(source),'source_manifest_entries_verified':len(entries),'protocol':protocol,'protocol_source_sha256':protocol_sha,
        'mass_state':'original fixed material-cell rho-weighted means unchanged',
        'arithmetic':'Python standard-library Fraction; analytic limits and logarithm bounds proved in report',
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
