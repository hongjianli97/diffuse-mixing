"""Exact rational certificates for a theory review, not a flow/backend run.

No graph, quadrature, state propagation, fitting, or candidate coefficients
are created. Analytic arguments accompanying these checks are in the report.
"""
from pathlib import Path
from fractions import Fraction as F
from math import comb
import argparse,hashlib,json,platform


def multiply(a,b):
    result=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):result[i+j]+=x*y
    return result


def integral_0_2(poly,power=0):
    return sum((c*F(2)**(j+power+1)/F(j+power+1) for j,c in enumerate(poly)),F(0))


def rational(x):
    return {'exact':str(x),'decimal_for_reading_only':float(x)}


def certificates():
    W=multiply([F(comb(6,j))*(-F(1,2))**j for j in range(7)],[F(1),F(3),F(35,12)])
    B=[-F(j+2)*W[j+2] for j in range(len(W)-2)]
    coded_B=[F(7,3)*c for c in multiply([F(2),F(5)],[F(comb(5,j))*(-F(1,2))**j for j in range(6)])]
    assert B==coded_B
    I0=integral_0_2(W);I1=integral_0_2(W,1)
    C2_times_pi=F(9,4)
    normalization=2*C2_times_pi*I1
    normal_second=C2_times_pi*integral_0_2(B,3)
    A_times_pi_h=2*C2_times_pi*I0
    assert I0==F(16,27) and I1==F(2,9) and normalization==normal_second==1 and A_times_pi_h==F(8,3)
    D=F(1,50);GammaL=D;GammaR=4*D;H=2*GammaL*GammaR/(GammaL+GammaR)
    traceL=H/GammaR-1;traceR=1-H/GammaL
    flux_ratio=H/F(2)*(1/GammaL+1/GammaR)
    assert H==F(4,125) and traceL==traceR==-F(3,5) and flux_ratio==1 and GammaL!=GammaR
    N=128;delta=F(1,N);epsilon=F(1,100)
    alphas={};orthogonality={}
    for phase,kappa in [('left',F(8,5)),('right',F(2,5))]:
        r=(kappa/N)**2;alpha=(4-r)/(2*(1-r))
        residual=1/(1-r)-2*alpha/(4-r)
        assert residual==0
        alphas[phase]=alpha;orthogonality[phase]=residual
    G=F(301,100)
    assert 1+max(alphas.values())<G and 1-epsilon*G>0
    R0=delta/(2*D)*(1+F(1,4))
    lower=1-F(2,5)*epsilon
    upper=1-F(21,55)*epsilon+epsilon**2*G**2/(1-epsilon*G)
    assert R0==F(125,512) and 0<lower<upper<1
    groups={
        'kernel_derivative_matches_unchanged_polynomial':{'verified':True,'W_polynomial_coefficients':[str(v) for v in W],'B_over_C2_polynomial_coefficients':[str(v) for v in B]},
        'raw_kernel_normalization_and_planar_moment':{'verified':True,'I0':rational(I0),'I1':rational(I1),'normal_second_moment':rational(normal_second),'A_h_times_pi_h':rational(A_times_pi_h)},
        'phase_constant_cross_coefficient_conditions':{'verified':True,'GammaL':rational(GammaL),'GammaR':rational(GammaR),'harmonic_K':rational(H),
            'left_required_K':rational(GammaR),'right_required_K':rational(GammaL),
            'harmonic_weighted_trace_over_2_A_J_left':rational(traceL),'harmonic_weighted_trace_over_2_A_J_right':rational(traceR),'harmonic_total_flux_ratio':rational(flux_ratio)},
        'same_scalar_mass_averages_orthogonality':{'verified':True,'N':N,'alpha_by_phase':{k:rational(v) for k,v in alphas.items()},'coefficient_residuals':{k:str(v) for k,v in orthogonality.items()}},
        'hidden_density_positivity':{'verified':True,'epsilon':rational(epsilon),'strict_g_upper_bound':rational(G),'density_ratio_strict_lower_bound':rational(1-epsilon*G)},
        'same_inputs_distinct_segment_resistance':{'verified':True,'R0_per_unit_transverse_area':rational(R0),
            'strict_lower_bound_R_epsilon_over_R0':rational(lower),'strict_upper_bound_R_epsilon_over_R0':rational(upper),
            'proof_uses':'3 < pi < 22/7 and exact reciprocal Taylor identity; no sampled density field or numerical PDE solve'}}
    assert len(groups)==6 and all(item['verified'] for item in groups.values())
    return {'certificate_groups':groups,'verified_exact_algebra_groups':6,
        'analytic_proofs_are_in_report':'research/reconnect/CONTACT_CONSTITUTIVE_REVIEW.md',
        'numerical_flow_or_unit_test_count_added':0,'previous_recorded_checks':341,'recorded_checks_total':341,
        'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete',
        'scope_limits':['phase-constant cross-coefficient trace consistency, not all positive exchange models',
            'finite input identifiability without subcell density prior, not fixed-profile refinement impossibility',
            'no h->0 weak-limit theorem, no new candidate or FV replacement']}


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-contact-continuous'
    with (source/'sha256.json').open() as f:entries=json.load(f)
    for rel,h in entries.items():
        with (root/rel).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,rel
    protocol=json.loads((source/'stage3-status.json').read_text())['next_contact_constitutive_review_predeclared']
    assert protocol['new_candidate_graphs']==protocol['new_candidate_transient_runs']==protocol['new_reference_transient_runs']==0
    assert protocol['no_implementation_before_review'] and not protocol['parameter_scan'] and not protocol['fluid_orc_enabled']
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    dump=lambda name,value:(out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    with (source/'stage3-status.json').open('rb') as f:protocol_sha=hashlib.file_digest(f,'sha256').hexdigest()
    dump('config.json',{'scope':'predeclared theory-only contact constitutive and input-identifiability review',
        'source':str(source),'protocol':protocol,'protocol_source_sha256':protocol_sha,
        'arithmetic':'fractions.Fraction for all algebraic certificates; decimal rendering only for reading',
        'input_witness':'analytic same-node/same-cell-mass/same-cbar density family; not a sampled or evolved replacement dataset',
        'candidate_unchanged':True,'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'fluid_orc_enabled':False})
    result=certificates();dump('proof-certificates.json',result)
    dump('environment.json',{'python':platform.python_version(),'dependencies':'Python standard library only for certificates'})
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--certificates-only',action='store_true');a=p.parse_args()
    if a.certificates_only:
        if a.output:p.error('choose certificates-only or a provenance-checked output directory')
        print(json.dumps(certificates(),indent=2,ensure_ascii=False))
    else:
        if not a.output:p.error('--output is required unless --certificates-only is used')
        main(a.output)
