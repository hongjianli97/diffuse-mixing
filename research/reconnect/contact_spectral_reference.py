"""Analytic weighted transmission reference for the original equal-half contact.

Standard-library Decimal arithmetic. Projections use the declared continuous
initial formula, never fitted cell values or FV/kernel responses. No graph or
time-stepper is constructed. The report proves completeness and tail bounds.
"""
from decimal import Decimal as Dec, getcontext, localcontext
from fractions import Fraction as F
from math import factorial


def dec(value):
    value=F(value)
    return Dec(value.numerator)/Dec(value.denominator)


def _atan(x):
    term=total=x;index=1;epsilon=Dec(1).scaleb(-getcontext().prec-5)
    while True:
        term*=-x*x;part=term/(2*index+1);total+=part
        if abs(part)<epsilon:return total
        index+=1
        if index>10000:raise ArithmeticError('atan did not converge')


def pi_machin():
    return 16*_atan(Dec(1)/5)-4*_atan(Dec(1)/239)


def pi_agm():
    a,b,t,p=Dec(1),1/Dec(2).sqrt(),Dec(1)/4,Dec(1)
    for _ in range(20):
        previous=a;a,b=(a+b)/2,(a*b).sqrt();t-=p*(previous-a)**2;p*=2
        if abs(a-b)<Dec(1).scaleb(-getcontext().prec+3):return (a+b)**2/(4*t)
    raise ArithmeticError('AGM did not converge')


class TrigPi:
    """Exact rational argument reduction; exact zeros are identities, not clips."""
    def __init__(self):
        self.pi=pi_machin();self.cache={F(0):(Dec(0),Dec(1))}

    def _small(self,r):
        if r not in self.cache:
            x=self.pi*dec(r);s=st=x;c=ct=Dec(1)
            eps=Dec(1).scaleb(-getcontext().prec-5)
            for k in range(1,10000):
                st*=-x*x/((2*k)*(2*k+1));ct*=-x*x/((2*k-1)*(2*k));s+=st;c+=ct
                if max(abs(st),abs(ct))<eps:break
            else:raise ArithmeticError('sine/cosine did not converge')
            self.cache[r]=(s,c)
        return self.cache[r]

    def sc(self,ratio):
        ratio=F(ratio)%2;q=int(ratio//F(1,2));r=ratio-q*F(1,2)
        if r>F(1,4):c,s=self._small(F(1,2)-r)
        else:s,c=self._small(r)
        return [(s,c),(c,-s),(-s,-c),(-c,s)][q]

    def sinc(self,ratio):
        return Dec(1) if ratio==0 else self.sc(ratio)[0]/(self.pi*dec(ratio))

    def integral_cos(self,freq,phase,lo,hi):
        return dec(hi-lo)*self.sc(phase)[1] if freq==0 else (
            self.sc(freq*hi+phase)[0]-self.sc(freq*lo+phase)[0])/(self.pi*dec(freq))

    def integral_sin(self,freq,phase,lo,hi):
        return dec(hi-lo)*self.sc(phase)[0] if freq==0 else (
            self.sc(freq*lo+phase)[1]-self.sc(freq*hi+phase)[1])/(self.pi*dec(freq))


# (left endpoint, right endpoint, rho, phase frequency/pi, phase offset/pi)
PIECES=((F(0),F(1,2),F(1),F(16,5),F(1,5)),
        (F(1,2),F(1),F(4),F(4,5),F(7,5)))


def projection_products(trig,K):
    A=dec(F(1,5));Z=dec(F(5,2));nc=dec(F(5,4));ns=dec(F(5,16))
    a0=sum((dec(rho)*A*trig.integral_cos(a,phi,lo,hi) for lo,hi,rho,a,phi in PIECES),Dec(0))/Z
    alpha=[a0];beta=[Dec(0)]
    for n in range(1,K+1):
        ac=bs=Dec(0)
        for lo,hi,rho,a,phi in PIECES:
            ac+=dec(rho)*A/2*(trig.integral_cos(a-2*n,phi,lo,hi)+trig.integral_cos(a+2*n,phi,lo,hi))
            bs+=A/2*(trig.integral_sin(2*n+a,phi,lo,hi)+trig.integral_sin(2*n-a,-phi,lo,hi))
        alpha.append(ac/nc);beta.append(bs/ns)
    return alpha,beta


def projection_boundary(trig,n):
    """Independent phasewise Green identity, without product-to-sum integrals."""
    k=2*trig.pi*n;nc=dec(F(5,4));ns=dec(F(5,16));ac=bs=Dec(0)
    for lo,hi,rho,a,phi in PIECES:
        ap=trig.pi*dec(a);den=k*k-ap*ap;ends=[]
        for x in [lo,hi]:
            sf,cf=trig.sc(a*x+phi);s,c=trig.sc(2*n*x)
            f=cf/5;fp=-ap*sf/5
            ends.append((fp*c+f*k*s,fp*s/dec(rho)-f*k*c/dec(rho)))
        ac+=dec(rho)*(ends[1][0]-ends[0][0])/den
        bs+=dec(rho)*(ends[1][1]-ends[0][1])/den
    return ac/nc,bs/ns


def material_factors(trig,alpha,beta,N,D,T):
    if N<4 or N%2:raise ValueError('even aligned observation grid required')
    if D<=0 or T<0:raise ValueError('positive D and nonnegative T required')
    tau=4*trig.pi**2*D*T;ey=(-tau).exp()
    attenuation=[(-(tau*n*n)).exp() for n in range(len(alpha))]
    normal=[];initial=[]
    for i in range(N):
        x=F(2*i+1,2*N);rho=F(1) if i<N//2 else F(4)
        lo,hi,rho,a,phi=PIECES[0 if i<N//2 else 1]
        initial.append(trig.sinc(a/(2*N))*trig.sc(a*x+phi)[1]/5)
        value=alpha[0]
        for n in range(1,len(alpha)):
            s,c=trig.sc(2*n*x);sn=trig.sinc(F(n,N))
            value+=attenuation[n]*sn*(alpha[n]*c+beta[n]*s/dec(rho))
        normal.append(ey*value)
    transverse=[trig.sinc(F(1,N))*trig.sc(F(2*j+1,N))[1] for j in range(N)]
    return normal,transverse,initial


def atan_rational_bracket(x,terms=12):
    value=sum(((-1)**k*x**(2*k+1)/(2*k+1) for k in range(terms)),F(0))
    next_term=(-1)**terms*x**(2*terms+1)/(2*terms+1)
    return min(value,value+next_term),max(value,value+next_term)


def tail_certificate(K,D,T):
    D,T=F(D),F(T)
    assert D>0 and T>0
    p,q=atan_rational_bracket(F(1,5)),atan_rational_bracket(F(1,239))
    pi_lo,pi_hi=16*p[0]-4*q[1],16*p[1]-4*q[0]
    assert pi_lo>F(157,50) and pi_hi<F(22,7)
    step=F(231,100)
    exponential_lower=sum((step**j/factorial(j) for j in range(13)),F(0))
    assert exponential_lower>10
    exponent_lower=4*F(157,50)**2*D*T*((K+1)**2+1)
    decades=int(exponent_lower//step)
    # ||(c0-1/2)||^2 <= (1/5)^2 * (5/2) * 1/2 = 1/20 < 1/16.
    assert F(1,20)<F(1,16)
    bound=F(1,4*10**decades)
    return {'K':K,'pi_lower_rational':'157/50','Machin_pi_lower_verified':str(pi_lo),
        'Machin_pi_upper_verified':str(pi_hi),'exp_step_exact':str(step),
        'exp_step_partial_sum_lower_verified':str(exponential_lower),
        'initial_norm_upper_exact':'1/4','decay_exponent_lower_exact':str(exponent_lower),
        'guaranteed_decimal_decades':decades,'material_mass_norm_tail_bound_exact':str(bound),
        'proof_requires_weighted_complete_eigenbasis_and_Jensen_projection':True}


def render(value,precision):
    if isinstance(value,Dec):
        with localcontext() as c:c.prec=precision;return str(+value)
    if isinstance(value,dict):return {key:render(item,precision) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [render(item,precision) for item in value]
    return value


def reference_case(model,K,precision):
    if model['rho_left']!='1' or model['rho_right']!='4' or model['N']!=128:
        raise ValueError('this reference is specialized to the predeclared equal-half contact')
    with localcontext() as context:
        context.prec=precision+30
        trig=TrigPi();D,T=dec(model['D_exact']),dec(model['T_exact']);N=model['N']
        alpha,beta=projection_products(trig,K)
        checks=[abs(trig.pi-pi_agm()),abs(alpha[0]-3*trig.sc(F(1,5))[0]/(4*trig.pi))]
        for n in range(1,K+1):
            a,b=projection_boundary(trig,n);checks.extend([abs(alpha[n]-a),abs(beta[n]-b)])
        assert all(alpha[n]==0 for n in range(1,K+1,2))
        assert all(beta[n]==0 for n in range(2,K+1,2))
        maximum=max(checks);assert maximum<Dec(1).scaleb(-precision+5)
        normal,transverse,initial=material_factors(trig,alpha,beta,N,D,T)
        tau=4*trig.pi**2*D*T
        approximate_tail=dec(F(1,4))*(-tau*((K+1)**2+1)).exp()
        cert=tail_certificate(K,model['D_exact'],model['T_exact'])
        assert approximate_tail<dec(cert['material_mass_norm_tail_bound_exact'])
        result={'K':K,'requested_decimal_digits':precision,'working_digits':context.prec,
            'alpha':alpha,'beta':beta,'normal_T':normal,'transverse_cell_means':transverse,
            'initial_normal_cell_means':initial,'normal_zero_mode':alpha[0],
            'matched_boundary_projection_and_pi_max_difference':maximum,
            'odd_alpha_and_even_beta_zeros_are_exact_symmetries_not_clipping':True,
            'tail_certificate':cert,'noninterval_exponential_tail_readout':approximate_tail,
            'reference_parameters_only_no_source_state_fit':True}
        return render(result,precision)


def field(case):
    with localcontext() as context:
        context.prec=case['requested_decimal_digits']+30
        return [Dec('0.5')+Dec(x)*Dec(y) for y in case['transverse_cell_means'] for x in case['normal_T']]


def weighted_norm(values,mass):
    return sum((dec(m)*v*v for m,v in zip(mass,values)),Dec(0)).sqrt()


def compare_cases(cases,source):
    with localcontext() as context:
        context.prec=120
        mass=source['mass'];c0=source['c0'];chosen=cases[-1];N=128
        values=[field(case) for case in cases]
        comparisons={}
        for i,j,name in [(0,1,'K64_vs_K128_at_60_digits'),(2,3,'K64_vs_K128_at_90_digits'),
                         (1,3,'60_vs_90_digits_at_K128')]:
            diff=[a-b for a,b in zip(values[i],values[j])]
            comparisons[name]={'max_absolute':max(map(abs,diff)),'mass_norm':weighted_norm(diff,mass)}
        initial=[Dec('0.5')+Dec(x)*Dec(y) for y in chosen['transverse_cell_means'] for x in chosen['initial_normal_cell_means']]
        gap=[dec(a)-b for a,b in zip(c0,initial)]
        serialization=[dec(float(v))-v for v in values[-1]]
        inventory=sum((dec(m)*v for m,v in zip(mass,values[-1])),Dec(0))
        source_inventory=sum((F(m)*F(c) for m,c in zip(mass,c0)),F(0))
        empirical_indicator=(weighted_norm(gap,mass)+weighted_norm(serialization,mass)
            +comparisons['60_vs_90_digits_at_K128']['mass_norm']
            +dec(chosen['tail_certificate']['material_mass_norm_tail_bound_exact']))
        result={'comparison_controls':comparisons,
            'source_vs_math_initial_mean_bridge_max_absolute':max(map(abs,gap)),
            'source_vs_math_initial_mean_bridge_mass_norm':weighted_norm(gap,mass),
            'source_initial_inventory_exact':str(source_inventory),
            'source_initial_inventory_minus_5_over_4_exact':str(source_inventory-F(5,4)),
            'reference_inventory_minus_5_over_4_not_forced_zero':inventory-Dec('1.25'),
            'reference_T_minimum':min(values[-1]),'reference_T_maximum':max(values[-1]),
            'reference_T_projected_fluctuation_norm':weighted_norm([v-Dec('.5') for v in values[-1]],mass),
            'source_initial_projected_fluctuation_norm':weighted_norm([dec(v)-Dec('.5') for v in c0],mass),
            'float64_serialization_mass_norm_difference':weighted_norm(serialization,mass),
            'float64_serialization_max_absolute_difference':max(map(abs,serialization)),
            'chosen_rigorous_modal_tail_bound_exact':chosen['tail_certificate']['material_mass_norm_tail_bound_exact'],
            'reference_initial_bridge_serialization_and_precision_indicator_mass_norm':empirical_indicator,
            'combined_indicator_is_not_a_certified_total_physical_error_bound':True,
            'bridge_contraction_requires_explicit_cellwise_constant_perturbation_lift':True,
            'cell_means_alone_do_not_identify_arbitrary_continuous_initial_subcell_fields':True,
            'initial_bridge_not_subtracted_and_reference_not_renormalized':True,
            'Decimal_controls_are_not_interval_certification':True,
            'new_candidate_transient_runs':0,'new_FV_transient_runs':0,'new_kernel_builds':0,
            'new_global_matrices':0,'new_reference_time_stepping_solves':0,
            'new_reference_numerical_integrations':0,'reference_physical_time_cases':1,
            'reference_truncation_and_precision_evaluations':4,
            'new_software_checks':4,'previous_checks_rerun':0,'recorded_checks_total':396,
            'new_exact_algebra_certificate_groups':0,'fluid_orc_enabled':False,
            'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete'}
        assert min(values[-1])>=Dec('.3') and max(values[-1])<=Dec('.7')
        return render(result,90)
