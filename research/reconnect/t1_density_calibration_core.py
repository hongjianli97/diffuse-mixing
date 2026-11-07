"""Post-seal T1 symbols and comparisons, retaining every frozen positive term."""
from fractions import Fraction as F
import math
import numpy as np
import mpmath as mp
from t1_uniform_core import h_indicator
from voronoi_time_action import metrics,operators

TIMES=(0.,.025,.05)
SOURCES=('oracle','kernel','ledger')
KS=(4,8,16)


def norm(m,v):return float(np.sqrt(m@(v*v)))


def independent_symbol(c0,D,source,digits):
    with mp.workdps(digits):
        f=lambda v:mp.mpf(v.numerator)/v.denominator
        if source=='kernel':
            terms=[]
            for a in range(-3,4):
                for b in range(-3,4):
                    s=a*a+b*b
                    if s<16:
                        r=mp.sqrt(s);terms.append((1-r/4)**6*(1+mp.mpf(3)/2*r+mp.mpf(35)/48*s))
            rho=mp.mpf(9)/(16*mp.pi)*mp.fsum(terms)
        else:rho=mp.mpf(1)
        lam=rho*8*f(F(D))*32**2*mp.sin(mp.pi/32)**2
        symbol=np.array([[float(mp.mpf(1)/2+(f(F(float(c)))-mp.mpf(1)/2)*mp.exp(-lam*f(F(t)))) for c in c0] for t in (F(0),F(1,40),F(1,20))])
        return float(rho),float(lam),symbol,mp.nstr(rho,digits),mp.nstr(lam,digits)


def symbols(m,c0,D,packets,densities,a_ref):
    result={};saved={}
    for source in SOURCES:
        rho,lam,hp,rhostr,lamstr=independent_symbol(c0,D,source,80)
        rho100,lam100,hp100,_,_=independent_symbol(c0,D,source,100)
        assert rho==rho100 and lam==lam100 and np.array_equal(hp,hp100)
        primary=np.array([.5+(c0-.5)*math.exp(-lam*t) for t in TIMES])
        evaluation=max(norm(m,u-v) for u,v in zip(primary,hp));assert evaluation<=a_ref
        packet=packets[source];op=operators(m,np.array(packet['edges'],dtype=np.int64),map(F,packet['conductances_exact']))
        root=np.sqrt(m);phi=c0-.5
        mode=norm(m,op.symmetric_generator@(root*phi)/root+lam*phi)
        constant=norm(m,.5*(op.symmetric_generator@root)/root)
        scale=1e-10*max(1.,lam*norm(m,phi));assert mode+constant<scale
        rho_res=float(np.max(abs(densities[source]-rho)));assert rho_res<1e-10*max(1.,abs(rho))
        result[source]={'predicted_density':rho,'predicted_density_80digits':rhostr,'discrete_lambda':lam,'discrete_lambda_80digits':lamstr,
            'high_precision_80_100_rounded_equal':True,'symbol_evaluation_difference_mass_norm':evaluation,
            'eigenmode_residue_mass_norm':mode,'constant_residue_mass_norm':constant,'original_algebra_guard':scale,
            'raw_density_vs_independent_prediction_Linf':rho_res,'raw_density_vs_true_one_Linf':float(np.max(abs(densities[source]-1))),
            'raw_density_minimum':float(np.min(densities[source])),'raw_density_maximum':float(np.max(densities[source])),
            'raw_density_spatial_range':float(np.ptp(densities[source])),'density_physical_accuracy_threshold_defined':False}
        saved[source+'_symbol']=primary;saved[source+'_symbol_80digits']=hp
    return result,saved


def compare(m,c0,reference,a_ref,sigma_common,runs,paths,symbol_records,symbol_arrays):
    H=lambda c:metrics(m,c,c0)['H'];hm=lambda e:h_indicator(m,c0,e)
    native_sigma=math.sqrt(metrics(m,c0,c0)['variance']);b0=norm(m,reference[0]-c0)
    e={};time_rows=[];symbol_rows=[];observations={};comparisons=[]
    for source in SOURCES:
        residue=symbol_records[source]['eigenmode_residue_mass_norm']+symbol_records[source]['constant_residue_mass_norm']
        for K in KS:
            for k,t in enumerate(TIMES):
                i=k*K//2;u=paths[source,K]['states'][i];v=paths[source,K]['independent_states'][i]
                row=runs[source,K]['steps'][i-1] if i else None
                machine=row['cumulative_machine_indicator_absolute'] if row else 0.
                tail=row['cumulative_tail_indicator_absolute'] if row else 0.
                observed=norm(m,u-v);e[source,K,k]=machine+tail+observed
                budget=e[source,K,k]+2*a_ref+t*residue;diff=norm(m,u-symbol_arrays[source+'_symbol'][k])
                hs=1-math.exp(-2*symbol_records[source]['discrete_lambda']*t);hd=abs(H(u)-hs)
                symbol_rows.append({'source':source,'K':K,'time':t,'machine':machine,'tail':tail,'observed_complete_sequence':observed,
                    'e':e[source,K,k],'symbol_state_difference':diff,'indicator':budget,'state_control_passed':diff<=budget,
                    'symbol_H_difference':hd,'H_indicator':hm(budget),'H_control_passed':hd<=hm(budget)})
        for A,B in ((4,8),(8,16)):
            for k,t in enumerate(TIMES):
                u,v=paths[source,A]['states'][k*A//2],paths[source,B]['states'][k*B//2]
                budget=e[source,A,k]+e[source,B,k];diff=norm(m,u-v);hd=abs(H(u)-H(v))
                time_rows.append({'source':source,'coarse_K':A,'fine_K':B,'time':t,'difference':diff,'indicator':budget,
                    'state_control_passed':diff<=budget,'H_difference':hd,'H_indicator':hm(budget),'H_control_passed':hd<=hm(budget)})
        observations[source]=[]
        for k,t in enumerate(TIMES):
            c=paths[source,16]['states'][k*8];err=norm(m,c-reference[k]);hc=H(c);hr=H(reference[k])
            observations[source].append({'time':t,'concentration_error_absolute':err,'concentration_error_common_sigma16':err/sigma_common,
                'concentration_error_native_sigma32':err/native_sigma,'H':hc,'reference_H':hr,'H_error':abs(hc-hr),
                'raw_mass_change':float(m@(c-c0)),'minimum':float(c.min()),'maximum':float(c.max())})
    def classify(change,budget):
        return 'error_larger_resolvably' if change>budget else ('error_smaller_resolvably' if change < -budget else 'unresolved')
    for source in ('kernel','ledger'):
        for k,t in enumerate(TIMES):
            if not k:continue
            components={}
            for s in (source,'oracle'):
                for A,B in ((4,8),(8,16)):
                    components[f'{s}_d{A}_{B}']=norm(m,paths[s,A]['states'][k*A//2]-paths[s,B]['states'][k*B//2])
                for K in KS:components[f'{s}_e{K}']=e[s,K,k]
            components['four_reference_evaluation']=4*a_ref;components['two_initial_bridge']=2*b0
            components['both_residual_time']=t*sum(symbol_records[s]['eigenmode_residue_mass_norm']+symbol_records[s]['constant_residue_mass_norm'] for s in (source,'oracle'))
            budget=sum(components.values());assert all(v>=0 for v in components.values())
            a,b=observations[source][k],observations['oracle'][k]
            gc=a['concentration_error_common_sigma16']-b['concentration_error_common_sigma16'];gh=a['H_error']-b['H_error']
            diff=norm(m,paths[source,16]['states'][8*k]-paths['oracle',16]['states'][8*k])
            comparisons.append({'source':source,'time':t,'budget_components':components,'B_absolute':budget,
                'concentration_error_change_common_sigma16':gc,'concentration_indicator':budget/sigma_common,
                'concentration_decision':classify(gc,budget/sigma_common),'H_error_change':gh,'H_indicator':hm(budget),
                'H_decision':classify(gh,hm(budget)),'actual_source_state_difference_absolute':diff,
                'state_difference_resolved':diff>budget,'ledger_path_reference_is_exact_coefficient_identity':source=='ledger'})
    passed=all(v['state_control_passed'] and v['H_control_passed'] for v in time_rows+symbol_rows)
    return {'implementation_controls_passed':passed,'time_controls':time_rows,'symbol_controls':symbol_rows,
        'common_sigma16':sigma_common,'native_sigma32':native_sigma,'initial_bridge_absolute':b0,
        'observations':observations,'source_comparisons':comparisons,'density_and_symbol_records':symbol_records,
        'ledger_paths_reused_not_rerun':True,'new_actual_paths':3,'new_independent_paths':3,'new_action_intervals':28,
        'new_density_fields':2,'new_coefficient_packets':2,'new_geometry_graphs':0,'new_reference_PDE_solves':0,
        'recorded_checks_total':504,'general_density_or_smooth_flow_acceptance':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}
