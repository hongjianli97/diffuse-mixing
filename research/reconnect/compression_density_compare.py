"""Frozen same-N64 source comparison with all original positive error terms."""
import math
import numpy as np
from voronoi_time_action import metrics

SOURCES=('oracle','kernel','ledger')
KS=(16,32)
TIMES=(0.,.025,.05)


def classification(change,budget):
    if change>budget:return 'error_larger_resolvably'
    if change < -budget:return 'error_smaller_resolvably'
    return 'unresolved'


def reference_parts(data,k,norm,H,c0):
    cp={'four_reference_space':4*norm(data['space'][k]),'four_reference_time':4*norm(data['time'][k]),
        'two_initial_bridge':2*norm(data['reference'][0]-c0),'same_N64_projection_identity':0.}
    hp={'two_reference_space_same_basis':2*abs(H(data['reference_space_coarse'][k])-H(data['reference_space_fine'][k])),
        'two_reference_time_same_basis':2*abs(H(data['reference_time_check'][k])-H(data['reference'][k])),
        'two_initial_bridge':2*abs(H(data['reference'][0])-H(c0)),'same_N64_projection_identity':0.}
    return cp,hp


def compare(data,runs,paths):
    m,c0=data['mass'],data['c0'];V0=metrics(m,c0,c0)['variance'];sigma=math.sqrt(V0)
    norm=lambda v:float(np.sqrt(m@(v*v)));H=lambda v:metrics(m,v,c0)['H']
    hm=lambda e:(2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))*e+e*e)/V0
    observations={};time_rows=[];numerical={}
    for source in SOURCES:
        observations[source]=[]
        for k,t in enumerate(TIMES):
            u=paths[source,32]['states'][16*k];ref=data['reference'][k];err=norm(u-ref)
            observations[source].append({'time':t,'concentration_error_absolute':err,'concentration_error_relative':err/sigma,
                'H':H(u),'reference_H':H(ref),'H_error':abs(H(u)-H(ref)),'raw_mass_change':float(m@(u-c0)),
                'minimum':float(u.min()),'maximum':float(u.max())})
            time_rows.append({'source':source,'time':t,'state_difference_16_32_absolute':norm(paths[source,16]['states'][8*k]-u),
                'H_difference_16_32':abs(H(paths[source,16]['states'][8*k])-H(u)),'time_order_or_decline_certified':False})
            for K in KS:
                a,b=paths[source,K]['states'][k*K//2],paths[source,K]['independent_states'][k*K//2]
                row=runs[source,K]['steps'][k*K//2-1] if k else None
                mt=0. if row is None else row['cumulative_machine_indicator_absolute']+row['cumulative_tail_indicator_absolute']
                numerical[source,K,k]=(mt,norm(a-b),abs(H(a)-H(b)))
    rows=[]
    for source in ('kernel','ledger'):
        for k,t in enumerate(TIMES):
            if not k:continue
            cp,hp=reference_parts(data,k,norm,H,c0)
            total_mt=0.
            for s in (source,'oracle'):
                a,b=paths[s,16]['states'][8*k],paths[s,32]['states'][16*k]
                cp[s+'_16_32']=norm(a-b);hp[s+'_16_32']=abs(H(a)-H(b))
                for K in KS:
                    mt,d,dh=numerical[s,K,k];cp[f'{s}_e{K}']=mt+d;hp[f'{s}_complete_sequence_H{K}']=dh;total_mt+=mt
            hp['both_machine_tail_to_H']=hm(total_mt)
            assert all(v>=0 for v in (*cp.values(),*hp.values()))
            bc=sum(cp.values())/sigma;bh=sum(hp.values());a,b=observations[source][k],observations['oracle'][k]
            gc=a['concentration_error_relative']-b['concentration_error_relative'];gh=a['H_error']-b['H_error']
            sd=norm(paths[source,32]['states'][16*k]-paths['oracle',32]['states'][16*k])
            rows.append({'source':source,'time':t,'concentration_parts_absolute':cp,'H_parts':hp,'concentration_indicator':bc,'H_indicator':bh,
                'concentration_error_change':gc,'H_error_change':gh,'concentration_decision':classification(gc,bc),'H_decision':classification(gh,bh),
                'actual_source_state_difference_absolute':sd,'state_source_difference_resolved':sd>bc*sigma})
    return {'common_sigma64':sigma,'observations':observations,'time_differences':time_rows,'source_comparisons':rows,
        'any_source_comparison_unresolved':any(r[name]=='unresolved' for r in rows for name in ('concentration_decision','H_decision')),
        'time_order_or_decline_certified':False,'static_semigroup_rule_applied':False,'indicators_are_certified_PDE_error_bounds':False,
        'new_density_fields':96,'new_coefficient_packets':96,'new_actual_paths':4,'new_independent_paths':4,'new_action_intervals':96,
        'new_geometry_graphs':0,'new_reference_PDE_solves':0,'old_paths_rerun':0,'recorded_checks_total':506,
        'general_density_physical_acceptance':False,'stage3':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}
