"""Three physical spatial levels for the same compression; raw union observations."""
import math
import numpy as np
from material_union import define_union,observe_path,group_sum
from voronoi_time_action import metrics
from shear_union64_study import classify


def native_data(d,ref):
    for name in ('mass','c0','ids','material_label_centers'):
        np.testing.assert_array_equal(d['parent_'+name],ref[name])
    np.testing.assert_array_equal(d['parent_reference_states'],ref['best_projected_states'])
    m,c,r=define_union(ref['mass'],ref['c0'],ref['best_projected_states'])
    project=lambda v:observe_path(v,ref['mass'],m)
    u={'mass':m,'c0':c,'reference':r,'space':project(ref['space_difference']),
       'time':project(ref['time_fine_difference']),
       'reference_space_coarse':project(ref['n512-k256_projected_states']),
       'reference_space_fine':project(ref['n1024-k256_projected_states']),
       'reference_time_check':project(ref['n1024-k512_projected_states'])}
    return {64:u,128:{'mass':ref['mass'],'c0':ref['c0'],'reference':ref['best_projected_states'],
          'space':ref['space_difference'],'time':ref['time_fine_difference']},
       256:{'mass':d['mass'],'c0':d['c0'],'reference':d['reference_n1024-k1024_states'],
          'space':d['reference_n512-k256_states']-d['reference_n1024-k256_states'],
          'time':d['reference_n1024-k512_states']-d['reference_n1024-k1024_states']}}


def compare(data,runs,paths):
    mp,cp=data[64]['mass'],data[64]['c0'];E0=metrics(mp,cp,cp)['variance'];sigma=math.sqrt(E0)
    norm=lambda m,v:float(np.sqrt(m@(v*v)))/sigma
    cnorm=lambda v:norm(mp,v)
    H=lambda c:metrics(mp,c,cp)['H']
    common={};projection={};initial={};factors={};native_sigma={};per_grid={}
    for N,d in data.items():
        m,c0=d['mass'],d['c0'];ratio=N//64;project=lambda v:observe_path(v,m,mp,ratio=ratio)
        ref=project(d['reference']);projection[N]=ref;initial[N]=project(c0[None,:])[0]
        factors[N]=math.sqrt(float(np.max(group_sum(m,64,ratio)/mp)))
        native_sigma[N]=math.sqrt(metrics(m,c0,c0)['variance'])
        common[f'n{N}_reference']=ref;common[f'n{N}_c0']=initial[N]
        for K in (16,32):
            for field in ('states','independent_states'):
                common[f'n{N}_k{K}_{field}']=project(paths[N,K][field])
        per_grid[str(N)]=[]
        for k,t in enumerate((0.,.025,.05)):
            c=paths[N,32]['states'][16*k];p=common[f'n{N}_k32_states'][16*k];refnative=d['reference'][k]
            per_grid[str(N)].append({'time':t,'native_error_common_denominator':norm(m,c-refnative),
              'native_error_native_denominator':float(np.sqrt(m@((c-refnative)**2)))/native_sigma[N],
              'common_error':cnorm(p-data[64]['reference'][k]),'native_H':metrics(m,c,c0)['H'],
              'native_reference_H':metrics(m,refnative,c0)['H'],'common_H':H(p),'reference_common_H':H(data[64]['reference'][k]),
              'native_raw_mass_change':float(m@(c-c0)),'common_raw_mass_change':float(mp@(p-cp))})
    rows=[]
    for C,N in ((64,128),(128,256)):
        mC,mN=data[C]['mass'],data[N]['mass'];bridges=np.array([data[64]['reference'][0]-cp,initial[N]-cp,projection[N][0]-initial[N]])
        common[f'pair{C}_{N}_initial_bridges']=bridges
        Hbridge=abs(H(data[64]['reference'][0])-H(cp))+abs(H(initial[N])-H(cp))+abs(H(projection[N][0])-H(initial[N]))
        for k,t in enumerate((0.,.025,.05)):
            c=common[f'n{C}_k32_states'][16*k];f=common[f'n{N}_k32_states'][16*k];ref=data[64]['reference'][k]
            c16=common[f'n{C}_k16_states'][8*k];f16=common[f'n{N}_k16_states'][8*k]
            solver_abs=solver_native_abs=observed=observed_native=observedH=0.
            for Q in (C,N):
                for K in (16,32):
                    i=k*K//2
                    if not i:continue
                    step=runs[Q,K]['steps'][i-1]
                    a=step['cumulative_machine_indicator_absolute']+step['cumulative_tail_indicator_absolute']
                    solver_abs+=factors[Q]*a;solver_native_abs+=a
                    u,v=common[f'n{Q}_k{K}_states'][i],common[f'n{Q}_k{K}_independent_states'][i]
                    observed+=cnorm(u-v);observedH+=abs(H(u)-H(v))
                    observed_native+=norm(data[Q]['mass'],paths[Q,K]['states'][i]-paths[Q,K]['independent_states'][i])
            components={'coarse16_32':cnorm(c16-c),'fine16_32':cnorm(f16-f),
              'twice_reference_space':2*cnorm(data[64]['space'][k]),'twice_reference_time':2*cnorm(data[64]['time'][k]),
              'coarse_reference_projection':cnorm(projection[C][k]-ref),'fine_reference_projection':cnorm(projection[N][k]-ref),
              'initial_bridges':sum(cnorm(v) for v in bridges),'solver_machine_and_tails':solver_abs/sigma,'observed_solver_sequences':observed}
            native_components={'coarse16_32':norm(mC,paths[C,16]['states'][8*k]-paths[C,32]['states'][16*k]),
              'fine16_32':norm(mN,paths[N,16]['states'][8*k]-paths[N,32]['states'][16*k]),
              'twice_reference_space_time':2*sum(norm(data[Q]['mass'],data[Q][field][k]) for Q in (C,N) for field in ('space','time')),
              'initial_reference_bridges':sum(norm(data[Q]['mass'],data[Q]['reference'][0]-data[Q]['c0']) for Q in (C,N)),
              'solver_machine_and_tails':solver_native_abs/sigma,'observed_solver_sequences':observed_native}
            rt=data[64]['reference_time_check'][k];rs=data[64]['reference_space_coarse'][k];rb=data[64]['reference_space_fine'][k]
            hp={'coarse16_32':abs(H(c16)-H(c)),'fine16_32':abs(H(f16)-H(f)),
              'twice_reference_space':2*abs(H(rs)-H(rb)),'twice_reference_time':2*abs(H(rt)-H(ref)),
              'coarse_reference_projection':abs(H(projection[C][k])-H(ref)),
              'fine_reference_projection':abs(H(projection[N][k])-H(ref)),
              'initial_bridges':Hbridge,'observed_solver_sequences':observedH,
              'solver_machine_to_H':(2*math.sqrt(float(mp.sum()))*max(1.,float(np.max(abs(cp))))*solver_abs+solver_abs**2)/E0}
            ec,ef=cnorm(c-ref),cnorm(f-ref)
            nc=norm(mC,paths[C,32]['states'][16*k]-data[C]['reference'][k]);nf=norm(mN,paths[N,32]['states'][16*k]-data[N]['reference'][k])
            hg=abs(H(c)-H(ref))-abs(H(f)-H(ref))
            ratio=lambda a,b:None if b==0 else a/b
            order=lambda a,b:None if min(a,b)<=0 else math.log2(a/b)
            row={'coarse_N':C,'fine_N':N,'time':t,'coarse_common_error':ec,'fine_common_error':ef,
              'common_gain':ec-ef,'common_indicator_components':components,'common_indicator':sum(components.values()),
              'native_coarse_error':nc,'native_fine_error':nf,'native_gain':nc-nf,'native_indicator_components':native_components,
              'native_indicator':sum(native_components.values()),'coarse_H_common':H(c),'fine_H_common':H(f),'reference_H_common':H(ref),
              'H_gain':hg,'H_indicator_components':hp,'H_indicator':sum(hp.values()),
              'common_error_ratio':ratio(ec,ef),'native_error_ratio':ratio(nc,nf),
              'common_observed_log2_ratio':order(ec,ef),'native_observed_log2_ratio':order(nc,nf)}
            for label in ('common','native','H'):row[label+'_decision']=classify(row[label+'_gain'],row[label+'_indicator'])
            rows.append(row)
    active=[r for r in rows if r['time']>0]
    resolved=all(r[label+'_decision']=='fine_better_resolvably' for r in active for label in ('common','native','H'))
    adverse=any(r[label+'_decision']=='fine_worse_resolvably' for r in active for label in ('common','native','H'))
    decision='compression_three_level_improvement_resolved_with_native_control' if resolved else ('adverse_at_a_checkpoint' if adverse else 'three_level_comparison_inconclusive')
    result={'common_initial_fluctuation_norm':sigma,'native_initial_fluctuation_norms':{str(k):v for k,v in native_sigma.items()},
      'raw_projection_solver_factors':{str(k):v for k,v in factors.items()},'per_grid_observations':per_grid,'adjacent_comparisons':rows,
      'comparison_decision':decision,'all_adjacent_times_common_native_H_resolved':resolved,
      'empirical_indicators_are_certified_bounds':False,'observed_ratios_certify_general_order':False,
      'new_reference_PDE_solves':0,'old_candidate_reruns':0,'general_smooth_flow_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    return result,common

