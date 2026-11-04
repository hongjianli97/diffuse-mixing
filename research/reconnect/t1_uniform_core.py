"""Fixed material means, analytic controls and three-level T1 comparisons."""
import math
import numpy as np
from material_union import observe_path,group_sum
from voronoi_time_action import metrics
from shear_union64_study import classify

TIMES=np.array([0.,.025,.05])
D=.02


def pure_labels(N):
    a=(np.arange(N)+.5)/N
    return np.stack(np.meshgrid(a,a,indexing='xy'),axis=-1).reshape(-1,2)


def analytic_data(N):
    a=pure_labels(N);m=np.full(N*N,1/(N*N))
    amplitude=.2*(math.sin(math.pi/N)/(math.pi/N))**2
    mode=amplitude*np.sin(2*math.pi*a[:,0])*np.cos(2*math.pi*a[:,1])
    c0=.5+mode;continuum=8*math.pi**2*D;discrete=8*D*N*N*math.sin(math.pi/N)**2
    reference=np.array([.5+mode*math.exp(-continuum*t) for t in TIMES])
    symbol=np.array([.5+(c0-.5)*math.exp(-discrete*t) for t in TIMES])
    allowance=64*np.finfo(float).eps*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))
    return {'mass':m,'c0':c0,'reference':reference,'symbol':symbol,
      'continuum_lambda':continuum,'discrete_lambda':discrete,'reference_evaluation_indicator_absolute':float(allowance)}


def h_indicator(mass,c0,absolute):
    E0=metrics(mass,c0,c0)['variance']
    return (2*math.sqrt(float(mass.sum()))*max(1.,float(np.max(abs(c0))))*absolute+absolute**2)/E0


def compare(data,runs,paths,base=16):
    levels=sorted(data);assert levels==[base,2*base,4*base]
    mp,cp=data[base]['mass'],data[base]['c0'];sigma=math.sqrt(metrics(mp,cp,cp)['variance'])
    norm=lambda m,v:float(np.sqrt(m@(v*v)))
    cn=lambda v:norm(mp,v)/sigma
    H=lambda v:metrics(mp,v,cp)['H']
    common={};initial={};projection={};observations={};controls=[];time_rows=[]
    def allowance(N,K,k):
        if k==0:return 0.
        step=runs[N,K]['steps'][k*K//2-1]
        return step['cumulative_machine_indicator_absolute']+step['cumulative_tail_indicator_absolute']
    for N,d in data.items():
        m,c0=d['mass'],d['c0'];project=lambda v:observe_path(v,m,mp,side=base,ratio=N//base)
        native_sigma=math.sqrt(metrics(m,c0,c0)['variance'])
        projection[N]=project(d['reference']);initial[N]=project(c0[None,:])[0]
        common[f'n{N}_reference']=projection[N];common[f'n{N}_c0']=initial[N]
        for K in (4,8,16):
            for field in ('states','independent_states'):common[f'n{N}_k{K}_{field}']=project(paths[N,K][field])
            for k,t in enumerate(TIMES):
                i=k*K//2;c=paths[N,K]['states'][i];other=paths[N,K]['independent_states'][i]
                solver=allowance(N,K,k);observed=norm(m,c-other)
                ref_abs=d['reference_evaluation_indicator_absolute']
                # Residue is measured only after geometry is sealed; never fitted away.
                defect=float(t)*(d.get('eigenmode_residue_mass_norm',0.)+d.get('constant_residue_mass_norm',0.))
                budget=solver+observed+2*ref_abs+defect
                err=norm(m,c-d['symbol'][k])
                Hsymbol=1-math.exp(-2*d['discrete_lambda']*float(t))
                Herror=abs(metrics(m,c,c0)['H']-Hsymbol)
                controls.append({'N':N,'K':K,'time':float(t),'actual_vs_discrete_symbol_absolute':err,
                  'fixed_indicator_absolute':budget,'within_indicator':err<=budget,
                  'actual_vs_discrete_symbol_H':Herror,'fixed_H_indicator':h_indicator(m,c0,budget),
                  'H_within_indicator':Herror<=h_indicator(m,c0,budget)})
        observations[str(N)]=[]
        for k,t in enumerate(TIMES):
            c=paths[N,16]['states'][8*k];p=common[f'n{N}_k16_states'][8*k]
            native=norm(m,c-d['reference'][k])
            observations[str(N)].append({'time':float(t),'native_error_common_denominator':native/sigma,
              'native_error_native_denominator':native/native_sigma,'common_error':cn(p-data[base]['reference'][k]),
              'native_H':metrics(m,c,c0)['H'],'native_reference_H':metrics(m,d['reference'][k],c0)['H'],
              'common_H':H(p),'reference_common_H':H(data[base]['reference'][k]),
              'predicted_symbol_spatial_error_common_denominator':abs(math.exp(-d['discrete_lambda']*float(t))-math.exp(-d['continuum_lambda']*float(t)))*native_sigma/sigma,
              'native_raw_mass_change':float(m@(c-c0))})
        for A,B in ((4,8),(8,16)):
            for k,t in enumerate(TIMES):
                a=paths[N,A]['states'][k*A//2];b=paths[N,B]['states'][k*B//2]
                obs=sum(norm(m,paths[N,K]['states'][k*K//2]-paths[N,K]['independent_states'][k*K//2]) for K in (A,B))
                budget=allowance(N,A,k)+allowance(N,B,k)+obs
                diff=norm(m,a-b)
                time_rows.append({'N':N,'coarse_K':A,'fine_K':B,'time':float(t),'raw_difference_absolute':diff,
                  'raw_difference_relative':diff/sigma,'fixed_numerical_indicator_absolute':budget,'within_numerical_indicator':diff<=budget})
    rows=[]
    for C,N in zip(levels[:-1],levels[1:]):
        for k,t in enumerate(TIMES):
            coarse=common[f'n{C}_k16_states'][8*k];fine=common[f'n{N}_k16_states'][8*k];ref=data[base]['reference'][k]
            solver=observed=observedH=0.;native_observed=0.
            for Q in (C,N):
                for K in (8,16):
                    i=k*K//2;solver+=allowance(Q,K,k)
                    p,q=common[f'n{Q}_k{K}_states'][i],common[f'n{Q}_k{K}_independent_states'][i]
                    observed+=cn(p-q);observedH+=abs(H(p)-H(q))
                    native_observed+=norm(data[Q]['mass'],paths[Q,K]['states'][i]-paths[Q,K]['independent_states'][i])/sigma
            bridge=[data[base]['reference'][0]-cp,initial[N]-cp,projection[N][0]-initial[N]]
            evaluation=sum(data[Q]['reference_evaluation_indicator_absolute'] for Q in (C,N))
            common_parts={'coarse8_16':cn(common[f'n{C}_k8_states'][4*k]-coarse),
              'fine8_16':cn(common[f'n{N}_k8_states'][4*k]-fine),'twice_reference_evaluation':2*evaluation/sigma,
              'both_reference_projections':sum(cn(projection[Q][k]-ref) for Q in (C,N)),
              'initial_bridges':sum(cn(v) for v in bridge),'machine_and_tails':solver/sigma,'observed_complete_sequences':observed}
            native_parts={'coarse8_16':norm(data[C]['mass'],paths[C,8]['states'][4*k]-paths[C,16]['states'][8*k])/sigma,
              'fine8_16':norm(data[N]['mass'],paths[N,8]['states'][4*k]-paths[N,16]['states'][8*k])/sigma,
              'twice_reference_evaluation':2*evaluation/sigma,
              'both_initial_reference_bridges':sum(norm(data[Q]['mass'],data[Q]['reference'][0]-data[Q]['c0']) for Q in (C,N))/sigma,
              'machine_and_tails':solver/sigma,'observed_complete_sequences':native_observed}
            Hparts={'coarse8_16':abs(H(common[f'n{C}_k8_states'][4*k])-H(coarse)),
              'fine8_16':abs(H(common[f'n{N}_k8_states'][4*k])-H(fine)),
              'twice_reference_evaluation':h_indicator(mp,cp,2*evaluation),
              'both_reference_projections':sum(abs(H(projection[Q][k])-H(ref)) for Q in (C,N)),
              'initial_bridges':abs(H(data[base]['reference'][0])-H(cp))+abs(H(initial[N])-H(cp))+abs(H(projection[N][0])-H(initial[N])),
              'machine_and_tails':h_indicator(mp,cp,solver),'observed_complete_sequences':observedH}
            ec,ef=cn(coarse-ref),cn(fine-ref)
            nc=norm(data[C]['mass'],paths[C,16]['states'][8*k]-data[C]['reference'][k])/sigma
            nf=norm(data[N]['mass'],paths[N,16]['states'][8*k]-data[N]['reference'][k])/sigma
            row={'coarse_N':C,'fine_N':N,'time':float(t),'coarse_common_error':ec,'fine_common_error':ef,
              'native_coarse_error':nc,'native_fine_error':nf,'common_gain':ec-ef,'native_gain':nc-nf,
              'coarse_H_common':H(coarse),'fine_H_common':H(fine),'reference_H_common':H(ref),
              'H_gain':abs(H(coarse)-H(ref))-abs(H(fine)-H(ref)),
              'common_indicator_components':common_parts,'native_indicator_components':native_parts,'H_indicator_components':Hparts,
              'common_error_ratio':None if ef==0 else ec/ef,'native_error_ratio':None if nf==0 else nc/nf}
            for label in ('common','native','H'):
                row[label+'_indicator']=sum(row[label+'_indicator_components'].values())
                row[label+'_decision']=classify(row[label+'_gain'],row[label+'_indicator'])
            rows.append(row)
    spatial=all(row[label+'_decision']=='fine_better_resolvably' for row in rows if row['time']>0 for label in ('common','native','H'))
    temporal=all(v['within_numerical_indicator'] for v in time_rows)
    symbol=all(v['within_indicator'] and v['H_within_indicator'] for v in controls)
    passed=spatial and temporal and symbol
    return {'common_initial_fluctuation_norm':sigma,'per_grid_observations':observations,'adjacent_comparisons':rows,
      'time_controls':time_rows,'discrete_symbol_controls':controls,'all_spatial_gains_resolved':spatial,
      'all_static_time_differences_within_numerical_indicators':temporal,'all_discrete_symbol_controls_passed':symbol,
      'T1_registered_case_passed':passed,'comparison_decision':'T1_static_uniform_material_mean_control_passed' if passed else 'T1_control_failed_or_unresolved',
      'general_temporal_order_claimed':False,'general_spatial_order_claimed':False,'indicators_are_certified_total_error_bounds':False,
      'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False},common
