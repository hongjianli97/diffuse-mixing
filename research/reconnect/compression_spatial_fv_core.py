"""Same-flow two-grid analysis: original raw material projection and native control."""
import math
import numpy as np
from voronoi_spatial_projection import validate_record,project_path
from voronoi_time_action import metrics
from shear_union64_study import classify


def compare(d,coarse_ref,runs,paths):
    child_sum,rounding,mass_audit=validate_record(d)
    mp,cp=d['parent_mass'],d['parent_c0'];mf,cf=d['mass'],d['c0']
    E0=metrics(mp,cp,cp)['variance'];sigma=math.sqrt(E0)
    norm=lambda m,v:float(np.sqrt(m@(v*v)))/sigma;cn=lambda v:norm(mp,v);H=lambda c:metrics(mp,c,cp)['H']
    hf=lambda c:metrics(mf,c,cf)['H'];factor=math.sqrt(float(np.max(child_sum/mp)))
    refs={128:coarse_ref['best_projected_states'],256:d['reference_n1024-k1024_states']}
    space={128:coarse_ref['space_difference'],256:d['reference_n512-k256_states']-d['reference_n1024-k256_states']}
    time={128:coarse_ref['time_fine_difference'],256:d['reference_n1024-k512_states']-refs[256]}
    common={'reference_fine_common':project_path(refs[256],d),'fine_c0_common':project_path(cf[None,:],d)[0]}
    for K in (16,32):
        for field in ('states','independent_states'):common[f'fine_k{K}_{field}']=project_path(paths[256,K][field],d)
    refp=refs[128];refproj=common['reference_fine_common']
    bridges=np.array([refp[0]-cp,common['fine_c0_common']-cp,refproj[0]-common['fine_c0_common']])
    common['initial_bridges']=bridges;common['reference_projection_difference']=refproj-refp
    Hb=abs(H(refp[0])-H(cp))+abs(H(common['fine_c0_common'])-H(cp))+abs(H(refproj[0])-H(common['fine_c0_common']))
    native_sigmas={128:sigma,256:math.sqrt(metrics(mf,cf,cf)['variance'])};observations={};rows=[];times=[]
    for N,m,c0 in [(128,mp,cp),(256,mf,cf)]:
        observations[str(N)]=[]
        for k,t in enumerate(coarse_ref['snapshot_time']):
            c=paths[N,32]['states'][16*k];p=c if N==128 else common['fine_k32_states'][16*k]
            observations[str(N)].append({'time':float(t),'native_error_common_denominator':norm(m,c-refs[N][k]),
              'native_error_native_denominator':float(np.sqrt(m@((c-refs[N][k])**2)))/native_sigmas[N],
              'common_error':cn(p-refp[k]),'native_H':metrics(m,c,c0)['H'],'native_reference_H':metrics(m,refs[N][k],c0)['H'],
              'common_H':H(p),'reference_common_H':H(refp[k]),'native_raw_mass_change':float(m@(c-c0)),
              'common_raw_mass_change':float(mp@(p-cp))})
    for k,t in enumerate(coarse_ref['snapshot_time']):
        coarse=paths[128,32]['states'][16*k];coarse16=paths[128,16]['states'][8*k]
        fine=common['fine_k32_states'][16*k];fine16=common['fine_k16_states'][8*k];ref=refp[k]
        solver_common=solver_native=obs_common=obs_native=obsH=0.
        for N,m,fact in [(128,mp,1.),(256,mf,factor)]:
            for K in (16,32):
                i=k*K//2
                if not i:continue
                row=runs[N,K]['steps'][i-1];allow=row['cumulative_machine_indicator_absolute']+row['cumulative_tail_indicator_absolute']
                solver_native+=allow;solver_common+=fact*allow
                a=paths[N,K]['states'][i];b=paths[N,K]['independent_states'][i]
                obs_native+=norm(m,a-b)
                ac=a if N==128 else common[f'fine_k{K}_states'][i]
                bc=b if N==128 else common[f'fine_k{K}_independent_states'][i]
                obs_common+=cn(ac-bc);obsH+=abs(H(ac)-H(bc))
        cp_parts={'coarse16_32':cn(coarse16-coarse),'fine16_32_common':cn(fine16-fine),
          'twice_parent_reference_space':2*cn(space[128][k]),'twice_parent_reference_time':2*cn(time[128][k]),
          'coarse_reference_projection':0.,'fine_reference_projection':cn(refproj[k]-ref),
          'initial_bridges':sum(cn(v) for v in bridges),'solver_machine_and_tails':solver_common/sigma,'observed_whole_solvers':obs_common}
        native_parts={'coarse16_32':norm(mp,coarse16-coarse),
          'fine16_32':norm(mf,paths[256,16]['states'][8*k]-paths[256,32]['states'][16*k]),
          'twice_both_reference_space_time':2*sum(norm(m,v[k]) for N,m in [(128,mp),(256,mf)] for v in (space[N],time[N])),
          'both_initial_reference_bridges':norm(mp,refs[128][0]-cp)+norm(mf,refs[256][0]-cf),
          'solver_machine_and_tails':solver_native/sigma,'observed_whole_solvers':obs_native}
        Hparts={'coarse16_32':abs(H(coarse16)-H(coarse)),'fine16_32_common':abs(H(fine16)-H(fine)),
          'twice_parent_reference_space':2*abs(H(coarse_ref['n512-k256_projected_states'][k])-H(coarse_ref['n1024-k256_projected_states'][k])),
          'twice_parent_reference_time':2*abs(H(coarse_ref['n1024-k512_projected_states'][k])-H(ref)),
          'coarse_reference_projection':0.,'fine_reference_projection':abs(H(refproj[k])-H(ref)),
          'initial_bridges':Hb,'observed_whole_solvers':obsH,
          'solver_machine_to_H':(2*math.sqrt(float(mp.sum()))*max(1.,float(np.max(abs(cp))))*solver_common+solver_common**2)/E0}
        ec,ef=cn(coarse-ref),cn(fine-ref);nc=native_f=0.
        nc=norm(mp,coarse-refs[128][k]);native_f=norm(mf,paths[256,32]['states'][16*k]-refs[256][k])
        ratio=lambda a,b:None if b==0 else a/b
        row={'time':float(t),'coarse32_error_common':ec,'fine32_error_common':ef,'common_gain':ec-ef,
          'common_indicator_components':cp_parts,'common_indicator':sum(cp_parts.values()),
          'native_coarse32_error':nc,'native_fine32_error':native_f,'native_gain':nc-native_f,
          'native_indicator_components':native_parts,'native_indicator':sum(native_parts.values()),
          'coarse32_H_common':H(coarse),'fine32_H_common':H(fine),'reference_H_common':H(ref),
          'H_gain':abs(H(coarse)-H(ref))-abs(H(fine)-H(ref)),'H_indicator_components':Hparts,'H_indicator':sum(Hparts.values()),
          'common_error_ratio':ratio(ec,ef),'native_error_ratio':ratio(nc,native_f)}
        for label in ('common','native','H'):row[label+'_decision']=classify(row[label+'_gain'],row[label+'_indicator'])
        rows.append(row)
        times.append({'time':float(t),'coarse16_32':cn(coarse16-coarse),'fine16_32_common':cn(fine16-fine),
          'fine16_32_native':norm(mf,paths[256,16]['states'][8*k]-paths[256,32]['states'][16*k])})
    active=rows[1:];resolved=all(v[label+'_decision']=='fine_better_resolvably' for v in active for label in ('common','native','H'))
    adverse=any(v[label+'_decision']=='fine_worse_resolvably' for v in active for label in ('common','native','H'))
    decision='compression_spatial_improvement_resolved_with_native_and_time_controls' if resolved else ('adverse_at_a_checkpoint' if adverse else 'compression_spatial_comparison_inconclusive')
    return {'parent_initial_fluctuation_norm':sigma,'native_initial_fluctuation_norms':{str(N):v for N,v in native_sigmas.items()},
      'parent_mass_audit':mass_audit,'projection_solver_norm_factor':factor,'per_grid_observations':observations,
      'comparison_observations':rows,'time_controls':times,'all_common_native_H_gains_resolved_at_both_times':resolved,
      'comparison_decision':decision,'indicators_are_certified_total_error_bounds':False,'two_levels_certify_general_spatial_order':False,
      'Richardson_or_fitted_state_used':False,'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False},common
