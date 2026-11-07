"""Fixed source identities, explicit-duration sparse actions and conservative diagnostics."""
from pathlib import Path
from fractions import Fraction as F
import sys,math
import numpy as np
from audit_route_decision import ROOT,read,digest
sys.path.insert(0,str(ROOT/'research/rebuild'))
from becker import Becker
from voronoi_time_action import operators,metrics
from explicit_duration_action import ordered_path_duration
from shear_union64_study import arrays
from t1_uniform_core import h_indicator

COUNTS=(4,8,16);T=F(1,10);SEED=20261101


def sample_frames(mu_edges,flow,rows=8):
    e=np.asarray(mu_edges);centres=(e[:-1]+e[1:])/2;cols=len(centres);y=(np.arange(rows)+.5)/rows;cases={}
    for K in COUNTS:
        frames=[]
        for i in range(2*K+1):
            t=T*i/(2*K);xx=flow.x_mu(centres,float(t));rho=flow.rho_mu(centres,float(t));bounds=flow.x_mu(e[[0,-1]],float(t))
            frames.append({'time_exact':str(t),'positions':np.stack(np.meshgrid(xx,y,indexing='xy'),axis=-1).reshape(-1,2).tolist(),
              'rho':np.tile(rho,rows).tolist(),'x_bounds':bounds.tolist(),'period_y':1.,'D':.02})
        cases[str(K)]={'K':K,'frames':frames}
    return {'ids':list(range(cols*rows)),'columns':cols,'rows':rows,'cases':cases,'point_rule':'fixed mass-label midpoint trajectory,not geometriccellcentre',
      'mass_c0_reference_internal_material_faces_or_F_C_supplied_to_builder':False}


def extrude(column,rows=8):
    m=np.asarray(column['mass']);c0=np.asarray(column['c0']);ref=np.asarray(column['reference'])
    if m.ndim!=1 or c0.shape!=m.shape or ref.shape!=m.shape or np.any(m<=0):raise ValueError('complete original material column means required')
    return {'mass':np.tile(m/rows,rows),'c0':np.tile(c0,rows),'reference':np.tile(ref,rows),'ids':np.arange(len(m)*rows,dtype=np.int64)}


def observe_columns(values,column_mass,rows=8):
    a=np.asarray(values);mc=np.asarray(column_mass);cols=len(mc)
    if a.shape[-1]!=cols*rows:raise ValueError('complete transverse material copies required')
    return (a.reshape(*a.shape[:-1],rows,cols)*(mc/rows)).sum(axis=-2)/mc


def native_run(out,K,data,index,guard=lambda *a:None,replay=False,T=T):
    folder=out/f'k{K}';m,c0=data['mass'],data['c0']
    def load(step):
        item=index[f'k{K}-step{step:02d}'];p=out/item['file'];assert digest(p)==item['sha256'];packet=read(p)
        return operators(m,np.array(packet['edges'],dtype=np.int64),map(F,packet['conductances_exact']))
    def persist(step,op,weights):
        a={'symmetric_data':op.symmetric_generator.data,'symmetric_indices':op.symmetric_generator.indices,'symmetric_indptr':op.symmetric_generator.indptr,
          'uniformization_data':op.uniformization_matrix.data,'uniformization_indices':op.uniformization_matrix.indices,'uniformization_indptr':op.uniformization_matrix.indptr,
          'shape':np.array(op.symmetric_generator.shape),'edges':op.edges,'g_float':op.conductance,'weights':weights}
        p=folder/f'action-step{step:02d}.npz'
        if replay:
            old=arrays(p)
            for key,value in a.items():assert np.array_equal(value,old[key]),key
        else:np.savez_compressed(p,**a)
    np.random.seed(SEED);run,path=ordered_path_duration(m,c0,load,K,persist,lambda stage,i,total:guard(f'k{K}-{stage}',i,total),T=T)
    for i,row in enumerate(run['steps']):row.update(coefficient_cache_sha256=index[f'k{K}-step{i:02d}']['sha256'],midpoint_time_saved=index[f'k{K}-step{i:02d}']['midpoint_time_saved'])
    return run,path


def evaluate(data,column,runs,paths,ref_meta):
    m,c0,ref=data['mass'],data['c0'],data['reference'];mc,cc=column['mass'],column['c0'];norm=lambda m,v:float(np.sqrt(m@(v*v)))
    sigma=math.sqrt(metrics(mc,cc,cc)['variance']);H=lambda u:metrics(m,u,c0)['H'];R=ref_meta['reference_space_time_indicator_absolute']
    p0=observe_columns(c0,mc);bridge=norm(mc,p0-cc);rows=[];numeric={}
    for K in COUNTS:
        run,path=runs[K],paths[K];u=path['states'][-1];v=path['independent_states'][-1];step=run['steps'][-1]
        machine=float(step['cumulative_machine_indicator_absolute']);tail=float(step['cumulative_tail_indicator_absolute']);delta=norm(m,u-v)
        numeric[K]=machine+tail+delta
        p=observe_columns(u,mc);rows.append({'K':K,'time':float(T),'native_error_relative':norm(m,u-ref)/sigma,'column_observation_error_relative':norm(mc,p-column['reference'])/sigma,
          'H':H(u),'reference_H':H(ref),'raw_mass_change':float(m@(u-c0)),'minimum':float(u.min()),'maximum':float(u.max()),
          'primary_control_difference_relative':delta/sigma,'numeric_indicator_relative':numeric[K]/sigma,'reference_indicator_relative':R/sigma,
          'minimum_all_times':min(row['minimum'] for row in run['steps']),'maximum_all_times':max(row['maximum'] for row in run['steps']),
          'all_primary_control_differences_within_original_allowance':all(row['primary_control_mass_norm_difference']<=row['cumulative_machine_indicator_absolute']+row['cumulative_tail_indicator_absolute'] for row in run['steps'])})
    u4,u8,u16=[paths[K]['states'][-1] for K in COUNTS];d48=norm(m,u4-u8);d816=norm(m,u8-u16);h48=abs(H(u4)-H(u8));h816=abs(H(u8)-H(u16))
    parts={'twice_all_path_numeric':2*sum(numeric.values()),'twice_saved_reference_space_time':2*R,'original_column_initial_observation_bridge':bridge}
    for K in COUNTS:parts[f'k{K}_column_observation_rounding_bridge']=norm(mc,observe_columns(paths[K]['states'][-1],mc)-paths[K]['states'][-1].reshape(8,-1)[0])
    budget=sum(parts.values());hbudget=float(h_indicator(m,c0,budget));state_pass=d48-d816>budget;Hpass=h48-h816>hbudget
    sequence=all(v['all_primary_control_differences_within_original_allowance'] for v in rows);passed=sequence and state_pass and Hpass
    return {'original_column_initial_fluctuation_norm':sigma,'observations':rows,'time_control':{'state4_8_relative':d48/sigma,'state8_16_relative':d816/sigma,
      'H4_8':h48,'H8_16':h816,'fixed_state_decrease_indicator_relative':budget/sigma,'fixed_H_decrease_indicator':hbudget,
      'components_absolute':parts,'state_time_decrease_resolved':state_pass,'H_time_decrease_resolved':Hpass},
      'all_complete_sequence_controls_passed':sequence,'registered_moving_Becker_oracle_discriminator_passed':passed,
      'comparison_decision':'moving_Becker_oracle_discriminator_passed' if passed else 'moving_Becker_control_failed_or_unresolved',
      'single_N_not_spatial_convergence':True,'no_absolute_PDE_accuracy_threshold_invented':True,'indicators_are_certified_total_error_bounds':False,
      'old_true_face_FV_not_current_point_backend':True,'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}
