"""Independent all-cell strip geometry, exact rates/full paths and saved-reference audit."""
from fractions import Fraction as F
from pathlib import Path
import argparse,json,math,resource,time
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save as raw_save,digest,preservation
from audit_bounded_geometry import audit_geometry
from becker_point_study import SOURCE,load_all,reference_indicators
from becker_point_completion import density_diagnostics
from becker_point_core import Becker,COUNTS,evaluate,extrude
from affine_shear_completion import plain_json
from shear_union64_study import arrays
from t2_d0_core import bitwise_same


def save(p,obj):raw_save(p,plain_json(obj))


def audit_paths(out,data,world,index):
    m,c0=data['mass'],data['c0'];mm=[F(float(v)) for v in m];norm=lambda u:float(np.sqrt(m@(u*u)))
    maxdelta=maxaction=maxdrift=0.;counts=0
    for K in COUNTS:
        folder=out/f'k{K}';path=arrays(folder/'trajectory-states.npz');run=read(folder/'native-results.json');previous=c0.copy();machine=tail=0.
        np.testing.assert_array_equal(path['states'][0],c0);np.testing.assert_array_equal(path['independent_states'][0],c0)
        variance=float(m@((c0-float(m@c0/m.sum()))**2));V0=variance
        for step,row in enumerate(run['steps']):
            item=index[f'k{K}-step{step:02d}'];f=world['cases'][str(K)]['frames'][2*step+1];packet=read(out/item['file'])
            assert digest(out/item['file'])==item['sha256']==row['coefficient_cache_sha256']
            assert item['midpoint_time_saved']==float(F(2*step+1,20*K))==row['midpoint_time_saved']
            edges,g,degree,cert=audit_geometry(packet,f['positions'],f['rho'],world['ids'],f['D'],f['x_bounds'])
            receipt=read(folder/f'geometry-audit-step{step:02d}.json');assert cert==receipt['certificate'] and receipt['cache_sha256']==item['sha256']
            assert receipt['pure_inputs_sha256']==digest(out/'pure-inputs.json') and not receipt['mass_c0_reference_loaded']
            assert receipt['audit_source_sha256']==digest(ROOT/'research/reconnect/audit_bounded_geometry.py')
            Lambda=math.ceil(max(d/a for d,a in zip(degree,mm)));assert Lambda==row['operator']['Lambda_exact_integer']
            rows=[{} for _ in m]
            for (i,j),value in zip(edges,g):
                assert j not in rows[i] and i not in rows[j];rows[i][j]=float(value/(Lambda*mm[i]));rows[j][i]=float(value/(Lambda*mm[j]))
            for i,(d,a) in enumerate(zip(degree,mm)):
                diagonal=1-d/(Lambda*a);assert diagonal>=0;rows[i][i]=float(diagonal)
            ptr=[0];cols=[];vals=[]
            for neighbours in rows:
                for j in sorted(neighbours):cols.append(j);vals.append(neighbours[j])
                ptr.append(len(cols))
            P=csr_matrix((vals,cols,ptr),shape=(len(m),len(m)));a=arrays(folder/f'action-step{step:02d}.npz')
            for key,v in [('uniformization_data',P.data),('uniformization_indices',P.indices),('uniformization_indptr',P.indptr),('edges',edges)]:np.testing.assert_array_equal(a[key],v)
            weights=a['weights'];M=len(weights)-1;dt=F(1,10*K);z=Lambda*dt;bound=z**(M+1)/math.factorial(M+1)/(1-z/(M+2))
            assert row['dt_exact']==str(dt) and row['poisson']['M']==M and row['poisson']['z_exact']==str(z)
            assert bound==F(row['poisson']['tail_upper_exact'])<=F(1,10**16)
            ratio=z/(M+1);assert ratio>=1 or z**M/math.factorial(M)/(1-ratio)>F(1,10**16)
            v=previous.copy();w=math.exp(-float(z));following=w*v;assert weights[0]==w
            for j in range(1,M+1):v=P@v;w*=float(z)/j;assert weights[j]==w;following+=w*v
            np.testing.assert_array_equal(following,path['independent_states'][step+1]);previous=following
            assert math.fsum(weights)==row['poisson']['raw_weight_sum'] and not row['poisson']['weights_or_state_normalized']
            maximum_degree=int(np.bincount(edges.ravel(),minlength=len(m)).max())
            allowance=float(64*np.finfo(float).eps*(maximum_degree+1)*(M+1)*(1+float(z))*max(1.,math.sqrt(float(m.sum())))*max(1.,float(np.max(abs(c0)))))
            machine+=allowance;tail+=float(bound)*math.sqrt(float(m.sum()))*float(np.max(abs(c0)))
            assert machine==row['cumulative_machine_indicator_absolute'] and tail==row['cumulative_tail_indicator_absolute']
            delta=norm(path['states'][step+1]-following);assert delta==row['primary_control_mass_norm_difference'] and delta<=machine+tail;maxdelta=max(maxdelta,delta)
            A=csr_matrix((a['symmetric_data'],a['symmetric_indices'],a['symmetric_indptr']),shape=tuple(a['shape']));root=np.sqrt(m);gf=np.array(list(map(float,g)));np.testing.assert_array_equal(a['g_float'],gf);i,j=edges.T
            for u,chi in zip((path['states'][step],path['exponential_half_states'][step],path['states'][step+1]),row['chi_start_half_end']):
                flux=gf*(u[j]-u[i]);net=np.zeros(len(m));np.add.at(net,i,flux);np.add.at(net,j,-flux)
                error=norm(A@(root*u)/root-net/m);assert error<=1e-10*max(1.,norm(net/m));maxaction=max(maxaction,error)
                assert float(gf@((u[j]-u[i])**2))==chi and chi>=0
            u=path['states'][step+1];drift=float(m@(u-c0));assert drift==row['raw_mass_change_from_c0'] and abs(drift)/float(m@c0)<1e-10;maxdrift=max(maxdrift,abs(drift))
            assert u.min()>=c0.min()-1e-10 and u.max()<=c0.max()+1e-10
            newvariance=float(m@((u-float(m@u/m.sum()))**2));assert newvariance<=variance+1e-10*V0;variance=newvariance
            save(folder/f'action-audit-step{step:02d}.json',{'cache_sha256':item['sha256'],'all_cell_boundary_geometry_independently_verified':True,
              'exact_rates_minimal_tail_rawweights_complete_sequence_verified':True,'primary_control_mass_norm_difference':delta,'shared_symmetric_action_difference':error})
            counts+=1;print(f'independent bounded Becker action:{counts}/28',flush=True)
    return {'graphs_and_full_actions_verified':counts,'maximum_primary_control_mass_norm_difference':maxdelta,
      'maximum_shared_vs_symmetric_action_mass_norm_difference':maxaction,'maximum_raw_mass_drift':maxdrift,
      'no_x_periodic_or_boundary_scalar_flux':True,'all_cell_halfplanes_and_non_nearest_images_verified':True}


def reference_and_material(out,data,col,paths):
    fine=arrays(out/'source-reference-n8192-k8192.npz');e=fine['mass_edges'];v=fine['c'];target=col['mass_edges'];projected=[]
    for l,h in zip(target[:-1],target[1:]):
        first=max(0,int(np.searchsorted(e,l,side='right'))-1);last=min(len(v),int(np.searchsorted(e,h,side='left'))+1);total=F(0)
        for k in range(first,last):
            lo=max(F(float(l)),F(float(e[k])));hi=min(F(float(h)),F(float(e[k+1])))
            if hi>lo:total+=(hi-lo)*F(float(v[k]))
        projected.append(float(total/(F(float(h))-F(float(l)))))
    projected=np.array(projected);m=col['mass'];norm=lambda u:float(np.sqrt(m@(u*u)))
    error=norm(projected-col['reference']);assert error<=1e-10*max(1.,norm(col['reference']))
    np.testing.assert_array_equal(col['reference'],col['reference_n8192-k8192']);np.testing.assert_array_equal(col['mass'],np.diff(target))
    np.testing.assert_array_equal(data['mass'].reshape(8,32),np.tile(col['mass']/8,(8,1)))
    np.testing.assert_array_equal(data['c0'].reshape(8,32),np.tile(col['c0'],(8,1)));np.testing.assert_array_equal(data['reference'].reshape(8,32),np.tile(col['reference'],(8,1)))
    checks=0;maxbridge=0.
    for K in COUNTS:
        for field in ('states','independent_states'):
            for state in paths[K][field]:
                means=[]
                for i in range(32):
                    exact=sum((F(float(data['mass'][j*32+i]))*F(float(state[j*32+i])) for j in range(8)),F(0))/F(float(col['mass'][i]));means.append(float(exact));checks+=1
                # Full row symmetry is a diagnostic, never a projection of actual states.
                maxbridge=max(maxbridge,norm(np.array(means)-state.reshape(8,32)[0]))
    save(out/'saved-reference-material-audit.json',{'independent_exact_overlap_projection_mass_norm_difference':error,
      'reference_coarse_means_not_recomputed_or_replaced':True,'all_original_column_mass_c0_reference_copies_exact':True,
      'transverse_exact_integral_checks':checks,'maximum_observation_vs_firstrow_mass_norm_difference':maxbridge,'new_PDE_solves':0})


def main(directory):
    started=time.monotonic();out=Path(directory).resolve();assert not (out/'sha256.json').exists();ex=read(out/'execution.json');status=read(out/'stage3-status.json');lock=read(out/'implementation-lock.json')
    assert not ex['main_first_attempt_passed'] and ex['tests_first_attempt_passed'] and ex['original_study_exit_code']==1 and ex['completion_exit_code']==0
    assert ex['elapsed_main_to_diagnostic_failure_seconds']<900 and ex['peak_main_RSS_MiB']<1024 and ex['scientific_paths_rerun_during_completion']==0
    for rel,h in read(out/'pre-completion-files-sha256.json').items():assert digest(out/rel)==h,rel
    assert read(out/'completion-lock.json')['completion_code_sha256']==digest(ROOT/'research/reconnect/becker_point_completion.py')
    assert status['recorded_checks_total']==500 and not status['fluid_orc_enabled'] and not status['moving_shock_T4_closed']
    assert read(out/'tests.json')=={'successful':True,'checks_run':4,'failures':0,'errors':0,'skipped':0}
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert lock['Becker_source_sha256']==digest(ROOT/'research/rebuild/becker.py')
    assert lock['protocol_sha256']==digest(out/'implementation-protocol.json')==digest(ROOT/'research/reconnect/becker_point_protocol.json')
    assert lock['contract_sha256']==digest(out/'boundary-duration-contract.md')==digest(ROOT/'research/reconnect/BECKER_BOUNDARY_DURATION_CONTRACT.md')
    assert read(out/'source-gate-spec.json')==read(SOURCE/'next-gate-spec.json')
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==digest(ROOT/item['source'])==item['sha256']
    world=read(out/'pure-inputs.json');index=read(out/'graph-index.json');seal=read(out/'coefficient-seal.json');labels=np.load(out/'source-mass-labels.npy',allow_pickle=False);flow=Becker(delta=.08)
    assert len(index)==seal['graphs']==28 and seal['all28_graphs_sealed_before_actual_mass_c0_reference_or_oldstate_load'] and not seal['internal_material_faces_or_F_C_received_by_builder']
    assert seal['graph_index_sha256']==digest(out/'graph-index.json') and world['columns']==32 and world['rows']==8 and world['ids']==list(range(256))
    centres=(labels[:-1]+labels[1:])/2
    for K in COUNTS:
        for i,f in enumerate(world['cases'][str(K)]['frames']):
            t=F(i,20*K);assert f['time_exact']==str(t) and f['period_y']==1. and f['D']==.02
            x=flow.x_mu(centres,float(t));rho=flow.rho_mu(centres,float(t));bounds=flow.x_mu(labels[[0,-1]],float(t))
            np.testing.assert_array_equal(np.array(f['positions']).reshape(8,32,2)[:,:,0],np.tile(x,(8,1)))
            np.testing.assert_array_equal(np.array(f['positions']).reshape(8,32,2)[:,:,1],np.tile(((np.arange(8)+.5)/8)[:,None],(1,32)))
            np.testing.assert_array_equal(np.array(f['rho']).reshape(8,32),np.tile(rho,(8,1)));np.testing.assert_array_equal(f['x_bounds'],bounds)
    args=load_all(out);data,col,runs,paths,rm=args;summary=audit_paths(out,data,world,index);reference_and_material(out,data,col,paths)
    assert reference_indicators(col)==rm;result=evaluate(*args);assert plain_json(result)==read(out/'results.json')
    assert plain_json(density_diagnostics(out,flow,labels,col,world))==read(out/'density-volume-diagnostics.json')
    # Direct old-H perturbation formula and predeclared positive time comparison.
    m,c0=data['mass'],data['c0'];mu=float(m@c0/m.sum());E0=float(m@((c0-mu)**2));b=sum(result['time_control']['components_absolute'].values())
    hb=(2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))*b+b*b)/E0
    assert hb==result['time_control']['fixed_H_decrease_indicator']
    preserved=preservation();known={i['manifest'] for i in preserved['formal_scientific_manifests_verified']}
    previous=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in previous:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==101;save(out/'preservation.json',preserved)
    summary.update({'independent_saved_reference_projection_and_original_material_audit_passed':True,'explicit_T1over10_and_original_H_verified':True,
      'comparison_decision':result['comparison_decision'],'registered_moving_Becker_oracle_discriminator_passed':bool(result['registered_moving_Becker_oracle_discriminator_passed']),
      'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':101,'new_software_checks':4,'recorded_checks_total':500,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    save(out/'numerical-audit.json',summary);save(out/'audit-execution.json',{'audit_exit_code':0,'audit_first_attempt_passed':True,'elapsed_audit_seconds':time.monotonic()-started,
      'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'audit_source_sha256':digest(Path(__file__))})
    print(json.dumps(plain_json(summary),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
