"""Independent saved-data audit; no actual exponential, geometry or PDE reruns."""
from pathlib import Path
from fractions import Fraction as F
import argparse,math,time,resource
import numpy as np
import mpmath as mp
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save,digest
from compression_density_audit_core import bin_density,verify_lifts,verify_ring_density,verify_coefficients
from shear_union64_study import arrays

SOURCE=ROOT/'runs/reconnect-compression-union64'


def audit_intervals(out,m,c0):
    norm=lambda v:float(np.sqrt(m@(v*v)));mu0=float(m@c0/m.sum());V0=float(m@((c0-mu0)**2))
    maximum_action_difference=maximum_mass_drift=0.;count=0
    for source in ('kernel','ledger'):
        index=read(out/source/'graph-index.json')
        for K in (16,32):
            folder=out/source/f'k{K}';path=arrays(folder/'trajectory-states.npz');run=read(folder/'native-results.json')
            assert path['states'].shape==path['independent_states'].shape==(K+1,len(m))
            np.testing.assert_array_equal(path['states'][0],c0);np.testing.assert_array_equal(path['independent_states'][0],c0)
            machine_sum=tail_sum=0.;last_variance=V0;dt=F(1,20*K)
            for step,row in enumerate(run['steps']):
                packet_file=folder/f'coefficients-step{step:02d}.json';packet=read(packet_file)
                edges=np.array(packet['edges'],dtype=np.int64);g=list(map(F,packet['conductances_exact']))
                n=len(m);mm=list(map(lambda x:F(float(x)),m));degree=[F(0)]*n
                for (i,j),w in zip(edges,g):degree[i]+=w;degree[j]+=w
                Lambda=math.ceil(max(a/b for a,b in zip(degree,mm)));adj=[{} for _ in m]
                for (i,j),w in zip(edges,g):adj[i][j]=float(w/(Lambda*mm[i]));adj[j][i]=float(w/(Lambda*mm[j]))
                for i,(a,b) in enumerate(zip(degree,mm)):
                    q=1-a/(Lambda*b);assert q>=0;adj[i][i]=float(q)
                vals=[];cols=[];ptr=[0]
                for adjacency_row in adj:
                    for j in sorted(adjacency_row):cols.append(j);vals.append(adjacency_row[j])
                    ptr.append(len(cols))
                P=csr_matrix((vals,cols,ptr),shape=(n,n));gf=np.array(list(map(float,g)));root=np.sqrt(m)
                act=arrays(folder/f'action-step{step:02d}.npz');count+=1
                for field,v in [('data',P.data),('indices',P.indices),('indptr',P.indptr)]:np.testing.assert_array_equal(v,act['uniformization_'+field])
                np.testing.assert_array_equal(edges,act['edges']);np.testing.assert_array_equal(gf,act['g_float'])
                assert row['coefficient_cache_sha256']==digest(packet_file)
                assert row['operator']['Lambda_exact_integer']==Lambda and row['dt_exact']==str(dt)
                assert row['midpoint_time_saved']==index[f'k{K}-step{step:02d}']['midpoint_time_saved']
                A=csr_matrix((act['symmetric_data'],act['symmetric_indices'],act['symmetric_indptr']),shape=tuple(act['shape']))
                weights=act['weights'];M=len(weights)-1;z=Lambda*dt
                tail=z**(M+1)/math.factorial(M+1)/(1-z/(M+2))
                assert tail==F(row['poisson']['tail_upper_exact'])<=F(1,10**16)
                assert z/(M+1)>=1 or z**M/math.factorial(M)/(1-z/(M+1))>F(1,10**16)
                assert row['poisson']['M']==M and F(row['poisson']['z_exact'])==z
                maximum_degree=int(np.bincount(edges.ravel(),minlength=n).max())
                machine=float(64*np.finfo(float).eps*(maximum_degree+1)*(M+1)*(1+float(z))*max(1.,math.sqrt(float(m.sum())))*max(1.,float(np.max(abs(c0)))))
                machine_sum+=machine;tail_sum+=float(tail)*math.sqrt(float(m.sum()))*float(np.max(abs(c0)))
                assert machine==row['machine_indicator_absolute_this_interval'] and machine_sum==row['cumulative_machine_indicator_absolute'] and tail_sum==row['cumulative_tail_indicator_absolute']
                v=path['independent_states'][step].copy();weight=math.exp(-float(z));following=weight*v
                assert weights[0]==weight
                for j in range(1,M+1):v=P@v;weight*=float(z)/j;assert weights[j]==weight;following+=weight*v
                np.testing.assert_array_equal(following,path['independent_states'][step+1])
                assert math.fsum(weights)==row['poisson']['raw_weight_sum'] and not row['poisson']['weights_or_state_normalized']
                delta=norm(path['states'][step+1]-following);assert delta==row['primary_control_mass_norm_difference']<=machine_sum+tail_sum
                assert np.max(abs(P@np.ones(n)-1))<1e-10 and np.max(abs(P.T@m-m))/m.max()<1e-10 and P.data.min()>=0
                for u,chi in zip((path['states'][step],path['exponential_half_states'][step],path['states'][step+1]),row['chi_start_half_end']):
                    i,j=edges.T;flux=gf*(u[j]-u[i]);net=np.zeros(n);np.add.at(net,i,flux);np.add.at(net,j,-flux)
                    diff=norm(A@(root*u)/root-net/m);assert diff<1e-10*max(1.,norm(net/m));maximum_action_difference=max(maximum_action_difference,diff)
                    assert float(gf@((u[j]-u[i])**2))==chi>=0
                    assert abs(-float((root*u)@(A@(root*u)))-chi)<1e-10*max(1.,chi)
                for key in ('states','independent_states'):
                    u=path[key][step+1];drift=float(m@(u-c0));assert abs(drift)/float(m@c0)<1e-10
                    assert np.isfinite(u).all() and u.min()>=c0.min()-1e-10 and u.max()<=c0.max()+1e-10
                    maximum_mass_drift=max(maximum_mass_drift,abs(drift))
                u=path['states'][step+1];variance=float(m@((u-float(m@u/m.sum()))**2))
                assert variance<=last_variance+1e-10*V0;last_variance=variance
                assert float(m@(u-c0))==row['raw_mass_change_from_c0']
    assert count==96
    return {'verified_intervals':count,'whole_independent_recurrences_bitwise_verified':True,
        'maximum_shared_vs_symmetric_action_difference':maximum_action_difference,'maximum_raw_mass_drift':maximum_mass_drift,
        'new_primary_propagations_in_audit':0,'old_oracle_paths_rerun':0}


def high_precision(data,runs,paths,result):
    reports=[]
    for digits in (80,100):
        with mp.workdps(digits):
            f=lambda x:mp.mpf(F(float(x)).numerator)/F(float(x)).denominator
            m=list(map(f,data['mass']));c=list(map(f,data['c0']));M=mp.fsum(m);mu=mp.fsum(a*b for a,b in zip(m,c))/M
            V0=mp.fsum(a*(b-mu)**2 for a,b in zip(m,c));sigma=mp.sqrt(V0)
            def H(u):
                mean=mp.fsum(a*b for a,b in zip(m,u))/M
                return 1-mp.fsum(a*(b-mean)**2 for a,b in zip(m,u))/V0
            norm=lambda v:mp.sqrt(mp.fsum(w*x*x for w,x in zip(m,v)))
            minus=lambda a,b:[x-y for x,y in zip(a,b)]
            hm=lambda e:(2*mp.sqrt(M)*max(1,max(map(abs,c)))*e+e*e)/V0
            a={name:[list(map(f,v)) for v in data[name]] for name in ('reference','space','time','reference_space_coarse','reference_space_fine','reference_time_check')}
            observations={(s,K,k,field):list(map(f,paths[s,K][field][k*K//2])) for s in ('oracle','kernel','ledger') for K in (16,32) for k in (1,2) for field in ('states','independent_states')}
            maximum=mp.mpf(0);rows=[]
            def close(actual,stored):
                nonlocal maximum
                diff=abs(actual-f(stored));maximum=max(maximum,diff)
                assert diff<=64*np.finfo(float).eps*max(1,abs(actual))
            close(sigma,result['common_sigma64'])
            for row in result['source_comparisons']:
                s=row['source'];k=round(row['time']/.025)
                cp={'four_reference_space':4*norm(a['space'][k]),'four_reference_time':4*norm(a['time'][k]),'two_initial_bridge':2*norm(minus(a['reference'][0],c)),'same_N64_projection_identity':mp.mpf(0)}
                hp={'two_reference_space_same_basis':2*abs(H(a['reference_space_coarse'][k])-H(a['reference_space_fine'][k])),
                    'two_reference_time_same_basis':2*abs(H(a['reference_time_check'][k])-H(a['reference'][k])),
                    'two_initial_bridge':2*abs(H(a['reference'][0])-H(c)),'same_N64_projection_identity':mp.mpf(0)}
                total=mp.mpf(0)
                for source in (s,'oracle'):
                    u=observations[source,16,k,'states'];v=observations[source,32,k,'states']
                    cp[source+'_16_32']=norm(minus(u,v));hp[source+'_16_32']=abs(H(u)-H(v))
                    for K in (16,32):
                        rr=runs[source,K]['steps'][k*K//2-1];mt=f(rr['cumulative_machine_indicator_absolute'])+f(rr['cumulative_tail_indicator_absolute']);total+=mt
                        u=observations[source,K,k,'states'];v=observations[source,K,k,'independent_states']
                        cp[f'{source}_e{K}']=mt+norm(minus(u,v));hp[f'{source}_complete_sequence_H{K}']=abs(H(u)-H(v))
                hp['both_machine_tail_to_H']=hm(total)
                for name,val in cp.items():close(val,row['concentration_parts_absolute'][name])
                for name,val in hp.items():close(val,row['H_parts'][name])
                bc=mp.fsum(cp.values())/sigma;bh=mp.fsum(hp.values())
                u=observations[s,32,k,'states'];v=observations['oracle',32,k,'states'];ref=a['reference'][k]
                gc=(norm(minus(u,ref))-norm(minus(v,ref)))/sigma;gh=abs(H(u)-H(ref))-abs(H(v)-H(ref))
                decide=lambda x,b:'error_larger_resolvably' if x>b else ('error_smaller_resolvably' if x < -b else 'unresolved')
                assert decide(gc,bc)==row['concentration_decision'] and decide(gh,bh)==row['H_decision']
                for actual,name in ((bc,'concentration_indicator'),(bh,'H_indicator'),(gc,'concentration_error_change'),(gh,'H_error_change')):close(actual,row[name])
                rows.append({'source':s,'time':row['time'],'concentration_decision':decide(gc,bc),'H_decision':decide(gh,bh),
                             'concentration_indicator':mp.nstr(bc,65),'H_indicator':mp.nstr(bh,65)})
            reports.append({'digits':digits,'maximum_float_scalar_difference':float(maximum),'all_positive_components_and_decisions_verified':True,'rows':rows})
    assert reports[0]['rows']==reports[1]['rows']
    return reports


def main(directory,write=False):
    start=time.monotonic();out=Path(directory).resolve();lock=read(out/'implementation-lock.json');p=read(out/'protocol.json')
    for name,h in lock['files'].items():assert digest(ROOT/name)==h,name
    for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
    test=read(out/'tests.json');assert test['successful'] and test['checks_run']==2
    execution=read(out/'execution.json');assert execution['completed'] and execution['elapsed_seconds']<700 and execution['peak_RSS_MiB']<1024
    seal=read(out/'coefficient-seal.json');assert seal['all_coefficients_sealed_before_scalar_loading'] and seal['new_coefficient_packets']==96
    for name,h in seal['files'].items():assert digest(out/name)==h,name
    # Independent saved-input reads; no density producer, lift producer or actual solver calls.
    with np.load(SOURCE/'trajectory-source.npz',allow_pickle=False) as z:ids=z['ids'];labels=z['labels'];rho0=z['rho0_point']
    with np.load(SOURCE/'builder-inputs.npz',allow_pickle=False) as z:b={f'k{K}_{n}':z[f'k{K}_{n}'] for K in (16,32) for n in ('time','positions')}
    d=arrays(SOURCE/'union-definition.npz');m=d['mass'];wind=arrays(out/'integer-winding.npz');anchor=read(out/'anchor-rings.json');index=read(SOURCE/'graph-index.json')
    np.testing.assert_array_equal(wind['k16'],wind['k32'][::2])
    for K in (16,32):assert verify_lifts(labels,b[f'k{K}_time'],b[f'k{K}_positions'],wind[f'k{K}'])==read(out/'lift-audit.json')[str(K)]
    receipts=[]
    for K in (16,32):
        for step in range(K):
            frame=2*step+1;x=b[f'k{K}_positions'][frame];original=read(SOURCE/index[f'k{K}-step{step:02d}']['file'])
            independent,counts=bin_density(ids,x,m,1/32)
            for s in ('kernel','ledger'):
                folder=out/s/f'k{K}';rho=arrays(folder/f'density-step{step:02d}.npz')['rho'];packet=read(folder/f'coefficients-step{step:02d}.json')
                equal=verify_coefficients(original,packet,ids,rho)
                diff=float(np.max(abs(rho-independent))) if s=='kernel' else None
                if s=='kernel':assert np.all(abs(rho-independent)<=1e-10*np.maximum(1.,abs(independent)))
                else:verify_ring_density(x,wind[f'k{K}'][frame],rho0,anchor['rings'],anchor['initial_areas_exact'],rho,read(folder/f'areas-step{step:02d}.json')['current_areas_exact'])
                receipts.append({'source':s,'K':K,'step':step,'all8192_rates_exactly_verified':True,'all_other_geometry_fields_unchanged':True,
                    'coefficient_identity_with_oracle':equal,'kernel_independent_difference':diff,
                    'kernel_support':{'support_count_min':min(counts),'support_count_max':max(counts),'self_retained':True,'h':1/32,'normalization':False} if s=='kernel' else None,
                    'scalar_or_reference_loaded':False})
            if (step+1)%8==0:print(f'saved density audit K{K} step{step+1}: {time.monotonic()-start:.1f}s',flush=True)
    assert receipts==read(out/'density-coefficient-audit.json')
    interval=audit_intervals(out,m,d['c0']);runs={};paths={}
    for s in ('oracle','kernel','ledger'):
        for K in (16,32):
            folder=(SOURCE if s=='oracle' else out/s)/f'k{K}';runs[s,K]=read(folder/'native-results.json');paths[s,K]=arrays(folder/'trajectory-states.npz')
    from compression_density_compare import compare
    result=read(out/'results.json');assert compare(d,runs,paths)==result
    hp=high_precision(d,runs,paths,result)
    findings={'passed':True,'all96_density_and_exact_rate_packets_verified':True,'raw_intervals':interval,'high_precision':hp,
        'recorded_checks_total':506,'stage3':'incomplete','fluid_orc_enabled':False,'no_actual_solver_or_reference_reruns':True}
    if write:save(out/'audit.json',{'passed':True,'findings':findings,'elapsed_seconds':time.monotonic()-start,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
    else:assert read(out/'audit.json')['findings']==findings
    print('Verified96 density/rate packets,96 saved raw intervals and80/100-digit complete source budgets.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();main(a.directory,a.write)
