"""Independent density/rate, saved interval and high precision budget audit."""
from pathlib import Path
from fractions import Fraction as F
import argparse,math,time,resource
import numpy as np
import mpmath as mp
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save,digest


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def audit_packets(out,ids,x,m,rho0,original):
    """Called before c0/reference/old paths are loaded; no scalar inputs."""
    out=Path(out);result={};box=list(map(F,original['periods_exact']));n=len(ids)
    assert n==1024 and ids.tolist()==original['ids'] and len(original['edges'])==2048
    assert all(F(float(v))==F(1,1024) for v in m)
    neighbors=[set() for _ in ids]
    for (i,j),shift in zip(original['edges'],original['image_shifts']):
        neighbors[i].add((j,*shift));neighbors[j].add((i,*[-v for v in shift]))
    for source in ('kernel','ledger'):
        packet=read(out/source/'coefficients.json');density=arrays(out/source/'density.npz')['rho']
        assert density.shape==(n,) and np.isfinite(density).all() and density.min()>0
        assert {k:v for k,v in packet.items() if k!='conductances_exact'}=={k:v for k,v in original.items() if k!='conductances_exact'}
        largest=0.
        if source=='kernel':
            for i in range(n):
                terms=[]
                for j in np.argsort(ids):
                    dx=float(x[j,0]-x[i,0]);dy=float(x[j,1]-x[i,1]);dx-=round(dx);dy-=round(dy)
                    q=math.hypot(dx,dy)*16
                    if q<2:terms.append(float(m[j])*9/(4*math.pi)*(16**2)*(1-q/2)**6*(1+3*q+35*q*q/12))
                expected=math.fsum(terms);largest=max(largest,abs(density[i]-expected))
                assert abs(density[i]-expected)<1e-10*max(1.,abs(expected))
                assert len(terms)==45
        else:
            ring=read(out/source/'rings.json')
            for i,order in enumerate(ring['rings']):
                assert set(map(tuple,order))==neighbors[i] and len(order)==4
                v=[tuple(F(float(x[j,k]))+shift*box[k]-F(float(x[i,k])) for k,shift in enumerate((sx,sy))) for j,sx,sy in order]
                # Independent shoelace on the saved fixed image lift, not the adapter.
                area=sum((a[0]*b[1]-a[1]*b[0] for a,b in zip(v,v[1:]+v[:1])),F(0))/2
                assert area==F(1,512)==F(ring['initial_areas_exact'][i])==F(ring['current_areas_exact'][i])
                assert all(a[0]*b[1]-a[1]*b[0]>0 for a,b in zip(v,v[1:]+v[:1]))
                assert F(float(density[i]))==F(float(rho0[i]))*area/area
                assert F(float(rho0[i]))*area/F(float(m[i]))==2
        D=F(original['D_exact']);g=[]
        for (i,j),ratio,stored in zip(packet['edges'],packet['face_over_distance_exact'],packet['conductances_exact']):
            resistance=1/(2*F(float(density[i]))*D)+1/(2*F(float(density[j]))*D)
            exact=F(packet['thickness_exact'])*F(ratio)/resistance
            assert exact==F(stored)>0;g.append(exact)
        identical=g==list(map(F,original['conductances_exact']))
        if source=='ledger':assert identical,'ledger coefficient identity failed; stop without replacement'
        result[source]={'all2048_exact_half_resistances_verified':True,'all_original_geometry_fields_unchanged':True,
            'raw_density_formula_max_independent_difference':largest,'exact_coefficients_equal_oracle':identical,
            'density_sha256':digest(out/source/'density.npz'),'coefficients_sha256':digest(out/source/'coefficients.json'),
            'mass_c0_reference_or_paths_used_to_fit_density':False,'c0_reference_or_old_paths_loaded':False}
    return result


def audit_intervals(out,m,c0):
    packet=read(out/'kernel/coefficients.json');edges=np.array(packet['edges'],dtype=np.int64);g=list(map(F,packet['conductances_exact']))
    n=len(m);mm=list(map(lambda x:F(float(x)),m));degree=[F(0)]*n
    for (i,j),w in zip(edges,g):degree[i]+=w;degree[j]+=w
    Lambda=math.ceil(max(a/b for a,b in zip(degree,mm)));adj=[{} for _ in m]
    for (i,j),w in zip(edges,g):adj[i][j]=float(w/(Lambda*mm[i]));adj[j][i]=float(w/(Lambda*mm[j]))
    for i,(a,b) in enumerate(zip(degree,mm)):
        q=1-a/(Lambda*b);assert q>=0;adj[i][i]=float(q)
    vals=[];cols=[];ptr=[0]
    for row in adj:
        for j in sorted(row):cols.append(j);vals.append(row[j])
        ptr.append(len(cols))
    P=csr_matrix((vals,cols,ptr),shape=(n,n));gf=np.array(list(map(float,g)));root=np.sqrt(m)
    norm=lambda v:float(np.sqrt(m@(v*v)));mu0=float(m@c0/m.sum());V0=float(m@((c0-mu0)**2))
    maximum_action_difference=maximum_mass_drift=0.;count=0
    for K in (4,8,16):
        folder=out/'kernel'/f'k{K}';path=arrays(folder/'trajectory-states.npz');run=read(folder/'native-results.json')
        assert path['states'].shape==path['independent_states'].shape==(K+1,n)
        np.testing.assert_array_equal(path['states'][0],c0);np.testing.assert_array_equal(path['independent_states'][0],c0)
        machine_sum=tail_sum=0.;last_variance=V0;dt=F(1,20*K)
        for step,row in enumerate(run['steps']):
            act=arrays(folder/f'action-step{step:02d}.npz');count+=1
            for field,v in [('data',P.data),('indices',P.indices),('indptr',P.indptr)]:np.testing.assert_array_equal(v,act['uniformization_'+field])
            np.testing.assert_array_equal(edges,act['edges']);np.testing.assert_array_equal(gf,act['g_float'])
            assert row['coefficient_cache_sha256']==digest(out/'kernel/coefficients.json')
            assert row['operator']['Lambda_exact_integer']==Lambda and row['dt_exact']==str(dt)
            assert row['midpoint_time_saved']==float((2*step+1)*.05/(2*K))
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
    assert count==28
    return {'verified_intervals':count,'whole_independent_recurrences_bitwise_verified':True,
        'maximum_shared_vs_symmetric_action_difference':maximum_action_difference,'maximum_raw_mass_drift':maximum_mass_drift,
        'new_primary_propagations_in_audit':0,'old_oracle_paths_rerun':0}


def high_precision_audit(out,m,c0,D,results,paths,runs,symbols):
    """Separate shell-multiplicity symbol and 80/100-digit norm/budget decisions."""
    reports=[]
    for digits in (80,100):
        with mp.workdps(digits):
            f=lambda x:mp.mpf(F(float(x)).numerator)/F(float(x)).denominator
            ff=lambda x:mp.mpf(F(x).numerator)/F(x).denominator
            mm=list(map(f,m));cc=list(map(f,c0));M=mp.fsum(mm)
            norm=lambda v:mp.sqrt(mp.fsum(a*b*b for a,b in zip(mm,v)))
            V0=mp.fsum(a*(b-mp.fsum(x*y for x,y in zip(mm,cc))/M)**2 for a,b in zip(mm,cc))
            hm=lambda e:(2*mp.sqrt(M)*max(1,max(map(abs,cc)))*e+e*e)/V0
            multiplicity={}
            for a in range(-3,4):
                for b in range(-3,4):
                    s=a*a+b*b
                    if s<16:multiplicity[s]=multiplicity.get(s,0)+1
            rho=mp.mpf(9)/(16*mp.pi)*mp.fsum(count*(1-mp.sqrt(s)/4)**6*(1+mp.mpf(3)/2*mp.sqrt(s)+mp.mpf(35)/48*s) for s,count in multiplicity.items())
            assert sum(multiplicity.values())==45
            for source in ('oracle','kernel','ledger'):
                lam=(rho if source=='kernel' else 1)*8*ff(D)*1024*mp.sin(mp.pi/32)**2
                assert float(lam)==results['density_and_symbol_records'][source]['discrete_lambda']
                for k,t in enumerate((F(0),F(1,40),F(1,20))):
                    independent=np.array([float(mp.mpf(1)/2+(c-mp.mpf(1)/2)*mp.exp(-lam*ff(t))) for c in cc])
                    np.testing.assert_array_equal(independent,symbols[source+'_symbol_80digits'][k])
            reference=arrays(out/'source/runs/reconnect-t1-uniform/n32/material-reference.npz')['reference']
            a_ref=read(out/'source/runs/reconnect-t1-uniform/n32/analytic-parameters.json')['reference_evaluation_indicator_absolute']
            H=lambda u:1-mp.fsum(w*(q-mp.fsum(x*y for x,y in zip(mm,u))/M)**2 for w,q in zip(mm,u))/V0
            maximum_scalar_difference=mp.mpf(0)
            for row in results['source_comparisons']:
                source=row['source'];k=round(row['time']/.025);t=ff(F(k,40));terms=[]
                for s in (source,'oracle'):
                    for A,B in ((4,8),(8,16)):
                        terms.append(norm([f(a)-f(b) for a,b in zip(paths[s,A]['states'][k*A//2],paths[s,B]['states'][k*B//2])]))
                    for K in (4,8,16):
                        step=runs[s,K]['steps'][k*K//2-1]
                        terms.append(f(step['cumulative_machine_indicator_absolute'])+f(step['cumulative_tail_indicator_absolute'])+norm([f(a)-f(b) for a,b in zip(paths[s,K]['states'][k*K//2],paths[s,K]['independent_states'][k*K//2])]))
                terms += [4*f(a_ref),2*norm([f(a)-b for a,b in zip(reference[0],cc)]),t*mp.fsum(f(results['density_and_symbol_records'][s][name]) for s in (source,'oracle') for name in ('eigenmode_residue_mass_norm','constant_residue_mass_norm'))]
                B=mp.fsum(terms);u=list(map(f,paths[source,16]['states'][k*8]));v=list(map(f,paths['oracle',16]['states'][k*8]));ref=list(map(f,reference[k]))
                gc=(norm([a-b for a,b in zip(u,ref)])-norm([a-b for a,b in zip(v,ref)]))/f(results['common_sigma16'])
                gh=abs(H(u)-H(ref))-abs(H(v)-H(ref))
                decision=lambda x,b:'error_larger_resolvably' if x>b else ('error_smaller_resolvably' if x < -b else 'unresolved')
                assert decision(gc,B/f(results['common_sigma16']))==row['concentration_decision']
                assert decision(gh,hm(B))==row['H_decision']
                for actual,stored in ((B,row['B_absolute']),(gc,row['concentration_error_change_common_sigma16']),(gh,row['H_error_change'])):
                    diff=abs(actual-f(stored));maximum_scalar_difference=max(maximum_scalar_difference,diff)
                    assert diff<=64*np.finfo(float).eps*max(1,abs(actual))
            reports.append({'digits':digits,'rho_star':mp.nstr(rho,70),'maximum_float_scalar_difference':float(maximum_scalar_difference),'all_frozen_source_decisions_reproduced':True})
    assert reports[0]['rho_star']==reports[1]['rho_star']
    return reports


def main(directory,write=False):
    start=time.monotonic();out=Path(directory).resolve();lock=read(out/'implementation-lock.json');protocol=read(out/'protocol.json')
    for name,h in lock['files'].items():assert digest(ROOT/name)==h,name
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==item['sha256']==protocol['sources'][item['source']]
    assert read(out/'tests.json')['successful'] and read(out/'tests.json')['checks_run']==2
    execution=read(out/'execution.json');assert execution['completed'] and execution['elapsed_seconds']<300 and execution['peak_RSS_MiB']<1024
    src=out/'source/runs/reconnect-t1-uniform';d=arrays(src/'n32/material-reference.npz')
    with np.load(src/'builder-inputs.npz',allow_pickle=False) as z:ids=z['n32_ids'];x=z['n32_positions'];rho0=z['n32_rho']
    original=read(src/'n32/geometry.json');cert=audit_packets(out,ids,x,d['mass'],rho0,original)
    assert cert==read(out/'density-coefficient-audit.json')
    seal=read(out/'coefficient-seal.json')
    assert seal['all_coefficients_sealed_before_scalar_loading'] and seal['new_geometry_graphs']==0
    for name,h in seal['files'].items():assert digest(out/name)==h,name
    evidence=read(out/'ledger-reuse.json');assert evidence['all2048_exact_coefficients_equal'] and evidence['new_ledger_runs']==0
    runs={};paths={}
    for K in (4,8,16):
        for source in ('oracle','ledger','kernel'):
            folder=out/'kernel'/f'k{K}' if source=='kernel' else src/'n32'/f'k{K}'
            runs[source,K]=read(folder/'native-results.json');paths[source,K]=arrays(folder/'trajectory-states.npz')
        assert evidence['old_paths'][str(K)]['sha256']==digest(src/'n32'/f'k{K}'/'trajectory-states.npz')
    interval=audit_intervals(out,d['mass'],d['c0']);result=read(out/'results.json');symbol=arrays(out/'symbol-observations.npz')
    from t1_density_calibration_core import compare
    a_ref=read(src/'n32/analytic-parameters.json')['reference_evaluation_indicator_absolute']
    sigma=read(src/'results.json')['common_initial_fluctuation_norm']
    assert compare(d['mass'],d['c0'],d['reference'],a_ref,sigma,runs,paths,result['density_and_symbol_records'],symbol)==result
    hp=high_precision_audit(out,d['mass'],d['c0'],original['D_exact'],result,paths,runs,symbol)
    assert result['implementation_controls_passed'] and not result['fluid_orc_enabled'] and result['stage3_physical_acceptance']=='incomplete'
    findings={'passed':True,'density_and_exact_rates_verified':True,'ledger_reuse_identity_verified':True,'raw_intervals':interval,
        'independent_high_precision':hp,'implementation_controls_passed':True,'source_comparisons':result['source_comparisons'],
        'recorded_checks_total':504,'stage3':'incomplete','fluid_orc_enabled':False}
    if write:save(out/'audit.json',{'passed':True,'findings':findings,'elapsed_seconds':time.monotonic()-start,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
    else:assert read(out/'audit.json')['findings']==findings
    print('Independent density/formula,28 saved intervals,ledger identity and80/100-digit source budgets verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();main(a.directory,a.write)
