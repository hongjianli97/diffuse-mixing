"""Independent K32 saved action/observation/positive-budget audit; no new PDE."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
import argparse,ast,inspect,json,math,resource,time
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,digest
from affine_shear_completion import plain_json
from audit_becker_spatial import Q,norm,save,arrays,audit_actions as prior_action_audit
from audit_aligned_becker import near
from becker_fine_time_core import analyze,verify_sampling_rule

def audit_actions_k32(out,N,write):
    folder=out/f'n{N}';world=read(folder/'pure-inputs.json');index=read(folder/'graph-index.json');data=arrays(folder/'material.npz')
    m,c0=data['mass'],data['c0'];mm=list(map(Q,m));count=0;maximum=0.
    for K in (32,):
        path=arrays(folder/f'k{K}/trajectory-states.npz');run=read(folder/f'k{K}/native-results.json');previous=c0.copy();machine=tail=0.
        np.testing.assert_array_equal(path['states'][0],c0);np.testing.assert_array_equal(path['independent_states'][0],c0)
        variance=float(m@((c0-float(m@c0/m.sum()))**2));V0=variance
        for step,row in enumerate(run['steps']):
            item=index[f'k{K}-step{step:02d}'];packet=read(folder/item['file']);receipt=read(folder/f'k{K}/geometry-audit-step{step:02d}.json')
            f=world['cases'][str(K)]['frames'][2*step+1]
            assert digest(folder/item['file'])==item['sha256']==row['coefficient_cache_sha256']==receipt['cache_sha256']
            assert receipt['pure_inputs_sha256']==digest(folder/'pure-inputs.json') and not receipt['mass_c0_reference_loaded']
            assert receipt['audit_source_sha256']==digest(ROOT/'research/reconnect/audit_bounded_geometry.py')
            cert=receipt['certificate'];assert cert['all_cell_world_halfplanes_verified'] and cert['all_vertices_rebuilt_from_boundary_lines_and_bisectors'] and cert['all_non_nearest_y_images_excluded']
            assert row['midpoint_time_saved']==item['midpoint_time_saved']==float(F(2*step+1,20*K))
            points=[tuple(map(Q,p)) for p in f['positions']];rho=list(map(Q,f['rho']));D=Q(f['D']);Ly=Q(f['period_y']);degree=[F(0)]*len(m);gg=[]
            edges=np.array(packet['edges'],dtype=np.int64)
            # Independent face-to-resistance calculation; all-cell geometric proofs
            # are already independently generated before scalar loading and sealed.
            for j,((i,k),(sx,sy)) in enumerate(zip(edges,packet['image_shifts'])):
                assert sx==0 and i<k
                vx=points[k][0]-points[i][0];vy=points[k][1]+sy*Ly-points[i][1]
                ends=[tuple(map(F,p)) for p in packet['face_endpoints_exact'][j]]
                dx=ends[1][0]-ends[0][0];dy=ends[1][1]-ends[0][1]
                if dy>Ly/2:dy-=Ly
                if dy<-Ly/2:dy+=Ly
                ratio=(vx*dy-vy*dx)/(vx*vx+vy*vy);assert ratio>0 and ratio==F(packet['face_over_distance_exact'][j])
                g=ratio/(1/(2*rho[i]*D)+1/(2*rho[k]*D));assert g==F(packet['conductances_exact'][j]) and g>0
                degree[i]+=g;degree[k]+=g;gg.append(g)
            Lambda=math.ceil(max(d/a for d,a in zip(degree,mm)));assert Lambda==row['operator']['Lambda_exact_integer'];rows=[{} for _ in m]
            for (i,j),g in zip(edges,gg):
                assert j not in rows[i] and i not in rows[j];rows[i][j]=float(g/(Lambda*mm[i]));rows[j][i]=float(g/(Lambda*mm[j]))
            for i,(d,a) in enumerate(zip(degree,mm)):
                q=1-d/(Lambda*a);assert q>=0;rows[i][i]=float(q)
            ptr=[0];cols=[];vals=[]
            for row_values in rows:
                for j in sorted(row_values):cols.append(j);vals.append(row_values[j])
                ptr.append(len(cols))
            P=csr_matrix((vals,cols,ptr),shape=(len(m),len(m)));saved=arrays(folder/f'k{K}/action-step{step:02d}.npz')
            for key,value in [('uniformization_data',P.data),('uniformization_indices',P.indices),('uniformization_indptr',P.indptr),('edges',edges)]:np.testing.assert_array_equal(saved[key],value)
            weights=saved['weights'];M=len(weights)-1;dt=F(1,10*K);z=Lambda*dt;bound=z**(M+1)/math.factorial(M+1)/(1-z/(M+2))
            assert row['dt_exact']==str(dt) and row['poisson']['z_exact']==str(z) and row['poisson']['M']==M
            assert bound==F(row['poisson']['tail_upper_exact'])<=F(1,10**16)
            ratio=z/(M+1);assert ratio>=1 or z**M/math.factorial(M)/(1-ratio)>F(1,10**16)
            v=previous.copy();w=math.exp(-float(z));following=w*v;assert weights[0]==w
            for j in range(1,M+1):v=P@v;w*=float(z)/j;assert weights[j]==w;following+=w*v
            np.testing.assert_array_equal(following,path['independent_states'][step+1]);previous=following
            assert math.fsum(weights)==row['poisson']['raw_weight_sum'] and not row['poisson']['weights_or_state_normalized']
            maxdegree=int(np.bincount(edges.ravel(),minlength=len(m)).max())
            allowance=float(64*np.finfo(float).eps*(maxdegree+1)*(M+1)*(1+float(z))*max(1.,math.sqrt(float(m.sum())))*max(1.,float(np.max(abs(c0)))))
            machine+=allowance;tail+=float(bound)*math.sqrt(float(m.sum()))*float(np.max(abs(c0)))
            assert machine==row['cumulative_machine_indicator_absolute'] and tail==row['cumulative_tail_indicator_absolute']
            delta=norm(m,path['states'][step+1]-following);assert delta==row['primary_control_mass_norm_difference'] and delta<=machine+tail;maximum=max(maximum,delta)
            A=csr_matrix((saved['symmetric_data'],saved['symmetric_indices'],saved['symmetric_indptr']),shape=tuple(saved['shape']));root=np.sqrt(m);gf=np.array(list(map(float,gg)));i,j=edges.T
            np.testing.assert_array_equal(gf,saved['g_float']);action_error=0.
            for u,chi in zip((path['states'][step],path['exponential_half_states'][step],path['states'][step+1]),row['chi_start_half_end']):
                flux=gf*(u[j]-u[i]);net=np.zeros(len(m));np.add.at(net,i,flux);np.add.at(net,j,-flux)
                err=norm(m,A@(root*u)/root-net/m);assert err<=1e-10*max(1.,norm(m,net/m));action_error=max(action_error,err)
                assert float(gf@((u[j]-u[i])**2))==chi and chi>=0
            u=path['states'][step+1];drift=float(m@(u-c0));assert drift==row['raw_mass_change_from_c0'] and abs(drift)/float(m@c0)<1e-10
            assert u.min()>=c0.min()-1e-10 and u.max()<=c0.max()+1e-10
            newvariance=float(m@((u-float(m@u/m.sum()))**2));assert newvariance<=variance+1e-10*V0;variance=newvariance
            result={'columns':N,'K':K,'step':step,'sealed_independent_all_cell_geometry_receipt_verified':True,
              'exact_face_resistance_rates_minimal_tail_raw_weights_complete_sequence_verified':True,
              'primary_control_mass_norm_difference':delta,'maximum_symmetric_shared_action_difference':action_error}
            target=folder/f'k{K}/action-audit-step{step:02d}.json'
            if write:save(target,result)
            else:assert read(target)==result
            count+=1
        print(f'independent saved n{N}/K{K} full sequence verified',flush=True)
    assert count==32
    return {'columns':N,'saved_actions_verified':count,'maximum_primary_control_difference':maximum,
      'all_cell_proof_policy':'verify pre-scalar independent full geometry proof receipts; independently recompute every face resistance, rate and sequence; do not rerun all-cell search'}


def verify_action_specialization():
    old=ast.parse(inspect.getsource(prior_action_audit)).body[0]
    new=ast.parse(inspect.getsource(audit_actions_k32)).body[0];old.name=new.name
    for node in ast.walk(old):
        if isinstance(node,ast.For) and isinstance(node.iter,ast.Name) and node.iter.id=='KS':node.iter=ast.parse('(32,)',mode='eval').body
        if isinstance(node,ast.Assert) and ast.dump(node.test)==ast.dump(ast.parse('count==28',mode='eval').body):node.test=ast.parse('count==32',mode='eval').body
    assert ast.dump(old)==ast.dump(new)


def variance_exact(m,u):
    m=list(map(Q,m));u=list(map(Q,u));M=sum(m,F(0));mean=sum((a*b for a,b in zip(m,u)),F(0))/M
    return sum((a*(b-mean)**2 for a,b in zip(m,u)),F(0))


def squared(m,a,b):
    return sum((Q(w)*(Q(x)-Q(y))**2 for w,x,y in zip(m,a,b)),F(0))


def observe_and_budget(out,r):
    src=out/'source';old=read(src/'results.json');data=arrays(out/'n128/material.npz');col=arrays(src/'n128/column.npz');base=arrays(src/'n32/column.npz')
    m,c0=data['mass'],data['c0'];mp,cp=base['mass'],base['c0'];mc=col['mass'];N=128
    np.testing.assert_array_equal(m,np.tile(mc/8,8));np.testing.assert_array_equal(c0,np.tile(col['c0'],8))
    path=arrays(out/'n128/k32/trajectory-states.npz');u,v=path['states'][-1],path['independent_states'][-1]
    u16=arrays(src/'n128/k16/trajectory-states.npz')['states'][-1];common=arrays(out/'common-observations.npz');pu=common['n128_k32']
    oldcommon=arrays(src/'common-observations.npz');p16=oldcommon['n128_k16'];obs=arrays(src/'reference/observations-l1-k16384.npz');ref=np.tile(obs['n128_projected'],8)
    exact=[];capacity=[]
    for i in range(32):
        ids=[row*128+4*i+j for row in range(8) for j in range(4)]
        exact.append(sum((Q(m[j])*Q(u[j]) for j in ids),F(0))/Q(mp[i]))
        capacity.append(sum((Q(m[j]) for j in ids),F(0))/Q(mp[i]))
    rounding=np.array([float(Q(a)-b) for a,b in zip(pu,exact)])
    assert np.max(abs(rounding))<=64*np.finfo(float).eps*max(1.,float(np.max(abs(u))))
    assert r['J128_star_parts']['new_K32_common_rounding']==norm(mp,rounding)
    assert r['J128_star_parts']['new_K32_capacity_final_bound']==norm(mp,np.array([float(abs(q-1))*float(np.max(abs(u))) for q in capacity]))
    factor=math.nextafter(math.sqrt(float(max(capacity))),math.inf)
    assert factor==r['capacity']['norm_factor_upper'] and float(max(abs(q-1) for q in capacity))==r['capacity']['defect']
    bridge=norm(mc,(u.reshape(8,128)*(mc/8)).sum(axis=0)/mc-u.reshape(8,128)[0])
    assert bridge==r['B128_star_parts']['new_K32_transverse_bridge']
    assert r['B128_star_parts']['retained_B128']==old['time_controls']['128']['budget_absolute']
    for key,value in old['spatial_budget_by_grid']['128']['common_parts'].items():
        if key not in ('observed8_16','capacity_times_original_budget'):assert r['J128_star_parts']['retained_'+key]==value
    for label,field in [('native','native_parts'),('common','common_parts'),('native_H','native_H_parts'),('common_H','common_H_parts')]:
        for key,value in old['spatial_budget_by_grid']['64'][field].items():assert r['positive_components'][label]['retained64_'+key]==value
    initial_variance=float(m@((c0-float(m@c0/m.sum()))**2));A=2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))
    common_initial_variance=float(mp@((cp-float(mp@cp/mp.sum()))**2));Ap=2*math.sqrt(float(mp.sum()))*max(1.,float(np.max(abs(cp))))
    hx=1-variance_exact(m,u)/variance_exact(m,c0);hpx=1-variance_exact(mp,pu)/variance_exact(mp,cp)
    assert abs(float(hx)-r['native_H_K32'])<=64*np.finfo(float).eps
    assert abs(float(hpx)-r['common_H_K32'])<=64*np.finfo(float).eps
    coarse=arrays(src/'n64/material.npz');coarse_u=arrays(src/'n64/k16/trajectory-states.npz')['states'][-1]
    sq={'native_error':squared(m,u,ref),'common_error':squared(mp,pu,obs['n32_projected']),
        'new16_32':squared(m,u16,u),'new_common16_32':squared(mp,p16,pu),'control':squared(m,u,v),
        'coarse_native':squared(coarse['mass'],coarse_u,np.tile(obs['n64_projected'],8)),
        'coarse_common':squared(mp,oldcommon['n64_k16'],obs['n32_projected'])}
    values=[]
    for digits in (80,100):
        with localcontext() as ctx:
            ctx.prec=digits;D=lambda x:Decimal.from_float(float(x));rational=lambda q:Decimal(q.numerator)/Decimal(q.denominator)
            roots={key:rational(q).sqrt() for key,q in sq.items()};sigma=D(r['common_initial_fluctuation_norm'])
            near(roots['native_error']/sigma,r['errors']['native']);near(roots['common_error']/sigma,r['errors']['common'])
            near(roots['coarse_native']/sigma,r['retained64_errors']['native']);near(roots['coarse_common']/sigma,r['retained64_errors']['common'])
            near(roots['new16_32'],r['state_time_control']['new16_32']);near(roots['new_common16_32'],r['positive_components']['common']['new128_observed16_32'])
            near(roots['control'],r['numeric_K32_parts']['complete_sequence_difference'])
            last=read(out/'n128/k32/native-results.json')['steps'][-1]
            assert r['numeric_K32_parts']['machine']==last['cumulative_machine_indicator_absolute'] and r['numeric_K32_parts']['tail']==last['cumulative_tail_indicator_absolute']
            numeric=sum(map(D,r['numeric_K32_parts'].values()),Decimal(0));near(2*numeric,r['B128_star_parts']['twice_new_path_numeric'])
            b=sum(map(D,r['B128_star_parts'].values()),Decimal(0));near(b,r['B128_star'])
            bp=D(factor)*b+sum(map(D,r['J128_star_parts'].values()),Decimal(0));near(bp,r['common_norm_budget'])
            hb=(D(A)*b+b*b)/D(initial_variance);hpb=(D(Ap)*bp+bp*bp)/D(common_initial_variance)
            near(hb,r['shifted_H_time_control']['current_original_H_indicator']);near(hpb,r['positive_components']['common_H']['new128_original_H_indicator'])
            assert all(x>=0 for x in r['B128_star_parts'].values()) and all(x>=0 for x in r['J128_star_parts'].values())
            comparisons=[]
            for key,parts in r['positive_components'].items():
                assert all(x>=0 for x in parts.values());budget=sum(map(D,parts.values()),Decimal(0))/(sigma if key in ('native','common') else 1)
                gain=D(r['retained64_errors'][key])-D(r['errors'][key]);near(gain,r['gains'][key]);near(budget,r['fixed_indicators'][key])
                decision='fine_better_resolvably' if gain>budget else ('fine_worse_resolvably' if gain < -budget else 'comparison_inconclusive')
                assert decision==r['decisions'][key];comparisons.append({'observable':key,'gain':str(gain),'budget':str(budget),'decision':decision})
            state=D(old['time_controls']['128']['state8_16'])-roots['new16_32'];near(state,r['state_time_control']['decrease'])
            assert (state>b)==r['state_time_control']['resolved']
            ceiling=D(old['time_controls']['128']['H8_16']);floor=D(old['time_controls']['128']['H_indicator'])
            assert ceiling<=floor<=hb and not r['shifted_H_time_control']['resolved']
            values.append({'digits':digits,'exact_saved_norms':{key:str(v) for key,v in roots.items()},'B128_star':str(b),'common_norm_budget':str(bp),
                'H_indicator':str(hb),'common_H_indicator':str(hpb),'comparisons':comparisons,'shifted_H_ceiling_below_retained_floor':True})
    return {'complete_mass_union_rounding_max':float(np.max(abs(rounding))),'old_terms_retained_without_cancellation':True,'precision_checks':values}


def main(directory,write=False):
    started=time.monotonic();out=Path(directory);r=read(out/'results.json');ex=read(out/'execution.json');seal=read(out/'coefficient-seal.json')
    assert ex['first_attempt_passed'] and ex['new_primary_paths']==ex['new_independent_paths']==1 and ex['new_reference_PDE_solves']==0
    assert seal['graphs']==32 and seal['all32_sealed_before_mass_scalar_oldstates_reference'] and not seal['internal_material_faces_given_to_builder']
    assert digest(out/'n128/graph-index.json')==seal['index_sha256']
    for name,h in read(out/'implementation-lock.json')['files'].items():assert digest(ROOT/name)==h,name
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==item['sha256']
    verify_sampling_rule();verify_action_specialization()
    pre=read(out/'labels-only-preflight.json');assert pre['old_K16_shared33_frames_bitwise_equal'] and not pre['mass_c0_reference_oldstates_loaded']
    world=read(out/'n128/pure-inputs.json')
    assert set(world['cases'])=={'32'} and all(F(frame['time_exact'])==F(i,640) for i,frame in enumerate(world['cases']['32']['frames']))
    actions=audit_actions_k32(out,128,write);budget=observe_and_budget(out,r)
    current,common=analyze(out);assert plain_json(current)==r
    saved=arrays(out/'common-observations.npz')
    for key,value in common.items():np.testing.assert_array_equal(value,saved[key])
    assert not r['new_full_spatial_or_Stage3_pass_allowed'] and not r['fluid_orc_enabled']
    findings={'passed':True,'actions':actions,'observations_and_positive_budgets':budget,'frozen_analysis_reproduced':True,
        'only_K32_sampling_verified_without_mutating_old_source':True,'old_source_hashes_verified':True,
        'comparison_decision':r['comparison_decision'],'shifted_H_unresolved_retained':True,'new_primary_or_reference_solves_in_audit':0,
        'new_software_checks':0,'recorded_checks_total':502,'scope':'saved independent sequence and complete-union observations/positive arithmetic; no new candidate or reference PDE and no certified continuum error bound'}
    if write:
        for name,h in read(ROOT/'runs/reconnect-becker-spatial-budget/sha256.json').items():assert digest(ROOT/name)==h,name
        save(out/'audit.json',{'passed':True,'first_attempt_passed':True,'findings':findings,'elapsed_seconds':time.monotonic()-started,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'auditor_sha256':digest(Path(__file__))})
    else:assert read(out/'audit.json')['findings']==findings
    print(json.dumps(plain_json(findings),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();main(a.directory,a.write)

