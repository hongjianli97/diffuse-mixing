"""Saved reference/observation budgets and exact candidate sequences; no new PDE."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
import argparse,json,math,resource,time
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save as raw_save,digest
from affine_shear_completion import plain_json
from becker_spatial_core import NS,KS,REFS,analyze
from aligned_becker_check import arrays
from audit_aligned_becker import near


def save(path,obj):raw_save(path,plain_json(obj))
def Q(v):return F(float(v))
def norm(m,u):return float(np.sqrt(m@(u*u)))


def audit_actions(out,N,write):
    folder=out/f'n{N}';world=read(folder/'pure-inputs.json');index=read(folder/'graph-index.json');data=arrays(folder/'material.npz')
    m,c0=data['mass'],data['c0'];mm=list(map(Q,m));count=0;maximum=0.
    for K in KS:
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
    assert count==28
    return {'columns':N,'saved_actions_verified':count,'maximum_primary_control_difference':maximum,
      'all_cell_proof_policy':'verify pre-scalar independent full geometry proof receipts; independently recompute every face resistance, rate and sequence; do not rerun all-cell search'}


def audit_references(out):
    columns={N:arrays(out/f'n{N}/column.npz') for N in NS};p=list(map(Q,columns[128]['mass_edges']));mc=list(map(Q,columns[128]['mass']))
    contract=read(out/'reference/mesh-contract.json');counts=[math.ceil(v/((p[-1]-p[0])/16384)) for v in mc];assert counts==contract['counts'];records=[];endpoints={}
    for level,k in REFS:
        name=f'l{level}-k{k}';raw=arrays(out/f'reference/raw-{name}.npz');inp=arrays(out/f'reference/input-{name}.npz');obs=arrays(out/f'reference/observations-{name}.npz');guard=read(out/f'reference/guard-{name}.json')
        for key in inp:np.testing.assert_array_equal(inp[key],raw[key])
        e=list(map(Q,raw['mass_edges']));mass=list(map(Q,raw['mass']));c0=list(map(Q,raw['c0']));c=list(map(Q,raw['c']));index={v:i for i,v in enumerate(e)}
        cursor=0
        for left,right,count in zip(p[:-1],p[1:],counts):
            n=count*2**level
            for j in range(n+1):assert float(left+(right-left)*j/n)==float(e[cursor+j])
            cursor+=n
        assert cursor==len(mass);np.testing.assert_array_equal(raw['mass'],np.diff(raw['mass_edges']))
        if level in endpoints:np.testing.assert_array_equal(endpoints[level],raw['mass_edges'])
        else:endpoints[level]=raw['mass_edges']
        assert guard['steps_completed']==guard['linear_solves']==k and guard['positivity_policy']=='monitor' and not guard['states_clipped_or_normalized']
        assert guard['max_relative_mass_drift']<=1e-9 and guard['min_all_times']>=guard['initial_min']-1e-10 and guard['max_all_times']<=guard['initial_max']+1e-10
        assert np.isfinite(raw['c']).all() and raw['c'].min()>=guard['min_all_times'] and raw['c'].max()<=guard['max_all_times']
        prefix=[[F(0)] for _ in range(5)]
        for j,m in enumerate(mass):
            defect=abs(m-(e[j+1]-e[j]));values=(m,m*c0[j],m*c[j],defect*abs(c0[j]),defect*abs(c[j]))
            for a,v in zip(prefix,values):a.append(a[-1]+v)
        drift=abs(prefix[2][-1]-prefix[1][-1])/abs(prefix[1][-1]);assert drift<=F(1,10**9)
        for N in NS:
            target=list(map(Q,columns[N]['mass_edges']));tm=list(map(Q,columns[N]['mass']));tc=list(map(Q,columns[N]['c0']))
            ri=[];rf=[];gap=[];gi=[];gf=[];closure=[]
            for i,(left,right) in enumerate(zip(target[:-1],target[1:])):
                lo=index[left];hi=index[right];mass_sum,i_sum,f_sum,i_geo,f_geo=[a[hi]-a[lo] for a in prefix]
                assert right-left==tm[i];im=i_sum/tm[i];fm=f_sum/tm[i]
                gap.append(im-tc[i]);ri.append(Q(obs[f'n{N}_initial_projected'][i])-im);rf.append(Q(obs[f'n{N}_projected'][i])-fm)
                gi.append(i_geo/tm[i]);gf.append(f_geo/tm[i]);closure.append(mass_sum-tm[i])
                assert float(im)==obs[f'n{N}_exact_initial_projected_rounded'][i] and float(fm)==obs[f'n{N}_exact_projected_rounded'][i]
            for key,values in [('exact_initial_mean_gap',gap),('initial_aggregation_rounding',ri),('terminal_aggregation_rounding',rf),('initial_geometry_bound',gi),('terminal_geometry_bound',gf)]:
                np.testing.assert_array_equal(obs[f'n{N}_{key}'],np.array(list(map(float,values))))
            records.append({'reference_level':level,'steps':k,'observed_columns':N,'maximum_exact_mass_closure':float(max(map(abs,closure))),
              'exact_endpoint_relative_mass_drift':float(drift),'all_complete_child_means_verified':True})
    assert np.array_equal(endpoints[0],endpoints[1][::2]);return records


def audit_common_and_budget(out,r):
    common=arrays(out/'common-observations.npz');base=arrays(out/'n32/column.npz');mp=list(map(Q,base['mass']));cp=list(map(Q,base['c0']));M=sum(mp,F(0))
    decimal_checks=[];observation_checks=[];exact_errors={}
    def variance(m,u):
        mean=sum((a*b for a,b in zip(m,u)),F(0))/sum(m,F(0))
        return sum((a*(b-mean)**2 for a,b in zip(m,u)),F(0))
    V0=variance(mp,cp)
    for N in NS:
        material=arrays(out/f'n{N}/material.npz');col=arrays(out/f'n{N}/column.npz');m=list(map(Q,material['mass']));ratio=N//32
        np.testing.assert_array_equal(material['mass'],np.tile(col['mass']/8,8));np.testing.assert_array_equal(material['c0'],np.tile(col['c0'],8))
        reference=arrays(out/'reference/observations-l1-k16384.npz')[f'n{N}_projected']
        vectors={'c0':material['c0'],'reference':np.tile(reference,8)}
        for K in KS:vectors[f'k{K}']=arrays(out/f'n{N}/k{K}/trajectory-states.npz')['states'][-1]
        maximum=0.
        for key,u in vectors.items():
            exact=[]
            for i in range(32):
                ids=[row*N+i*ratio+j for row in range(8) for j in range(ratio)]
                exact.append(sum((m[j]*Q(u[j]) for j in ids),F(0))/mp[i])
            error=max(float(abs(Q(a)-b)) for a,b in zip(common[f'n{N}_{key}'],exact));maximum=max(maximum,error)
            assert error<=64*np.finfo(float).eps*max(1.,float(np.max(abs(u))))
        observation_checks.append({'columns':N,'maximum_complete_union_component_rounding':maximum,'no_partial_interpolation_or_reweighting':True})
        native_ref=list(map(Q,np.tile(reference,8)));native_u=list(map(Q,vectors['k16']))
        common_ref=list(map(Q,arrays(out/'reference/observations-l1-k16384.npz')['n32_projected']))
        common_u=list(map(Q,common[f'n{N}_k16']))
        exact_errors[N]={'native_squared':sum((a*(u-v)**2 for a,u,v in zip(m,native_u,native_ref)),F(0)),
          'common_squared':sum((a*(u-v)**2 for a,u,v in zip(mp,common_u,common_ref)),F(0)),
          'native_H':1-variance(m,native_u)/variance(m,list(map(Q,material['c0']))),
          'native_reference_H':1-variance(m,native_ref)/variance(m,list(map(Q,material['c0']))),
          'common_H':1-variance(mp,common_u)/V0,'common_reference_H':1-variance(mp,common_ref)/V0}
        # Original numeric indicator also bounds the saved H evaluation rounding.
        for key in ('native_H','native_reference_H','common_H','common_reference_H'):
            assert abs(float(exact_errors[N][key])-r['observations'][str(N)][key])<=r['time_controls'][str(N)]['H_indicator']
    for digits in (80,100):
        with localcontext() as ctx:
            ctx.prec=digits;D=lambda v:Decimal.from_float(float(v));checks=[];endpoint_errors=[]
            for N in NS:
                values={}
                for kind,key in [('native','native_error_common_denominator'),('common','common_error')]:
                    q=exact_errors[N][kind+'_squared'];value=(Decimal(q.numerator)/Decimal(q.denominator)).sqrt()/D(r['common_initial_fluctuation_norm'])
                    near(value,r['observations'][str(N)][key]);values[kind]=str(value)
                endpoint_errors.append({'columns':N,'exact_squared_error_with_saved_normalizer':values})
            # Independently sum all named positive budget parts and reconstruct
            # original H polynomial; the exact reference integrals were checked above.
            for N in NS:
                tc=r['time_controls'][str(N)];values=list(tc['components_absolute'].values());assert all(v>=0 for v in values)
                b=sum(map(D,values),Decimal(0));near(b,tc['budget_absolute'])
                z=arrays(out/f'n{N}/material.npz');mn,cn=z['mass'],z['c0'];mean=float(mn@cn/mn.sum());E0=float(mn@((cn-mean)**2));A=2*math.sqrt(float(mn.sum()))*max(1.,float(np.max(abs(cn))))
                hb=(D(A)*b+b*b)/D(E0);near(hb,tc['H_indicator'])
                assert (D(tc['state_decrease'])>b)==tc['state_decrease_resolved'] and (D(tc['H_decrease'])>hb)==tc['H_decrease_resolved']
                checks.append({'columns':N,'state_budget':str(b),'H_indicator':str(hb)})
            adjacent=[]
            for row in r['adjacent_spatial_comparisons']:
                values={}
                for key,parts in row['positive_components'].items():
                    assert all(v>=0 for v in parts.values());b=sum(map(D,parts.values()),Decimal(0))
                    if key in ('native','common'):b/=D(r['common_initial_fluctuation_norm'])
                    near(b,row['fixed_indicators'][key]);gain=D(row['gains'][key]);decision='fine_better_resolvably' if gain>b else ('fine_worse_resolvably' if gain < -b else 'comparison_inconclusive')
                    assert decision==row['decisions'][key];values[key]={'budget':str(b),'decision':decision}
                adjacent.append({'coarse':row['coarse_columns'],'fine':row['fine_columns'],'checks':values})
            decimal_checks.append({'digits':digits,'independent_endpoint_errors':endpoint_errors,'time_checks':checks,'spatial_checks':adjacent})
    return observation_checks,decimal_checks


def main(directory,write=False):
    start=time.monotonic();out=Path(directory);r=read(out/'results.json');ex=read(out/'execution.json');lock=read(out/'implementation-lock.json');seal=read(out/'coefficient-seal.json')
    assert ex['first_attempt_passed'] and ex['new_candidate_paths']==6 and ex['new_reference_PDE_solves']==3
    assert seal['graphs']==56 and seal['all56_sealed_before_mass_scalar_oldstates_or_reference'] and not seal['internal_material_faces_given_to_builder']
    for rel,h in lock['files'].items():assert digest(ROOT/rel)==h,rel
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==item['sha256']
    for N in (64,128):assert digest(out/f'n{N}/graph-index.json')==seal['indices'][str(N)]
    actions=[audit_actions(out,N,write) for N in (64,128)];refs=audit_references(out);obs,decimal=audit_common_and_budget(out,r)
    current,common=analyze(out);assert plain_json(current)==r
    saved=arrays(out/'common-observations.npz')
    for key,value in common.items():np.testing.assert_array_equal(value,saved[key])
    findings={'passed':True,'actions':actions,'reference_integrals':refs,'common_integrals':obs,'decimal_checks':decimal,
      'frozen_comparison_reproduced':True,'source_and_coefficient_hashes_verified':True,'old32_candidate_reruns':0,'new_PDE_solves_in_audit':0,
      'new_software_checks':0,'recorded_checks_total':502,'comparison_decision':r['comparison_decision'],
      'scope':'independent saved face/rate/full-uniformization reconstruction and sealed pre-scalar all-cell proofs; saved reference endpoints and positive-budget arithmetic,not an independent reference PDE trajectory or certified continuum bound'}
    if write:
        for rel,h in read(ROOT/'runs/reconnect-aligned-becker/sha256.json').items():assert digest(ROOT/rel)==h,rel
        save(out/'audit.json',{'passed':True,'first_attempt_passed':True,'findings':findings,'elapsed_seconds':time.monotonic()-start,
          'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'auditor_sha256':digest(Path(__file__))})
    else:assert findings==read(out/'audit.json')['findings']
    print(json.dumps(plain_json(findings),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--write',action='store_true');a=parser.parse_args();main(a.directory,a.write)
