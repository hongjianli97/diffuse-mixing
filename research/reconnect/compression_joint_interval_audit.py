"""Independent reconstruction of all128 saved positive action intervals."""
from fractions import Fraction as F
import math
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import read,digest
from shear_union64_study import arrays

def audit_intervals(out,m,c0):
    norm=lambda v:float(np.sqrt(m@(v*v)));mu0=float(m@c0/m.sum());V0=float(m@((c0-mu0)**2))
    maximum_action_difference=maximum_mass_drift=0.;count=0
    for source in ('oracle','ledger'):
        index=read(out/source/'graph-index.json')
        for K in (64,):
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
    assert count==128
    return {'verified_intervals':count,'whole_independent_recurrences_bitwise_verified':True,
        'maximum_shared_vs_symmetric_action_difference':maximum_action_difference,'maximum_raw_mass_drift':maximum_mass_drift,
        'new_primary_propagations_in_audit':0,'old_oracle_paths_rerun':0}

