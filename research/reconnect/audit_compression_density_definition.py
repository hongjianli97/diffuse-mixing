"""Independent saved-data/topology and symbolic compression contract audit."""
from pathlib import Path
from fractions import Fraction as F
import argparse,ast,math,resource,time
import numpy as np
import mpmath as mp
from audit_route_decision import ROOT,read,save,digest


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def symbolic():
    # Sparse formal polynomials: no evaluation of any new density or trajectory.
    def var(s):return {(s,):F(1)}
    def add(*polys):
        out={}
        for p in polys:
            for k,v in p.items():out[k]=out.get(k,F(0))+v
        return {k:v for k,v in out.items() if v}
    def scale(p,v):return {k:c*v for k,c in p.items()}
    def mul(a,b):
        out={}
        for k,v in a.items():
            for j,w in b.items():q=tuple(sorted(k+j));out[q]=out.get(q,F(0))+v*w
        return {k:v for k,v in out.items() if v}
    zero={};d,xp,xm=map(var,('d','xp','xm'));vertices=[(xp,zero),(zero,d),(xm,zero),(zero,scale(d,-1))]
    area=scale(add(*(add(mul(a[0],b[1]),scale(mul(a[1],b[0]),-1)) for a,b in zip(vertices,vertices[1:]+vertices[:1]))),F(1,2))
    assert area==add(mul(d,xp),scale(mul(d,xm),-1))
    st,ct,sz,cz=map(var,('sin_theta','cos_theta','sin_z','cos_z'))
    plus=add(mul(st,cz),mul(ct,sz));minus=add(mul(st,cz),scale(mul(ct,sz),-1))
    assert add(plus,scale(minus,-1))==scale(mul(ct,sz),2)
    a,s,c,r=map(var,('alpha','s','cos','rho0'));one={():F(1)}
    oracle_denom=add(one,mul(a,c));proxy_denom=add(one,mul(mul(a,s),c))
    difference=mul(r,add(oracle_denom,scale(proxy_denom,-1)))
    assert difference==mul(mul(mul(r,a),c),add(one,scale(s,-1)))
    assert F(2,64)==F(1,32) and 2*F(1,32)<F(1,2)
    return {'oriented_diamond_area_identity':True,'sine_difference_secant_identity':True,
        'density_difference_cross_multiplication_identity':True,'new_density_finite_sum_or_field_evaluated':False}


def floors_high_precision(source,stored):
    u=arrays(source/'union-definition.npz');paths={K:arrays(source/f'k{K}/trajectory-states.npz') for K in (16,32)}
    runs={K:read(source/f'k{K}/native-results.json') for K in (16,32)};records=[]
    for digits in (80,100):
        with mp.workdps(digits):
            f=lambda v:mp.mpf(F(float(v)).numerator)/F(float(v)).denominator
            m=list(map(f,u['mass']));c=list(map(f,u['c0']));M=mp.fsum(m)
            norm=lambda v:mp.sqrt(mp.fsum(w*q*q for w,q in zip(m,v)))
            def V(v):
                mean=mp.fsum(w*q for w,q in zip(m,v))/M
                return mp.fsum(w*(q-mean)**2 for w,q in zip(m,v))
            V0=V(c);H=lambda v:1-V(v)/V0
            hm=lambda e:(2*mp.sqrt(M)*max(1,max(map(abs,c)))*e+e*e)/V0
            diff=lambda a,b:[f(x)-f(y) for x,y in zip(a,b)]
            check=[]
            for k,row in enumerate(stored['rows'],1):
                cp={'oracle_16_32':norm(diff(paths[16]['states'][8*k],paths[32]['states'][16*k])),
                    'four_reference_space':4*norm(list(map(f,u['space'][k]))),'four_reference_time':4*norm(list(map(f,u['time'][k]))),
                    'two_initial_bridge':2*norm(diff(u['reference'][0],u['c0'])),'same_N64_projection_identity':mp.mpf(0)}
                hp={'oracle_16_32':abs(H(list(map(f,paths[16]['states'][8*k])))-H(list(map(f,paths[32]['states'][16*k])))),
                    'two_reference_space_same_basis':2*abs(H(list(map(f,u['reference_space_coarse'][k])))-H(list(map(f,u['reference_space_fine'][k])))),
                    'two_reference_time_same_basis':2*abs(H(list(map(f,u['reference_time_check'][k])))-H(list(map(f,u['reference'][k])))),
                    'two_initial_bridge':2*abs(H(list(map(f,u['reference'][0])))-H(c)),'same_N64_projection_identity':mp.mpf(0)}
                mt=mp.mpf(0)
                for K in (16,32):
                    step=runs[K]['steps'][k*K//2-1];a,b=paths[K]['states'][k*K//2],paths[K]['independent_states'][k*K//2]
                    machine_tail=f(step['cumulative_machine_indicator_absolute'])+f(step['cumulative_tail_indicator_absolute']);mt+=machine_tail
                    cp[f'oracle_e{K}']=machine_tail+norm(diff(a,b));hp[f'oracle_complete_sequence_H{K}']=abs(H(list(map(f,a)))-H(list(map(f,b))))
                hp['oracle_machine_tail_to_H']=hm(mt)
                maxdiff=mp.mpf(0)
                for computed,key in ((cp,'concentration_fixed_parts_absolute'),(hp,'H_fixed_parts')):
                    assert set(computed)==set(row[key])
                    for name,v in computed.items():
                        assert v>=0;delta=abs(v-f(row[key][name]));maxdiff=max(maxdiff,delta)
                        assert delta<=64*np.finfo(float).eps*max(1,abs(v))
                bc=mp.fsum(cp.values())/f(stored['common_sigma64']);bh=mp.fsum(hp.values())
                assert abs(bc-f(row['concentration_floor_relative']))<64*np.finfo(float).eps
                assert abs(bh-f(row['H_floor']))<64*np.finfo(float).eps
                check.append({'time':row['time'],'concentration_floor':float(bc),'H_floor':float(bh),'maximum_float_term_difference':float(maxdiff)})
            records.append({'digits':digits,'rows':check})
    assert records[0]['rows']==records[1]['rows']
    return records


def main(directory,write=False):
    started=time.monotonic();out=Path(directory).resolve();p=read(out/'protocol.json');definition=read(out/'definition.json')
    for name,h in p['sources'].items():assert digest(ROOT/name)==h,name
    for name,h in read(out/'implementation-lock.json')['files'].items():assert digest(ROOT/name)==h,name
    assert digest(out/'source-gate-spec.json')==p['source_gate_sha256'] and digest(ROOT/p['contract'])==definition['contract_sha256']
    ex=read(out/'execution.json');assert ex['completed'] and ex['elapsed_seconds']<120 and ex['peak_RSS_MiB']<1024
    src=ROOT/'runs/reconnect-compression-union64';b=arrays(src/'builder-inputs.npz');s=arrays(src/'trajectory-source.npz');u=arrays(src/'union-definition.npz')
    inv=read(out/'input-inventory.json');index=read(src/'graph-index.json');seal=read(src/'coefficient-seal.json')
    assert inv['nodes']==4096 and inv['saved_geometries_verified']==48 and inv['faces_per_geometry']==8192
    assert inv['coarse_frames_bitwise_subset_of_fine_frames'] and inv['distinct_midpoint_times']==48
    for q in range(33):
        for name in ('positions','rho'):np.testing.assert_array_equal(b['k16_'+name][q],b['k32_'+name][2*q])
        assert b['k16_time'][q]==b['k32_time'][2*q]
    for i,xy in enumerate(s['labels']):
        row,col=divmod(i,64);assert xy[0]==(2*col+1)/128 and xy[1]==(2*row+1)/128 and s['ids'][i]==i
    assert np.array_equal(b['k32_positions'][0],s['labels']) and np.array_equal(b['k32_rho'][0],s['rho0_point'])
    for item in inv['saved_motion_diagnostics']:
        K=item['K'];x=b[f'k{K}_positions'];delta=x[1:]-x[:-1]
        short=np.where(delta>.5,delta-1,np.where(delta<-.5,delta+1,delta))
        assert not np.any(np.abs(delta)==.5)
        assert np.max(abs(short),axis=(0,1)).tolist()==item['maximum_short_displacement_per_axis']
        assert np.count_nonzero(delta>.5,axis=(0,1)).tolist()==item['positive_wrapped_jumps']
        assert np.count_nonzero(delta<-.5,axis=(0,1)).tolist()==item['negative_wrapped_jumps']
        assert item['maximum_saved_dt']<F(1,2) and not item['new_lift_array_or_winding_field_generated']
    assert set(map(float,b['k16_midpoint_time'])).isdisjoint(set(map(float,b['k32_midpoint_time'])))
    for item in inv['graphs']:
        K,step=item['K'],item['step'];entry=index[f'k{K}-step{step:02d}'];g=read(src/item['file'])
        receipt=read(src/f'k{K}/geometry-only-audit-step{step:02d}.json')
        assert digest(src/item['file'])==item['sha256']==entry['sha256']==receipt['cache_sha256']
        assert receipt['builder_packet_sha256']==seal['builder_packet_sha256'] and receipt['pure_geometry_only_no_mass_c0_reference_loaded']
        degree=[0]*4096;seen=set()
        for i,j in g['edges']:
            assert 0<=i<j<4096 and (i,j) not in seen;seen.add((i,j));degree[i]+=1;degree[j]+=1
            ri,ci=divmod(i,64);rj,cj=divmod(j,64)
            assert (ri==rj and (ci-cj)%64 in (1,63)) or (ci==cj and (ri-rj)%64 in (1,63))
        assert all(d==4 for d in degree) and g['ids']==s['ids'].tolist() and F(g['D_exact'])==F(.02)
    for K in (16,32):
        path=arrays(src/f'k{K}/trajectory-states.npz');run=read(src/f'k{K}/native-results.json');saved=inv['oracle_paths'][str(K)]
        assert digest(src/f'k{K}/trajectory-states.npz')==saved['path_sha256'] and digest(src/f'k{K}/native-results.json')==saved['run_sha256']
        np.testing.assert_array_equal(path['states'][0],u['c0']);np.testing.assert_array_equal(path['independent_states'][0],u['c0'])
        assert path['states'].shape==(K+1,4096) and all(F(x['dt_exact'])==F(1,20*K) for x in run['steps'])
    case=definition['proposed_case'];assert case==p['proposed_case']
    assert case['new_density_fields']==case['new_coefficient_packets']==96
    assert case['new_actual_paths']==case['new_independent_paths']==4 and case['new_action_intervals']==96
    assert case['new_geometry_graphs']==case['new_reference_PDE_solves']==case['old_paths_rerun']==0
    assert case['K']==[16,32] and case['h_exact']=='1/32' and case['new_software_groups']==2 and case['future_recorded_checks_if_pass']==506
    assert definition['recorded_checks_total']==504 and not definition['future_execution_performed'] and not definition['fluid_orc_enabled']
    for name in ('compression_density_definition.py','audit_compression_density_definition.py'):
        tree=ast.parse((ROOT/'research/reconnect'/name).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert set(imports)<={'pathlib','fractions','audit_route_decision'}
    proof=symbolic();floors=floors_high_precision(src,read(out/'fixed-budget-floor.json'))
    findings={'passed':True,'all48_saved_graph_hashes_and_cardinal_topologies_verified':True,'saved_shared_frames_and_periodic_crossings_verified':True,
        'original_mass_c0_reference_and_oracle_paths_reused':True,'symbolic_checks':proof,'independent_budget_floors':floors,
        'new_density_coefficient_lift_graph_or_state_arrays':0,'recorded_checks_total':504,'stage3':'incomplete','fluid_orc_enabled':False}
    if write:save(out/'audit.json',{'passed':True,'findings':findings,'elapsed_seconds':time.monotonic()-started,
        'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
    else:assert read(out/'audit.json')['findings']==findings
    print('Independent saved48-topology/frame audit,formal secant identities and80/100-digit fixed budget floors verified; no new physical fields.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();main(a.directory,a.write)
