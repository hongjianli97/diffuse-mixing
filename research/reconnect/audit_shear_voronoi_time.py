"""Independent bisector/half-resistance and full uniformization-sequence audit."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import gc
import json
import math
import numpy as np
from scipy.sparse import csr_matrix

from audit_route_decision import ROOT, read, save, digest, preservation
from audit_shear_voronoi_fv import fixed_binary_enclosure

SOURCE=ROOT/'runs/reconnect-shear-voronoi-FV-frozen'


def dot(a,b):return a[0]*b[0]+a[1]*b[1]
def cross(a,b):return a[0]*b[1]-a[1]*b[0]
def sub(a,b):return (a[0]-b[0],a[1]-b[1])


def audit_geometry(packet,positions,rho,ids,probe_ids):
    points=tuple(tuple(F(float(x)) for x in row) for row in positions)
    density=tuple(F(float(x)) for x in rho)
    periods=tuple(map(F,packet['periods_exact']));D=F(packet['D_exact']);b=F(packet['thickness_exact'])
    assert list(map(int,ids))==packet['ids'] and D==F(float(.02)) and b==1 and periods==(1,1)
    vertices=tuple(tuple(map(F,row)) for row in packet['vertices_exact'])
    directed={};areas=[];selected={}
    def vector(i,label):
        j,sx,sy=label
        return (points[j][0]+sx*periods[0]-points[i][0],points[j][1]+sy*periods[1]-points[i][1])
    for i,(vi,labels) in enumerate(zip(packet['cell_vertex_indices'],packet['cell_incoming_labels'])):
        assert len(vi)==len(labels)>=3
        poly=[]
        for k,label in enumerate(labels):
            a=vector(i,label);v=vector(i,labels[(k+1)%len(labels)])
            ca=dot(a,a)/2;cv=dot(v,v)/2;det=cross(a,v);assert det
            q=((ca*v[1]-a[1]*cv)/det,(a[0]*cv-ca*v[0])/det)
            world=((points[i][0]+q[0])%periods[0],(points[i][1]+q[1])%periods[1])
            assert world==vertices[vi[k]]
            poly.append(q)
        twice=pyramid=F(0)
        for k,(q,label) in enumerate(zip(poly,labels)):
            prev=poly[k-1];nxt=poly[(k+1)%len(poly)]
            assert cross(sub(q,prev),sub(nxt,q))>0
            v=vector(i,label);d2=dot(v,v);edge=sub(q,prev)
            assert d2>0 and dot(v,edge)==0 and 2*dot(v,prev)==2*dot(v,q)==d2
            ratio=cross(v,edge)/d2;assert ratio>0
            twice+=cross(prev,q);pyramid+=ratio*d2/4
            tag=(i,*label);assert tag not in directed
            directed[tag]=(ratio,(vi[k-1],vi[k]))
        area=twice/2;assert area>0 and area==pyramid==F(packet['cell_areas_exact'][i])
        areas.append(area)
        if int(ids[i]) in probe_ids:selected[int(ids[i])]=(i,poly)
    lo,hi=fixed_binary_enclosure(areas)
    assert lo<=periods[0]*periods[1]<=hi
    original=packet['geometry_diagnostics']['raw_total_area_enclosure']
    assert F(original['lower_exact'])<=hi and lo<=F(original['upper_exact'])
    g=[];edges=[];degree=[F(0)]*len(ids);face_keys=set()
    for k,((i,j),(sx,sy)) in enumerate(zip(packet['edges'],packet['image_shifts'])):
        assert ids[i]<ids[j] and (i,j,sx,sy) not in face_keys
        face_keys.add((i,j,sx,sy))
        first=directed[(i,j,sx,sy)];other=directed[(j,i,-sx,-sy)]
        assert first[0]==other[0]==F(packet['face_over_distance_exact'][k])
        assert first[1]==tuple(packet['face_vertex_indices'][k])==other[1][::-1]
        # Independent sum of two half-resistances; not the builder's harmonic product.
        conductance=b*first[0]/(1/(2*density[i]*D)+1/(2*density[j]*D))
        assert conductance>0 and conductance==F(packet['conductances_exact'][k])
        g.append(conductance);edges.append((i,j));degree[i]+=conductance;degree[j]+=conductance
    assert 2*len(edges)==len(directed)
    probes=[]
    for ident in probe_ids:
        i,poly=selected[ident];R=max(abs(q[0])+abs(q[1]) for q in poly)
        assert R<min(periods)/4
        possible=0;worst=None
        for j,p in enumerate(points):
            if j==i:continue
            delta=[]
            for k,L in enumerate(periods):
                q=p[k]-points[i][k]
                if q>L/2:q-=L
                elif q<-L/2:q+=L
                delta.append(q)
            if abs(delta[0])>2*R or abs(delta[1])>2*R:continue
            v=tuple(delta);d2=dot(v,v);possible+=1
            residual=max(2*dot(v,q)-d2 for q in poly);assert residual<=0
            worst=residual if worst is None else max(worst,residual)
        probes.append({'id':ident,'other_points_checked':len(ids)-1,'possible_cutters_tested':possible,
            'maximum_signed_halfplane_residual_exact':str(worst),'final_radius_L1_exact':str(R)})
    result={'all_vertices_independently_rebuilt_from_bisectors':True,
        'all_face_counterparts_and_two_half_resistances_exact':True,
        'shared_positive_faces':len(edges),'area_sum_256bit_enclosure':{'lower_exact':str(lo),'upper_exact':str(hi)},
        'fixed_all_point_halfplane_probes':{'fixed_ids':probe_ids,'all_selected_cells_passed':True,'records':probes}}
    return np.array(edges,dtype=np.int64),g,degree,result


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    spec=read(out/'source-gate-spec.json');protocol=read(out/'implementation-protocol.json')
    lock=read(out/'implementation-lock.json');execution=read(out/'execution.json')
    assert spec==read(SOURCE/'next-gate-spec.json') and not spec['already_executed']
    assert read(out/'tests.json')=={'successful':True,'checks_run':6,'failures':0,'errors':0,'skipped':0}
    assert execution['scientific_main_first_attempt_passed'] and execution['test_first_attempt_passed']
    assert execution['elapsed_main_seconds']<600 and execution['peak_RSS_MiB']<512
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert digest(out/'implementation-protocol.json')==lock['protocol_sha256']==digest(ROOT/'research/reconnect/voronoi_time_protocol.json')
    assert digest(out/'control-at-lock.md')==lock['control_note_sha256']
    assert digest(out/'builder-inputs.npz')==spec['builder_input_packet_sha256']
    assert digest(out/'original-comparison.npz')==spec['comparison_packet_sha256']
    seal=read(out/'coefficient-seal.json');index=read(out/'graph-index.json')
    assert seal['graph_index_sha256']==digest(out/'graph-index.json') and seal['graphs']==len(index)==12
    assert seal['all12_geometry_and_conductance_caches_sealed_before_mass_c0_reference_load']
    assert not seal['geometry_builder_received_mass_scalar_reference_or_true_faces']
    with np.load(out/'builder-inputs.npz',allow_pickle=False) as z:builder={k:z[k].copy() for k in z.files}
    with np.load(out/'original-comparison.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with np.load(out/'trajectory-states.npz',allow_pickle=False) as z:states={k:z[k].copy() for k in z.files}
    m,c0=data['mass'],data['c0'];exact_m=tuple(F(float(x)) for x in m)
    mu0=float(m@c0/m.sum());E0=float(m@((c0-mu0)**2));sigma=math.sqrt(E0)
    result=read(out/'results.json');assert result['original_initial_mass_fluctuation_norm']==sigma
    audits=[];max_action_delta=0.;max_raw_mass=0.;max_control_delta=0.
    H=lambda c:1-float(m@((c-float(m@c/m.sum()))**2))/E0
    rel=lambda v:float(np.sqrt(m@(v*v))/sigma)
    for K in (4,8):
        primary=states[f'k{K}_states'];independent=states[f'k{K}_independent_states']
        np.testing.assert_array_equal(primary[0],c0);np.testing.assert_array_equal(independent[0],c0)
        previous_control=c0.copy();dt=F(1,20*K);cumulative_machine=0.;cumulative_tail=0.
        for step in range(K):
            item=index[f'k{K}-step{step:02d}'];row=result['cases'][str(K)]['steps'][step]
            assert item['K']==K and item['step']==step and item['frame']==2*step+1
            assert item['midpoint_time_saved']==float(builder[f'k{K}_midpoint_time'][step])==row['midpoint_time_saved']
            assert row['dt_exact']==str(dt)
            assert digest(out/item['file'])==item['sha256']==row['coefficient_cache_sha256']
            packet=read(out/item['file']);frame=2*step+1
            edges,g,degree,audit=audit_geometry(packet,builder[f'k{K}_positions'][frame],builder[f'k{K}_rho'][frame],builder['ids'],protocol['fixed_geometry_probe_ids'])
            assert audit['fixed_all_point_halfplane_probes']==item['independent_fixed_probes']
            del packet;gc.collect()
            Lambda=math.ceil(max(d/a for d,a in zip(degree,exact_m)))
            assert Lambda==row['operator']['Lambda_exact_integer']
            rows=[{} for _ in m]
            for (i,j),w in zip(edges,g):
                assert j not in rows[i] and i not in rows[j]
                rows[i][j]=float(w/(Lambda*exact_m[i]));rows[j][i]=float(w/(Lambda*exact_m[j]))
            for i,(d,a) in enumerate(zip(degree,exact_m)):
                diagonal=1-d/(Lambda*a);assert diagonal>=0
                rows[i][i]=float(diagonal)
            indptr=[0];columns=[];values=[]
            for values_by_column in rows:
                for j in sorted(values_by_column):columns.append(j);values.append(values_by_column[j])
                indptr.append(len(columns))
            P=csr_matrix((values,columns,indptr),shape=(len(m),len(m)))
            with np.load(out/f'action-k{K}-step{step:02d}.npz',allow_pickle=False) as z:
                assert np.array_equal(P.data,z['uniformization_data'])
                assert np.array_equal(P.indices,z['uniformization_indices']) and np.array_equal(P.indptr,z['uniformization_indptr'])
                np.testing.assert_array_equal(z['edges'],edges)
                gf=np.array(list(map(float,g)));np.testing.assert_array_equal(z['g_float'],gf)
                weights=z['weights'].copy()
                A=csr_matrix((z['symmetric_data'],z['symmetric_indices'],z['symmetric_indptr']),shape=tuple(z['shape']))
            zexact=Lambda*dt;M=len(weights)-1;assert M==row['poisson']['M'] and str(zexact)==row['poisson']['z_exact']
            bound=zexact**(M+1)/math.factorial(M+1)/(1-zexact/(M+2))
            assert bound==F(row['poisson']['tail_upper_exact'])<=F(1,10**16)
            maximum_degree=int(np.bincount(edges.ravel(),minlength=len(m)).max())
            machine=float(64*np.finfo(float).eps*(maximum_degree+1)*(M+1)*(1+float(zexact))
                          *max(1.,math.sqrt(float(m.sum())))*max(1.,float(np.max(abs(c0)))))
            cumulative_machine+=machine
            cumulative_tail+=float(bound)*math.sqrt(float(m.sum()))*float(np.max(abs(c0)))
            assert machine==row['machine_indicator_absolute_this_interval']
            assert cumulative_machine==row['cumulative_machine_indicator_absolute']
            assert cumulative_tail==row['cumulative_tail_indicator_absolute']
            if M:
                ratio=zexact/(M+1)
                assert ratio>=1 or zexact**M/math.factorial(M)/(1-ratio)>F(1,10**16)
            weight=math.exp(-float(zexact));v=previous_control.copy();following=weight*v
            assert weight==weights[0]
            for k in range(1,M+1):
                v=P@v;weight*=float(zexact)/k
                assert weight==weights[k]
                following+=weight*v
            np.testing.assert_array_equal(following,independent[step+1]);previous_control=following
            assert math.fsum(weights)==row['poisson']['raw_weight_sum'] and not row['poisson']['weights_or_state_normalized']
            delta=rel(primary[step+1]-following)
            assert delta==row['primary_control_relative_difference'];max_control_delta=max(max_control_delta,delta)
            root=np.sqrt(m);u=primary[step];net=np.zeros(len(m))
            i,j=edges.T;flux=gf*(u[j]-u[i]);np.add.at(net,i,flux);np.add.at(net,j,-flux)
            action_delta=float(np.sqrt(m@((A@(root*u)/root-net/m)**2)))
            action_scale=max(1.,float(np.sqrt(m@((net/m)**2))))
            assert action_delta<=1e-10*action_scale;max_action_delta=max(max_action_delta,action_delta)
            saved_half=states[f'k{K}_exponential_half_states'][step]
            for c,chi in zip((u,saved_half,primary[step+1]),row['chi_start_half_end']):
                assert float(gf@((c[j]-c[i])**2))==chi and chi>=0
                projection=-float((root*c)@(A@(root*c)))
                assert abs(projection-chi)<1e-10*max(1.,chi)
            raw=float(m@(primary[step+1]-c0));assert raw==row['raw_mass_change_from_c0']
            max_raw_mass=max(max_raw_mass,abs(raw))
            assert abs(raw)/float(m@c0)<1e-10
            assert primary[step+1].min()>=c0.min()-1e-10 and primary[step+1].max()<=c0.max()+1e-10
            assert H(primary[step+1])>=H(u)-1e-10
            audit.update({'K':K,'step':step,'uniformization_matrix_and_whole_sequence_bitwise_verified':True,
                          'Poisson_minimal_degree_and_unscaled_weights_verified':True,
                          'symmetric_vs_shared_edge_action_mass_norm_difference':action_delta})
            audits.append(audit)
            print(f'independent geometry/action/sequence audit:k{K} step{step+1}/{K}',flush=True)
            del A,P,rows,g,degree,edges;gc.collect()
        for k in range(3):
            c=primary[k*K//2];ref=data['reference_states'][k];r=result['cases'][str(K)]['observations'][k]
            assert rel(c-ref)==r['relative_concentration_error']
            assert H(c)==r['candidate_metrics']['H'] and H(ref)==r['reference_metrics']['H']
    for k,row in enumerate(result['observations']):
        a=states['k8_states'][4*k];a4=states['k4_states'][2*k]
        b=data['k8_states'][4*k];b4=data['k4_states'][2*k];ref=data['reference_states'][k]
        assert rel(a-ref)==row['FV_relative_concentration_error'] and rel(b-ref)==row['saved_kernel_relative_concentration_error']
        assert rel(a4-a)==row['FV_4_8_state_difference'] and rel(b4-b)==row['saved_kernel_4_8_state_difference']
        assert H(a)==row['FV_H'] and H(b)==row['saved_kernel_H'] and H(ref)==row['reference_H']
        solver_absolute=solver_observed=solver_H_observed=0.
        for K,idx in ((4,2*k),(8,4*k)):
            if idx:
                step=result['cases'][str(K)]['steps'][idx-1]
                solver_absolute+=step['cumulative_machine_indicator_absolute']+step['cumulative_tail_indicator_absolute']
                solver_observed+=step['primary_control_relative_difference']
                solver_H_observed+=abs(step['state_metrics']['H']-step['independent_metrics']['H'])
        bridge=rel(data['reference_states'][0]-c0)
        old_machine=64*np.finfo(float).eps*(4+8+2)*max(1.,float(np.sqrt(m@(c0*c0))))/sigma
        reference_space=rel(data['reference_space_difference'][k]);reference_time=rel(data['reference_time_difference'][k])
        expected_indicator=rel(a4-a)+rel(b4-b)+2*(reference_space+reference_time+bridge)+solver_absolute/sigma+solver_observed+old_machine
        assert expected_indicator==row['combined_state_comparison_indicator']
        coarse_time=ref+data['reference_time_difference'][k]
        H_space=abs(H(coarse_time+data['reference_space_difference'][k])-H(coarse_time))
        H_time=abs(H(coarse_time)-H(ref));H_bridge=abs(H(data['reference_states'][0])-H(c0))
        state_absolute=solver_absolute+old_machine*sigma
        H_machine=(2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))*state_absolute+state_absolute**2)/(sigma*sigma)
        expected_H_indicator=abs(H(a4)-H(a))+abs(H(b4)-H(b))+2*(H_space+H_time+H_bridge)+solver_H_observed+H_machine
        assert expected_H_indicator==row['combined_H_comparison_indicator']
        if k:
            gain=rel(b-ref)-rel(a-ref);indicator=row['combined_state_comparison_indicator']
            expected='FV_better_resolvably' if gain>indicator else ('FV_worse_resolvably' if gain < -indicator else 'comparison_inconclusive')
            assert expected==row['state_decision']
            H_gain=abs(H(b)-H(ref))-abs(H(a)-H(ref))
            expected_H='FV_better_resolvably' if H_gain>expected_H_indicator else ('FV_worse_resolvably' if H_gain < -expected_H_indicator else 'comparison_inconclusive')
            assert expected_H==row['H_decision']
    old=read(ROOT/'runs/reconnect-ordered-shear/results.json')
    for K in (4,8):
        for k,r in enumerate(old['cases'][str(K)]['observations']):
            assert rel(data[f'k{K}_states'][k*K//2]-data['reference_states'][k])==r['relative_concentration_error_initial_fluctuation_norm']
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
        {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),
         'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in prior:
        if item['manifest'] in known:continue
        path=ROOT/item['manifest'];assert digest(path)==item['manifest_sha256']
        entries=read(path);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==80
    status=read(out/'stage3-status.json');assert status['recorded_checks_total']==439+6==445
    assert status['new_candidate_graphs']==12 and status['new_candidate_transient_runs']==2
    assert status['new_PDE_reference_solves']==0 and status['stage3_physical_acceptance']=='incomplete' and not status['fluid_orc_enabled']
    save(out/'preservation.json',preserved)
    audit={'graph_audits':audits,'all12_geometry_caches_and_independent_whole_control_sequences_verified':True,
        'all_original_packets_and_reference_unchanged':True,'maximum_symmetric_vs_shared_action_mass_norm_difference':max_action_delta,
        'maximum_primary_control_relative_difference':max_control_delta,'maximum_primary_raw_mass_change_from_c0':max_raw_mass,
        'old_kernel_error_definition_and_results_matched':True,'new_checks':6,'recorded_checks_total':445,
        'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':80,
        'new_PDE_reference_solves':0,'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'numerical-audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='graph_audits'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);args=p.parse_args();main(args.directory)
