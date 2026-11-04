"""Independent bisector intersections, exact node sums and dyadic global sums."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import inspect
import json
import numpy as np

from audit_route_decision import ROOT,read,save,digest,preservation
from periodic_voronoi_fv import PeriodicFaces,build_periodic_faces,coefficient_packet,dot,cross,sub,rate_digest
from shear_voronoi_fv_study import SOURCE,PROTOCOL,CONSTRUCTION,record,evaluate,verify_probe_cells


def fixed_binary_enclosure(values,bits=256):
    lo=hi=0;scale=1<<bits
    for value in values:
        numerator=value.numerator*scale;floor,remainder=divmod(numerator,value.denominator)
        lo+=floor;hi+=floor+bool(remainder)
    return F(lo,scale),F(hi,scale)


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    cfg=read(out/'config.json');lock=read(out/'implementation-lock.json');spec=read(out/'source-gate-spec.json')
    parent=read(SOURCE/'sha256.json');assert len(parent)==cfg['parent_manifest_entries_verified']==1803
    assert spec==read(SOURCE/'next-gate-spec.json') and not spec['already_executed']
    assert cfg['parent_protocol_sha256']==digest(SOURCE/'next-gate-spec.json')
    assert lock['source_protocol_sha256']==digest(PROTOCOL)==digest(out/'implementation-protocol.json')
    assert lock['construction_sha256']==digest(CONSTRUCTION)==digest(out/'construction-at-lock.md')
    assert lock['builder_source_sha256']==digest(ROOT/'research/reconnect/periodic_voronoi_fv.py')
    assert lock['test_source_sha256']==digest(ROOT/'research/reconnect/test_periodic_voronoi_fv.py')
    assert lock['real_geometry_or_scalar_not_evaluated_before_lock']
    assert read(out/'tests.json')=={'successful':True,'checks_run':6,'failures':0,'errors':0,'skipped':0}
    run=read(out/'execution.json');assert run['first_attempts_passed'] and run['study_exit_code']==run['tests_exit_code']==0
    assert run['elapsed_main_seconds']<600 and run['peak_RSS_MiB']<512
    assert list(inspect.signature(build_periodic_faces).parameters)==['ids','positions','rho','D','periods','thickness','progress']
    seal=read(out/'coefficient-seal.json');cache=read(out/'coefficient-cache.json')
    assert seal['coefficient_cache_sha256']==digest(out/'coefficient-cache.json')
    assert seal['geometry_and_all_coefficients_sealed_before_scalar_reference_access']
    assert not seal['actual_mass_scalar_reference_true_geometry_received_by_builder']
    with np.load(out/'original-state.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(out/'builder-inputs.npz',allow_pickle=False) as z:
        for k in z.files:assert record(z[k])==record(data[k])==seal['builder_array_records'][k]
    p=read(out/'input-provenance.json');assert p['original_state_sha256']==digest(out/'original-state.npz')
    assert digest(ROOT/spec['source_npz'])==p['source_npz_sha256']==spec['source_npz_sha256']
    with np.load(ROOT/spec['source_npz'],allow_pickle=False) as z:
        for k,v in data.items():
            original=z[k][1] if k in ['positions','rho'] else z[k]
            assert record(v)==record(original)==p['array_records'][k],k
    ids=tuple(map(int,data['ids']));points=tuple(tuple(F(float(x)) for x in row) for row in data['positions'])
    rho=tuple(F(float(x)) for x in data['rho']);mass=tuple(F(float(x)) for x in data['mass']);u=tuple(F(float(x)) for x in data['c0'])
    periods=tuple(map(F,cache['periods_exact']));D=F(cache['D_exact']);b=F(cache['thickness_exact'])
    verts=tuple(tuple(map(F,row)) for row in cache['vertices_exact'])
    cells=[];areas=[];centroids=[];seconds=[];directed={}
    def vector(i,label):
        j,sx,sy=label
        return (points[j][0]+sx*periods[0]-points[i][0],points[j][1]+sy*periods[1]-points[i][1])
    for i,(vi,labels) in enumerate(zip(cache['cell_vertex_indices'],cache['cell_incoming_labels'])):
        assert len(vi)==len(labels)>=3
        poly=[]
        for k,label in enumerate(labels):
            a=vector(i,label);v=vector(i,labels[(k+1)%len(labels)])
            ca=dot(a,a)/2;cv=dot(v,v)/2;det=cross(a,v)
            assert det!=0
            q=((ca*v[1]-a[1]*cv)/det,(a[0]*cv-ca*v[0])/det)
            world=((points[i][0]+q[0])%periods[0],(points[i][1]+q[1])%periods[1])
            assert world==verts[vi[k]]
            poly.append((q,tuple(label)))
        twice=mx=my=trace=F(0)
        for k,(q,label) in enumerate(poly):
            prev=poly[k-1][0];nxt=poly[(k+1)%len(poly)][0]
            assert cross(sub(q,prev),sub(nxt,q))>0
            z=cross(prev,q);twice+=z;mx+=(prev[0]+q[0])*z;my+=(prev[1]+q[1])*z
            trace+=(prev[0]**2+prev[0]*q[0]+q[0]**2+prev[1]**2+prev[1]*q[1]+q[1]**2)*z
            v=vector(i,label);d2=dot(v,v);edge=sub(q,prev)
            assert d2>0 and dot(v,edge)==0 and 2*dot(v,prev)==2*dot(v,q)==d2
            ratio=cross(v,edge)/d2;assert ratio>0
            assert label not in directed.get(i,{})
            directed.setdefault(i,{})[label]=(ratio,(vi[k-1],vi[k]))
        area=twice/2;assert area>0 and area==F(cache['cell_areas_exact'][i])
        cells.append(tuple(poly));areas.append(area);centroids.append((mx/(6*area),my/(6*area)));seconds.append(trace/(12*area))
    area_lo,area_hi=fixed_binary_enclosure(areas)
    assert area_lo<=periods[0]*periods[1]<=area_hi
    stored_area=cache['geometry_diagnostics']['raw_total_area_enclosure']
    assert F(stored_area['lower_exact'])<=area_hi and area_lo<=F(stored_area['upper_exact'])
    edges=tuple(map(tuple,cache['edges']));shifts=tuple(map(tuple,cache['image_shifts']));ratios=tuple(map(F,cache['face_over_distance_exact']))
    conductances=tuple(map(F,cache['conductances_exact']));displacements=[];endpoints=[];rows=[[] for _ in ids];seen=set()
    for k,((i,j),(sx,sy),ratio,g) in enumerate(zip(edges,shifts,ratios,conductances)):
        assert ids[i]<ids[j] and (i,j,sx,sy) not in seen;seen.add((i,j,sx,sy))
        first=directed[i][(j,sx,sy)];other=directed[j][(i,-sx,-sy)]
        assert first[0]==other[0]==ratio and first[1]==other[1][::-1]
        assert list(first[1])==cache['face_vertex_indices'][k]
        expected=b*ratio/(1/(2*rho[i]*D)+1/(2*rho[j]*D));assert expected==g>0
        displacements.append(vector(i,(j,sx,sy)));endpoints.append(tuple(verts[v] for v in first[1]))
        rows[i].append((j,g));rows[j].append((i,g))
    assert 2*len(edges)==sum(map(len,cells)) and all(len(v)==len(cells[i]) for i,v in enumerate(rows))
    graph=PeriodicFaces(ids,points,rho,periods,D,b,tuple(cells),tuple(areas),tuple(centroids),tuple(seconds),
        tuple(map(F,cache['initial_search_radius_L1_exact'])),edges,shifts,tuple(endpoints),ratios,
        tuple(displacements),conductances,cache['geometry_diagnostics'])
    assert coefficient_packet(graph)==cache
    assert verify_probe_cells(graph,read(PROTOCOL)['independent_actual_geometry_probe_ids'])==read(out/'independent-geometry-probes.json')
    q=tuple(sum((g*(u[j]-u[i]) for j,g in row),F(0))/mass[i] for i,row in enumerate(rows))
    assert rate_digest(q)==read(out/'action-controls.json')['exact_rate_digest']
    energy_lo,energy_hi=fixed_binary_enclosure(g*(u[j]-u[i])**2 for (i,j),g in zip(edges,conductances))
    result=read(out/'results.json');energy=result['graph_dissipation_exact_sum_enclosure']
    assert F(energy['lower_exact'])<=energy_hi and energy_lo<=F(energy['upper_exact'])
    result2,controls,capacity,arrays=evaluate(graph,data)
    assert result2==result and controls==read(out/'action-controls.json') and capacity==read(out/'capacity-diagnostics.json')
    with np.load(out/'actions.npz',allow_pickle=False) as z:
        assert np.array_equal(z['FV_rate'],np.array(list(map(float,q))))
        for k,v in arrays.items():assert np.array_equal(z[k],v),k
    assert result['comparison_decision']=='FV_better_than_saved_kernel_resolvably'
    assert result['error_reduction']>result['combined_comparison_indicator_relative']
    old=read(out/'source-kernel-results.json')
    assert abs(old['relative_mass_weighted_instantaneous_rate_error']-result['saved_kernel_relative_rate_error_recomputed'])<=64*np.finfo(float).eps
    s=read(out/'stage3-status.json');assert s['recorded_checks_total']==433+6==439
    assert s['new_actual_frozen_state_cases']==s['new_candidate_graphs']==1 and s['new_candidate_conductances']==len(edges)
    for k in ['new_candidate_transient_runs','new_PDE_reference_solves','new_kernel_builds','new_density_path_builds','new_full_global_matrices','new_exact_algebra_certificate_groups']:
        assert s[k]==0,k
    assert s['shock_extension_deferred']
    for obj in [s,result]:
        assert not obj['general_smooth_flow_admitted'] and not obj['finite_node_candidate_admitted']
        assert obj['stage3_physical_acceptance']=='incomplete' and not obj['fluid_orc_enabled']
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
        {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),
            'entries_verified':len(parent)}]
    for item in prior:
        if item['manifest'] in known:continue
        path=ROOT/item['manifest'];assert digest(path)==item['manifest_sha256']
        entries=read(path);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==79
    save(out/'preservation.json',preserved)
    audit={'implementation_locked_before_geometry_and_state_seal_verified':True,
        'all_original_arrays_and_independent_full_reference_match_by_bytes':True,
        'all_vertices_rebuilt_by_independent_pairs_of_bisectors':True,
        'all_periodic_shared_faces_and_resistances_verified_exactly':True,
        'all_rates_rebuilt_by_independent_node_sums_and_hash':True,
        'fixed_global_halfplane_probes_and_metrics_verified':True,
        'independent_area_sum_256bit_enclosure':{'lower_exact':str(area_lo),'upper_exact':str(area_hi)},
        'independent_energy_sum_256bit_enclosure':{'lower_exact':str(energy_lo),'upper_exact':str(energy_hi)},
        'new_software_checks':6,'recorded_checks_total':439,'original_files_verified':782,
        'historical_attachments_verified':15,'prior_formal_manifests_verified':79,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'frozen-audit.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);args=p.parse_args();main(args.directory)
