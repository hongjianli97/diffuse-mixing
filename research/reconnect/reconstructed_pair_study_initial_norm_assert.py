"""One fixed original-c0 full-support pair diagnostic, not a graph or evolution."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import shutil
import numpy as np
import scipy

from audit_route_decision import ROOT, digest, read, save
from reconstructed_path_rows import build_reconstructed_path_rows, apply_reconstructed_rows

SOURCE = ROOT/'runs/reconnect-point-contract-review'


def array_record(a):
    return {'shape':list(a.shape),'dtype':a.dtype.str,'C_order_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def encode_path(p):
    return {'start_id':p.start_id,'end_id':p.end_id,'displacement':list(map(str,p.displacement)),
        'mean_inverse_density_exact':str(p.mean_inverse_density),
        'intervals':[{'start':str(v.start),'stop':str(v.stop),'source_id':v.source_id,'density':str(v.density)} for v in p.intervals],
        'breakpoint_owners':[[str(t),i] for t,i in p.breakpoint_owners]}


def pi_decimal(precision):
    with localcontext() as ctx:
        ctx.prec=precision+12
        def atan_inverse(n):
            x=Decimal(1)/n;power=x;total=x;k=1
            while True:
                power*=-x*x;term=power/(2*k+1);updated=total+term
                if updated==total:return total
                total=updated;k+=1
        return +(16*atan_inverse(5)-4*atan_inverse(239))


def high_B(radius2,h,precision,pi):
    with localcontext() as ctx:
        ctx.prec=precision
        dec=lambda v:Decimal(v.numerator)/Decimal(v.denominator)
        width=dec(h);z=dec(radius2).sqrt()/width
        if z>=2:return F(0)
        return F(7*(Decimal(9)/(4*pi))/3/width**4*(2+5*z)*(1-z/2)**5)


def evaluate(data, controls, kernel_cache, progress=None):
    # This call sees no scalar state, PDE response, face/volume geometry or cache.
    stencil=build_reconstructed_path_rows(data['positions'],data['ids'],data['mass'],data['rho'],
        float(data['D']),float(data['h']),data['periods'],[63,64],progress=progress)
    lookup={int(v):i for i,v in enumerate(data['ids'])}
    indices=np.array([lookup[r.target_id] for r in stencil.rows],dtype=int)
    mass=data['mass'][indices]
    means={int(v):F(float(c)) for v,c in zip(data['ids'],data['c0'])}
    rates_exact=apply_reconstructed_rows(stencil,means)
    rates=np.array([float(rates_exact[r.target_id]) for r in stencil.rows])
    reference=data['reference_rate'][indices];old=data['kernel_rate'][indices]
    denominator=float(np.sqrt(data['mass']@(data['reference_rate']**2)))
    assert denominator==0.2479286450687567
    relative=lambda v:float(np.sqrt(mass@(np.asarray(v,dtype=float)**2)))/denominator
    cache={};pi={p:pi_decimal(p) for p in (60,100)}
    high_g={60:[],100:[]};float_g=[];edge_records=[]
    for e in stencil.edges:
        i,j=lookup[e.left_id],lookup[e.right_id]
        base=2*stencil.quadrature_area[i]*stencil.quadrature_area[j]*stencil.diffusivity/e.path.mean_inverse_density
        high={}
        for p in (60,100):
            key=(e.distance_squared,p)
            if key not in cache:cache[key]=high_B(e.distance_squared,stencil.h,p,pi[p])
            high[p]=cache[key];high_g[p].append(base*high[p])
        float_mean=math.fsum(float(v.stop-v.start)/float(v.density) for v in e.path.intervals)
        gf=2*float(stencil.quadrature_area[i])*float(stencil.quadrature_area[j])*float(stencil.diffusivity)/float_mean*e.kernel_B
        float_g.append(gf)
        edge_records.append({'ids':[e.left_id,e.right_id],
            'displacement_exact':list(map(str,e.displacement)),'distance_squared_exact':str(e.distance_squared),
            'distance_display':math.sqrt(float(e.distance_squared)),'retained_ids':list(e.retained_ids),
            'path':encode_path(e.path),'B_h_original_float':e.kernel_B,
            'B_h_high_precision_100_exact_decimal':str(high[100]),
            'B_h_float_minus_high_precision_display':float(F(e.kernel_B)-high[100]),
            'mean_inverse_density_float_reduction':float_mean,
            'mean_inverse_density_float_minus_exact_display':float(F(float_mean)-e.path.mean_inverse_density),
            'conductance_exact_given_binary_B_h':str(e.conductance),
            'conductance_display':float(e.conductance),'conductance_float_control':gf,
            'high_precision_60_conductance_exact_decimal':str(high_g[60][-1]),
            'high_precision_100_conductance_exact_decimal':str(high_g[100][-1])})
    high_rates={};floating=[];row_l1=[];rows=[]
    for p in (60,100):
        high_rates[p]=np.array([float(sum((high_g[p][k]*(means[j]-means[r.target_id])
            for j,k in r.neighbors),F(0))/r.mass) for r in stencil.rows])
    for r in stencil.rows:
        floating.append(math.fsum(float_g[k]*(float(means[j])-float(means[r.target_id]))
                                  for j,k in r.neighbors)/float(r.mass))
        row_l1.append(float(2*sum((stencil.edges[k].conductance for _,k in r.neighbors),F(0))/r.mass))
    floating=np.array(floating)
    old_dot=[];old_l1=[]
    for side,i in zip(['left','right'],indices):
        values=kernel_cache[side+'_L'];columns=kernel_cache[side+'_columns']
        old_dot.append(-float(values@data['c0'][columns])/data['mass'][i])
        old_l1.append(float(np.sum(np.abs(values)))/data['mass'][i])
    mean_difference=controls['exact_c0_vs_Gauss16_max_absolute']
    uncertainty={
        'saved_face_vs_volume16':relative(reference-data['volume16_rate'][indices]),
        'saved_volume8_vs_volume16':relative(data['volume8_rate'][indices]-data['volume16_rate'][indices]),
        'saved_kernel_vs_cached_row_dot':relative(np.array(old_dot)-old),
        'new_exact_binary_B_h_vs_float_flux_reduction':relative(rates-floating),
        'new_binary_B_h_vs_100_digit_kernel_row_action':relative(rates-high_rates[100]),
        'new_60_vs_100_digit_kernel_row_action':relative(high_rates[60]-high_rates[100]),
        'new_row_norm_times_saved_mean_difference':relative(np.array(row_l1)*mean_difference),
        'old_row_norm_times_saved_mean_difference':relative(np.array(old_l1)*mean_difference),
    }
    old_error=relative(old-reference);new_error=relative(rates-reference)
    assert old_error==0.009314548617023083
    benefit=old_error-new_error;indicator=sum(uncertainty.values())
    # No absolute physical threshold is introduced. These are empirical controls.
    decision='resolvable_pair_improvement' if benefit>indicator else ('resolvable_pair_deterioration' if benefit < -indicator else 'no_resolvable_benefit')
    for k,r in enumerate(stencil.rows):
        rows.append({'target_id':r.target_id,'original_mass':float(r.mass),'original_cbar':float(means[r.target_id]),
            'neighbors':[{'id':j,'edge_index':index} for j,index in r.neighbors],
            'support_neighbors':len(r.neighbors),'reference_rate':float(reference[k]),'saved_kernel_rate':float(old[k]),
            'reconstructed_rate_exact_given_binary_B_h':str(rates_exact[r.target_id]),'reconstructed_rate':float(rates[k]),
            'float_flux_control_rate':float(floating[k]),'kernel_100_digit_control_rate':float(high_rates[100][k]),
            'reconstructed_signed_error':float(rates[k]-reference[k]),'saved_kernel_signed_error':float(old[k]-reference[k]),
            'raw_row_l1_norm':row_l1[k]})
    shared=[k for k,e in enumerate(stencil.edges) if {e.left_id,e.right_id}=={63,64}]
    assert len(shared)==1 and all(sum(index==shared[0] for _,index in r.neighbors)==1 for r in stencil.rows)
    assert all(e.conductance>=0 for e in stencil.edges)
    open_rate=sum((r.mass*rates_exact[r.target_id] for r in stencil.rows),F(0))
    main={'rows':rows,'fixed_global_PDE_norm':denominator,'saved_kernel_pair_error':old_error,
        'reconstructed_pair_error':new_error,'saved_FV_pair_error_context_only':0.0028831076306128703,
        'benefit_old_minus_reconstructed':benefit,'percent_reduction_of_saved_kernel_pair_error':100*benefit/old_error,
        'empirical_difference_indicators':uncertainty,'sum_of_empirical_difference_indicators':indicator,
        'benefit_over_empirical_indicator':benefit/indicator,
        'indicator_is_not_certified_physical_bound_or_acceptance_threshold':True,
        'decision':decision,'pair_error_not_global_error':True,'time_discretization_present':False,
        'open_pair_mass_rate_exact':str(open_rate),'open_pair_mass_rate_display':float(open_rate),
        'open_pair_mass_rate_not_forced_to_zero':True,'shared_target_edge_index':shared[0],
        'shared_target_edge_reused_exactly':True,'exact_constant_state_rate':[0,0],
        'new_distinct_conductances':len(stencil.edges),'row_neighbor_counts':[len(r.neighbors) for r in stencil.rows],
        'all_global_nodes_validated':len(stencil.ids),'exact_integer_broad_phase':stencil.exact_integer_broad_phase,
        'unique_radius_squared_count':len(cache)//2}
    return edge_records,main


def main(output,tests,test_log):
    manifest=read(SOURCE/'sha256.json')
    for rel,expected in manifest.items():assert digest(ROOT/rel)==expected,rel
    protocol=read(SOURCE/'gate-spec.json')
    assert not protocol['already_executed'] and protocol['target_ids']==[63,64]
    snapshot=ROOT/protocol['source_snapshot'];assert digest(snapshot)==protocol['source_snapshot_sha256']
    checks=read(tests);assert checks=={'successful':True,'checks_run':7,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',{'study':'one predeclared reconstructed-density-path full-support pair',
        'protocol':protocol,'source_protocol_sha256':digest(SOURCE/'gate-spec.json'),
        'source_manifest_entries_verified':len(manifest),
        'primary_arithmetic':'exact rational action with original binary input, exact path integral and unchanged binary B_h',
        'control_arithmetic':'60/100 digit same analytic B_h with independently computed Machin pi; float flux reduction',
        'uncertainty_combination_fixed_before_evaluation':'sum saved reference differences, old row-dot difference, new float/high-precision controls and old/new row norms times saved mean difference',
        'localization':'exact start closed-ball broad phase followed by unchanged endpoint-ball localization and primitive',
        'no_new_acceptance_threshold':True})
    for name,path in [('source-gate-spec.json',SOURCE/'gate-spec.json'),('source-stage3-status.json',SOURCE/'stage3-status.json'),
                      ('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(path,out/name)
    keys=['ids','mass','c0','reference_rate','kernel_rate','volume8_rate','volume16_rate']
    with np.load(snapshot,allow_pickle=False) as z:
        data={k:z[k].copy() for k in keys};data['positions']=z['positions'][0].copy();data['rho']=z['rho'][0].copy()
    originals={k:array_record(v) for k,v in data.items()}
    data.update(D=np.array(float(F(protocol['D_exact']))),h=np.array(float(F(protocol['h_exact']))),periods=np.array(protocol['periods']))
    controls=read(ROOT/'runs/reconnect-contact-frozen/results.json')['reference_controls']
    cache={}
    cache_path=ROOT/'runs/reconnect-interface-pair/interface-pair.npz'
    with np.load(cache_path,allow_pickle=False) as z:
        for side in ['left','right']:
            cache[side+'_columns']=z[side+'_row_columns'].copy();cache[side+'_L']=z[side+'_saved_L_row'].copy()
    np.savez_compressed(out/'original-inputs.npz',**data,**cache)
    save(out/'saved-reference-controls.json',controls)
    edges,result=evaluate(data,controls,cache,lambda n,total:print(f'fixed original pair: {n}/{total} shared edge receipts',flush=True))
    assert originals=={k:array_record(data[k]) for k in originals}
    assert digest(snapshot)==protocol['source_snapshot_sha256']
    save(out/'edges.json',edges);save(out/'main-c0-comparison.json',result)
    save(out/'input-provenance.json',{'source_snapshot':protocol['source_snapshot'],'source_snapshot_sha256':digest(snapshot),
        'source_frame':0,'original_members':originals,'kernel_cache_path':str(cache_path.relative_to(ROOT)),
        'kernel_cache_sha256':digest(cache_path),'input_artifact_sha256':digest(out/'original-inputs.npz'),
        'state_unchanged':True,'coefficient_builder_did_not_receive_c0_PDE_or_geometry':True})
    status={'updated':'2026-11-01','timezone':'Etc/UTC','gate_executed':True,
        'new_software_checks':7,'previous_recorded_checks':403,'recorded_checks_total':410,'previous_checks_rerun':0,
        'new_fixed_snapshot_two_row_diagnostics':1,'new_edge_weights':len(edges),'new_candidate_rows':2,
        'new_candidate_graphs':0,'new_global_matrices':0,'new_candidate_transient_runs':0,'new_reference_time_solutions':0,
        'new_flow_solves':0,'LP_solves':0,'new_exact_algebra_certificate_groups':0,
        'decision':result['decision'],'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete',
        'fluid_orc_enabled':False,'ordinary_central_resistance_FV_baseline_retained':True,
        'mass_average_resistance_remains_experimental':True,
        'full_graph_or_transient_authorized_by_pair_result':False,'pdf_generated':False,'published':False}
    if result['decision']=='resolvable_pair_improvement':
        status['next_work']='read-only scope and cost review before any independent full-domain gate; no immediate full graph or evolution'
    else:
        status['next_work']='stop escalation of this fixed finite-node configuration; preserve signed error and review closure without retuning'
    save(out/'stage3-status.json',status)
    save(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'network_used':False,'environment_configuration_changed':False})
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


def replay(directory):
    """Rebuild only the same two rows from copied inputs, not a new experiment."""
    out=Path(directory);provenance=read(out/'input-provenance.json')
    assert digest(out/'original-inputs.npz')==provenance['input_artifact_sha256']
    with np.load(out/'original-inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    cache={k:data[k] for k in ['left_columns','left_L','right_columns','right_L']}
    edges,result=evaluate(data,read(out/'saved-reference-controls.json'),cache,
        lambda n,total:print(f'replay same pair: {n}/{total}',flush=True))
    assert edges==read(out/'edges.json')
    assert result==read(out/'main-c0-comparison.json')
    print('Full-support original pair receipts and comparison replay exactly. Stage 3 incomplete; ORC disabled.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output');parser.add_argument('--tests');parser.add_argument('--test-log');parser.add_argument('--replay')
    args=parser.parse_args()
    if args.replay:replay(args.replay)
    else:
        if not all([args.output,args.tests,args.test_log]):parser.error('--output, --tests and --test-log required')
        main(args.output,args.tests,args.test_log)
