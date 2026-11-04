"""One predeclared zero-normal-flux probe on two saved interface rows."""
from pathlib import Path
import argparse,json,platform,resource
import numpy as np
from audit_contact_constitutive import digest
from package_compression_midpoint import dump
from point_row_response import row_response
from interface_pair_probe import quadratic_cell_reference,region_summary


def main(output):
    root=Path(__file__).resolve().parents[2];source=root/'runs/reconnect-point-row'
    entries=json.loads((source/'sha256.json').read_text())
    for rel,h in entries.items():assert digest(root/rel)==h,rel
    protocol=json.loads((source/'stage3-status.json').read_text())['next_saved_interface_pair_predeclared']
    assert protocol['target_indices']==[63,64] and (protocol['N_per_direction'],protocol['h_m'],protocol['D_m2_per_s'],protocol['contact_origin_a_m'])==(128,.04,.02,.5)
    assert protocol['no_new_edge_weights_or_graph_or_matrix_scan'] and protocol['no_pure_phase_ideal_excess_correction']
    assert not protocol['fluid_orc_enabled']
    path=root/protocol['source_graph'];assert digest(path)==protocol['source_graph_sha256']
    keys=['mass','ids','c0','material_label_centers','rho0_point','quadrature_area','true_cell_area_reference_only','reference_rate','kernel_rate']
    with np.load(path) as z:
        ptr=z['L_indptr'];L=z['L_data'];idx=z['L_indices'];shape=z['L_shape'];original={key:z[key] for key in keys}
    ell=1/128;h=.04;D=.02;origin=.5;records={};arrays={};row_configs={};masses=[];rates=[];references=[]
    assert np.array_equal(shape,[128**2,128**2])
    reference_cache={}
    for side,target in zip(['left','right'],protocol['target_indices']):
        start,stop=map(int,ptr[target:target+2]);cols=idx[start:stop];values=L[start:stop]
        saved={key:value[cols] for key,value in original.items()};at=int(np.flatnonzero(cols==target)[0])
        mass=float(saved['mass'][at]);center=saved['material_label_centers'][at]
        assert int(saved['ids'][at])==target and float(saved['rho0_point'][at])==(1. if side=='left' else 4.)
        np.testing.assert_array_equal(saved['mass']/saved['rho0_point'],saved['quadrature_area'])
        np.testing.assert_array_equal(saved['quadrature_area'],saved['true_cell_area_reference_only'])
        np.testing.assert_array_equal(saved['quadrature_area'],ell**2)
        assert set(saved['rho0_point'])=={1.,4.}
        dx=saved['material_label_centers']-center;dx-=np.rint(dx)
        assert np.all(np.linalg.norm(dx,axis=1)<=2*h)
        row_arrays,row_metrics=row_response(values,cols,saved['material_label_centers'][:,0],target,mass,ell,origin=origin)
        for col in cols:
            k=int(col)
            if k not in reference_cache:
                reference_cache[k]=quadratic_cell_reference(float(original['material_label_centers'][k,0]),ell,float(original['rho0_point'][k]),float(original['mass'][k]),D,origin)
        exact_means=np.array([reference_cache[int(k)]['values']['mass_average'] for k in cols])
        target_reference=reference_cache[target]
        assert target_reference['original_mass_matches_constant_phase_square'] and target_reference['values']['PDE_mass_average_rate_per_s']==D
        row_arrays.update({'original_'+key:value for key,value in saved.items()})
        row_arrays['periodic_row_displacement']=dx;row_arrays['independent_integrated_quadratic_mass_means']=exact_means
        for key,value in row_arrays.items():arrays[side+'_'+key]=value
        raw=row_metrics['raw_CSR_quadratic_response_per_s']
        integrated_action=-float(values@exact_means)/mass
        records[side]={'target_index':target,'target_id':int(saved['ids'][at]),'target_rho':float(saved['rho0_point'][at]),
            'row':row_metrics,'independent_reference':target_reference,'PDE_response_per_s':D,
            'response_error_per_s':raw-D,'relative_error':(raw-D)/D,'relative_error_percent':100*((raw-D)/D),
            'independent_integrated_state_action_per_s':integrated_action,
            'integrated_state_action_minus_preselected_CSR_per_s':integrated_action-raw,
            'integrated_mean_minus_formula_max_abs':float(np.max(abs(exact_means-row_arrays['probe_quadratic_mass_means']))),
            'opposite_phase_entries':int(np.sum(saved['rho0_point']!=saved['rho0_point'][at])),
            'maximum_row_neighbor_distance_m':float(np.max(np.linalg.norm(dx,axis=1)))}
        row_configs[side]={'target_index':target,'source_start_stop':[start,stop],'source_row_columns_dtype':str(cols.dtype),
            'source_row_values_dtype':str(values.dtype),'original_array_dtypes':{key:str(v.dtype) for key,v in saved.items()},
            'original_c0_at_target':float(saved['c0'][at])}
        masses.append(mass);rates.append(raw);references.append(D)
    left_off=float(arrays['left_saved_L_row'][arrays['left_row_columns']==64][0])
    right_off=float(arrays['right_saved_L_row'][arrays['right_row_columns']==63][0])
    assert left_off==right_off
    target_means=[records[side]['row']['target_probe_mass_average'] for side in ['left','right']]
    region=region_summary(masses,rates,references)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    dump(out/'config.json',{'scope':'predeclared read-only interface-adjacent two-row zero-flux quadratic probe',
        'source':str(source),'source_manifest_entries_verified':len(entries),'protocol':protocol,'protocol_source_sha256':digest(source/'stage3-status.json'),
        'source_graph_sha256':digest(path),'CSR_shape':shape.tolist(),'rows':row_configs,
        'reference':'independent Fraction exact square-state and normal-face-flux integrals; original dyadic geometry/mass and decimal D',
        'main_action':'unchanged row_response raw CSR on formula-defined exact quadratic mass means; independent integrated means kept as control',
        'original_c0_mass_ids_saved_cosine_PDE_and_candidate_unchanged':True,'fluid_orc_enabled':False})
    result={'rows':records,'two_cell_region':region,'saved_shared_offdiagonal_L_left_to_right_kg_per_s':left_off,
        'saved_shared_offdiagonal_L_right_to_left_kg_per_s':right_off,'shared_offdiagonal_bytes_equal':True,
        'two_target_quadratic_mass_mean_difference':target_means[1]-target_means[0],
        'exact_interface_left_and_right_normal_flux':0.,'probe_origin_m':origin,
        'same_probe_correct_PDE_response_both_sides_per_s':D,'same_saved_graph_and_original_data_unchanged':True,
        'pure_phase_ideal_eta_used_as_correction':False,'raw_CSR_main_value_selected_before_inspection':True,
        'saved_row_evaluations':2,'new_candidate_graphs':0,'new_edge_weights':0,'new_operator_quadrature_evaluations':0,
        'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'finite_node_candidate_admitted':False,'contact_transient_admitted':False,
        'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete','peak_process_RSS_MiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'limits':['one fixed contact and one local quadratic probe only','two-cell region is not a closed exchange system; region error is not a global conservation residual',
            'arithmetic mean is diagnostic only, not the material mass observation','not a quantitative explanation of the original cosine-state residuals']}
    np.savez_compressed(out/'interface-pair.npz',**arrays);dump(out/'results.json',result)
    dump(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'threads':4})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
