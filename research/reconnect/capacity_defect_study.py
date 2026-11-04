"""Freeze theory and audit one existing snapshot using only permitted input fields."""
from pathlib import Path
import argparse
import hashlib
import shutil
import sys

from audit_route_decision import ROOT,read,save,digest
from capacity_defect_theory import SOURCE,THEORY,INPUT,ZERO_FIELDS,contract,certificates,array_records,analyse_packet,replay


def main(output):
    parent=read(SOURCE/'sha256.json')
    for rel,h in parent.items():assert digest(ROOT/rel)==h,rel
    assert read(SOURCE/'stage3-status.json')['recorded_checks_total']==428
    assert read(SOURCE/'next-theory-scope.json')['no_mass_density_position_or_response_normalization']
    assert INPUT in parent
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    spec={'source_npz':INPUT,'source_npz_sha256':parent[INPUT],'N':128,'snapshot_index':1,
        'source_time_display_expected':0.025,'time_label_exact':'1/40','periods':[1,1],'unit_thickness':1,
        'read_members':['ids','mass','positions','rho','time'],
        'forbidden_members_for_screen':['c0','reference_rate','kernel_rate','conductances','L_data',
            'true_cell_area_from_unwrapped_faces','unwrapped_left_faces','unwrapped_right_faces','J'],
        'no_new_points_density_or_mass':True,'one_existing_snapshot_only':True,
        'structural_controls':['original canonical IDs','finite positive mass and density','exact product positions',
            'positive single-cycle periodic spacing without sorting or coordinate repair'],
        'outputs':['exact auxiliary capacities and ratios','raw global mass mismatch','eta and R',
            'eta*(2+R) same-L capacity bound','centroid L1 upper and second-moment geometric quantities'],
        'stop_on_unsupported_input_without_new_tolerance_or_snapshot':True,
        'no_conductances_graph_actions_scalar_or_PDE_solves':True,'not_physical_acceptance':True}
    save(out/'input-screen-spec.json',spec);save(out/'capacity-contract.json',contract())
    shutil.copyfile(THEORY,out/'theory-at-definition.md')
    save(out/'definition-receipt.json',{'theory_sha256_before_evaluation':digest(THEORY),
        'contract_sha256_before_evaluation':digest(out/'capacity-contract.json'),
        'input_screen_spec_sha256_before_evaluation':digest(out/'input-screen-spec.json'),
        'all_written_before_certificate_or_capacity_evaluation':True})
    save(out/'config.json',{'scope':'static capacity theorem plus one predeclared existing-input diagnostic',
        'parent_manifest_entries_verified':len(parent),'parent_manifest_sha256':digest(SOURCE/'sha256.json'),
        'source_next_scope_sha256':digest(SOURCE/'next-theory-scope.json'),'no_new_flow_case_or_backend':True})
    save(out/'algebra-certificates.json',certificates())
    # NumPy is used solely to read the five specified existing NPZ members.
    import numpy as np
    with np.load(ROOT/INPUT,allow_pickle=False) as z:
        assert float(z['time'][1])==spec['source_time_display_expected']
        arrays={'ids':z['ids'].copy(),'mass':z['mass'].copy(),
            'positions':z['positions'][1].copy(),'rho':z['rho'][1].copy()}
        source_time=float(z['time'][1])
    packet={k:a.tolist() for k,a in arrays.items()};packet['time']=source_time
    raw={k:{'shape':list(a.shape),'dtype':a.dtype.str,'C_order_sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}
        for k,a in arrays.items()}
    assert raw==array_records(packet)
    save(out/'saved-input.json',packet)
    save(out/'input-provenance.json',{'source_npz':INPUT,'source_npz_sha256':digest(ROOT/INPUT),
        'snapshot_index':1,'saved_packet_sha256':digest(out/'saved-input.json'),'copied_array_records':raw,
        'only_predeclared_input_members_read_no_scalar_or_reference_loaded':True})
    try:
        result,exact=analyse_packet(packet)
        assert raw==array_records(packet)
        save(out/'input-screen-results.json',result);save(out/'input-screen-exact.json',exact)
    except BaseException as error:
        save(out/'input-screen-failure.json',{'type':type(error).__name__,'reason':str(error),
            'no_snapshot_tolerance_or_input_repair':True,'fluid_orc_enabled':False})
        raise
    save(out/'stage3-status.json',{'updated':'2026-11-01','timezone':'Etc/UTC',**{k:0 for k in ZERO_FIELDS},
        'new_exact_algebra_certificate_groups':8,'new_saved_input_diagnostics':1,'recorded_checks_total':428,
        'static_capacity_uniform_time_bound_proved_in_report':True,'same_L_capacity_bound_not_full_PDE_error':True,
        'one_saved_midpoint_input_supported_for_capacity_budget':True,'candidate_PDE_accuracy_evaluated':False,
        'finite_node_candidate_admitted':False,'contact_transient_physically_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
        'next_work':'one predeclared existing compression-midpoint fixed-state auxiliary FV action, not another theoretical branch',
        'pdf_generated':False,'software_release_created':False,'published':False})
    replay(out)
    print({k:result[k] for k in ['minimum_capacity_ratio','maximum_capacity_ratio','eta_maximum_local_capacity_defect',
        'ratio_spread_R','uniform_same_L_capacity_comparison_bound_over_initial_fluctuation',
        'auxiliary_minus_original_total_mass','epsilon_c_L1_upper','epsilon_2']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args();main(args.output)
