"""Independently audit saved endpoint variance algebra using raw moments."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np

from audit_route_decision import ROOT,read,save,digest,preservation
from point_endpoint_review import SOURCE,BLOCKED,ZERO_FIELDS,record,replay


def moments(m,v,w,indices):
    total=Sv=Sw=V2=W2=VW=F(0)
    for i in indices:
        a,b,c=F(float(m[i])),F(float(v[i])),F(float(w[i]))
        total+=a;Sv+=a*b;Sw+=a*c;V2+=a*b*b;W2+=a*c*c;VW+=a*b*c
    return total,Sv,Sw,V2,W2,VW


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists();replay(out)
    config=read(out/'config.json');parent=read(SOURCE/'sha256.json');p=read(out/'input-provenance.json')
    assert len(parent)==config['source_manifest_entries_verified']==1683
    assert config['source_manifest_sha256']==digest(SOURCE/'sha256.json')
    for target,src in [('source-results.json',SOURCE/'results.json'),('source-solver-controls.json',SOURCE/'solver-controls.json'),
        ('source-FV-results.json',SOURCE/'source-FV-results.json'),('source-stage3-status.json',SOURCE/'stage3-status.json')]:
        assert (out/target).read_bytes()==src.read_bytes(),target
    assert p['source_original_inputs_sha256']==digest(SOURCE/'original-inputs.npz')
    assert p['source_point_endpoint_sha256']==digest(SOURCE/'point-T0.05.npz')
    with np.load(out/'saved-endpoints.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(SOURCE/'original-inputs.npz',allow_pickle=False) as z:
        for k in data:
            if k!='point_cbar_T':assert record(data[k])==record(z[k]),k
    with np.load(SOURCE/'point-T0.05.npz',allow_pickle=False) as z:assert record(data['point_cbar_T'])==record(z['point_semidiscrete_T'])
    b=read(out/'variance-budget.json');m=data['mass'];w=data['reference_cbar_T'];c0=data['c0'];n=len(m)
    M,S0,_,C02,_,_=moments(m,c0,c0,range(n));E0=(C02-S0*S0/M)/2
    assert str(M)==b['mass_exact'] and str(E0)==b['initial_variance_exact']
    maximum_norm_display_difference=0.
    for label,key in [('point','point_cbar_T'),('saved_FV','saved_FV_semidiscrete_T')]:
        v=data[key];M,Sv,Sw,V2,W2,VW=moments(m,v,w,range(n));mu=Sw/M
        cross=VW-W2-mu*(Sv-Sw);quad=(V2+W2-2*VW)/2;mean_term=-(Sv-Sw)**2/(2*M)
        delta=(V2-W2-(Sv*Sv-Sw*Sw)/M)/2
        assert delta==cross+quad+mean_term
        expected={'signed_cross_term':cross,'half_error_squared':quad,'raw_mean_correction':mean_term,
            'raw_mean_difference':(Sv-Sw)/M,'endpoint_variance_difference':delta,
            'H_bias_from_exact_saved_data':-delta/E0,'H_cross_contribution':-cross/E0,
            'H_square_contribution':-quad/E0,'H_mean_contribution':-mean_term/E0,
            'resolved_variance_loss_from_initial':E0-(V2-Sv*Sv/M)/2}
        method=b['methods'][label]
        for name,value in expected.items():
            assert F(method[name]['exact_of_saved_binary_data'])==value,name
            assert method[name]['display']==float(value),name
        assert method['exact_identity_residual']=='0'
        for region in ['near_contact','far_from_contact']:
            _,s,t,a,c,d=moments(m,v,w,np.flatnonzero(data[region]));item=method['original_regions'][region]
            assert F(item['signed_cross_term']['exact_of_saved_binary_data'])==d-c-mu*(s-t)
            assert F(item['half_error_squared']['exact_of_saved_binary_data'])==(a+c-2*d)/2
        difference=abs(method['norm_display_difference_not_used_to_replace_saved_value'])
        assert difference<=64*np.finfo(float).eps*max(1.,method['saved_relative_error_unchanged'])
        maximum_norm_display_difference=max(maximum_norm_display_difference,difference)
    decision=read(out/'route-decision.json');status=read(out/'stage3-status.json')
    assert not decision['new_numerical_gate_selected'] and not status['new_numerical_gate_selected']
    assert status['recorded_checks_total']==428
    for key in ZERO_FIELDS:assert status[key]==0,key
    for obj in [decision,status]:
        assert not obj['finite_node_candidate_admitted'] and not obj['contact_transient_physically_admitted']
        assert obj['stage3_physical_acceptance']=='incomplete' and not obj['fluid_orc_enabled']
    for name in BLOCKED:assert name not in sys.modules,name
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
    assert len(known)==74
    save(out/'preservation.json',preserved)
    audit={'source_1683_scientific_entries_and_copied_endpoint_members_verified':True,
        'independent_raw_moment_variance_identity_exact':True,'original_region_signed_sums_exact':True,
        'raw_mean_correction_kept_no_state_or_mean_adjustment':True,
        'maximum_saved_vs_exact_gram_norm_display_difference':maximum_norm_display_difference,
        'variance_identity_not_claimed_as_causal_kernel_projection_split':True,
        'no_new_graph_operator_propagation_reference_or_old_proof_executed':True,
        'no_numerical_gate_selected_no_retuning':True,**{k:0 for k in ZERO_FIELDS},'recorded_checks_total':428,
        'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':74,
        'stage3_physical_acceptance':'incomplete','finite_node_candidate_admitted':False,'fluid_orc_enabled':False}
    save(out/'review-audit.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);args=p.parse_args();main(args.directory)
