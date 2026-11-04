"""Seal a read-only rotation-domain review; fresh standard-library receipt verification only."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure
from t2_rotation_domain_review import ZERO_FIELDS


def main():
    source=ROOT/'runs/reconnect-t2-d0';out=ROOT/'runs/reconnect-t2-rotation-domain'
    archive=ROOT/'exports/ltn-t2-rotation-domain.zip';delivery=ROOT/'exports/ltn-t2-rotation-domain-delivery.json'
    downloads=ROOT/'exports/ltn-t2-rotation-domain-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)),'closed evidence is never overwritten'
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'review-audit.json')
    status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert ex['review_exit_code']==ar['audit_exit_code']==0 and ex['review_first_attempt_passed'] and ar['audit_first_attempt_passed']
    assert audit['receipt_consistent'] and audit['prior_formal_manifests_verified']==91 and audit['original_files_verified']==782
    assert status['recorded_checks_total']==474 and all(status[k]==0 for k in ZERO_FIELDS)
    assert not status['adapter_implemented'] and not status['rotation_test_executed'] and not status['fluid_orc_enabled'] and not gate['already_executed']
    prior=read(source/'sha256.json');assert len(prior)==3825
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+n for n in ['t2_rotation_domain_review.py','t2_rotation_domain_protocol.json',
      'T2_ROTATION_DOMAIN_CONTRACT.md','audit_t2_rotation_domain.py','package_t2_rotation_domain.py']]
    source_seal=str((source/'sha256.json').relative_to(ROOT));entries=dict(prior);entries[source_seal]=digest(source/'sha256.json')
    for rel in files:assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for p in out.rglob('*'):
        if p.is_file():
            rel=str(p.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(p)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)|{str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()}|dependency_closure([p for p in files if p.endswith('.py')])
    selected.update([source_seal,'docs/RESEARCH_STATUS.md'])
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'] if not item['path'].endswith('.npz'))
    manifest={p:digest(ROOT/p) for p in sorted(selected)}
    note=(
      'LTN T2 ROTATION DOMAIN CONTRACT: READ-ONLY REVIEW, NO NEW NUMERICAL EXPERIMENT.\n'
      'Decision:boundary_representation_gap_with_minimal_isometry_adapter_proposed.\n'
      'Current geometry API accepts two axis-aligned periodic lengths; a general continuous rotation requires the periodic lattice to rotate too.\n'
      'Written derivations cover material means,mass,scalarPDE,lattice automorphisms,rigidvelocity,ordinaryg/Q/P/H andfixedmetric covariance.\n'
      'A fixedsquare quarter-turn is a discrete static symmetry,not a continuous rotation experiment.\n'
      'A separate optional exact orthogonal boundary adapter is proposed; originalgeometry/ordinaryresistance/timebackends unchanged.\n'
      'Boundaryorientation is given globaldomaingeometry,not fittedlocalF/C or trueinternalmaterialfaces.\n'
      'First proposed mode uses exact rational boundary/worldpoints,losslesslocalfloat conversion andindependentworldgeometry audit.\n'
      'Next gate only registered: implement adapter and3 necessary checkcategories,then oneN32,D=.02,T=.05,K16 continuous rational rotation onthecorotating torus.\n'
      'No adapter implementation,newrotationpoints,graphs,states ornewchecks executed inthisreview.\n'
      'Newpoints/graphs/weights/actions/paths/PDE/LP/softwarechecks/machinealgebracertificategroups all0; recordedchecks remain474.\n'
      'Written proofs are not labelled as machineverified or numericaltest results.\n'
      '782originals,15attachments,91priorformalmanifests andoldD0delivery preserved; fourknownhistoricaldiagnostic differences retained.\n'
      'python -S research/reconnect/audit_t2_rotation_domain.py --directory runs/reconnect-t2-rotation-domain --receipt-only\n'
      'Fresh extracted review is checked withstandardlibrary only; no oldnumericalpath replayed.\n'
      'Included files verified by review-manifest.json; cumulative seal references omittedhistory. Full historical preservation audit needs originalworkspace.\n'
      'OriginalREADME/METHOD/CODEX_TASK/PHASE andoldcode/data/reports unchanged; onlymutable RESEARCH_STATUS updated.\n'
      'GeneralT2/smoothflow/finite-node admissionfalse,Stage3incomplete,fluidORCoff; no kappa orfluidORC run.\n'
      '2026-11-02 Asia/Shanghai; no PDF,version,commit/push,configurationchange orpublication.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-rotation-domain-receipt-',dir='/tmp') as directory:
            z.extractall(directory)
            result=subprocess.run([sys.executable,'-S',str(Path(directory)/'research/reconnect/audit_t2_rotation_domain.py'),
              '--directory','runs/reconnect-t2-rotation-domain','--receipt-only'],cwd=directory,
              env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
            assert result.returncode==0,result.stdout+result.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'archive_CRC_and_all_included_SHA256_verified':True,'fresh_standard_library_receipt_exit_code':result.returncode,
      'fresh_standard_library_receipt_log':result.stdout,'fresh_receipt_files_unchanged':True,
      'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'review_decision':status['decision'],
      'recorded_checks_total':474,'previous_checks_rerun':0,**dict.fromkeys(ZERO_FIELDS,0),
      'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':91,
      'adapter_implemented':False,'backend_or_api_changed':False,'rotation_test_executed':False,'next_gate_executed':False,
      'proofs_are_written_not_machine_verified':True,'old_numerical_paths_replayed':False,
      'general_T2_admitted':False,'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# T2刚体旋转：周期域契约审查\n\n[推导、当前接口证据、验收账本、下一项方案及标准库来源审计](ltn-t2-rotation-domain.zip)。\n\n'
          f'SHA-256：`{info["archive"]["sha256"]}`。\n\n[交付核验](ltn-t2-rotation-domain-delivery.json)。CRC/SHA和全新解压标准库收据核验通过；本轮不重放数值路径。\n\n'
          '新点/图/状态/数值实验/软件检查均0，记录仍474。适配层及旋转试验尚未实现或运行；Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
