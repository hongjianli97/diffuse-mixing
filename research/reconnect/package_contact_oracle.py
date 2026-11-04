"""Separate downloadable PDF and source/results ZIP; CRC/SHA and fresh replay verified."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest


def main():
    out=ROOT/'runs/reconnect-contact-oracle';source=ROOT/'runs/reconnect-affine-shear-spatial';exports=ROOT/'exports'
    pdf=exports/'LTN_contact_oracle_report_20261104.pdf';archive=exports/'LTN_contact_oracle_code_results_20261104.zip';delivery=exports/'LTN_contact_oracle_delivery_20261104.json';downloads=exports/'LTN_contact_oracle_downloads_20261104.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (pdf,archive,delivery,downloads,seal)),'new delivery only; never overwrite sealed evidence'
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'numerical-audit.json');result=read(out/'results.json');gate=read(out/'next-gate-spec.json')
    assert ex['study_exit_code']==ex['tests_exit_code']==ar['audit_exit_code']==0 and ex['main_first_attempt_passed']
    assert not ex['tests_first_attempt_passed'] and ex['tests_second_attempt_passed'] and not ar['audit_first_attempt_passed'] and ar['audit_second_attempt_passed']
    assert audit['registered_same_backend_contact_oracle_control_passed'] and audit['prior_formal_manifests_verified']==99 and audit['recorded_checks_total']==493
    assert not gate['already_executed'] and not result['fluid_orc_enabled'] and not result['moving_shock_T4_closed']
    for rel,h in read(out/'pre-reporting-type-recovery-sha256.json').items():assert digest(out/rel)==h,rel
    prior=read(source/'sha256.json');assert len(prior)==6374
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    assert digest(exports/'ltn-affine-shear-spatial.zip')=='7466b3e90291dbb1132858d0f9521dda6f06f7d87b9c234d7d66e13ab496d183'
    pdf_source=out/pdf.name;log=(out/'LTN_contact_oracle_report_20261104.log').read_text()
    assert 'Overfull' not in log and 'Missing character' not in log
    info=subprocess.run(['pdfinfo',str(pdf_source)],capture_output=True,text=True,check=True).stdout
    assert 'Pages:           4' in info
    shutil.copyfile(pdf_source,pdf);shutil.copyfile('/tmp/ltn-contact-oracle-xelatex-final2.log',out/'PDF-build.log')
    save(out/'PDF-generation.json',{'PDF_explicitly_requested_by_user':True,'report_pdf':str(pdf.relative_to(ROOT)),'sha256':digest(pdf),
      'pages':4,'xelatex_exit_code':0,'overfull_boxes':0,'missing_characters':0,'all_four_pages_rendered_and_visually_reviewed':True,
      'contains_original_failure_records_and_Stage3_scope':True,'software_or_environment_published':False})
    files=['research/reconnect/'+name for name in ['contact_oracle_core.py','contact_oracle_study.py','test_contact_oracle.py','contact_oracle_protocol.json',
      'audit_contact_oracle.py','audit_contact_oracle_completed.py','contact_oracle_reporting.py','plot_contact_oracle.py',
      'document_contact_oracle.py','package_contact_oracle.py','CONTACT_ORACLE_COMPARISON.md']]
    entries=dict(prior);source_seal=str((source/'sha256.json').relative_to(ROOT));entries[source_seal]=digest(source/'sha256.json')
    for rel in files:assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for p in out.rglob('*'):
        if p.is_file():
            rel=str(p.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(p)
    entries[str(pdf.relative_to(ROOT))]=digest(pdf);save(seal,dict(sorted(entries.items())))
    originals=read('/workspace/original-handoff-sha256.json');assert len(originals)==782
    selected=set(originals)|set(files)|{source_seal,'docs/RESEARCH_STATUS.md','requirements-reference.txt',str(pdf.relative_to(ROOT))}
    for folder in ('research','docs','prototypes','reference_baseline'):
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    selected.update(str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file())
    selected.update(item['path'] for item in read(out/'evidence-index.json')['sources'] if Path(item['path']).suffix!='.zip')
    manifest={rel:digest(ROOT/rel) for rel in sorted(selected)}
    for rel,h in originals.items():assert manifest[rel]==h,rel
    note='''LTN / diffuse-mixing: 2026-11-04, one registered same-backend N32 static periodic contact oracle control.
Authoritative results: runs/reconnect-contact-oracle/results.json.
Decision T4_same_backend_static_contact_oracle_control_passed. Stage3 incomplete; fluid ORC OFF; kappa notcomputed.
Unitperiodic square, equalhalvesrho1/4, originalDfloat.02, T1/20, one staticpointVoronoigraph,K16actual/fullindependentpath.
One graph sealed before mass/c0/reference; original geometry/ordinaryhalfresistance/action sources unchanged.
True finite-cell masses/c0/reference newly evaluated,not recoveredv080 norinterpolatedN128means.
Stored128mode90digit continuous coefficients reused, no spectralfit or PDEsolve; t0directformula, tails evaluatedseparatelyat.025/.05.
Independentmp80/100 references, exactworldfaces/interfaceg, exactrates/minimalrawtail/weights/fullsequence and17Padé states audited.
All1024Cartesian modes retained, negative raw zeroeigenvalue not reset, no clipping/normalization/F_Cfit orphysical/Hthresholdchange.
First testfixtureN4 hit oldauditor strictL1radius<1/4 boundary; only fixturechangedtoN8; firstsource/log/JSONretained.
First numericalaudit failed lowbit JSON equality because originalnp.float64total becamePythonfloat andPython3.12sumcompensation changed.
Append-only contact_oracle_reporting.py restores original scalarTYPE only, samevalues/formulas/thresholds; all48pre-recoveryfiles unchanged.
Three checks passed secondattempt;493recordeduniquechecks,old490notrerun. Scientificmainfirstpassed; finalauditsecondpassed.
Use this authoritative replay entry (old frozenstudy--replay retained asdiagnostic,notdeliveryentry):
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/contact_oracle_reporting.py --directory runs/reconnect-contact-oracle --replay
Replay checks the whole actual/fullindependentpath,halfstates,rawCSR/weights,finitecellmeans,storedcoefficientreferences,Cartesianfullmodes bitwise.
Typed budget decisions reproduce originalJSON exactly; no geometry/oldcandidate/PDE/ORC rerun.
Includes ALL projectresearchcode/docs/prototypes/referencebaseline, ALL782originalfiles byteidentical, currentreplayabledata/figures/LaTeX/PDF/audits/failurelogs.
Does NOT include all18GBhistoricalruns,21GBhistoricalexportcopies,15originalattachments or unavailablev080.
review-manifest.json verifies includedfiles. original-handoff-sha256.json verifiesall782 originals.
99priorformalmanifests and15attachments verified inoriginalworkspace; cumulative science manifestreferences omittedhistoricaldata; fullhistoricalaudit needsworkspace.
Next gate NOT EXECUTED: samecontactN16/32/64 spacecomparison,reuseN32,addonlytwo staticgraphs/newpaths,no angle/density/flow/modelscan.
Moving shockT4 andoracle/kernel/ledgerthree-densityclosure remain. OneN32control isnotfullspatialconvergence orStage3pass.
PDF explicitlyrequestedbyuser; nopublication,release/version,commit/push orenvironmentconfigchange.
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'review-manifest.json','original-handoff-sha256.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        for rel,h in originals.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-contact-oracle-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            replay=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/contact_oracle_reporting.py'),
              '--directory','runs/reconnect-contact-oracle','--replay'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=600)
            assert replay.returncode==0,replay.stdout+replay.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    delivery_info={'date':'2026-11-04','timezone':'Asia/Shanghai','PDF':{'path':str(pdf.relative_to(ROOT)),'bytes':pdf.stat().st_size,'sha256':digest(pdf),'pages':4},
      'code_and_current_results_zip':{'path':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'all_zip_CRC_and_included_SHA256_verified':True,'all782_original_files_included_and_byteidentical':True,'all_project_research_code_and_documents_included':True,
      'all_historical_run_data_or_export_archives_included':False,'original_attachments_included':False,
      'archive_files':len(manifest),'cumulative_scientific_manifest_entries':len(entries),'fresh_replay_exit_code':replay.returncode,'fresh_replay_log':replay.stdout,'fresh_replay_files_unchanged':True,
      'comparison_decision':result['comparison_decision'],'final_observation':result['observations'][-1],
      'new_software_checks':3,'recorded_checks_total':493,'old_checks_rerun':0,'main_first_attempt_passed':True,'tests_first_attempt_passed':False,'tests_second_attempt_passed':True,
      'audit_first_attempt_passed':False,'audit_second_attempt_passed':True,'all48_pre_reporting_recovery_files_unchanged':True,
      'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':99,
      'new_candidate_graphs':1,'new_candidate_transient_runs':1,'new_complete_independent_uniformization_sequences':1,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_spectral_projection_fits':0,'original_backends_changed':False,
      'next_gate_type':gate['type'],'next_gate_executed':False,'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False,'published':False}
    save(delivery,delivery_info)
    with downloads.open('x') as f:f.write('# LTN周期接触oracle判别：独立下载\n\n'
      '[四页中文报告PDF](LTN_contact_oracle_report_20261104.pdf)\n\n[全部研究代码、782原始文件与本轮可重放结果ZIP](LTN_contact_oracle_code_results_20261104.zip)\n\n'
      f'PDF SHA-256：`{digest(pdf)}`。\n\nZIP SHA-256：`{digest(archive)}`。\n\n'
      '[交付核验](LTN_contact_oracle_delivery_20261104.json)：CRC/SHA及全新解压完整重放通过。不是全部历史运行数据备份，旧附件另在工作区核验。Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(delivery_info,indent=2),flush=True)


if __name__=='__main__':main()
