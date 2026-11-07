"""Seal the unresolved Becker case, preserve reporting failure, verify download artifacts."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest


def main():
    out=ROOT/'runs/reconnect-becker-point';source=ROOT/'runs/reconnect-contact-spatial';exports=ROOT/'exports'
    pdf=exports/'LTN_becker_point_report_20261104.pdf';archive=exports/'LTN_becker_point_code_results_20261104.zip'
    delivery=exports/'LTN_becker_point_delivery_20261104.json';downloads=exports/'LTN_becker_point_downloads_20261104.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (pdf,archive,delivery,downloads,seal)),'never overwrite sealed files'
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'numerical-audit.json');r=read(out/'results.json');gate=read(out/'next-gate-spec.json')
    assert ex['tests_exit_code']==ex['completion_exit_code']==ar['audit_exit_code']==0 and ex['original_study_exit_code']==1 and not ex['main_first_attempt_passed']
    assert ex['tests_first_attempt_passed'] and ar['audit_first_attempt_passed'] and audit['prior_formal_manifests_verified']==101
    assert not r['registered_moving_Becker_oracle_discriminator_passed'] and r['time_control']['state_time_decrease_resolved'] and not r['time_control']['H_time_decrease_resolved']
    assert not gate['already_executed'] and gate['new_candidate_graphs']==gate['new_reference_PDE_solves']==0 and not r['fluid_orc_enabled']
    for rel,h in read(out/'pre-completion-files-sha256.json').items():assert digest(out/rel)==h,rel
    prior=read(source/'sha256.json');assert len(prior)==6576
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    log=(out/'LTN_becker_point_report_20261104.log').read_text();assert 'Overfull' not in log and 'Missing character' not in log
    info=subprocess.run(['pdfinfo',str(out/pdf.name)],capture_output=True,text=True,check=True).stdout;assert 'Pages:           4' in info
    shutil.copyfile(out/pdf.name,pdf);shutil.copyfile('/tmp/ltn-becker-point-xelatex-final.log',out/'PDF-build.log')
    save(out/'PDF-generation.json',{'PDF_explicitly_requested_by_user':True,'file':str(pdf.relative_to(ROOT)),'sha256':digest(pdf),'pages':4,
      'xelatex_exit_code':0,'overfull_boxes':0,'missing_characters':0,'all_pages_rendered_and_visually_reviewed':True,'unresolved_H_and_original_failure_disclosed':True})
    names=['BECKER_BOUNDARY_DURATION_CONTRACT.md','bounded_periodic_voronoi.py','audit_bounded_geometry.py','explicit_duration_action.py','becker_point_core.py',
      'becker_point_study.py','becker_point_protocol.json','test_becker_point.py','becker_point_completion.py','audit_becker_point.py','replay_becker_point.py',
      'plot_becker_point.py','document_becker_point.py','package_becker_point.py','BECKER_POINT_COMPARISON.md']
    files=['research/reconnect/'+n for n in names];entries=dict(prior);source_seal=str((source/'sha256.json').relative_to(ROOT));entries[source_seal]=digest(source/'sha256.json')
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
    for rel,h in originals.items():assert manifest[rel]==h
    note='''LTN movingBecker point-FV oracle discriminator,2026-11-04.
DECISION: moving_Becker_control_failed_or_unresolved. State time-decrease resolved; Hdecrease BELOW originalpositiveindicator. Do not report passed.
Delta.08,D originalfloat.02,T1/10,original32materialcolumns x8transverserows,K4/8/16. No extraN/K/centre/parameter scan.
28 new bounded-x/periodic-y pointVoronoi graphs closedbefore actualm/c0/reference/oldstate; onlygivenmasslabels used forpointtrajectories beforehand.
Given movingoutermaterialendpoints noflux; no periodicx,shockisnotwall. Allinternalpositivefaces retained,normalboundaryscalarflux0.
Optionalgeometry usesoriginalexactEuclideanclip/moments; KDneighbours initialonly,allpointnearimage radiusconstraints checked,farimages excludedbyfinalradius.
Independentallcellvertices/halfplanes/areas/reciprocalfaces/halfresistances/outerwalls verified.
Optionaldurationwrapper onlychanges exactdt=T/K,sharesoriginalexponential/rawuniformization/H. T.05 fixturebitwisematches oldhardcodedT.05helper; T.1 reallyused.
Original32columnmass/c0/reference means unchanged,rowsm/8; physicalunknowns aretruematerialmeans,notpointVoronoimeans. No F_Cfit ornormalization/clipping.
Existing8192space/timePDEreference andsaveduncertaintyreused,no newPDEsolve; independentexactoverlapreferenceprojection checked.
OriginalpositiveHcriterion retained,reference-spacebudgetdominates; no retrospectiveHformula/thresholdtightening orsignedcancellation.
Fournewsoftwarechecksfirstpassed,record500; old496notrerun. MainFAILEDafter allscientificpaths/results saved atdiagnostic rationalarea float parsing.
OriginalValueError log/frozensource/results retained. Append-onlybecker_point_completion.py parses float(Fraction(value)) fordiagnosticoutput only.
All109precompletionfiles unchanged; no scientificgeometry/path rerun orcriterionchange duringcompletion. Independentnumericalauditfirstpassed.
AUTHORITATIVE replay:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/replay_becker_point.py --directory runs/reconnect-becker-point
Replay checks actual/fullindependentpaths,halfstates,rawCSR/weights,originalcolumn/rowmeans,time/Hdecisions bitwise,thencompleteddiagnostics andoriginalfailurepreservation.
No geometry/PDE/oldcandidate/ORC rerun. Fullhistorypreservationaudit requiresoriginalworkspace.
Includesallresearchcode/docs,all782originalfiles byteidentical,current28graphpaths/reference/audits/PDF/LaTeX/failurelogs; notfull39GBworkspace,oldexports or15attachments.
review-manifest.json coversincludedfiles; original-handoff-sha256.json covers782originals; cumulativescience seal referencesomittedhistory.
101priorformalmanifests and15attachments verifiedinworkspace; fourhistoricaldiagnosticdifferences retained.
Nextgate NOT EXECUTED: readonlysavedH/reference component review,0graphs/0candidate/0PDE/0newchecks; originalunresolveddecisionmustremain.
Stage3 incomplete,fluid ORC OFF,kappa notcomputed; moving-shockT4 andthree-densityclosure NOT complete. No environment/softwareversion publication.
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','original-handoff-sha256.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-becker-point-replay-',dir='/tmp') as directory:
            z.extractall(directory);replay=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/replay_becker_point.py'),'--directory','runs/reconnect-becker-point'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=600)
            assert replay.returncode==0,replay.stdout+replay.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    di={'date':'2026-11-04','timezone':'Asia/Shanghai','PDF':{'path':str(pdf.relative_to(ROOT)),'bytes':pdf.stat().st_size,'sha256':digest(pdf),'pages':4},
      'code_and_current_results_zip':{'path':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'all_zip_CRC_and_included_SHA256_verified':True,'all782_original_files_included_and_byteidentical':True,'archive_files':len(manifest),'cumulative_scientific_manifest_entries':len(entries),
      'fresh_replay_exit_code':replay.returncode,'fresh_replay_log':replay.stdout,'fresh_replay_files_unchanged':True,'comparison_decision':r['comparison_decision'],
      'registered_moving_Becker_oracle_discriminator_passed':False,'time_control':r['time_control'],'final_K16_observation':r['observations'][-1],
      'new_software_checks':4,'recorded_checks_total':500,'old_checks_rerun':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':False,'audit_first_attempt_passed':True,
      'original_study_exit_code':1,'reporting_completion_exit_code':0,'all109_pre_completion_files_unchanged':True,'scientific_paths_rerun_during_completion':0,
      'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':101,
      'new_candidate_graphs':28,'new_candidate_transient_runs':3,'new_complete_independent_uniformization_sequences':3,'old_candidate_reruns':0,'new_reference_PDE_solves':0,
      'original_periodic_exponential_uniformization_H_backends_changed':False,'next_gate_type':gate['type'],'next_gate_executed':False,
      'stage3_physical_acceptance':'incomplete','moving_shock_T4_closed':False,'all_three_density_sources_closed':False,'fluid_orc_enabled':False,'kappa_computed':False,'environment_or_software_version_published':False}
    save(delivery,di)
    with downloads.open('x') as f:f.write('# 移动Becker点FV oracle：未决结果下载\n\n[四页中文PDF](LTN_becker_point_report_20261104.pdf)\n\n[代码与完整本例结果ZIP](LTN_becker_point_code_results_20261104.zip)\n\n'
      f'PDF SHA-256：`{digest(pdf)}`。\n\nZIP SHA-256：`{digest(archive)}`。\n\n[交付核验](LTN_becker_point_delivery_20261104.json)：CRC/SHA、新解压逐位重放通过。状态时间控制分辨，H未分辨；不宣告物理判别通过。Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(di,indent=2),flush=True)


if __name__=='__main__':main()
