"""Seal contact-space evidence, separate PDF/ZIP, verify complete fresh replay."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest


def main():
    out=ROOT/'runs/reconnect-contact-spatial';source=ROOT/'runs/reconnect-contact-oracle';exports=ROOT/'exports'
    pdf=exports/'LTN_contact_spatial_report_20261104.pdf';archive=exports/'LTN_contact_spatial_code_results_20261104.zip'
    delivery=exports/'LTN_contact_spatial_delivery_20261104.json';downloads=exports/'LTN_contact_spatial_downloads_20261104.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (pdf,archive,delivery,downloads,seal)),'newsealed files only'
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'numerical-audit.json');r=read(out/'results.json');gate=read(out/'next-gate-spec.json')
    assert ex['study_exit_code']==ex['tests_exit_code']==ar['audit_exit_code']==0 and ex['main_first_attempt_passed'] and ex['tests_first_attempt_passed'] and ar['audit_first_attempt_passed']
    assert audit['prior_formal_manifests_verified']==100 and audit['registered_three_space_level_contact_oracle_control_passed'] and audit['recorded_checks_total']==496
    assert not gate['already_executed'] and not r['fluid_orc_enabled']
    prior=read(source/'sha256.json');assert len(prior)==6459
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    log=(out/'LTN_contact_spatial_report_20261104.log').read_text();assert 'Overfull' not in log and 'Missing character' not in log
    info=subprocess.run(['pdfinfo',str(out/pdf.name)],capture_output=True,text=True,check=True).stdout;assert 'Pages:           3' in info
    shutil.copyfile(out/pdf.name,pdf);shutil.copyfile('/tmp/ltn-contact-spatial-xelatex-final.log',out/'PDF-build.log')
    save(out/'PDF-generation.json',{'PDF_explicitly_requested_by_user':True,'file':str(pdf.relative_to(ROOT)),'sha256':digest(pdf),'pages':3,
      'xelatex_exit_code':0,'overfull_boxes':0,'missing_characters':0,'all_pages_rendered_and_visually_reviewed':True,'environment_or_software_release_published':False})
    files=['research/reconnect/'+n for n in ('contact_spatial_core.py','contact_spatial_study.py','test_contact_spatial.py','contact_spatial_protocol.json',
      'audit_contact_spatial.py','plot_contact_spatial.py','document_contact_spatial.py','package_contact_spatial.py','CONTACT_SPATIAL_COMPARISON.md')]
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
    for rel,h in originals.items():assert manifest[rel]==h
    note=f'''LTN / diffuse-mixing: registered N16/32/64 static periodic contact oracle space comparison,2026-11-04.
Decision: {r['comparison_decision']}. Stage3 incomplete; fluid ORC OFF;kappa not computed.
Same unitperiodicbox,rho1/4,D originalfloat.02,T1/20,K16. Addonly N16/N64two graphs,two actual/fullindependentpaths.
Bothgraphs sealed beforeallactualmaterial/c0/reference/oldstates. N32loadedonly,notrerun; originaldecision exactpreserved.
Original pointVoronoi/ordinaryhalfresistance/action/time/H codeunchanged. No clipping/normalization/F_Cfit/positiveface deletion/thresholdchange.
Truefinite-cellmeans derivedfromsame originalcontinuousformula; stored128mode90digitcoefficients reused,no newprojection/PDEsolve.
Independent80/100digit refs,separatepositive-time tails,exactworldgeometry/rawrates/minimal1e-16tails/rawweights/fullsequences and17Padé states pernewgridverified.
Common originalderivedN16 material capacities; allinitial/refprojectionbridges retained; native/common concentration andnative/commonHgains exceed predeclaredpositivebudgets atbothnonzerotimes andbothadjacentpairs.
Highfrequency nativeerror maskedbyaggregation explicitly rejectedinsoftwarefixture; no signedcancellation,Richardsonstate oruniversalorderclaim.
Three newsoftwarechecks,main andaudit allfirstpassed;496recorded,old493notrerun. ParentN32failure/recovery unchanged.
Replay:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/contact_spatial_study.py --replay runs/reconnect-contact-spatial
Replay rebuildsactions fromtwo savedcaches,checksactual/fullindependentpaths,halfstates,CSR/rawweights,Cartesianfullmodes,cellmeans/references/commonintegrals/budgets bitwise.
N32states loadedonly. No geometry/oldcandidate/PDE/ORC rerun.
Includes allresearchcode/docs/prototypes/referencebaseline,all782 originalfiles byteidentical,currentthreegriddata/figures/PDF/TeX/audits,nextBeckerselectedsavedsources.
Doesnotinclude full39GBworkspace,18GBhistoricalruns/21GBolderexports,15attachments ormissingoldv080.
100priorformalmanifests and15attachments verified inoriginalworkspace;cumulativeseal referencesomittedhistory,notclaimof fullrecovery.
review-manifest.json verifiesincludedfiles;original-handoff-sha256.json verifiesall782originals.
Nextgate NOT EXECUTED: oneBecker movingouterboundaryxnoflux/periodicy andexplicitdurationadapter contract/unitchecks thenminimal32column/8transverserow,T.1,delta.08 oracle discriminator.
CurrentT.05helpercannot silentlystandinforT.1; newoptionalwrapper mustfixturematch oldT.05 andshareunalteredexponential/uniformization/H.
Reuseoldfixedmass/c0/saved8192reference; no newBeckerpoint/graph/state/refevaluationthisround. Moving shock andthree-densityclosure remain.
PDF andGitHubbackup authorized; noenvironment/softwareversion publication. Currentfreshreplaydoesnotrequirehistoricalmassive datasets.
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('original-handoff-sha256.json',json.dumps(originals,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','original-handoff-sha256.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        for rel,h in originals.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-contact-spatial-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            replay=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/contact_spatial_study.py'),'--replay','runs/reconnect-contact-spatial'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=600)
            assert replay.returncode==0,replay.stdout+replay.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    di={'date':'2026-11-04','timezone':'Asia/Shanghai','PDF':{'path':str(pdf.relative_to(ROOT)),'bytes':pdf.stat().st_size,'sha256':digest(pdf),'pages':3},
      'code_and_current_results_zip':{'path':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'all_zip_CRC_and_included_SHA256_verified':True,'all782_original_files_included_and_byteidentical':True,'archive_files':len(manifest),'cumulative_scientific_manifest_entries':len(entries),
      'fresh_replay_exit_code':replay.returncode,'fresh_replay_log':replay.stdout,'fresh_replay_files_unchanged':True,'comparison_decision':r['comparison_decision'],
      'per_grid_final_observations':{N:v[-1] for N,v in r['per_grid_observations'].items()},'new_software_checks':3,'recorded_checks_total':496,'old_checks_rerun':0,
      'tests_first_attempt_passed':True,'main_first_attempt_passed':True,'audit_first_attempt_passed':True,'original_backends_changed':False,
      'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':100,
      'new_candidate_graphs':2,'new_candidate_transient_runs':2,'new_complete_independent_uniformization_sequences':2,'N32_paths_loaded_only':1,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_spectral_projection_fits':0,'next_gate_type':gate['type'],'next_gate_executed':False,
      'all_historical_runs_or_attachments_included':False,'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False,'environment_or_software_version_published':False}
    save(delivery,di)
    with downloads.open('x') as f:f.write('# 周期接触oracle三档空间对照\n\n[三页中文报告PDF](LTN_contact_spatial_report_20261104.pdf)\n\n[代码与本轮结果ZIP](LTN_contact_spatial_code_results_20261104.zip)\n\n'
      f'PDF SHA-256：`{digest(pdf)}`。\n\nZIP SHA-256：`{digest(archive)}`。\n\n[交付核验](LTN_contact_spatial_delivery_20261104.json)：CRC/SHA及新解压两新路径逐位重放通过，N32只加载。不是全工作区镜像；Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(di,indent=2),flush=True)


if __name__=='__main__':main()
