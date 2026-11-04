"""Seal exact-boundary continuous rotation and verify a fresh full action replay."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from audit_route_decision import ROOT,read,save,digest
from package_shear_voronoi_time16 import dependency_closure


def main():
    source=ROOT/'runs/reconnect-t2-rotation-domain';out=ROOT/'runs/reconnect-t2-rotation'
    archive=ROOT/'exports/ltn-t2-rotation.zip';delivery=ROOT/'exports/ltn-t2-rotation-delivery.json';downloads=ROOT/'exports/ltn-t2-rotation-downloads.md';seal=out/'sha256.json'
    assert not any(p.exists() for p in (archive,delivery,downloads,seal)),'closed evidence is never overwritten'
    ex=read(out/'execution.json');ar=read(out/'audit-execution.json');audit=read(out/'numerical-audit.json');r=read(out/'results.json')
    status=read(out/'stage3-status.json');gate=read(out/'next-gate-spec.json')
    assert ex['tests_exit_code']==ex['study_exit_code']==ar['audit_exit_code']==0
    assert ex['main_first_attempt_passed'] and ex['tests_first_attempt_passed'] and ar['audit_first_attempt_passed']
    assert audit['all16_independent_world_geometries_exact_rates_and_full_raw_sequences_verified'] and audit['prior_formal_manifests_verified']==92
    assert status['recorded_checks_total']==477 and not status['fluid_orc_enabled'] and not status['general_smooth_flow_admitted'] and not gate['already_executed']
    prior=read(source/'sha256.json');assert len(prior)==3849
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    files=['research/reconnect/'+n for n in ['oriented_periodic_boundary.py','audit_oriented_world_geometry.py','rotation_world_trajectory.py',
      'test_oriented_periodic_boundary.py','t2_rotation_core.py','t2_rotation_study.py','t2_rotation_protocol.json','T2_ROTATION_CONTROL.md',
      'audit_t2_rotation_geometry.py','audit_t2_rotation.py','plot_t2_rotation.py','package_t2_rotation.py','T2_ROTATION_COMPARISON.md']]
    source_seal=str((source/'sha256.json').relative_to(ROOT));entries=dict(prior);entries[source_seal]=digest(source/'sha256.json')
    for rel in files:assert rel not in entries;entries[rel]=digest(ROOT/rel)
    for path in out.rglob('*'):
        if path.is_file():
            rel=str(path.relative_to(ROOT));assert rel not in entries;entries[rel]=digest(path)
    save(seal,dict(sorted(entries.items())))
    selected=set(files)|{str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()}|dependency_closure([p for p in files if p.endswith('.py')])
    selected.update([source_seal,'docs/RESEARCH_STATUS.md']);selected.update(v['path'] for v in read(out/'evidence-index.json')['sources'])
    manifest={p:digest(ROOT/p) for p in sorted(selected)}
    note=(
      'LTN T2: ONE EXACT-BOUNDARY CONTINUOUS ROTATION ON THE CORRESPONDING ROTATING TORUS.\n'
      f'Decision:{r["comparison_decision"]};Stage3incomplete,fluidORCoff.\n'
      'N32,rho1,D=.02,T=.05,K16;originalmaterialm,c0,IDs,reference andstaticpaths copied byte-identically.\n'
      'GivenboundaryE(s)=[[1-s^2,-2s],[2s,1-s^2]]/(1+s^2),s=t/(2T);worldpoints andperiodlattice rotate together.\n'
      'Smooth variableangularspeed,final2atan(1/2),not aquarter-turn orconstantangularvelocity test.\n'
      'New opt-in exactorthogonal boundaryadapter;original geometry,ordinaryresistance andtimebackends unchanged.\n'
      'No localF/Cfit ortruesourcefaces;adapter doesnot receive source-local-labels,mass,c0 orreference.\n'
      'Exactproperisometry andlosslesslocalfloat/periodlength conversion required;unsupported inputs rejected,norepair.\n'
      'All16 freshoriginalgeometry builds andworldlifts sealedbeforemass/scalar/reference/staticgeometry/path load; no staticg copied.\n'
      'Independentworld audit directly reconstructs worldbisectors,periodicfacepairs,halfresistances,areas andall fixedworldimageprobes.\n'
      'AllIDedges/exactg,rawCSR/weights and17actual/independent endpoints plus16halfstates match staticbaseline bitwise.\n'
      'PositiveD nonzeroH andoriginalreferenceerror retained;not forcedzero orreplacedbystaticstates.\n'
      'Oneactual andonecompleteindependentuniformization path,original1e-16tail,1e-10algebra scale andseed20261101.\n'
      'All16worldgeometry,exactrate,minimalrawwighttail,completeindependentsequence andcovariance audits verified.\n'
      'Three newchecks passed firstattempt,recorded477;old474 notrerun;oldv0.8.0/246 notrecovered.\n'
      '782originals,15attachments and92priorformalmanifests preserved;fourknownhistoricaldiagnosticdifferences retained.\n'
      'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/t2_rotation_study.py --replay runs/reconnect-t2-rotation\n'
      'Freshreplay rebuilds actual/completeindependent actions from16sealedlocalcaches andcompares staticpaths bitwise;worldaudit inputs/receipts included.\n'
      'No geometry,oldcandidate orPDE reruns inreplay;full historical audit requires originalworkspace.\n'
      'Includedfiles checked byreview-manifest.json;cumulativeseal references omittedhistory.\n'
      'Exactboundary/losslesslocalinputmode only;arbitraryfloatworlddata andnonorthogonallattices notadmitted.\n'
      'NextT3pairedconstantD/Gamma affinecompression gate registeredonly,unexecuted;no nextpoints/graphs/references/states generated.\n'
      'No clipping,pointrepair,mass/rho/state/row/weightnormalization,F/Cfit,thresholdchange,PDF,version,commit/push,configurationchange orpublication.\n'
      '2026-11-02 Asia/Shanghai;generalT2/smoothflow/Stage3 notaccepted,fluidORCoff,kappa notcomputed.\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(ROOT/rel,rel)
        z.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n');z.writestr('DELIVERY_README.txt',note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(manifest)|{'review-manifest.json','DELIVERY_README.txt'}
        for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
        with tempfile.TemporaryDirectory(prefix='ltn-t1-fresh-replay-',dir='/tmp') as directory:
            z.extractall(directory)
            result=subprocess.run([sys.executable,str(Path(directory)/'research/reconnect/t2_rotation_study.py'),'--replay','runs/reconnect-t2-rotation'],cwd=directory,
              env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=600)
            assert result.returncode==0,result.stdout+result.stderr
            for rel,h in manifest.items():assert digest(Path(directory)/rel)==h,rel
    for rel,h in read(seal).items():assert digest(ROOT/rel)==h,rel
    info={'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},
      'archive_CRC_and_all_included_SHA256_verified':True,'fresh_replay_exit_code':result.returncode,'fresh_replay_log':result.stdout,'fresh_replay_files_unchanged':True,
      'scientific_manifest_entries':len(entries),'archive_entries':len(manifest),'comparison_decision':r['comparison_decision'],
      'T2_registered_rotation_case_passed':r['T2_registered_rotation_case_passed'],'final_observation':r['observations'][-1],'all_path_arrays_bitwise_static':r['path_arrays_bitwise_static'],
      'new_software_checks':3,'recorded_checks_total':477,'previous_checks_rerun':0,'new_candidate_graphs':16,'new_candidate_transient_runs':1,
      'new_independent_uniformization_sequences':1,'new_raw_action_intervals':16,'old_candidate_reruns':0,'new_reference_PDE_solves':0,
      'original_files_preserved':782,'historical_attachments_preserved':15,'prior_formal_manifests_preserved':92,'next_gate_executed':False,
      'exact_boundary_adapter_implemented':True,'original_backends_changed':False,'general_T2_admitted':False,'kappa_computed':False,'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'published':False}
    save(delivery,info)
    with downloads.open('x') as f:
        f.write('# T2精确旋转周期域：连续刚体旋转对照\n\n[报告、适配层、16张局部/世界图、原始输入与路径、审计和重放代码](ltn-t2-rotation.zip)。\n\n'
          f'SHA-256：`{info["archive"]["sha256"]}`。\n\n[交付核验](ltn-t2-rotation-delivery.json)。全新解压，实际/完整独立路径及半步状态逐位重放通过，与原静止路径一致。\n\n'
          '本例精确边界旋转对照通过；Stage3 incomplete，fluid ORC关闭。下一项T3恒D/恒Gamma仿射压缩仅登记，尚未执行。\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
