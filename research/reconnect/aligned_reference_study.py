"""Run the two registered software fixtures and preserve the historical ledger."""
from pathlib import Path
import argparse,io,json,resource,shutil,signal,time,traceback,unittest
from audit_route_decision import ROOT,read,save,digest
import test_aligned_reference as fixtures

SOURCE=ROOT/'runs/reconnect-initial-projection'


def main(output):
    start=time.monotonic();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    def timeout(*args):raise TimeoutError('registered120s prototype fixture cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        gate=read(SOURCE/'next-gate-spec.json');protocol=ROOT/'research/reconnect/aligned_reference_protocol.json'
        assert not gate['already_executed'] and gate['planned_new_software_checks']==2
        assert digest(SOURCE/'next-gate-spec.json')==read(protocol)['source_gate_sha256']
        for item in gate['sources']+[gate['contract']]:assert digest(ROOT/item['path'])==item['sha256']
        parent=read(SOURCE/'sha256.json')
        for rel,h in parent.items():assert digest(ROOT/rel)==h,rel
        shutil.copyfile(protocol,out/'protocol.json');shutil.copyfile(SOURCE/'next-gate-spec.json',out/'source-gate-spec.json')
        files=['aligned_mass_reference.py','test_aligned_reference.py','aligned_reference_study.py','aligned_reference_protocol.json','ALIGNED_REFERENCE_PROTOTYPE_CONTRACT.md']
        save(out/'implementation-lock.json',{'files':{name:digest(ROOT/'research/reconnect'/name) for name in files},
          'legacy_reference_sha256':digest(ROOT/'research/rebuild/reference.py'),'legacy_initial_sha256':digest(ROOT/'research/rebuild/becker.py'),'fixed_before_fixture_inputs_and_solves':True})
        fixtures.OUTPUT=out;fixtures.CALLS.clear();fixtures.METRICS.clear();stream=io.StringIO()
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(fixtures.AlignedReferenceFixtures))
        (out/'tests.log').write_text(stream.getvalue());save(out/'calls.json',fixtures.CALLS);save(out/'fixture-metrics.json',fixtures.METRICS)
        save(out/'tests.json',{'successful':result.wasSuccessful(),'checks_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)})
        assert result.wasSuccessful() and result.testsRun==2,stream.getvalue()
        calls=fixtures.CALLS;assert len(calls)==7 and sum(v['linear_solves'] for v in calls)==33
        save(out/'results.json',{'decision':'material_aligned_reference_prototype_software_fixtures_passed','software_checks_passed':2,'previous_checks_rerun':0,
          'recorded_checks_total':502,'uniform_limit_bitwise':True,'complete_child_integral_mapping_verified':True,
          'conditional_CN_positivity_and_raw_rejection_verified':True,'arbitrary_nonuniform_second_order_claimed':False,
          'synthetic_solver_attempts':len(calls),'synthetic_linear_system_solves':33,'synthetic_expected_rejections':2,
          'new_Becker_reference_solves':0,'new_physical_reference_experiments':0,'new_candidate_graphs':0,'new_candidate_transient_runs':0,
          'old_decisions_budgets_and_H_preserved':True,'moving_shock_T4_closed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
        old=read(SOURCE/'preservation.json');formal=old['formal_manifests_verified']+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(parent)}]
        assert len(formal)==105
        for item in formal:
            p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256']
            for rel,h in read(p).items():assert digest(ROOT/rel)==h,rel
        originals=read('/workspace/original-handoff-sha256.json');attachments=read('/workspace/setup-ltn/attachment-manifest.json');assert len(originals)==782 and len(attachments)==15
        for rel,h in originals.items():assert digest(ROOT/rel)==h
        for item in attachments:assert digest(Path('/workspace')/item['path'])==item['sha256']
        save(out/'preservation.json',{'formal_manifests_verified':formal,'originals_verified':782,'attachments_verified':15,
          'historical_diagnostic_differences_retained':old['historical_diagnostic_differences_retained'],'all_parent_scientific_files_unchanged':True})
        elapsed=time.monotonic()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert elapsed<120 and peak<1024
        save(out/'execution.json',{'first_attempt_passed':True,'exit_code':0,'elapsed_seconds':elapsed,'peak_RSS_MiB':peak,
          'new_Becker_or_candidate_solves':0,'published_or_pushed':False})
        print(stream.getvalue(),flush=True);print(json.dumps(read(out/'results.json'),indent=2),flush=True)
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});raise
    finally:signal.alarm(0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
