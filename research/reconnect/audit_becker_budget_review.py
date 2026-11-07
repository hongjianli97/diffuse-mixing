"""Independent exact-rational/Decimal accounting audit; no scientific solver calls."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
import argparse,ast,csv,json,math,resource,signal,struct,time,zipfile
from audit_route_decision import ROOT,read,save,digest


def vector(path,key):
    with zipfile.ZipFile(path) as z:raw=z.read(key+'.npy')
    assert raw[:6]==b'\x93NUMPY' and raw[6:8] in (b'\x01\x00',b'\x02\x00')
    size=2 if raw[6]==1 else 4;length=int.from_bytes(raw[8:8+size],'little');start=8+size
    header=ast.literal_eval(raw[start:start+length].decode('latin1'))
    assert header['descr']=='<f8' and not header['fortran_order'];shape=header['shape'];assert len(shape)==1
    values=tuple(F(v[0]) for v in struct.iter_unpack('<d',raw[start+length:]));assert len(values)==shape[0]
    return values


def audit(directory):
    started=time.monotonic()
    def timeout(*args):raise TimeoutError('120s saved-budget accounting audit cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    out=Path(directory).resolve();src=out/'source';r=read(out/'results.json');old=read(src/'results.json');ref=read(src/'reference-indicators.json')
    protocol=read(out/'review-protocol.json');lock=read(out/'implementation-lock.json')
    assert digest(ROOT/'research/reconnect/becker_budget_review.py')==lock['review_code_sha256']
    assert digest(out/'review-protocol.json')==lock['protocol_sha256']
    for item in protocol['sources']:assert digest(ROOT/item['path'])==item['sha256']
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==digest(ROOT/item['source'])==item['sha256']
    # Recompose positives from saved source JSON without importing the review implementation.
    expected={'twice_reference_space':2*F(ref['components_absolute']['saved_reference_space_difference']),
      'twice_reference_time':2*F(ref['components_absolute']['saved_reference_time_difference']),
      'twice_machine':F(0),'twice_tail':F(0),'twice_observed_full_sequence':F(0)}
    for K in (4,8,16):
        step=read(src/f'k{K}/native-results.json')['steps'][-1]
        for name,field in [('machine','cumulative_machine_indicator_absolute'),('tail','cumulative_tail_indicator_absolute'),('observed_full_sequence','primary_control_mass_norm_difference')]:expected['twice_'+name]+=2*F(step[field])
    parts=old['time_control']['components_absolute']
    expected.update({k:F(v) for k,v in parts.items() if 'bridge' in k})
    actual={row['component']:F(row['exact']) for row in r['components']};assert actual==expected and all(v>=0 for v in actual.values())
    b=F(sum(parts.values()));total=sum(expected.values(),F(0));residue=b-total
    assert F(r['positive_components_exact_sum']['exact'])==total and F(r['retained_accumulation_roundoff_not_a_scientific_cancellation']['exact'])==residue
    assert float(b)==r['original_binary64_budget_absolute'] and r['original_binary64_H_indicator']==old['time_control']['fixed_H_decrease_indicator']
    A,E0=F(r['original_H_prefactor']),F(r['original_initial_variance_binary64']);h=(A*b+b*b)/E0
    lo=hi=r['original_binary64_H_indicator']
    for _ in range(8):lo=math.nextafter(lo,-math.inf);hi=math.nextafter(hi,math.inf)
    assert F(lo)<=h<=F(hi)
    m=vector(src/'material-inputs.npz','mass');c0=vector(src/'material-inputs.npz','c0')
    M=sum(m,F(0));mean=sum((a*c for a,c in zip(m,c0)),F(0))/M;variance=sum((a*(c-mean)**2 for a,c in zip(m,c0)),F(0))
    # The exact stored-input variance is diagnostic, not a replacement denominator.
    variance_relative_rounding=float(abs(variance-E0)/variance)
    mc=vector(src/'source-column-inputs.npz','mass')
    cols=[vector(src/'source-column-inputs.npz',key) for key in ('reference_n4096-k4096','reference_n8192-k4096','reference_n8192-k8192')]
    squares=[sum((mass*(a-bb)**2 for mass,a,bb in zip(mc,cols[i],cols[i+1])),F(0)) for i in (0,1)]
    gap=F(old['time_control']['H4_8']-old['time_control']['H8_16']);nonref=b-F(parts['twice_saved_reference_space_time'])
    checkpoints=[]
    for precision in (80,100):
        with localcontext() as ctx:
            ctx.prec=precision;D=lambda v:Decimal(v.numerator)/Decimal(v.denominator)
            a,e,d=D(A),D(E0),D(gap);critical=2*e*d/(a+(a*a+4*e*d).sqrt());ceiling=(critical-D(nonref))/2
            vals=[D(v).sqrt() for v in squares]
            checkpoints.append({'precision':precision,'critical_budget':str(critical),'reference_ceiling':str(ceiling),'stored_space_time_norms':[str(v) for v in vals]})
            assert float(critical)==float(Decimal(r['feasibility']['same_formula_total_budget_at_equality']))
            assert float(ceiling)==float(Decimal(r['feasibility']['same_formula_reference_space_time_ceiling']))
            for value,field in zip(vals,('saved_reference_space_difference','saved_reference_time_difference')):
                observed=ref['components_absolute'][field];assert abs(float(value)-observed)<=8*math.ulp(observed)
    assert all(float(checkpoints[0][key])==float(checkpoints[1][key]) for key in ('critical_budget','reference_ceiling'))
    assert (A*nonref+nonref*nonref)/E0<gap<(A*expected['twice_reference_space']+expected['twice_reference_space']**2)/E0
    assert expected['twice_reference_space']/b>F(99,100)
    assert r['source_decision_retained']==old['comparison_decision']=='moving_Becker_control_failed_or_unresolved' and not r['source_case_passed']
    gate=read(out/'next-gate-spec.json');assert not gate['already_executed'] and gate['new_reference_PDE_solves']==3
    assert gate['reference_levels']==[{'cells':16384,'steps':8192},{'cells':32768,'steps':8192},{'cells':32768,'steps':16384}]
    assert gate['new_candidate_graphs']==gate['new_candidate_transient_runs']==gate['new_operator_actions']==gate['new_software_checks']==0
    for item in gate['sources']:assert digest(ROOT/item['path'])==item['sha256']
    csvrows=list(csv.DictReader((out/'budget-components.csv').open()));assert {v['component']:F(v['exact']) for v in csvrows}==expected
    preservation=read(out/'preservation.json');assert len(preservation['formal_manifests_verified'])==102
    for item in preservation['formal_manifests_verified']:
        path=ROOT/item['manifest'];assert digest(path)==item['manifest_sha256'];entries=read(path);assert len(entries)==item['entries_verified']
        for rel,expected_hash in entries.items():assert digest(ROOT/rel)==expected_hash
    elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert elapsed<120 and peak<1024
    save(out/'accounting-audit.json',{'audit_passed':True,'audit_first_attempt_passed':True,'elapsed_seconds':elapsed,'peak_RSS_MiB':peak,
      'auditor_sha256':digest(Path(__file__)),'independent_standard_library_exact_accounting':True,'Decimal80_and100_checks':checkpoints,
      'stored_initial_variance_exact':str(variance),'original_binary64_variance_relative_rounding':variance_relative_rounding,
      'positive_accounting_original_H_and_unresolved_preserved':True,'prior_formal_manifests_verified':102,'new_software_checks':0,'recorded_checks_total':500,
      'new_graphs_actions_PDE_solves':0,'next_reference_check_executed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})
    signal.alarm(0);print('Exact positive accounting, original H, independent80/100-digit arithmetic and102 prior formal manifests verified. No solver calls.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();audit(a.directory)
