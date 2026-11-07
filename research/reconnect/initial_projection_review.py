"""Saved initial means only: analytic boundary representation and rounding accounting."""
from pathlib import Path
from fractions import Fraction as F
import argparse,bisect,csv,json,resource,shutil,signal,time,traceback
import numpy as np
import mpmath as mp
from audit_route_decision import ROOT,read,save,digest

SOURCE=ROOT/'runs/reconnect-becker-reference'
PROTOCOL=ROOT/'research/reconnect/initial_projection_protocol.json'
LEVELS=((16384,8192),(32768,8192),(32768,16384))


def initial_arrays(path,fine=False):
    keys=('mass_edges','c0','initial_projected') if fine else ('mass_edges','mass','c0')
    with np.load(path,allow_pickle=False) as z:return {key:z[key].copy() for key in keys}


def exact_projection(edges,values,target):
    e=list(map(lambda x:F(float(x)),edges));v=list(map(lambda x:F(float(x)),values));out=[];boundary=[]
    for x in target:
        xx=F(float(x));j=min(len(v)-1,max(0,bisect.bisect_right(e,xx)-1));boundary.append((j,xx-e[j],e[j+1]-e[j]))
    for a,b in zip(target[:-1],target[1:]):
        aa,bb=F(float(a)),F(float(b));j=max(0,bisect.bisect_right(e,aa)-1);total=F(0)
        while j<len(v) and e[j]<bb:
            l=max(e[j],aa);h=min(e[j+1],bb)
            if h>l:total+=(h-l)*v[j]
            j+=1
        out.append(total/(bb-aa))
    return out,boundary


def analyze(folder):
    folder=Path(folder);coarse=initial_arrays(folder/'column.npz');target=coarse['mass_edges'];mass=coarse['mass'];coarse0=coarse['c0']
    cases=[];records={};cache={}
    for n,k in LEVELS:
        z=initial_arrays(folder/f'n{n}-k{k}.npz',True)
        signature=(z['mass_edges'].tobytes(),z['c0'].tobytes(),z['initial_projected'].tobytes())
        if signature in cache:
            base=cache[signature];cases.append({**base,'steps':k,'same_initial_data_as':f'n{n}-k8192'});continue
        raw,boundary=exact_projection(z['mass_edges'],z['c0'],target)
        for value,saved in zip(raw,z['initial_projected']):assert float(value)==saved
        by_precision=[];last_vectors=None
        for precision in (80,100):
            with mp.workdps(precision):
                q=lambda x:mp.mpf(x.numerator)/x.denominator if isinstance(x,F) else mp.mpf(float(x))
                e=list(map(q,z['mass_edges']));a=list(map(q,target));M=a[-1];L=mp.pi/(4*M)
                primitive=lambda x:x/2+M*mp.sin(mp.pi*x/M)/(4*mp.pi)
                mean=lambda l,r:(primitive(r)-primitive(l))/(r-l)
                defects=[];bounds=[];cuts=[]
                for x,(j,d,h) in zip(a,boundary):
                    l,r=e[j],e[j+1];s=(x-l)*mean(l,r)-(primitive(x)-primitive(l));bound=L*(x-l)*(r-x)/2
                    assert abs(s)<=bound+mp.mpf('1e-60')
                    defects.append(s);bounds.append(bound);cuts.append({'fine_cell':j,'fraction_exact':str(d/h),'defect':mp.nstr(s,60),'Lipschitz_bound':mp.nstr(bound,60)})
                vectors={name:[] for name in ('representation','fine_mean_rounding','coarse_mean_rounding','projection_rounding','stored_total','representation_bound')}
                identity=[]
                for i,(lo,hi) in enumerate(zip(a[:-1],a[1:])):
                    true=mean(lo,hi);representation=(defects[i+1]-defects[i])/(hi-lo);ideal_projected=true+representation
                    pieces=[representation,q(raw[i])-ideal_projected,true-q(coarse0[i]),q(z['initial_projected'][i])-q(raw[i])]
                    total=q(z['initial_projected'][i])-q(coarse0[i]);identity.append(total-sum(pieces))
                    for name,value in zip(('representation','fine_mean_rounding','coarse_mean_rounding','projection_rounding'),pieces):vectors[name].append(value)
                    vectors['stored_total'].append(total);vectors['representation_bound'].append((bounds[i+1]+bounds[i])/(hi-lo))
                norm=lambda v:mp.sqrt(mp.fsum(q(m)*x*x for m,x in zip(mass,v)))
                norms={name:mp.nstr(norm(v),60) for name,v in vectors.items()}
                maximum=max(map(abs,identity));assert maximum<mp.mpf('1e-60')
                round_sum=sum(mp.mpf(norms[name]) for name in ('fine_mean_rounding','coarse_mean_rounding','projection_rounding'))
                assert abs(mp.mpf(norms['stored_total'])-mp.mpf(norms['representation']))<=round_sum+mp.mpf('1e-60')
                by_precision.append({'precision':precision,'mass_norms':norms,'max_vector_identity_residual':mp.nstr(maximum,10),
                  'positive_rounding_norm_sum':mp.nstr(round_sum,60),'representation_over_rounding_norm_sum':mp.nstr(mp.mpf(norms['representation'])/round_sum,40)})
                last_vectors={name:[mp.nstr(v,60) for v in values] for name,values in vectors.items()};last_cuts=cuts
        assert all(float(by_precision[0]['mass_norms'][name])==float(by_precision[1]['mass_norms'][name]) for name in last_vectors)
        record={'cells':n,'steps':k,'interior_nonaligned_boundaries':sum(d!=0 and d!=h for _,d,h in boundary[1:-1]),
          'precision_results':by_precision,'same_initial_data_as':None}
        cache[signature]=record;cases.append(record);records[f'n{n}']={'vectors':last_vectors,'boundary_defects':last_cuts,'exact_stored_projection':[str(v) for v in raw]}
    old=read(folder/'parent-results.json')
    for case,oldrow in zip(cases,old['reference_rows']):
        v=float(case['precision_results'][-1]['mass_norms']['stored_total']);oldvalue=oldrow['initial_mean_bridge_absolute']
        # Reduction audit only: the original physical decision/budget is copied verbatim.
        assert abs(v-oldvalue)<=64*np.finfo(float).eps*oldvalue
    mismatches=sum(F(float(m))!=F(float(b))-F(float(a)) for m,a,b in zip(mass,target[:-1],target[1:]))
    result={'decision':'initial_bridge_is_nonaligned_piecewise_constant_representation_not_physical_initial_change',
      'cases':cases,'coarse_mass_storage_differs_from_exact_endpoint_subtraction_count':mismatches,
      'fixed_initial_function':'1/2+1/4*cos(pi*mu/M)','analytic_knowledge_used_only_at_initial_time':True,
      'initial_cell_mean_rounding_separated_from_representation':True,'signed_vectors_only_for_identity_not_acceptance':True,
      'parent_decision_retained':old['decision'],'parent_positive_budget_components_unchanged':old['positive_budget_components_absolute'],
      'parent_H_indicator_unchanged':old['H_indicator_unchanged_formula'],'parent_H_decline_unchanged':old['H_decrease_unchanged'],
      'old_case_reclassified':False,'reference_terminal_or_candidate_arrays_parsed':False,
      'new_reference_PDE_solves':0,'new_candidate_graphs':0,'new_candidate_transient_runs':0,'new_software_checks':0,'recorded_checks_total':500,
      'reference_observation_contract_only_proposed':True,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}
    return result,records


def main(output):
    start=time.monotonic();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    def timeout(*args):raise TimeoutError('registered120s initial projection review cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        gate=read(SOURCE/'next-gate-spec.json');protocol=read(PROTOCOL)
        assert not gate['already_executed'] and digest(SOURCE/'next-gate-spec.json')==protocol['source_gate_sha256']
        for item in gate['sources']:assert digest(ROOT/item['path'])==item['sha256']
        parent=read(SOURCE/'sha256.json')
        for rel,h in parent.items():assert digest(ROOT/rel)==h,rel
        shutil.copyfile(PROTOCOL,out/'protocol.json');shutil.copyfile(SOURCE/'next-gate-spec.json',out/'source-gate-spec.json')
        save(out/'implementation-lock.json',{'code_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),'before_new_accounting':True})
        (out/'source').mkdir();pairs=[('source/column.npz','column.npz'),('results.json','parent-results.json')]+[(f'reference-n{n}-k{k}.npz',f'n{n}-k{k}.npz') for n,k in LEVELS]
        copied=[]
        for src,dest in pairs:
            shutil.copyfile(SOURCE/src,out/'source'/dest);copied.append({'source':str((SOURCE/src).relative_to(ROOT)),'copy':'source/'+dest,'sha256':digest(SOURCE/src)})
        save(out/'source-copies.json',copied);result,vectors=analyze(out/'source');save(out/'results.json',result);save(out/'diagnostic-vectors.json',vectors)
        with (out/'components.csv').open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['cells','steps','component','mass_norm'])
            for case in result['cases']:
                for name,value in case['precision_results'][-1]['mass_norms'].items():w.writerow([case['cells'],case['steps'],name,value])
        pres=read(SOURCE/'preservation.json');formal=pres['formal_manifests_verified']+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(parent)}]
        assert len(formal)==104
        for item in formal:
            p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256']
            for rel,h in read(p).items():assert digest(ROOT/rel)==h,rel
        originals=read('/workspace/original-handoff-sha256.json');attachments=read('/workspace/setup-ltn/attachment-manifest.json')
        assert len(originals)==782 and len(attachments)==15
        for rel,h in originals.items():assert digest(ROOT/rel)==h
        for item in attachments:assert digest(Path('/workspace')/item['path'])==item['sha256']
        save(out/'preservation.json',{'formal_manifests_verified':formal,'originals_verified':782,'attachments_verified':15,
          'historical_diagnostic_differences_retained':pres['historical_diagnostic_differences_retained'],'all_parent_scientific_files_unchanged':True})
        elapsed=time.monotonic()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert elapsed<120 and peak<1024
        save(out/'execution.json',{'first_attempt_passed':True,'exit_code':0,'elapsed_seconds':elapsed,'peak_RSS_MiB':peak,
          'reference_terminal_states_parsed':False,'new_PDE_or_candidate_solves':0,'published_or_pushed':False})
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});raise
    finally:signal.alarm(0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
