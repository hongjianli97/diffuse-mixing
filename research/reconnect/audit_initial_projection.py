"""Independent sinc/partial-cell accounting for saved initial means; no PDE solve."""
from pathlib import Path
from fractions import Fraction as F
import argparse,bisect,resource,time
import numpy as np
import mpmath as mp
from audit_route_decision import ROOT,read,save,digest
from initial_projection_review import initial_arrays,LEVELS


def audit(directory):
    start=time.monotonic();out=Path(directory).resolve();src=out/'source';r=read(out/'results.json');vectors=read(out/'diagnostic-vectors.json')
    lock=read(out/'implementation-lock.json');assert digest(ROOT/'research/reconnect/initial_projection_review.py')==lock['code_sha256']
    assert digest(out/'protocol.json')==lock['protocol_sha256'];assert read(out/'execution.json')['first_attempt_passed']
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==digest(ROOT/item['source'])==item['sha256']
    coarse=initial_arrays(src/'column.npz');tt=list(map(lambda v:F(float(v)),coarse['mass_edges']));mass=coarse['mass'];receipts=[]
    for n,k in ((16384,8192),(32768,8192)):
        z=initial_arrays(src/f'n{n}-k{k}.npz',True);ee=list(map(lambda v:F(float(v)),z['mass_edges']));zz=list(map(lambda v:F(float(v)),z['c0']))
        raw=[];spans=[];j=0
        for lo,hi in zip(tt[:-1],tt[1:]):
            while ee[j+1]<=lo:j+=1
            h=j;total=F(0)
            while h<len(zz) and ee[h]<hi:
                a=max(lo,ee[h]);b=min(hi,ee[h+1])
                if b>a:total+=(b-a)*zz[h]
                h+=1
            raw.append(total/(hi-lo));spans.append((j,h-1))
        assert [str(v) for v in raw]==vectors[f'n{n}']['exact_stored_projection']
        evaluated=[]
        for precision in (100,120):
            with mp.workdps(precision):
                q=lambda x:mp.mpf(x.numerator)/x.denominator if isinstance(x,F) else mp.mpf(float(x))
                M=q(tt[-1]);e=list(map(q,ee));targets=list(map(q,tt))
                def average(a,b):
                    t=mp.pi*(b-a)/(2*M)
                    return mp.mpf(1)/2+mp.cos(mp.pi*(a+b)/(2*M))*mp.sin(t)/(4*t)
                v={name:[] for name in ('representation','fine_mean_rounding','coarse_mean_rounding','projection_rounding','stored_total')}
                for i,(a,b) in enumerate(zip(targets[:-1],targets[1:])):
                    j,h=spans[i]
                    if j==h:integral=(b-a)*average(e[j],e[j+1])
                    else:
                        integral=(e[j+1]-a)*average(e[j],e[j+1])+(b-e[h])*average(e[h],e[h+1])
                        if h>j+1:integral+=(e[h]-e[j+1])*average(e[j+1],e[h])
                    ideal=integral/(b-a);truth=average(a,b)
                    items=[ideal-truth,q(raw[i])-ideal,truth-q(coarse['c0'][i]),q(z['initial_projected'][i])-q(raw[i]),q(z['initial_projected'][i])-q(coarse['c0'][i])]
                    for name,value in zip(v,items):
                        v[name].append(value);assert abs(value-mp.mpf(vectors[f'n{n}']['vectors'][name][i]))<mp.mpf('1e-60')
                norms={name:mp.nstr(mp.sqrt(mp.fsum(q(m)*x*x for m,x in zip(mass,values))),60) for name,values in v.items()}
                expected=next(c for c in r['cases'] if c['cells']==n)['precision_results'][-1]['mass_norms']
                for name in norms:assert float(norms[name])==float(expected[name])
                evaluated.append({'precision':precision,'mass_norms':norms})
        receipts.append({'cells':n,'independent_partial_cell_plus_full_interior_integral':True,'evaluations':evaluated})
    # A bounded zero-whole-mean perturbation changes the mean over a cut cell.
    x=tt[1];j=bisect.bisect_right(ee,x)-1;l,h=ee[j],ee[j+1];width=h-l;amplitude=1/(4*width)
    whole=amplitude*((h-(l+h)/2)**2-(l-(l+h)/2)**2)/2
    partial=amplitude*(x-l)*(x-h)/2
    assert whole==0 and partial!=0 and amplitude*width/2==F(1,8)
    witness={'fine_cells':32768,'fine_cell':j,'whole_cell_integral_perturbation_exact':str(whole),
      'partial_cell_integral_perturbation_exact':str(partial),'perturbation':'q(mu)=(mu-midpoint)/(4*cell_width)',
      'base_scalar':'1/2','both_fields_within':[.375,.625],'same_full_cell_average_does_not_determine_cut_average':True,
      'only_an_information_witness_not_a_new_physical_state':True}
    save(out/'information-witness.json',witness)
    parent=ROOT/'runs/reconnect-becker-reference';old=read(parent/'results.json')
    assert old['positive_budget_components_absolute']==r['parent_positive_budget_components_unchanged']
    assert old['H_indicator_unchanged_formula']==r['parent_H_indicator_unchanged'] and not r['old_case_reclassified']
    for rel,h in read(parent/'sha256.json').items():assert digest(ROOT/rel)==h
    assert len(read(out/'preservation.json')['formal_manifests_verified'])==104
    save(out/'audit.json',{'passed':True,'first_attempt_passed':True,'elapsed_seconds':time.monotonic()-start,
      'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'auditor_sha256':digest(Path(__file__)),
      'independent_sinc_partial_cell_evaluations':receipts,'initial_only_terminal_arrays_not_parsed':True,
      'old_decisions_budgets_and_H_preserved':True,'reference_or_candidate_solves':0,'new_software_checks':0,'recorded_checks_total':500,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})
    print('Independent sinc/partial-cell100/120-digit accounting and exact information witness passed; no old decision changed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();audit(a.directory)
