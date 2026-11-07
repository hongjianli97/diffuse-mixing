"""Independent80/100-digit reconstruction of joint budgets and time descent."""
from fractions import Fraction as F
import numpy as np
import mpmath as mp


def audit_budgets(data,states,independent,mt,result):
    reports=[]
    for digits in (80,100):
        with mp.workdps(digits):
            f=lambda x:mp.mpf(F(float(x)).numerator)/F(float(x)).denominator
            vec=lambda a:list(map(f,a))
            m,c=vec(data['mass']),vec(data['c0']);M=mp.fsum(m)
            mean=lambda u:mp.fsum(w*x for w,x in zip(m,u))/M
            def variance(u):
                mu=mean(u);return mp.fsum(w*(x-mu)**2 for w,x in zip(m,u))
            V=variance(c);sigma=mp.sqrt(V);H=lambda u:1-variance(u)/V
            sub=lambda a,b:[x-y for x,y in zip(a,b)]
            norm=lambda a:mp.sqrt(mp.fsum(w*x*x for w,x in zip(m,a)))
            hm=lambda e:(2*mp.sqrt(M)*max(1,max(map(abs,c)))*e+e*e)/V
            U={key:[vec(a) for a in value] for key,value in states.items()}
            I={key:[vec(a) for a in value] for key,value in independent.items()}
            R={key:[vec(a) for a in data[key]] for key in ('reference','reference_space_coarse','reference_space_fine','reference_time_check')}
            numerical={(s,K,k):(f(mt[s,K][k]),norm(sub(U[s,K][k],I[s,K][k])),abs(H(U[s,K][k])-H(I[s,K][k])))
                       for s in ('oracle','ledger') for K in (16,32,64) for k in (1,2)}
            maximum=mp.mpf(0)
            def close(x,y):
                nonlocal maximum
                err=abs(x-f(y));maximum=max(maximum,err)
                assert err<=64*np.finfo(float).eps*max(1,abs(x))
            decide=lambda x,b:'error_larger_resolvably' if x>b else ('error_smaller_resolvably' if x < -b else 'unresolved')
            close(sigma,result['common_sigma64']);rows=[];time_rows=[]
            for row in result['source_comparisons']:
                k=round(row['time']/.025);ra,rb,rr=(R[key][k] for key in ('reference_space_coarse','reference_space_fine','reference'))
                cp={'four_reference_space':4*norm(sub(ra,rb)),'four_reference_time':4*norm(sub(rb,rr)),
                    'two_initial_bridge':2*norm(sub(R['reference'][0],c)),'same_N64_projection_identity':mp.mpf(0)}
                hp={'two_reference_space_same_basis':2*abs(H(ra)-H(rb)),'two_reference_time_same_basis':2*abs(H(rb)-H(rr)),
                    'two_initial_bridge':2*abs(H(R['reference'][0])-H(c)),'same_N64_projection_identity':mp.mpf(0)}
                total=mp.mpf(0)
                for s in ('ledger','oracle'):
                    cp[s+'_32_64']=norm(sub(U[s,32][k],U[s,64][k]));hp[s+'_32_64']=abs(H(U[s,32][k])-H(U[s,64][k]))
                    for K in (32,64):
                        machine,delta,dh=numerical[s,K,k];total+=machine
                        cp[f'{s}_e{K}']=machine+delta;hp[f'{s}_complete_sequence_H{K}']=dh
                hp['both_machine_tail_to_H']=hm(total)
                for key,value in cp.items():close(value,row['concentration_parts_absolute'][key])
                for key,value in hp.items():close(value,row['H_parts'][key])
                bc,bh=mp.fsum(cp.values())/sigma,mp.fsum(hp.values())
                gc=(norm(sub(U['ledger',64][k],rr))-norm(sub(U['oracle',64][k],rr)))/sigma
                gh=abs(H(U['ledger',64][k])-H(rr))-abs(H(U['oracle',64][k])-H(rr))
                for x,key in ((bc,'concentration_indicator'),(bh,'H_indicator'),(gc,'concentration_error_change'),(gh,'H_error_change')):close(x,row[key])
                assert decide(gc,bc)==row['concentration_decision'] and decide(gh,bh)==row['H_decision']
                rows.append({'time':row['time'],'concentration_decision':decide(gc,bc),'H_decision':decide(gh,bh),
                    'concentration_indicator':mp.nstr(bc,65),'H_indicator':mp.nstr(bh,65)})
            for row in result['time_differences']:
                s=row['source'];k=round(row['time']/.025);u,v,w=(U[s,K][k] for K in (16,32,64))
                dc=[norm(sub(u,v)),norm(sub(v,w))];dh=[abs(H(u)-H(v)),abs(H(v)-H(w))]
                ec=mp.fsum(weight*sum(numerical[s,K,k][:2]) for K,weight in ((16,1),(32,2),(64,1)))
                eh=mp.fsum(weight*numerical[s,K,k][2] for K,weight in ((16,1),(32,2),(64,1)))+hm(
                    mp.fsum(weight*numerical[s,K,k][0] for K,weight in ((16,1),(32,2),(64,1))))
                for x,key in ((dc[0],'state_difference_16_32'),(dc[1],'state_difference_32_64'),(dh[0],'H_difference_16_32'),
                              (dh[1],'H_difference_32_64'),(ec,'concentration_decrease_indicator'),(eh,'H_decrease_indicator')):close(x,row[key])
                assert (dc[0]-dc[1]>ec)==row['concentration_decrease_resolved']
                assert (dh[0]-dh[1]>eh)==row['H_decrease_resolved']
                time_rows.append({'source':s,'time':row['time'],'C':bool(dc[0]-dc[1]>ec),'H':bool(dh[0]-dh[1]>eh)})
            reports.append({'digits':digits,'maximum_float_scalar_difference':float(maximum),'rows':rows,'time_rows':time_rows})
    assert reports[0]['rows']==reports[1]['rows'] and reports[0]['time_rows']==reports[1]['time_rows']
    return reports
