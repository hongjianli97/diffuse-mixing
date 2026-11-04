"""Independent cell integration and a native-error masking counterexample."""
import argparse,json,math,unittest
import numpy as np
from t1_uniform_core import analytic_data,pure_labels,compare,TIMES
from material_union import block_index


def fixture(hidden=0.):
    base=4;cp=np.tile([.3,.7],8);data={};runs={};paths={}
    for N,delta in [(4,.06),(8,.015),(16,.004)]:
        idx=block_index(base,N//base);c0=cp[idx];m=np.full(N*N,1/(N*N));phi=c0-.5;lam=1.5-delta
        ref=np.array([.5+phi*math.exp(-1.5*t) for t in TIMES])
        symbol=np.array([.5+phi*math.exp(-lam*t) for t in TIMES])
        data[N]={'mass':m,'c0':c0,'reference':ref,'symbol':symbol,'discrete_lambda':lam,'continuum_lambda':1.5,'reference_evaluation_indicator_absolute':1e-14}
        for K in (4,8,16):
            states=np.array([.5+phi*math.exp(-lam*t) for t in np.arange(K+1)*(.05/K)])
            if N==16:states+=np.outer(np.arange(K+1)/K*hidden,np.tile([1.,-1.,1.,-1.],N*N//4))
            paths[N,K]={'states':states,'independent_states':states.copy()}
            runs[N,K]={'steps':[{'cumulative_machine_indicator_absolute':(i+1)*1e-16,'cumulative_tail_indicator_absolute':(i+1)*1e-18} for i in range(K)]}
    return data,runs,paths


class Checks(unittest.TestCase):
    def test_true_cell_integrals_and_continuum_H(self):
        N=4;d=analytic_data(N);a=pure_labels(N);x,w=np.polynomial.legendre.leggauss(12)
        means=[]
        for cx,cy in a:
            xx=cx+x/(2*N);yy=cy+x/(2*N)
            value=.5+.2*np.sin(2*math.pi*xx[:,None])*np.cos(2*math.pi*yy[None,:])
            means.append(float(w@value@w/4))
        np.testing.assert_allclose(d['c0'],means,rtol=0,atol=8*np.finfo(float).eps)
        point=.5+.2*np.sin(2*math.pi*a[:,0])*np.cos(2*math.pi*a[:,1])
        self.assertGreater(float(np.max(abs(point-d['c0']))),.01)
        for t,c in zip(TIMES,d['reference']):
            variance=float(d['mass']@((c-float(d['mass']@c))**2));E0=float(d['mass']@((d['c0']-.5)**2))
            self.assertAlmostEqual(1-variance/E0,1-math.exp(-16*math.pi**2*.02*t),places=14)

    def test_time_checkpoint_budget_and_native_mask_rejection(self):
        d,r,p=fixture();values=[v for q in d.values() for v in q.values() if isinstance(v,np.ndarray)]+[v for q in p.values() for v in q.values()]
        before=[v.tobytes() for v in values]
        for v in values:v.setflags(write=False)
        result,common=compare(d,r,p,base=4)
        self.assertTrue(result['T1_registered_case_passed'])
        self.assertTrue(all(row['raw_difference_absolute']==0 for row in result['time_controls']))
        self.assertAlmostEqual(result['per_grid_observations']['4'][-1]['native_error_native_denominator'],math.exp(-1.44*.05)-math.exp(-1.5*.05),places=14)
        self.assertEqual(before,[v.tobytes() for v in values])
        bad,_=compare(*fixture(hidden=.01),base=4);row=bad['adjacent_comparisons'][-1]
        self.assertEqual(row['common_decision'],'fine_better_resolvably')
        self.assertEqual(row['native_decision'],'fine_worse_resolvably')
        self.assertFalse(bad['T1_registered_case_passed'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
