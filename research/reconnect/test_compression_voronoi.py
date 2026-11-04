"""New pure compression adapter and complete retention of original local checks."""
import argparse,inspect,json,unittest
import numpy as np
from compression_probe import make_compression_samples
from compression_voronoi_core import sample_points,kernel_local_controls


class Checks(unittest.TestCase):
    def fixture(self):
        a=np.array([[.2,.2],[.7,.2],[.2,.7],[.7,.7]])
        return a,np.array([8,3,11,6]),1+.25*np.cos(2*np.pi*a[:,0])

    def test_original_producer_bitwise_irregular_times(self):
        a,ids,r=self.fixture();t=np.array([0.,.003,.019,.05]);got=sample_points(a,ids,r,t)
        expected,_,_=make_compression_samples(a,ids,np.array([1.,2.,3.,4.]),np.array([.2,.4,.6,.8]),r,t)
        np.testing.assert_array_equal(got['positions'],expected.positions);np.testing.assert_array_equal(got['rho'],expected.rho)
        np.testing.assert_array_equal(got['ids'],ids);np.testing.assert_array_equal(got['time'],t)

    def test_nested_records_and_nonconstant_intermediate_density(self):
        a,ids,r=self.fixture();fine=sample_points(a,ids,r,np.linspace(0.,.05,33))
        for K,stride in [(4,4),(8,2)]:
            old=sample_points(a,ids,r,np.linspace(0.,.05,2*K+1))
            for field in ('time','positions','rho'):np.testing.assert_array_equal(old[field],fine[field][::stride])
        self.assertGreater(np.ptp(fine['rho'][16]),.05)
        cosine=np.cos(2*np.pi*a[:,0]);J=1+.125*cosine
        np.testing.assert_allclose(fine['rho'][16]*J,r,rtol=0,atol=2e-16)
        # No forcing of endpoint density: equality follows from the original formula.
        np.testing.assert_array_equal(fine['rho'][-1],r/(1+.25*cosine))

    def test_invalid_and_forbidden_inputs_rejected(self):
        a,ids,r=self.fixture();t=np.array([0.,.05])
        self.assertEqual(list(inspect.signature(sample_points).parameters),['labels','ids','rho0','times'])
        for kw in ('mass','c0','J','F','reference'):
            with self.assertRaises(TypeError):sample_points(a,ids,r,t,**{kw:np.ones(4)})
        for times in ([0.,.05,.02],[0.,.051],[0.,0.],[-.01,.05],[0.,float('nan')]):
            with self.assertRaises(ValueError):sample_points(a,ids,r,times)
        with self.assertRaises(ValueError):sample_points(a,np.array([8,8,11,6]),r,t)
        with self.assertRaises(ValueError):sample_points(a,ids,-r,t)

    def test_original_controls_do_not_cancel_and_missing_controls_reject(self):
        m=np.array([1.,3.]);c=np.array([.25,.75]);mu=float(m@c/m.sum());sigma=float(np.sqrt(m@((c-mu)**2)))
        saved={};old={'cases':{}}
        for K in (4,8):
            states=np.broadcast_to(c,(K+1,2)).copy();direct=np.broadcast_to(c,(K,2)).copy();half=direct.copy()
            direct[0,0]+=2**-20;direct[1,0]-=2**-20;half[0,1]+=2**-21;half[1,1]-=2**-21
            rows=[]
            for i in range(K):rows.append({'direct_vs_sampled_relative_difference':float(np.sqrt(m@((direct[i]-c)**2)))/sigma,
              'two_half_vs_sampled_relative_difference':float(np.sqrt(m@((half[i]-c)**2)))/sigma})
            saved[f'k{K}_states']=states;saved[f'k{K}_direct_endpoints']=direct;saved[f'k{K}_two_half_endpoints']=half
            old['cases'][str(K)]={'steps':rows}
        initial=kernel_local_controls(saved,old,m,c,0);middle=kernel_local_controls(saved,old,m,c,1);final=kernel_local_controls(saved,old,m,c,2)
        self.assertEqual(initial['absolute_sum'],0);self.assertGreater(middle['absolute_sum'],0)
        self.assertEqual(middle['absolute_sum'],final['absolute_sum']);self.assertEqual(len(final['details']),24)
        self.assertFalse(final['original_local_checks_are_independent_complete_paths'])
        del saved['k4_two_half_endpoints']
        with self.assertRaises(KeyError):kernel_local_controls(saved,old,m,c,2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
