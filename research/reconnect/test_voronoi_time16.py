"""Small independent controls for pure sampling and sixteen-step orchestration."""
from fractions import Fraction as F
import argparse,inspect,json,math,unittest
import numpy as np
from scipy.linalg import expm
from shear_probe import make_shear_samples
from voronoi_time_action import operators
from voronoi_time16_core import sample_points,ordered_path


class Checks(unittest.TestCase):
    def fixture(self):
        a=np.array([[.2,.2],[.7,.2],[.2,.7],[.7,.7]])
        return a,np.array([8,3,11,6]),np.array([.8,1.1,.9,1.2])

    def test_same_original_producer_irregular_times(self):
        a,ids,r=self.fixture();times=np.array([0.,.003,.019,.05])
        actual=sample_points(a,ids,r,times)
        expected,_,_=make_shear_samples(a,ids,np.ones(4),np.array([.2,.4,.6,.8]),r,times)
        np.testing.assert_array_equal(actual['positions'],expected.positions)
        np.testing.assert_array_equal(actual['rho'],expected.rho)
        self.assertEqual(list(inspect.signature(sample_points).parameters),['labels','ids','rho0','times'])
        with self.assertRaises(TypeError):sample_points(a,ids,r,times,c0=np.ones(4))

    def test_nested_original_time_records(self):
        a,ids,r=self.fixture();fine=sample_points(a,ids,r,np.linspace(0.,.05,33))
        for K,stride in [(4,4),(8,2)]:
            old=sample_points(a,ids,r,np.linspace(0.,.05,2*K+1))
            for key in ['time','positions','rho']:np.testing.assert_array_equal(fine[key][::stride],old[key])
        with self.assertRaises(ValueError):sample_points(a,ids,r,np.array([0.,.05,.02]))

    def test_sixteen_steps_unequal_mass_analytic_and_zero_D(self):
        m=np.array([1.,3.]);c=np.array([.2,.8]);g=F(2,5);op=operators(m,np.array([[0,1]]),[g])
        run,path=ordered_path(m,c,lambda k:op)
        mean=float(m@c/m.sum())
        for k in [0,8,16]:
            expected=mean+(c-mean)*math.exp(-float(g)*(1/m[0]+1/m[1])*.05*k/16)
            np.testing.assert_allclose(path['states'][k],expected,atol=2e-14,rtol=0)
            np.testing.assert_allclose(path['independent_states'][k],expected,atol=2e-14,rtol=0)
        self.assertEqual(len(run['steps']),16)
        zero=operators(m,np.array([[0,1]]),[F(0)])
        _,raw=ordered_path(m,c,lambda k:zero)
        np.testing.assert_array_equal(raw['states'],np.broadcast_to(c,(17,2)))

    def test_sixteen_step_chronological_noncommuting_product(self):
        m=np.array([1.,2.,3.]);c=np.array([.1,.4,.9]);edges=np.array([[0,1],[1,2]])
        a=operators(m,edges,[F(1),F(0)]);b=operators(m,edges,[F(0),F(2)])
        Qa=np.array([[-1.,1,0],[.5,-.5,0],[0,0,0]])
        Qb=np.array([[0.,0,0],[0,-1,1],[0,2/3,-2/3]])
        expected=expm(.025*Qb)@expm(.025*Qa)@c
        _,path=ordered_path(m,c,lambda k:a if k<8 else b)
        np.testing.assert_allclose(path['states'][-1],expected,atol=2e-14,rtol=0)
        np.testing.assert_allclose(path['independent_states'][-1],expected,atol=2e-14,rtol=0)
        self.assertGreater(np.max(abs(expected-expm(.025*Qa)@expm(.025*Qb)@c)),1e-7)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:
        json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
