"""Independent checks; no missing Becker inputs or old metrics fabricated."""
from pathlib import Path
import json, unittest
import numpy as np
from material_fv import center_resistance, step, propagate


class Checks(unittest.TestCase):
    def test_resistance_variable_density(self):
        # rho=1/(1+x), D=.2 gives analytic integrated resistance.
        faces=np.array([0.,.1,.5,1.]); x=(faces[:-1]+faces[1:])/2
        exact=.2/(np.diff(x)+np.diff(x*x)/2)
        g,e=center_resistance(faces,lambda x,t:1/(1+x),.2,0.)
        np.testing.assert_allclose(g,exact,rtol=1e-12)
        self.assertLess(e.max(),1e-10)
        # c linear in resistance coordinate gives constant interior flux.
        c=(x+x*x/2)/.2
        np.testing.assert_allclose(g*np.diff(c),1.,rtol=1e-12)

    def test_unequal_mass_exact_backward_euler(self):
        m=np.array([1.,3.]); c=np.array([.1,.9]); g=np.array([2.])
        mean=m@c/m.sum(); difference=(c[1]-c[0])/(1+.3*g[0]*(1/m[0]+1/m[1]))
        expected=[mean-m[1]/m.sum()*difference,mean+m[0]/m.sum()*difference]
        np.testing.assert_allclose(step(c,m,g,.3),expected,rtol=1e-14)

    def test_moving_geometry_invariants(self):
        m=np.array([.1,.4,.2,.3]); c=np.array([0.,1.,.2,.8]); times=np.linspace(0,.1,65)
        faces=lambda t:np.array([0.,.1,.4,.7,1.])*(1-.2*t)
        for initial in (c,np.full(4,.37)):
            h=propagate(initial,m,times,faces,lambda x,t:1/(1-.2*t),.02)
            np.testing.assert_allclose(h@m,initial@m,rtol=1e-13,atol=1e-14)
            self.assertGreaterEqual(h.min(),initial.min()-1e-13)
            self.assertLessEqual(h.max(),initial.max()+1e-13)
            mean=m@initial/m.sum(); energy=((h-mean)**2)@m/2
            self.assertTrue(np.all(np.diff(energy)<=1e-13))
        h=propagate(c,m,times,faces,lambda x,t:1.,0.)
        np.testing.assert_allclose(h,np.broadcast_to(c,h.shape),atol=1e-14)

    def test_temporal_order_against_exact_semidiscrete(self):
        m=np.array([1.,3.]); c0=np.array([0.,1.]); g=np.array([2.]); T=.1
        mean=m@c0/m.sum(); d=np.exp(-g[0]*(1/m[0]+1/m[1])*T)
        exact=np.array([mean-.75*d,mean+.25*d]); errors=[]
        for k in (128,256,512):
            c=c0.copy()
            for _ in range(k):c=step(c,m,g,T/k)
            errors.append(float(np.sqrt(m@((c-exact)**2)/m.sum())))
        ratios=np.array(errors[:-1])/errors[1:]
        self.assertTrue(np.all((ratios>1.98)&(ratios<2.02)))
        TEMPORAL.update(steps=[128,256,512],mass_weighted_rms_error=errors,ratios=ratios.tolist(),reference='exact two-cell semidiscrete exponential; not Becker')

    def test_reject_invalid_inputs(self):
        for faces in ([0.,0.,1.],[0.,np.nan,1.]):
            with self.assertRaises(ValueError):center_resistance(faces,lambda x,t:1.,.02,0.)
        with self.assertRaises(ValueError):step([0.,1.],[1.,-1.],[1.],.1)
        with self.assertRaises(ValueError):step([0.,1.],[1.,1.],[-1.],.1)
        with self.assertRaises(ValueError):center_resistance([0.,1.,2.],lambda x,t:-1.,.02,0.)


TEMPORAL={}
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (output/'checks.json').write_text(json.dumps({'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'temporal':TEMPORAL,'becker_comparison':'blocked: original inputs absent','fluid_orc_enabled':False},indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
