"""Independent early-layer, series-tail and identifiability checks."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc
from contact import Contact,mesh
from contact_step_reference import StepReference,right_phase_inventory_bounds


class ContactStepChecks(unittest.TestCase):
    def test_initial_step_exact_mass_average_and_projection_loss(self):
        f=Contact();r=StepReference(f);e=mesh(32,'cut');m=f.mass(e);c0=r.average(e,0.)
        idx=np.flatnonzero((e[:-1]<0)&(e[1:]>0))[0]
        self.assertAlmostEqual(c0[idx],6/7,places=14)
        self.assertAlmostEqual(float(m@c0),2.,places=14)
        E=.5*m@((c0-r.mean)**2);loss=1/3-E
        self.assertAlmostEqual(float(loss),(2/32)/7,places=14)
        self.assertAlmostEqual(float((f.rho_right*(np.maximum(e[1:],0)-np.maximum(e[:-1],0)))@c0),2*(2/32)/7,places=14)

    def test_spectral_tail_bound_dominates_positive_time_tail(self):
        r=StepReference(Contact());e=mesh(64,'cut');t=.001;K=64
        bound=r.tail_bound(t,K)
        self.assertLess(np.max(abs(r.average(e,t,K)-r.average(e,t,4*K))),bound)
        selected=r.terms_for(1e-4)
        self.assertLess(r.tail_bound(1e-4,selected),1e-13)
        a=np.array([-.01,-.003,.003,.01])
        self.assertLess(np.max(abs(r.scalar(a,t,K)-r.scalar(a,t,4*K))),bound)

    def test_early_series_against_independent_erfc_layer_and_flux(self):
        f=Contact();r=StepReference(f);t=1e-4;x=np.array([-.01,-.003,0.,.003,.01]);w=2*np.sqrt(f.D*t)
        expected=np.where(x<0,1-(1-r.mean)*erfc(-x/w),r.mean*erfc(x/w))
        np.testing.assert_allclose(r.scalar(x,t),expected,atol=2e-13,rtol=0)
        flux=f.rho_left*f.rho_right/(f.rho_left+f.rho_right)*np.sqrt(f.D/(np.pi*t))
        self.assertAlmostEqual(float(r.flux(np.array(0.),t)),float(flux),places=11)

    def test_finite_domain_mass_and_neumann_endpoints_at_late_time(self):
        f=Contact();r=StepReference(f);e=mesh(64,'cut');m=f.mass(e)
        for t in (.001,.1,20.):
            self.assertAlmostEqual(float(m@r.average(e,t)),2.,places=12)
            self.assertLess(np.max(abs(r.flux(np.array([-1.,1.]),t))),1e-11)
            self.assertAlmostEqual(float(r.scalar(np.array(0.),t)),2/3,places=14)
        np.testing.assert_allclose(r.average(e,1e5),2/3,atol=1e-13)
        np.testing.assert_array_equal(StepReference(Contact(D=0.)).average(e,.1),r.initial_average(e))

    def test_image_inventory_matches_spectral_integral_and_its_flux_derivative(self):
        f=Contact();r=StepReference(f)
        for t in (.001,.1,20.):
            inv,tail,n=r.right_inventory(0.,t)
            spec=f.rho_right*r.average(np.array([0.,1.]),t)[0]
            self.assertAlmostEqual(inv,float(spec),delta=2e-12)
        t=.01;eps=1e-6
        derivative=(r.right_inventory(0.,t+eps)[0]-r.right_inventory(0.,t-eps)[0])/(2*eps)
        self.assertAlmostEqual(derivative,float(r.flux(np.array(0.),t)),delta=1e-8)

    def test_tiny_image_inventory_positive_without_reference_clipping(self):
        f=Contact();r=StepReference(f);start=(2/32)/3;t=1e-4
        inv,tail,n=r.right_inventory(start,t)
        # Physical erfc tail is small but positive; independent scaled
        # quadrature avoids loss of tiny absolute values in its integrator.
        w=2*np.sqrt(f.D*t)
        scaled=quad(lambda x:1e50*r.mean*(erfc(x/w)+erfc((2-x)/w)),start,1,epsabs=1e-9)[0]/1e50
        self.assertGreater(inv,0.);self.assertGreater(inv,tail)
        self.assertAlmostEqual(inv/scaled,1.,places=10)

    def test_inventory_identifiability_bounds_are_not_a_uniform_reconstruction(self):
        m=np.array([3.]);ml=np.array([2.]);mr=np.array([1.]);c=np.array([2/3])
        lower,upper=right_phase_inventory_bounds(c,m,ml,mr)
        self.assertEqual(lower,0.);self.assertEqual(upper,1.)
        # left/right values (1,0) and (.5,1) have identical total mean.
        self.assertEqual((2*1+1*0)/3,(2*.5+1*1)/3)
        self.assertNotEqual(float(mr@c),lower)
        self.assertNotEqual(float(mr@c),upper)

    def test_underflowed_positive_image_tail_is_not_reported_as_exact_zero(self):
        r=StepReference(Contact());t=1e-4
        log_bound=r.image_log_tail_bound(t,1)
        self.assertLess(log_bound,np.log(np.nextafter(0.,1.)))
        bound=r.image_tail_bound(t,1)
        self.assertGreater(bound,0.)
        self.assertGreater(np.log(bound),log_bound)
        self.assertGreater(r.right_inventory(.01,t)[1],0.)

    def test_initial_step_reference_rejects_invalid_inputs(self):
        r=StepReference(Contact())
        with self.assertRaises(ValueError):StepReference(Contact(),left=1.1)
        with self.assertRaises(ValueError):r.average([0.,0.,1.],.1)
        with self.assertRaises(ValueError):r.average([-1.,1.],-.1)
        with self.assertRaises(ValueError):r.right_inventory(-.1,.1)
        with self.assertRaises(ValueError):r.terms_for(0.)


if __name__=='__main__':unittest.main(verbosity=2)
