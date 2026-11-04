"""Independent algebra and physical time checks for the cached BE solver."""
import unittest
import numpy as np
from scipy.linalg import eigh,expm
from fixed_be_time import FixedBackwardEuler,integrate,spectral_backward_euler
from contact import laplacian
from experiment import step


class FixedBEChecks(unittest.TestCase):
    def test_cached_step_matches_original_banded_solver_over_repeated_steps(self):
        m=np.array([.1,.3,.2,.4]);g=np.array([.2,1.3,.5]);c0=np.array([.1,.9,.3,.6]);dt=.003
        cached=FixedBackwardEuler(m,g,dt);a=c0.copy();b=c0.copy()
        for _ in range(200):a=cached.advance(a);b=step(b,m,g,dt)
        np.testing.assert_allclose(a,b,rtol=0,atol=2e-13)

    def test_unequal_two_cell_actual_BE_matches_closed_transfer_factor(self):
        m=np.array([1.,3.]);g=np.array([2.]);c0=np.array([.2,.8]);T=.1;steps=512
        mean=m@c0/m.sum();decay=(1+T/steps*g[0]*(1/m[0]+1/m[1]))**(-steps)
        exact=mean+.6*decay*np.array([-.75,.25])
        actual,d=integrate(c0,m,g,T,steps)
        np.testing.assert_allclose(actual,exact,atol=2e-13)

    def test_stiff_modal_BE_and_stable_difference_match_independent_dense_evolution(self):
        m=np.array([.05,.2,.75]);g=np.array([.8,10.]);c0=np.array([1.,.2,0.]);L=laplacian(g);root=np.sqrt(m)
        vals,U=eigh(L/root[:,None]/root[None,:]);T=10.
        for steps in (1,8,512):
            with np.errstate(over='raise',invalid='raise'):
                predicted,delta=spectral_backward_euler(vals,U,m,c0,T,steps)
            direct,d=integrate(c0,m,g,T,steps);continuous=expm(-T*L/m[:,None])@c0
            np.testing.assert_allclose(predicted,direct,atol=3e-13)
            np.testing.assert_allclose(delta,direct-continuous,atol=3e-13)

    def test_modal_difference_has_correct_small_time_cancellation_behavior(self):
        m=np.array([1.,2.]);g=np.array([.5]);c0=np.array([0.,1.]);L=laplacian(g);root=np.sqrt(m)
        vals,U=eigh(L/root[:,None]/root[None,:])
        state,delta=spectral_backward_euler(vals,U,m,c0,1e-4,65536)
        # The true leading error is ~T²/N; it must not become a large
        # roundoff difference of two O(1) concentration states.
        self.assertLess(np.linalg.norm(delta),1e-12)
        self.assertGreater(np.linalg.norm(delta),1e-15)
        self.assertAlmostEqual(float(m@delta),0.,delta=1e-20)

    def test_cached_map_preserves_constants_mass_nonnegativity_and_energy_norm(self):
        m=np.array([.1,1.,.3,4.]);g=np.array([2.,.2,1.]);solver=FixedBackwardEuler(m,g,.03)
        P=np.column_stack([solver.advance(np.eye(4)[:,j]) for j in range(4)])
        np.testing.assert_allclose(P.sum(axis=1),1.,atol=2e-14)
        np.testing.assert_allclose(m@P,m,atol=2e-14)
        self.assertGreaterEqual(P.min(),0.)
        transformed=np.sqrt(m)[:,None]*P/np.sqrt(m)[None,:]
        self.assertLessEqual(np.linalg.svd(transformed,compute_uv=False).max(),1+1e-14)

    def test_zero_diffusion_and_zero_one_constants_are_preserved(self):
        m=np.array([.1,.2,.7]);c0=np.array([.1,.9,.3]);g=np.zeros(2)
        c,d=integrate(c0,m,g,.1,128);np.testing.assert_allclose(c,c0,atol=1e-14)
        for value in (0.,.37,1.):
            c,d=integrate(np.full(3,value),m,np.array([1.,2.]),.1,128)
            np.testing.assert_allclose(c,value,atol=2e-14)

    def test_actual_error_is_first_order_against_exact_exponential(self):
        m=np.array([1.,3.]);g=np.array([2.]);c0=np.array([.2,.8]);T=.1;L=laplacian(g)
        exact=expm(-T*L/m[:,None])@c0;errors=[]
        for n in (64,128,256):
            c,d=integrate(c0,m,g,T,n);errors.append(np.linalg.norm(c-exact))
        np.testing.assert_allclose(np.array(errors[:-1])/errors[1:],2.,rtol=.003)

    def test_reject_invalid_masses_rates_factorization_and_step_counts(self):
        for m,g,dt in [([1.,0.],[1.],.1),([1.,1.],[-1.],.1),([1.,1.],[1.],0.),([1.,1.],[1.],np.nan)]:
            with self.assertRaises(ValueError):FixedBackwardEuler(m,g,dt)
        with self.assertRaises(ValueError):integrate(np.array([0.,1.]),np.ones(2),np.ones(1),.1,0)
        with self.assertRaises(ValueError):integrate(np.array([0.,1.]),np.ones(2),np.ones(1),.1,1.5)


if __name__=='__main__':unittest.main(verbosity=2)
