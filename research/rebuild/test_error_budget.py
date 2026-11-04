"""Manufactured and independent dense checks of error attribution."""
import unittest
import numpy as np
from error_budget import budget,resolvent,record_reference
from reference import solve


class ErrorBudgetChecks(unittest.TestCase):
    def test_multi_rhs_resolvent_against_dense_unequal_mass_system(self):
        m=np.array([.2,.7,.1]);g=np.array([.8,.3]);dt=.07
        L=np.array([[g[0],-g[0],0.],[-g[0],g.sum(),-g[1]],[0.,-g[1],g[1]]])
        rhs=np.array([[.1,.3],[.8,-.1],[.2,.5]])
        exact=np.linalg.solve(np.diag(m)+dt*L,m[:,None]*rhs)
        np.testing.assert_allclose(resolvent(rhs,m,g,dt),exact,rtol=2e-14)
        with self.assertRaises(ValueError):resolvent(np.zeros((2,2)),m,g,dt)

    def test_exact_two_cell_transient_has_only_time_error_and_first_order(self):
        m=np.array([1.,3.]);c0=np.array([.2,.8]);mean=m@c0/m.sum();lam=1/m[0]+1/m[1]
        v=np.array([-.75,.25]);norms=[]
        for steps in (64,128,256):
            t=np.linspace(0,.1,steps+1);g=(.6+.3*t[:-1])[:,None]
            d=.6*np.exp(-lam*(.6*t+.15*t*t))
            r=mean+d[:,None]*v
            f=(-lam*(.6+.3*t)*d)[:,None]*v
            s,a=budget(m,c0,t,r,f,g,g)
            self.assertLess(s['contribution_norms']['average_closure'],1e-14)
            self.assertEqual(s['contribution_norms']['center_scale'],0.)
            self.assertLess(s['contribution_norms']['initial_projection'],1e-14)
            d_be=.6*np.prod(1/(1+np.diff(t)*lam*g[:,0]))
            np.testing.assert_allclose(a['numerical_c'][-1],mean+d_be*v,rtol=3e-14)
            norms.append(s['contribution_norms']['time_remainder'])
        np.testing.assert_allclose(np.array(norms[:-1])/norms[1:],2.,rtol=.003)

    def test_initial_projection_difference_is_not_discarded(self):
        m=np.array([1.,3.]);c0=np.array([.2,.8]);t=np.linspace(0,.1,9)
        r=np.tile(c0-.1,(len(t),1));f=np.zeros_like(r);g=np.zeros((len(t)-1,1))
        s,a=budget(m,c0,t,r,f,g,g)
        np.testing.assert_allclose(a['contributions'][:,:,0],.1,rtol=1e-14)
        np.testing.assert_allclose(a['contributions'][:,:,1:],0.,atol=1e-14)
        self.assertAlmostEqual(s['terminal_normalized_error'],s['contribution_norms']['initial_projection'],places=13)

    def test_stationary_manufactured_reference_has_only_spatial_defects(self):
        m=np.array([.2,.8]);c0=np.array([.1,.9]);t=np.linspace(0,.1,17)
        r=np.tile(c0,(len(t),1));f=np.zeros_like(r)
        g=np.full((16,1),.4);ga=np.full((16,1),.7)
        s,a=budget(m,c0,t,r,f,g,ga)
        self.assertEqual(s['contribution_norms']['time_remainder'],0.)
        self.assertEqual(s['contribution_norms']['initial_projection'],0.)
        self.assertLess(s['average_center_cross_term'],0.)
        np.testing.assert_allclose(a['contributions'][-1].sum(axis=1),a['numerical_c'][-1]-c0,atol=1e-14)

    def test_noncommuting_three_cell_case_matches_ordered_dense_product(self):
        m=np.array([.2,.7,.1]);c0=np.array([.2,.6,.9]);t=np.linspace(0,.1,19)
        r=np.column_stack((.3+.1*t,.6-.02*t,.7-.1*t*t))
        f=np.column_stack((np.full(len(t),.1),np.full(len(t),-.02),-.2*t))
        g=np.column_stack((1+t[:-1],.1+.4*t[:-1]))
        ga=np.column_stack((.8+.7*t[:-1],.2-.3*t[:-1]))
        s,a=budget(m,c0,t,r,f,g,ga)
        dense=c0.copy();Ms=np.diag(m)
        matrices=[]
        for dt,(g0,g1) in zip(np.diff(t),g):
            L=np.array([[g0,-g0,0.],[-g0,g0+g1,-g1],[0.,-g1,g1]])
            P=np.linalg.solve(Ms+dt*L,Ms);matrices.append(P);dense=P@dense
        self.assertGreater(np.linalg.norm(matrices[0]@matrices[-1]-matrices[-1]@matrices[0]),1e-7)
        np.testing.assert_allclose(a['numerical_c'][-1],dense,atol=2e-14)
        np.testing.assert_allclose(a['contributions'][-1].sum(axis=1),dense-r[-1],atol=2e-14)
        self.assertLess(s['max_history_reconstruction_residual'],1e-12)

    def test_recording_preserves_existing_reference_and_projected_derivative(self):
        class Uniform:
            total=2.
            def rho_mu(self,mu,t):return np.ones_like(mu)
        coarse=np.array([0.,.17,.53,1.14,2.])
        history,d=record_reference(Uniform(),64,128,coarse,samples=32)
        e,c0,c,checks=solve(Uniform(),64,128)
        np.testing.assert_array_equal(history['fine_final'],c)
        self.assertLess(d['projection_divergence_identity_max_residual'],1e-13)
        self.assertEqual(history['c'].shape,(33,4))


if __name__=='__main__':unittest.main(verbosity=2)
