"""Checks that spatial refinement changes resolution, not the physical problem."""
import unittest
import numpy as np
from becker import Becker,initial
from experiment import conductance,step
from error_budget import apply_q
from spatial_study import normalized_error,empirical_order
from flux_diagnosis import METHODS


class SpatialChecks(unittest.TestCase):
    def test_nested_material_cells_preserve_mass_and_exact_initial_averages(self):
        f=Becker();previous=None
        for n in (32,64,128):
            labels=np.linspace(-1,1,n+1);edges=f.mu(labels);m=np.diff(edges);c0=initial(edges)
            self.assertAlmostEqual(float(m.sum()),f.total,places=13)
            self.assertAlmostEqual(float(m@c0),.5*f.total,places=13)
            np.testing.assert_allclose(f.x_mu(edges,0),labels,rtol=0,atol=2e-14)
            if previous is not None:
                pe,pm,pc=previous
                np.testing.assert_array_equal(edges[::2],pe)
                np.testing.assert_allclose(m.reshape(-1,2).sum(axis=1),pm,rtol=2e-14)
                restricted=(m*c0).reshape(-1,2).sum(axis=1)/pm
                np.testing.assert_allclose(restricted,pc,rtol=2e-14)
            previous=edges,m,c0

    def test_error_norm_uses_initial_mass_fluctuation_and_preserves_mass_scale(self):
        m=np.array([1.,3.]);c0=np.array([.1,.9]);d=np.array([.03,-.01])
        # Mean=.7, initial fluctuation norm squared=.48, error norm squared=.0012.
        self.assertAlmostEqual(normalized_error(m,c0,d),.05,places=14)
        self.assertAlmostEqual(normalized_error(m*13,c0,d),.05,places=14)
        with self.assertRaises(ValueError):normalized_error(m,np.ones(2),d)

    def test_all_center_definitions_reduce_to_second_order_uniform_heat_operator(self):
        class Uniform:
            B=j=u1=1.
            def rho_mu(self,mu,t):return np.ones_like(mu)
            def x_mu(self,mu,t):return mu-1
            def z_at_xi(self,xi):return np.asarray(xi)
        f=Uniform();D=.02;T=.1;errors=[]
        for n in (16,32,64):
            e=np.linspace(0,2,n+1);m=np.diff(e);c0=initial(e)
            rate=4*D/m[0]**2*np.sin(np.pi/(2*n))**2
            for method in METHODS:
                g=conductance(f,e,0.,D,method)
                np.testing.assert_allclose(g,D/m[0],rtol=2e-14)
                np.testing.assert_allclose(apply_q(c0,m,g),-rate*(c0-.5),atol=2e-14)
            semi=.5+(c0-.5)*np.exp(-rate*T)
            continuum=.5+(c0-.5)*np.exp(-D*(np.pi/2)**2*T)
            errors.append(normalized_error(m,c0,semi-continuum))
        for a,b in zip(errors[:-1],errors[1:]):self.assertGreater(empirical_order(a,b),1.99)

    def test_zero_diffusion_preserves_scalar_for_all_material_resolutions(self):
        f=Becker()
        for n in (32,64,128):
            e=f.mu(np.linspace(-1,1,n+1));c0=initial(e)
            for method in METHODS:
                # Exercise the physical conductance and existing FV core.
                # The older experiment.candidate diagnostic computes 0/0
                # for a relative quadrature check at D=0; it is not used here.
                c=c0.copy()
                for t in np.arange(8)*.1/8:
                    g=conductance(f,e,t,0.,method)
                    np.testing.assert_array_equal(g,0.)
                    c=step(c,np.diff(e),g,.1/8)
                np.testing.assert_allclose(c,c0,atol=3e-15)


if __name__=='__main__':unittest.main(verbosity=2)
