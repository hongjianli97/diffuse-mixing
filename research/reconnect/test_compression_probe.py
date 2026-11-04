"""Prescribed mass-conserving motion and independent frozen PDE controls."""
import unittest
import numpy as np
from compression_probe import map_x,make_compression_samples,material_point_fields,analytic_material_rate,compression_geometry,compression_gauss_reference
from smooth_density_probe import make_samples,analytic_reference
from positive_kernel_2d import build_periodic_exchange
from sparse_exchange_audit import audit_laplacian


class CompressionChecks(unittest.TestCase):
    def samples(self,N=8,alpha=.25):
        initial,r=make_samples(N,.25,.02)
        d,g,a=make_compression_samples(r['labels'],initial.ids,initial.mass,initial.c0,initial.rho[0],[0.,.05],alpha,.05)
        return initial,r,d,g,a

    def test_rho0_over_J_continuity_and_uniform_frozen_identity(self):
        initial,r,d,g,a=self.samples()
        np.testing.assert_array_equal(d.mass,initial.mass);np.testing.assert_array_equal(d.c0,initial.c0)
        np.testing.assert_allclose(d.rho*g['J'],np.broadcast_to(initial.rho[0],d.rho.shape),atol=3e-16,rtol=0.)
        np.testing.assert_array_equal(d.rho[1],1.)
        self.assertLess(a['material_continuity_max_residual'],2e-15);self.assertGreater(a['minimum_map_J'],.75)

    def test_material_face_balance_matches_independent_eulerian_Gauss(self):
        initial,r,d,g,a=self.samples();ref,split=analytic_material_rate(8,.25,.25,.02,d.mass,d.c0,r['labels'])
        q=compression_gauss_reference(8,.25,.25,.02,r['labels'],d.mass,16)
        np.testing.assert_allclose(ref,q['rate'],atol=4e-15)
        np.testing.assert_allclose(q['state'],d.c0,atol=5e-16);np.testing.assert_allclose(q['mass'],d.mass,atol=2e-17)
        np.testing.assert_array_equal(q['drift_rate'],0.)

    def test_material_flux_divergence_matches_full_point_PDE(self):
        a=np.array([.11,.29,.53,.81]);b=np.array([.17,.33,.49,.73]);eps=.25;D=.02;k=2*np.pi;step=1e-25
        for alpha in [.12,.25]:
            rho0,J,rho,c,lap,drift,total=material_point_fields(a,b,eps,alpha,D)
            def gradient_flux(x):return -D*.2*k*(1+eps*np.cos(k*x))/(1+alpha*np.cos(k*x))**2*np.sin(k*x)*np.cos(k*b)
            independent=np.imag(gradient_flux(a+1j*step))/step/rho0-D*k*k*(c-.5)
            np.testing.assert_allclose(total,independent,atol=5e-16,rtol=2e-15)
            if alpha==.12:self.assertGreater(np.max(abs(drift)),.001)

    def test_zero_deformation_recovers_existing_translation_reference(self):
        initial,r,d,g,a=self.samples(alpha=0.)
        np.testing.assert_array_equal(d.positions,initial.positions);np.testing.assert_array_equal(d.rho,initial.rho)
        ref,_=analytic_material_rate(8,.25,0.,.02,d.mass,d.c0,r['labels'])
        np.testing.assert_allclose(ref,r['rate'],atol=3e-15)

    def test_true_area_identity_and_node_center_offset_are_not_repaired(self):
        initial,r,d,g,a=self.samples();geometry=compression_geometry(8,r['labels'],.25,.05)
        np.testing.assert_allclose(geometry['true_cell_area_from_unwrapped_faces'],d.mass,atol=3e-17)
        self.assertGreater(np.max(abs(geometry['node_minus_geometric_center_x'])),1e-4)
        self.assertGreater(geometry['true_cell_width'].max()/geometry['true_cell_width'].min(),1.5)
        np.testing.assert_array_equal(d.positions[1,:,0],np.mod(geometry['unwrapped_node_x'],1.))

    def test_fixed_point_backend_positive_graph_and_zero_D_on_deformed_cloud(self):
        initial,r,d,g,a=self.samples();saved=d.positions.copy();L,p,diag=build_periodic_exchange(d,1,.02,.19,[1.,1.])
        self.assertTrue(audit_laplacian(d.mass,L)['graph_algebra_eligible']);np.testing.assert_array_equal(p['quadrature_area'],d.mass)
        np.testing.assert_array_equal(d.positions,saved)
        zero,pz,_=build_periodic_exchange(d,1,0.,.19,[1.,1.]);np.testing.assert_array_equal(zero.data,0.);np.testing.assert_array_equal(pz['conductances'],0.)

    def test_reference_map_J_is_the_derivative_of_the_given_periodic_map(self):
        a=np.array([.03,.18,.39,.74]);alpha=.25;t=.05;step=1e-25
        derivative=np.imag(map_x(a+1j*step,alpha,t))/step
        np.testing.assert_allclose(derivative,1+alpha*np.cos(2*np.pi*a),atol=3e-16)
        np.testing.assert_allclose(map_x(a+1,alpha,t)-map_x(a,alpha,t),1.,atol=3e-16)

    def test_nonphysical_map_or_unexplained_time_extrapolation_is_rejected(self):
        initial,r=make_samples(8,.25,.02)
        with self.assertRaises(ValueError):make_compression_samples(r['labels'],initial.ids,initial.mass,initial.c0,initial.rho[0],[0.,.05],1.,.05)
        with self.assertRaises(ValueError):make_compression_samples(r['labels'],initial.ids,initial.mass,initial.c0,initial.rho[0],[0.,.1],.25,.05)


if __name__=='__main__':unittest.main(verbosity=2)
