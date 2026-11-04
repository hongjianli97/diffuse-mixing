"""Independent image reference, forcing identities and weak patch moments."""
import unittest
import numpy as np
from scipy.sparse import diags
from contact import Contact
from contact_step_reference import StepReference
from fixed_width_kernel import build_constant_exchange
from interface_error_forcing import scaled_forcing
from interface_layer_study import affine_weak_moments
from step_image_profile import HeatStepImages


class InterfaceLayerChecks(unittest.TestCase):
    def setUp(self):
        self.f = Contact(); self.ref = StepReference(self.f)
        self.images = HeatStepImages(); self.faces = np.linspace(-1., 1., 65)
        self.x = (self.faces[:-1]+self.faces[1:])/2
        self.V = np.diff(self.faces); self.m = self.f.mass(self.faces)
        self.a = np.where(self.x < 0, 1., 2.)/3

    def test_image_averages_match_independent_finite_interval_spectral_reference(self):
        for t in (1e-4, .001, .1, 1.):
            chi, _ = self.images.average_and_tau_derivative(self.faces, np.sqrt(t))
            np.testing.assert_allclose(2/3-self.a*chi, self.ref.average(self.faces, t), atol=1e-12, rtol=0.)

    def test_tau_derivative_matches_independent_spectral_face_fluxes(self):
        for t in (.001, .1, 1.):
            _, derivative = self.images.average_and_tau_derivative(self.faces, np.sqrt(t))
            flux = self.ref.flux(self.faces, t)
            expected = 2*np.sqrt(t)*(flux[:-1]-flux[1:])/self.m
            np.testing.assert_allclose(-self.a*derivative, expected, atol=1e-11, rtol=0.)

    def test_analytic_tau_zero_endpoint_is_finite_and_conservative(self):
        chi, derivative = self.images.average_and_tau_derivative(self.faces, 0.)
        np.testing.assert_array_equal(chi, np.sign(self.x))
        self.assertTrue(np.isfinite(derivative).all())
        self.assertEqual(np.count_nonzero(derivative), 2)
        self.assertLess(abs(float(self.V@derivative)), 1e-15)
        np.testing.assert_allclose(2/3-self.a*chi, self.ref.average(self.faces, 0.), atol=2e-16)

    def operators(self, rho_left=2., rho_right=1.):
        rho = np.where(self.x < 0, rho_left, rho_right); mass = rho*self.V
        L, ar, _ = build_constant_exchange(self.x, mass, self.V, rho, .02, .125, (-1., 1.))
        L0, _, _ = build_constant_exchange(self.x, self.V, self.V, np.ones_like(self.x), .02, .125, (-1., 1.))
        return -diags(1/mass)@L, -diags(1/self.V)@L0, mass, ar

    def test_forcing_identity_phase_support_and_individual_mass_conservation(self):
        Q, Q0, mass, ar = self.operators()
        for tau in (0., .001, .05, np.sqrt(.1)):
            chi, derivative = self.images.average_and_tau_derivative(self.faces, tau)
            forcing, ref, d = scaled_forcing(Q, Q0, mass, chi, derivative, tau, self.x,
                         ar['pair_i'], ar['pair_j'], ar['conductance'], 2., 1.)
            self.assertLess(d['algebra_closure_max_abs'], 1e-11)
            self.assertLess(abs(d['interface_mass_forcing']), 1e-13)
            self.assertLess(abs(d['bulk_mass_forcing']), 1e-13)
            np.testing.assert_array_equal(forcing[abs(self.x) >= .25, 0], 0.)

    def test_density_equal_interface_forcing_vanishes_identically(self):
        Q, Q0, mass, ar = self.operators(1., 1.)
        chi, derivative = self.images.average_and_tau_derivative(self.faces, .2)
        forcing, ref, d = scaled_forcing(Q, Q0, mass, chi, derivative, .2, self.x,
                            ar['pair_i'], ar['pair_j'], ar['conductance'], 1., 1.)
        np.testing.assert_array_equal(forcing[:, 0], 0.)
        self.assertLess(d['algebra_closure_max_abs'], 1e-11)

    def test_affine_patch_has_zero_weak_mass_but_quadratic_moment_of_order_h2(self):
        for h in (.03125, .015625):
            r = affine_weak_moments(h, .001, 1/3)
            self.assertLess(abs(r['mass_moment']), 1e-15)
            self.assertAlmostEqual(r['weighted_trace_residual'], -27/16*.001/h/3, places=13)
            self.assertAlmostEqual(r['second_moment']/r['second_moment_exact'], 1., places=11)

    def test_nested_average_reference_is_the_same_fixed_material_observable(self):
        fine = np.linspace(-1., 1., 257); widths = np.diff(fine)
        chi, _ = self.images.average_and_tau_derivative(fine, np.sqrt(.1))
        coarse = (widths*chi).reshape(64, 4).sum(axis=1)/self.V
        expected, _ = self.images.average_and_tau_derivative(self.faces, np.sqrt(.1))
        np.testing.assert_allclose(coarse, expected, atol=1e-13, rtol=0.)

    def test_invalid_reference_inputs_rejected(self):
        with self.assertRaises(ValueError):HeatStepImages(D=0.)
        with self.assertRaises(ValueError):HeatStepImages(images=0)
        with self.assertRaises(ValueError):self.images.average_and_tau_derivative(self.faces, -.1)
        with self.assertRaises(ValueError):self.images.average_and_tau_derivative(np.array([-1., -.1, .2, 1.]), .1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
