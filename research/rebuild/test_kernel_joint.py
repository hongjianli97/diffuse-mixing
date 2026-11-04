"""Checks of sparse pair completeness, physics identities and fixed-h limits."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from positive_kernel_1d import build_exchange, slope_over_radius
from fixed_width_kernel import build_constant_exchange
from kernel_joint_study import subdivide, evolve


class JointKernelChecks(unittest.TestCase):
    def test_sparse_equals_prior_all_pairs_with_nonuniform_nodes_and_images(self):
        # Independent old dense pair enumeration exercises both reflected
        # endpoints and nonuniform physical volumes without a fitted solution.
        faces = np.r_[np.linspace(-1, 0, 19), np.linspace(0, 1, 26)[1:]]
        V = np.diff(faces); x = (faces[:-1]+faces[1:])/2
        rho = np.where(x < 0, 2., 1.); m = rho*V; h = .13
        L, a, d = build_constant_exchange(x, m, V, rho, .02, h, (-1., 1.))
        old, _ = build_exchange(x, m, V, rho, .02, np.full_like(x, h), (-1., 1.))
        np.testing.assert_allclose(L.toarray(), old['L'], rtol=3e-14, atol=1e-14)
        np.testing.assert_allclose(a['moment_first'], old['moment_first'], atol=1e-14)
        np.testing.assert_allclose(a['moment_diffusivity'], old['moment_diffusivity'], atol=1e-14)
        self.assertGreater(d['cross_contact_edges'], 0)

    def test_Neumann_uniform_cosine_has_independent_lattice_decay_rate(self):
        n = 128; faces = np.linspace(-1., 1., n+1)
        V = np.diff(faces); x = (faces[:-1]+faces[1:])/2; h = .03125; D = .02; k = np.pi/2
        L, a, d = build_constant_exchange(x, V, V, np.ones(n), D, h, (-1., 1.))
        r = np.arange(1, 5)*V[0]
        rate = 4*D*V[0]*np.sum(slope_over_radius(r, h)*(1-np.cos(k*r)))
        avg = np.cos(k*(x+1))*np.sinc(V/4)
        np.testing.assert_allclose(-(L@avg)/V, -rate*avg, atol=2e-12, rtol=0.)
        self.assertLess(d['interior_first_moment_max_abs'], 1e-12)

    def test_raw_propagator_mass_positive_energy_and_independent_evolution(self):
        faces = np.linspace(-1., 1., 49); V = np.diff(faces); x = (faces[:-1]+faces[1:])/2
        rho = np.where(x < 0, 2., 1.); m = rho*V
        L, a, d = build_constant_exchange(x, m, V, rho, .02, .12, (-1., 1.))
        P = expm(-.1*L.toarray()/m[:, None])
        self.assertGreaterEqual(P.min(), 0.)
        np.testing.assert_allclose(P.sum(axis=1), 1., atol=3e-13)
        np.testing.assert_allclose(m@P, m, atol=3e-13)
        c0 = (x < 0).astype(float)
        c, checks = evolve(L, m, c0, .1, full_eigen_check=True)
        np.testing.assert_allclose(c, P@c0, atol=3e-13)
        self.assertLess(checks['final_half_variance'], checks['initial_half_variance'])

    def test_D0_and_physical_density_scaling(self):
        faces = np.linspace(-1., 1., 33); V = np.diff(faces); x = (faces[:-1]+faces[1:])/2
        rho = np.where(x < 0, 2., 1.); m = rho*V
        L, _, _ = build_constant_exchange(x, m, V, rho, .02, .15, (-1., 1.))
        scaled, _, _ = build_constant_exchange(x, 7*m, V, 7*rho, .02, .15, (-1., 1.))
        np.testing.assert_allclose(scaled.toarray(), 7*L.toarray(), rtol=3e-14, atol=1e-14)
        zero, _, _ = build_constant_exchange(x, m, V, rho, 0., .15, (-1., 1.))
        self.assertEqual(zero.nnz, 0)

    def test_subdivision_preserves_original_faces_and_exact_phase_initial_mass(self):
        parent = np.array([-1., -.2, 0., .13, 1.])
        fine = subdivide(parent, 4)
        self.assertTrue(np.all(np.isin(parent, fine)))
        rho = np.where(parent[:-1] < 0, 2., 1.)
        m = rho*np.diff(parent)
        self.assertEqual(len(fine), 4*len(m)+1)
        weights = np.diff(fine)/np.repeat(np.diff(parent), 4)
        np.testing.assert_allclose(np.add.reduceat(np.repeat(m, 4)*weights, np.arange(0, 16, 4)), m, atol=1e-15)

    def test_constant_width_continuum_odd_moment_and_quadratic_action(self):
        # Integrate an affine and a quadratic field independently of node
        # enumeration. These targets motivate the fixed-h quadrature control.
        h = .03125
        # Integrate dimensionless coordinates, so the absolute quadrature
        # tolerance concerns an O(1) integrand instead of an O(1/h) odd one.
        odd = 2/h*quad(lambda u: float(slope_over_radius(u, 1.))*u, -2., 2., epsabs=1e-13)[0]
        second = quad(lambda r: float(slope_over_radius(r, h))*r*r, -2*h, 2*h, epsabs=1e-13)[0]
        self.assertAlmostEqual(odd, 0., places=13)
        self.assertAlmostEqual(second, 1., places=13)

    def test_contact_constant_flux_passes_but_continuous_affine_trace_is_not_stationary(self):
        # A local steady transmission solution has matched flux and unequal
        # slopes. Test the continuous pair integral independently of the mesh.
        h = .03125; GL = .04; GR = .02; F = .001
        H = 2*GL*GR/(GL+GR); sL = -F/GL; sR = -F/GR
        I1 = quad(lambda r: float(slope_over_radius(r, h))*r, 0., 2*h, epsabs=1e-13)[0]
        cross_flux = 2*H*quad(lambda r: float(slope_over_radius(r, h))*(-(sL+sR)*r*r/2),
                             0., 2*h, epsabs=1e-13)[0]
        self.assertAlmostEqual(cross_flux, F, places=14)
        left_trace = 2*(-GL*sL+H*sR)*I1
        right_trace = 2*(-H*sL+GR*sR)*I1
        exact = -2*F*(GL-GR)/(GL+GR)*(27/32)/h
        self.assertAlmostEqual(left_trace, exact, places=13)
        self.assertAlmostEqual(right_trace, exact, places=13)
        self.assertNotEqual(exact, 0.)

    def test_invalid_inputs_rejected_without_repair(self):
        faces = np.linspace(-1., 1., 17); V = np.diff(faces); x = (faces[:-1]+faces[1:])/2
        args = (x, V, V, np.ones_like(x), .02)
        with self.assertRaises(ValueError):build_constant_exchange(*args, np.ones_like(x), (-1., 1.))
        with self.assertRaises(ValueError):build_constant_exchange(*args, np.nan, (-1., 1.))
        with self.assertRaises(ValueError):build_constant_exchange(*args, 1., (-1., 1.))
        with self.assertRaises(ValueError):build_constant_exchange(x[::-1], V, V, np.ones_like(x), .02, .1, (-1., 1.))
        with self.assertRaises(ValueError):build_constant_exchange(x, 2*V, V, np.ones_like(x), .02, .1, (-1., 1.))


if __name__ == '__main__':
    unittest.main(verbosity=2)
