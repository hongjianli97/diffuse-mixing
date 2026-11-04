"""Independent mass-average identities and exact rational cone certificates."""
import unittest
import numpy as np
from becker import Becker
from mass_moment_cone import moment_features,solve_moment_cone,exact_basis_witness,require_bulk_context,transverse_drift_separator
from mass_cone_study import stream_cell_moments,primitive_targets,independent_boundary_targets,build_support


def homogeneous(radius):
    means=np.arange(15)/16;var=np.full(15,1/16**2/12);column=7
    return build_support(means,var,column,range(column-radius,column+radius+1))[0]


class MassMomentConeChecks(unittest.TestCase):
    def test_transverse_drift_certificate_survives_widening_and_handles_both_signs(self):
        A=homogeneous(5);b=np.array([0.,.09,.04,0.,.04])
        for sign in (-1,1):
            b[1]=sign*.09;certificate=transverse_drift_separator(A,b)
            self.assertTrue(certificate['verified'])
            self.assertGreaterEqual(certificate['minimum_exact_column_slack']['float'],0.)
            self.assertAlmostEqual(certificate['exact_target_dot_functional']['float'],-.005)
        b[1]=.05;self.assertFalse(transverse_drift_separator(A,b)['verified'])

    def test_features_match_quadratic_mass_means_about_an_arbitrary_origin(self):
        center=np.array([.3,-.4]);d=np.array([[.2,.5],[-.3,-.5]])
        S=np.array([[.1,.02],[.02,.2]]);Sj=np.array([[[.15,.03],[.03,.3]],[[.2,-.01],[-.01,.25]]])
        A=moment_features(d,Sj-S);b=center+d
        # Exact expected polynomial mass averages: E[(x-center)(x-center)^T].
        raw=b[:,:,None]*b[:,None,:]+Sj
        centered=raw-center[None,:,None]*b[:,None,:]-b[:,:,None]*center[None,None,:]+center[:,None]*center[None,:]
        difference=centered-S
        np.testing.assert_allclose(A[2:],difference[:,[0,0,1],[0,1,1]].T,atol=1e-16)

    def test_orthogonal_cone_has_independent_exact_positive_basis(self):
        d=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        A=moment_features(d,np.zeros((4,2,2)));b=np.array([0.,0.,.04,0.,.04])
        report,r=solve_moment_cone(A,b)
        self.assertTrue(report['solver_reports_feasible']);self.assertTrue(report['exact_basis_certificate']['verified'])
        self.assertGreaterEqual(float(r.min()),0.);np.testing.assert_allclose(r,.02,atol=1e-16)
        self.assertFalse(report['globally_conservative_operator_constructed'])

    def test_narrow_shear_has_exact_separating_functional(self):
        A=homogeneous(1);b=np.array([0.,0.,.04,0.,.04]);report,r=solve_moment_cone(A,b)
        self.assertFalse(report['solver_reports_feasible']);self.assertIsNone(r)
        certificate=report['cross_ratio_separator'];self.assertTrue(certificate['verified'])
        self.assertGreaterEqual(certificate['minimum_exact_column_slack']['float'],0.)
        self.assertLess(certificate['exact_target_dot_functional']['float'],0.)
        np.testing.assert_allclose(certificate['cross_ratio_interval'],[.475,.725])

    def test_single_wide_shear_control_has_exact_cone_witness(self):
        A=homogeneous(5);b=np.array([0.,0.,.04,0.,.04]);report,r=solve_moment_cone(A,b)
        self.assertTrue(report['solver_reports_feasible']);self.assertTrue(report['exact_basis_certificate']['verified'])
        self.assertGreaterEqual(float(r.min()),0.);np.testing.assert_allclose(A@r,b,atol=2e-15)

    def test_becker_primitive_targets_match_independent_boundary_quadrature(self):
        f=Becker();edges=f.mu(np.linspace(-1.,1.,33));means,var=stream_cell_moments(f,edges,.05)
        beta,Gamma,J=primitive_targets(f,edges,means,16,.05)
        direct_beta,direct_Gamma=independent_boundary_targets(f,edges,16,means[16],.05)
        np.testing.assert_allclose(beta,direct_beta,atol=3e-13,rtol=0.)
        np.testing.assert_allclose(Gamma,direct_Gamma,atol=3e-14,rtol=0.)
        self.assertAlmostEqual(Gamma[1,1],.04);self.assertAlmostEqual(beta[1],-.6*beta[0])
        self.assertTrue(np.all(var>0.))

    def test_covariance_difference_is_required_for_quadratic_mass_states(self):
        d=np.array([[1.,0.],[-1.,0.]]);S=np.zeros((2,2,2));S[:,0,0]=.2
        A=moment_features(d,S);r=np.ones(2)
        self.assertAlmostEqual(float((A@r)[2]),2.4)
        self.assertAlmostEqual(float(np.sum((1+.3-.1)*r)),2.4)
        self.assertGreater(abs(float((A@r)[2])-float((d[:,0]**2)@r)),.39)

    def test_boundary_guard_forbids_unmarked_bulk_substitution(self):
        with self.assertRaises(ValueError):require_bulk_context(True,'actual_interior')
        with self.assertRaises(ValueError):require_bulk_context(True,'closed_boundary_ignored')
        require_bulk_context(True,'explicit_virtual_transverse_extension');require_bulk_context(False,'actual_interior')

    def test_rigid_quarter_turn_preserves_cone_feasibility(self):
        A,raw=build_support(np.arange(15)/16,np.full(15,.001),7,range(2,13))
        R=np.array([[0.,-1.],[1.,0.]])
        d=raw['displacements']@R.T;S=R@raw['covariance_difference']@R.T
        rotated=moment_features(d,S);b=np.array([0.,0.,.04,0.,.04])
        report,r=solve_moment_cone(rotated,b)
        self.assertTrue(report['exact_basis_certificate']['verified']);np.testing.assert_allclose(rotated@r,b,atol=2e-15)

    def test_units_and_density_scale_do_not_recalibrate_physics(self):
        d=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]);A=moment_features(d,np.zeros((4,2,2)));b=np.array([0.,0.,.04,0.,.04])
        _,r=solve_moment_cone(A,b,.5);_,r2=solve_moment_cone(A,b,2.)
        np.testing.assert_allclose(r,r2,atol=1e-16)
        f=Becker();g=Becker(rho1=7.,p1=7.);edges=f.mu(np.linspace(-1.,1.,33));eg=g.mu(np.linspace(-1.,1.,33))
        mean,_=stream_cell_moments(f,edges,.05);mg,_=stream_cell_moments(g,eg,.05)
        beta,Gamma,_=primitive_targets(f,edges,mean,16,.05);bg,Gg,_=primitive_targets(g,eg,mg,16,.05)
        np.testing.assert_allclose(beta,bg,atol=3e-14);np.testing.assert_allclose(Gamma,Gg,atol=3e-15)

    def test_invalid_input_and_false_exact_basis_rejected(self):
        with self.assertRaises(ValueError):moment_features([[1.,0.]],np.full((1,2,2),np.nan))
        with self.assertRaises(ValueError):solve_moment_cone(np.zeros((4,3)),np.zeros(5))
        with self.assertRaises(ValueError):solve_moment_cone(np.zeros((5,3)),np.zeros(5),-1.)
        A=np.ones((5,1));b=np.array([1.,1.,1.,1.,2.])
        self.assertFalse(exact_basis_witness(A,b,[0])['verified'])


if __name__=='__main__':unittest.main(verbosity=2)
