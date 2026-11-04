"""Known minimax problems and independent exact posterior certificates."""
from fractions import Fraction
import unittest
import numpy as np
from scipy.sparse import csr_matrix,hstack
from moment_defect_budget import budget_problem,solve_budget
from budget_certificates import separator_budget_lower,posterior_budget_upper


def toy(displacements,target,D=.02,mass=1.):
    d=np.asarray(displacements);A=csr_matrix(np.stack((d[:,0],d[:,1],d[:,0]**2,d[:,0]*d[:,1],d[:,1]**2)))
    b=mass*np.asarray(target);return A,b,budget_problem(A,b,np.array([mass]),np.array([0]),D)


def drift_toy(mass=1.):return toy([[0.,.25],[0.,-.25]],[0.,.2,0.,0.,.04],mass=mass)


class MomentBudgetChecks(unittest.TestCase):
    def test_known_transverse_drift_requires_quarter_budget(self):
        A,b,p=drift_toy();report,raw=solve_budget(p)
        self.assertTrue(report['solver_reports_feasible']);self.assertAlmostEqual(report['epsilon_solver'],.25)
        np.testing.assert_allclose(p['E']@raw['raw_conductances'],p['e'],atol=1e-16)
        self.assertFalse(report['physical_acceptance']);self.assertFalse(report['physical_targets_changed'])

    def test_exact_separator_lower_bound_matches_known_stored_problem(self):
        A,b,p=drift_toy();lower=separator_budget_lower(A,b,p['second_rows'],p['scale'],{1:-Fraction(1,4),4:Fraction(1)})
        self.assertTrue(lower['verified'])
        expected=(Fraction(1,4)*Fraction.from_float(float(b[1]))-Fraction.from_float(float(b[4])))/Fraction.from_float(float(p['scale'][2]))
        self.assertEqual(Fraction(int(lower['exact_lower_bound']['numerator']),int(lower['exact_lower_bound']['denominator'])),expected)

    def test_posterior_drift_correction_proves_an_upper_bound_without_changing_rates(self):
        A,b,p=drift_toy();g=np.array([.8000000001,0.]);before=g.copy();upper,arrays=posterior_budget_upper(p,g)
        self.assertTrue(upper['verified']);self.assertLess(upper['exact_inverse_residual_norm']['float'],1.)
        self.assertGreaterEqual(upper['exact_upper_bound']['float'],.25);self.assertLess(upper['exact_upper_bound']['float'],.25000001)
        self.assertGreater(upper['correction_bound_kg_per_s']['float'],0.);np.testing.assert_array_equal(g,before)

    def test_exact_zero_budget_is_certified_without_a_correction_basis(self):
        A,b,p=toy([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],[0.,0.,.04,0.,.04])
        report,raw=solve_budget(p);self.assertAlmostEqual(report['epsilon_solver'],0.)
        upper,_=posterior_budget_upper(p,raw['raw_conductances']);self.assertTrue(upper['verified'])
        self.assertEqual(upper['exact_upper_bound']['float'],0.)

    def test_signed_cross_defect_is_not_assumed_to_be_added_psd_diffusion(self):
        A,b,p=toy([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],[0.,0.,.04,.2,.04])
        report,raw=solve_budget(p);self.assertAlmostEqual(report['epsilon_solver'],5.)
        delta=((p['S']@raw['raw_conductances']-p['s'])/1.).reshape(1,3)[0]
        tensor=np.array([[delta[0],delta[1]],[delta[1],delta[2]]]);self.assertLess(np.linalg.eigvalsh(tensor).min(),0.)

    def test_physical_mass_scale_and_length_units_do_not_change_relative_budget(self):
        A,b,p=drift_toy();report,raw=solve_budget(p);As,bs,ps=drift_toy(mass=7.);other,rs=solve_budget(ps)
        self.assertAlmostEqual(report['epsilon_solver'],other['epsilon_solver'])
        np.testing.assert_allclose(rs['raw_conductances'],7*raw['raw_conductances'],atol=2e-15)
        pu=budget_problem(A,b,np.array([1.]),np.array([0]),.02,2.);unit,ru=solve_budget(pu)
        self.assertAlmostEqual(report['epsilon_solver'],unit['epsilon_solver'])

    def test_drift_is_not_relaxed_to_make_the_budget_feasible(self):
        A,b,p=toy([[1.,0.]],[ -1.,0.,.04,0.,.04]);report,raw=solve_budget(p)
        self.assertFalse(report['solver_reports_feasible']);self.assertNotIn('raw_conductances',raw)

    def test_invalid_physics_or_negative_witness_is_rejected_without_clipping(self):
        A,b,p=drift_toy()
        with self.assertRaises(ValueError):budget_problem(A,b,np.array([1.]),np.array([0]),0.)
        with self.assertRaises(ValueError):budget_problem(A,b,np.array([-1.]),np.array([0]),.02)
        upper,_=posterior_budget_upper(p,[-.1,1.]);self.assertFalse(upper['verified'])

    def test_unidentified_boundary_column_can_change_weights_without_changing_budget(self):
        A,b,p=drift_toy();A=hstack((A,csr_matrix((5,1))),format='csr');p=budget_problem(A,b,np.array([1.]),np.array([0]),.02)
        r1,_=posterior_budget_upper(p,[.8,0.,0.]);r2,_=posterior_budget_upper(p,[.8,0.,2.])
        self.assertTrue(r1['verified']);self.assertTrue(r2['verified']);self.assertEqual(r1['exact_upper_bound'],r2['exact_upper_bound'])

    def test_false_lower_or_singular_positive_proof_basis_does_not_pass(self):
        A,b,p=drift_toy();lower=separator_budget_lower(A,b,p['second_rows'],p['scale'],{1:Fraction(1,4),4:Fraction(-1)})
        self.assertFalse(lower['verified'])
        # Two independent drift equations cannot be corrected with one active column.
        A,b,p=toy([[1.,1.]],[1.,2.,.04,0.,.04]);upper,_=posterior_budget_upper(p,[1.])
        self.assertFalse(upper['verified'])


if __name__=='__main__':unittest.main(verbosity=2)
