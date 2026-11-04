"""Independent exact face reference and unequal-mass pair observations."""
from fractions import Fraction as F
import unittest
import numpy as np
from interface_pair_probe import quadratic_cell_reference,region_summary


class InterfacePairChecks(unittest.TestCase):
    def test_exact_cell_integrals_and_shared_zero_flux_on_both_sides(self):
        ell=F(1,128);origin=F(1,2);records=[]
        for center,rho in [(origin-ell/2,F(1)),(origin+ell/2,F(4))]:
            r=quadratic_cell_reference(float(center),float(ell),float(rho),float(rho*ell**2),.02,.5)
            self.assertEqual(F(r['exact']['mass_average']),ell**2/6)
            self.assertEqual(F(r['exact']['PDE_mass_average_rate_per_s']),F(1,50))
            self.assertTrue(r['original_mass_matches_constant_phase_square']);records.append(r)
        self.assertEqual(F(records[0]['exact']['right_q_normal']),0)
        self.assertEqual(F(records[1]['exact']['left_q_normal']),0)
        self.assertEqual(F(records[1]['exact']['PDE_mass_rate_kg_per_s']),4*F(records[0]['exact']['PDE_mass_rate_kg_per_s']))

    def test_unequal_mass_region_does_not_replace_itself_with_arithmetic_mean(self):
        m=np.array([1.,4.]);before=m.copy();r=region_summary(m,[.024,.016],[.02,.02])
        np.testing.assert_array_equal(m,before)
        self.assertLess(abs(r['arithmetic_mean_diagnostic_only_per_s']-.02),1e-17)
        self.assertLess(abs(r['mass_average_rate_per_s']-.0176),1e-17)
        self.assertLess(abs(r['mass_average_relative_error']+.12),2e-15)
        self.assertLess(abs(r['mass_weighted_local_relative_L2_error']-.2),2e-15)
        self.assertFalse(r['arithmetic_mean_is_material_observation'])

    def test_reference_keeps_supplied_mass_denominator_and_reports_mismatch(self):
        ell=1/128;mass=ell**2
        a=quadratic_cell_reference(.5-ell/2,ell,1.,mass,.02,.5)
        b=quadratic_cell_reference(.5-ell/2,ell,1.,2*mass,.02,.5)
        self.assertEqual(F(b['exact']['state_mass_numerator_kg']),F(a['exact']['state_mass_numerator_kg']))
        self.assertEqual(F(b['exact']['mass_average']),F(a['exact']['mass_average'])/2)
        self.assertEqual(F(b['exact']['PDE_mass_average_rate_per_s']),F(1,100))
        self.assertFalse(b['original_mass_matches_constant_phase_square'])


if __name__=='__main__':unittest.main(verbosity=2)
