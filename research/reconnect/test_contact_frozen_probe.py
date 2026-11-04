"""Independent contact transmission, conservation and material means checks."""
import unittest
import numpy as np
from contact_frozen_probe import contact_inputs,point_fields,analytic_face_reference,volume_reference,contact_regions


class ContactFrozenChecks(unittest.TestCase):
    def test_exact_material_means_match_independent_Gauss_not_point_values(self):
        data,a=contact_inputs(16);q=volume_reference(16,.02,a['material_label_centers'],a['mass'],16)
        np.testing.assert_allclose(a['mass'],q['mass'],atol=4e-18,rtol=0.)
        np.testing.assert_allclose(a['c0'],q['state'],atol=5e-16,rtol=0.)
        self.assertGreater(np.max(abs(a['c0']-a['point_scalar_reference_only'])),1e-3)
        np.testing.assert_array_equal(data.positions[0],data.positions[1])
        np.testing.assert_array_equal(a['mass']/a['rho0_point'],a['true_cell_area_reference_only'])

    def test_concentration_and_nonzero_flux_match_at_both_density_contacts(self):
        for left,right in [(np.nextafter(.5,0.),np.nextafter(.5,1.)),(np.nextafter(1.,0.),0.)]:
            v=point_fields(left,.07);w=point_fields(right,.07)
            self.assertLess(abs(float(v['c']-w['c'])),3e-15)
            self.assertLess(abs(float(v['q_a']-w['q_a'])),3e-15)
            self.assertGreater(abs(float(v['q_a'])),.01)
            self.assertNotEqual(float(v['rho']),float(w['rho']))
            self.assertGreater(abs(float(v['c_a']-w['c_a'])),.1)

    def test_analytic_conservative_faces_match_full_piecewise_PDE_volumes(self):
        d,a=contact_inputs(16);ref=analytic_face_reference(16,.02,a['material_label_centers'],a['mass'])
        for order in [8,16]:
            q=volume_reference(16,.02,a['material_label_centers'],a['mass'],order)
            for key in ['rate_a','rate_b','rate']:np.testing.assert_allclose(ref[key],q[key],atol=4e-15,rtol=0.)

    def test_full_physical_inventory_and_energy_identity_include_contacts(self):
        d,a=contact_inputs(16);q=volume_reference(16,.02,a['material_label_centers'],a['mass'],16)
        self.assertLess(abs(float(a['mass']@q['rate'])),3e-17)
        self.assertLess(abs(float(q['c_rate_integral'].sum()+q['physical_dissipation'].sum())),3e-16)
        self.assertGreater(float(q['physical_dissipation'].sum()),0.)

    def test_fixed_two_h_regions_include_periodic_contact_and_partition_cells(self):
        d,a=contact_inputs(128);r=contact_regions(a['material_label_centers'],.04)
        self.assertEqual(int(r['near_contact'].sum()),5120)
        for column in [0,63,64,127]:self.assertTrue(r['near_contact'][column])
        self.assertFalse(r['near_contact'][32]);self.assertFalse(np.any(r['near_contact']&r['far_from_contact']))
        self.assertTrue(np.all(r['near_contact']|r['far_from_contact']))

    def test_unaligned_or_invalid_reference_inputs_are_rejected(self):
        for N,D in [(15,.02),(16,0.),(16,-.02),(2,.02)]:
            with self.assertRaises(ValueError):contact_inputs(N,D)
        with self.assertRaises(ValueError):point_fields(-.01,0.)


if __name__=='__main__':unittest.main(verbosity=2)
