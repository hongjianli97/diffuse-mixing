"""Independent nested physical means and conserved common observations."""
import unittest
import numpy as np
from compression_spatial_observation import fine_inputs,restrict_path
from smooth_density_probe import analytic_reference,gauss_reference
from variable_parent_observation import observe_parent,parent_mass_audit
from compression_reference import material_columns,project_to_saved_cells


class SpatialObservationChecks(unittest.TestCase):
    def inputs(self):
        parent=analytic_reference(8,.25,.02);fine,audit=fine_inputs(16,np.arange(64),parent['mass'])
        return parent,fine,audit

    def test_initial_mass_averages_match_independent_Gauss_and_original_parent(self):
        p,f,a=self.inputs();q=gauss_reference(16,.25,.02,f['material_label_centers'],16)
        np.testing.assert_allclose(f['mass'],q['mass'],atol=3e-18,rtol=0.)
        np.testing.assert_allclose(f['c0'],q['state'],atol=5e-16,rtol=0.)
        np.testing.assert_allclose(observe_parent(f['c0'],f['mass'],p['mass']),p['state'],atol=4e-16,rtol=0.)

    def test_child_parent_identity_is_fixed_and_covers_exactly_four_children(self):
        p,f,a=self.inputs();counts=np.bincount(f['parent_index'],minlength=64)
        np.testing.assert_array_equal(counts,4);np.testing.assert_array_equal(f['parent_id_per_child'],f['parent_index'])
        centers=p['labels'][f['parent_index']]
        self.assertLessEqual(np.max(abs(f['material_label_centers']-centers)),1/32)
        self.assertEqual(len(np.unique(f['ids'])),256)

    def test_arbitrary_component_inventory_is_preserved_on_original_mass(self):
        p,f,a=self.inputs();c=.1+.7*np.sin(np.arange(256)*.37)**2
        parent=observe_parent(c,f['mass'],p['mass'])
        self.assertLess(abs(p['mass']@parent-f['mass']@c),2e-16)
        np.testing.assert_allclose(restrict_path([c,1-c],f['mass'],p['mass'])[0],parent,atol=0.,rtol=0.)

    def test_mass_variance_coarsening_matches_within_cell_variance(self):
        p,f,a=self.inputs();c=.1+.7*np.sin(np.arange(256)*.37)**2;cp=observe_parent(c,f['mass'],p['mass'])
        mu=f['mass']@c/f['mass'].sum();Ef=f['mass']@((c-mu)**2);Ep=p['mass']@((cp-mu)**2)
        within=f['mass']@((c-cp[f['parent_index']])**2)
        self.assertLess(abs(Ef-Ep-within),1e-16);self.assertGreater(within,0.)

    def test_reference_projection_commutes_with_material_nesting(self):
        p,f,a=self.inputs();axis,mf,B=material_columns(64,.25)
        B=B+.03*np.cos(6*np.pi*axis)
        own,_=project_to_saved_cells(B,mf,.05,.02,16,f['mass'])
        direct,_=project_to_saved_cells(B,mf,.05,.02,8,p['mass'])
        np.testing.assert_allclose(observe_parent(own,f['mass'],p['mass']),direct,atol=3e-16,rtol=0.)

    def test_incompatible_child_mass_is_rejected_instead_of_rescaled(self):
        p,f,a=self.inputs();changed=f['mass'].copy();changed[0]*=1.01
        with self.assertRaises(ValueError):parent_mass_audit(changed,p['mass'])
        with self.assertRaises(ValueError):fine_inputs(12,np.arange(64),p['mass'])


if __name__=='__main__':unittest.main(verbosity=2)
