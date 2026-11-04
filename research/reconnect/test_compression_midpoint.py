"""Checks for the coupled-density midpoint and vector-only time ordering."""
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from compression_probe import make_compression_samples,analytic_material_rate,compression_geometry,compression_gauss_reference
from compression_midpoint import commutator_action,pair_action
from smooth_density_probe import make_samples
from positive_kernel_2d import build_periodic_exchange
from sparse_exchange_audit import audit_laplacian


class MidpointChecks(unittest.TestCase):
    def samples(self):
        initial,r=make_samples(8,.25,.02)
        data,g,a=make_compression_samples(r['labels'],initial.ids,initial.mass,initial.c0,initial.rho[0],[0.,.025])
        return initial,r,data,g

    def test_midpoint_nonuniform_density_and_saved_material_state(self):
        initial,r,data,g=self.samples()
        np.testing.assert_array_equal(data.mass,initial.mass);np.testing.assert_array_equal(data.c0,initial.c0)
        np.testing.assert_array_equal(data.rho,initial.rho[0][None,:]/g['J'])
        self.assertLess(data.rho[1].min(),.9);self.assertGreater(data.rho[1].max(),1.1)
        np.testing.assert_array_equal(g['alpha'],[0.,.125])

    def test_midpoint_full_reference_and_nonzero_density_drift(self):
        initial,r,data,g=self.samples()
        ref,_=analytic_material_rate(8,.25,.125,.02,data.mass,data.c0,r['labels'])
        q=compression_gauss_reference(8,.25,.125,.02,r['labels'],data.mass,16)
        np.testing.assert_allclose(ref,q['rate'],atol=4e-15,rtol=2e-15)
        np.testing.assert_allclose(q['state'],data.c0,atol=5e-16,rtol=0.)
        self.assertGreater(np.max(abs(q['drift_rate'])),.01)
        self.assertGreater(np.max(abs(q['rate']-q['laplacian_rate'])),.01)

    def test_area_discrepancy_and_node_offsets_remain_raw(self):
        initial,r,data,g=self.samples();geometry=compression_geometry(8,r['labels'],.125,.025)
        raw=data.mass/data.rho[1];true=geometry['true_cell_area_from_unwrapped_faces']
        self.assertGreater(np.max(abs(raw/true-1)),1e-3)
        self.assertGreater(np.max(abs(geometry['node_minus_geometric_center_x'])),1e-4)
        np.testing.assert_array_equal(data.positions[1,:,0],np.mod(geometry['unwrapped_node_x'],1.))

    def test_midpoint_point_graph_has_raw_positive_shared_flux(self):
        initial,r,data,g=self.samples();L,a,d=build_periodic_exchange(data,1,.02,.19,[1.,1.])
        self.assertTrue(audit_laplacian(data.mass,L)['graph_algebra_eligible'])
        np.testing.assert_array_equal(a['quadrature_area'],data.mass/data.rho[1])
        np.testing.assert_allclose(pair_action(data.mass,a['pairs'],a['conductances'],data.c0),-(L@data.c0)/data.mass,atol=2e-15,rtol=1e-13)

    def test_known_unequal_mass_commutator_orientation(self):
        m=np.array([1.,2.,3.]);c=np.array([.1,.4,.8])
        L0=csr_matrix([[1.,-1.,0.],[-1.,1.,0.],[0.,0.,0.]])
        L1=csr_matrix([[0.,0.,0.],[0.,2.,-2.],[0.,-2.,2.]])
        forward=commutator_action(m,L0,L1,c);reverse=commutator_action(m,L1,L0,c)
        np.testing.assert_allclose(forward['commutator'],[-.4,.35,-.1],atol=2e-16,rtol=0.)
        np.testing.assert_array_equal(reverse['commutator'],-forward['commutator'])
        self.assertLess(abs(m@forward['commutator']),2e-16)

    def test_identical_endpoints_and_constant_state_commute(self):
        m=np.array([1.,2.,3.]);L=csr_matrix([[1.,-1.,0.],[-1.,3.,-2.],[0.,-2.,2.]])
        np.testing.assert_array_equal(commutator_action(m,L,L,[.1,.4,.8])['commutator'],0.)
        np.testing.assert_array_equal(commutator_action(m,L,2*L,np.ones(3))['commutator'],0.)

    def test_commutator_rejects_missing_mass_identity_or_dense_operator(self):
        L=csr_matrix([[1.,-1.],[-1.,1.]])
        for m,c,A in [([1.,0.],[.2,.4],L),([1.,2.],[.2],L),([1.,2.],[.2,.4],L.toarray())]:
            with self.assertRaises(ValueError):commutator_action(m,A,L,c)


if __name__=='__main__':unittest.main(verbosity=2)
