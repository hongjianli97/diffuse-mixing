"""Sparse admission checks and conservative comparison of nested averages."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from trajectory_chain import laplacian,operator_audit
from sparse_exchange_audit import sparse_laplacian,audit_laplacian,sparse_max_abs,restrict_equal_nested_squares
from positive_kernel_2d import build_periodic_exchange
from kernel2d_study import fixture


class RefinementChecks(unittest.TestCase):
    def test_sparse_raw_audit_matches_original_dense_graph_interface(self):
        p=np.array([[0,1],[1,2],[0,3]]);g=np.array([.8,.4,.2]);m=np.array([1.,2.,3.,4.]);L=sparse_laplacian(4,p,g)
        dense=operator_audit(m,-L.toarray()/m[:,None]);sparse=audit_laplacian(m,L)
        for key in ['constant_residual_per_s','mass_residual_kg_per_s','weighted_symmetry_residual_kg_per_s','minimum_raw_offdiagonal_per_s']:
            self.assertAlmostEqual(sparse[key],dense[key],places=14)
        self.assertEqual(sparse['graph_algebra_eligible'],dense['graph_algebra_eligible']);self.assertFalse(sparse['physical_acceptance_inferred'])

    def test_offdiagonal_min_includes_implicit_zero_but_not_diagonal(self):
        L=sparse_laplacian(2,np.array([[0,1]]),[2.]);a=audit_laplacian([1.,3.],L)
        self.assertAlmostEqual(a['minimum_raw_offdiagonal_per_s'],2/3)
        L3=sparse_laplacian(3,np.array([[0,1]]),[2.]);self.assertEqual(audit_laplacian([1.,2.,3.],L3)['minimum_raw_offdiagonal_per_s'],0.)

    def test_signed_and_nonreversible_raw_operators_rejected_without_repair(self):
        signed=csr_matrix([[-1.,1.,0.],[1.,0.,-1.],[0.,-1.,1.]])
        saved=signed.data.copy();a=audit_laplacian([1.,2.,3.],signed);self.assertFalse(a['graph_algebra_eligible']);self.assertLess(a['minimum_raw_offdiagonal_per_s'],0.)
        np.testing.assert_array_equal(signed.data,saved)
        nonsym=csr_matrix([[1.,-1.,0.],[0.,1.,-1.],[0.,0.,0.]])
        self.assertFalse(audit_laplacian([1.,2.,3.],nonsym)['graph_algebra_eligible'])

    def test_snapshot_assembly_and_audit_do_not_request_dense_conversion(self):
        d,_,_=fixture(8)
        with patch.object(csr_matrix,'toarray',side_effect=AssertionError('dense conversion forbidden')):
            L,a,_=build_periodic_exchange(d,1,.02,.19,[1.,1.]);core=sparse_laplacian(len(d.mass),a['pairs'],a['conductances'])
            self.assertTrue(audit_laplacian(d.mass,L)['graph_algebra_eligible']);self.assertLess(sparse_max_abs(core-L),1e-14)

    def test_sparse_shared_pair_assembly_zero_graph_and_invalid_edges(self):
        p=np.array([[0,1],[1,2],[0,3]]);g=[.8,0.,.2]
        np.testing.assert_allclose(sparse_laplacian(4,p,g).toarray(),laplacian(4,p,g),atol=1e-15)
        zero=sparse_laplacian(4,np.empty((0,2),int),[]);self.assertTrue(audit_laplacian([1.,2.,3.,4.],zero)['graph_algebra_eligible'])
        with self.assertRaises(ValueError):sparse_laplacian(4,p,[-1.,0.,1.])
        with self.assertRaises(ValueError):sparse_laplacian(4,np.array([[0,1],[0,1]]),[1.,1.])

    def test_analytical_material_means_and_PDE_rates_restrict_to_same_parent(self):
        coarse,_,_=fixture(4);fine,_,_=fixture(8)
        restrict=lambda v:restrict_equal_nested_squares(v,fine.mass,coarse.mass)
        np.testing.assert_allclose(restrict(fine.c0),coarse.c0,atol=2e-16)
        np.testing.assert_allclose(restrict(-8*np.pi**2*.02*(fine.c0-.5)),-8*np.pi**2*.02*(coarse.c0-.5),atol=4e-16)
        self.assertAlmostEqual(float(coarse.mass@restrict(fine.c0)),float(fine.mass@fine.c0))
        np.testing.assert_array_equal(restrict(np.ones(len(fine.mass))),1.)

    def test_projection_refuses_child_mass_mismatch_instead_of_renormalizing(self):
        coarse,_,_=fixture(4);fine,_,_=fixture(8);bad=fine.mass.copy();bad[0]*=2
        with self.assertRaises(ValueError):restrict_equal_nested_squares(fine.c0,bad,coarse.mass)
        with self.assertRaises(ValueError):restrict_equal_nested_squares(fine.c0[:-1],fine.mass,coarse.mass)
        with self.assertRaises(ValueError):restrict_equal_nested_squares(fine.c0,fine.mass,-coarse.mass)

    def test_actual_kernel_rate_restriction_preserves_scalar_mass_and_mode(self):
        coarse,_,_=fixture(4);fine,_,_=fixture(8);L,a,_=build_periodic_exchange(fine,1,.02,.19,[1.,1.]);rate=-(L@fine.c0)/fine.mass
        projected=restrict_equal_nested_squares(rate,fine.mass,coarse.mass)
        self.assertLess(abs(float(fine.mass@rate-coarse.mass@projected)),2e-15)
        i,j=a['pairs'].T;touch=(i==0)|(j==0);dx=a['minimum_image_displacement'][touch]
        symbol=np.sum(a['conductances'][touch]*(1-np.cos(2*np.pi*(dx[:,0]+dx[:,1]))))/fine.mass[0]
        np.testing.assert_allclose(projected,-symbol*(coarse.c0-.5),atol=2e-14)


if __name__=='__main__':unittest.main(verbosity=2)
