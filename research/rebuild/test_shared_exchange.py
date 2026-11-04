"""Shared unknowns, mass invariants and sparse coupled infeasibility proofs."""
from fractions import Fraction
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from becker import Becker
from mass_cone_study import stream_cell_moments,primitive_targets,independent_boundary_targets
from transverse_partition import logical_child_layout
from shared_exchange_feasibility import fixed_logical_pairs,assemble_shared_moments,solve_shared_moments,exact_sparse_separator,conservative_graph


def two_cell():
    pairs=np.array([[0,1]]);cols=np.array([0,1]);rows=np.zeros(2,dtype=int);means=np.array([0.,1.]);v=np.zeros(2);mass=np.array([2.,5.])
    targets=np.array([[.2,0.,.2,0.,0.],[-.08,0.,.08,0.,0.]])
    A,b=assemble_shared_moments(pairs,cols,rows,means,v,mass,np.arange(2),targets,shear=0.)
    return pairs,mass,A,b


class SharedExchangeChecks(unittest.TestCase):
    def test_one_shared_unknown_gives_unequal_mass_reciprocal_rates(self):
        pairs,mass,A,b=two_cell();L,Q=conservative_graph(pairs,[.4],mass)
        self.assertAlmostEqual(Q[0,1],.2);self.assertAlmostEqual(Q[1,0],.08)
        np.testing.assert_allclose(mass@Q,0.,atol=1e-16);np.testing.assert_allclose(Q@np.ones(2),0.,atol=1e-16)
        np.testing.assert_array_equal(L,L.T);self.assertLess(float(np.linalg.eigvalsh(-L).max()),1e-15)

    def test_shared_pair_column_enters_both_mass_scaled_moment_blocks(self):
        pairs,mass,A,b=two_cell();expected=np.array([1.,0.,1.,0.,0.,-1.,0.,1.,0.,0.])
        np.testing.assert_array_equal(A.toarray()[:,0],expected)
        np.testing.assert_allclose(A@np.array([.4]),b,atol=1e-16)

    def test_small_coupled_system_has_positive_shared_witness(self):
        pairs,mass,A,b=two_cell();report,g=solve_shared_moments(A,b)
        self.assertTrue(report['solver_reports_feasible']);np.testing.assert_allclose(g,.4,atol=1e-16)
        self.assertFalse(report['physical_acceptance']);self.assertGreaterEqual(float(g.min()),0.)
        zero,g0=solve_shared_moments(A,np.zeros_like(b));self.assertTrue(zero['solver_reports_feasible'])
        L0,Q0=conservative_graph(pairs,g0,mass);np.testing.assert_array_equal(L0,0.);np.testing.assert_array_equal(Q0,0.)

    def test_one_row_separator_embeds_in_a_coupled_system(self):
        columns=np.tile(np.arange(5),4);rows=np.repeat(np.arange(4),5);pairs=fixed_logical_pairs(columns,rows,1,1)
        inside=np.array([7,8]);mass=np.full(20,2.);targets=np.array([[0.,.2,.04,0.,.04],[0.,0.,.04,0.,.04]])
        A,b=assemble_shared_moments(pairs,columns,rows,np.arange(5)*.1,np.zeros(5),mass,inside,targets,shear=0.)
        proof=exact_sparse_separator(A,b,{1:-Fraction(1,4),4:Fraction(1)})
        self.assertTrue(proof['verified']);self.assertGreaterEqual(proof['minimum_exact_pair_slack']['float'],0.)
        self.assertLess(proof['exact_target_dot_dual']['float'],0.)
        report,g=solve_shared_moments(A,b);self.assertFalse(report['solver_reports_feasible']);self.assertIsNone(g)

    def test_fixed_128_cell_topology_has_1806_unique_pairs(self):
        columns,rows,boundary=logical_child_layout();pairs=fixed_logical_pairs(columns,rows)
        self.assertEqual(len(pairs),1806);self.assertEqual(len(set(map(tuple,pairs.tolist()))),1806)
        self.assertTrue(np.all(pairs[:,0]<pairs[:,1]));self.assertTrue(np.all(abs(columns[pairs[:,0]]-columns[pairs[:,1]])<=6))
        self.assertTrue(np.all(abs(rows[pairs[:,0]]-rows[pairs[:,1]])<=1));self.assertEqual(int((~boundary).sum()),60)
        neighbours=set(pairs[pairs[:,0]==33,1].tolist()+pairs[pairs[:,1]==33,0].tolist());self.assertEqual(len(neighbours),38)

    def test_cell_permutation_changes_only_identifiers_and_pair_order(self):
        columns=np.tile(np.arange(3),3);rows=np.repeat(np.arange(3),3);means=np.array([-.2,.1,.5]);v=np.array([.001,.002,.004])
        mass=np.arange(1.,10.);inside=np.array([3,4,5]);targets=np.arange(15.).reshape(3,5)/100
        pairs=fixed_logical_pairs(columns,rows,1,1);A,b=assemble_shared_moments(pairs,columns,rows,means,v,mass,inside,targets)
        perm=np.array([4,0,8,2,6,1,7,5,3]);inv=np.argsort(perm)
        pp=fixed_logical_pairs(columns[perm],rows[perm],1,1)
        Ap,bp=assemble_shared_moments(pp,columns[perm],rows[perm],means,v,mass[perm],inv[inside],targets)
        lookup={tuple(e):i for i,e in enumerate(pairs.tolist())};order=[lookup[tuple(sorted(perm[e].tolist()))] for e in pp]
        np.testing.assert_allclose(Ap.toarray(),A.toarray()[:,order],atol=1e-16);np.testing.assert_array_equal(bp,b)

    def test_length_units_and_physical_mass_scale_preserve_graph_rates(self):
        pairs,mass,A,b=two_cell();_,g=solve_shared_moments(A,b,.5);_,gu=solve_shared_moments(A,b,2.)
        _,gs=solve_shared_moments(A,7*b,.5);np.testing.assert_allclose(g,gu,atol=1e-16)
        _,Q=conservative_graph(pairs,g,mass);_,Qs=conservative_graph(pairs,gs,7*mass)
        np.testing.assert_allclose(Q,Qs,atol=1e-16)

    def test_infeasible_or_negative_weights_cannot_become_a_zero_or_clipped_graph(self):
        pairs,mass,A,b=two_cell()
        with self.assertRaises(ValueError):conservative_graph(pairs,None,mass)
        with self.assertRaises(ValueError):conservative_graph(pairs,[-.1],mass)
        with self.assertRaises(ValueError):conservative_graph(pairs,[np.nan],mass)
        with self.assertRaises(ValueError):conservative_graph(pairs,[.1],mass[:,None])
        with self.assertRaises(ValueError):fixed_logical_pairs([.5,1.],[0,0])

    def test_adjacent_becker_column_obstruction_matches_independent_boundary_integral(self):
        f=Becker();edges=f.mu(np.linspace(-1.,1.,33));means,var=stream_cell_moments(f,edges,.05)
        b,G,J=primitive_targets(f,edges,means,17,.05);bd,Gd=independent_boundary_targets(f,edges,17,means[17],.05,hy=.25,order=64)
        np.testing.assert_allclose(b,bd,atol=3e-13);np.testing.assert_allclose(G,Gd,atol=3e-14)
        self.assertLess(G[1,1]-.25*abs(b[1]),0.)
        oldb,oldG,J=primitive_targets(f,edges,means,16,.05);self.assertGreater(oldG[1,1]-.25*abs(oldb[1]),0.)

    def test_false_dual_and_invalid_shared_pairs_are_rejected(self):
        pairs,mass,A,b=two_cell();self.assertFalse(exact_sparse_separator(A,b,{0:Fraction(0)})['verified'])
        self.assertFalse(exact_sparse_separator(A,b,{0:Fraction(-1)})['verified'])
        with self.assertRaises(ValueError):exact_sparse_separator(A,b,{100:Fraction(1)})
        with self.assertRaises(ValueError):assemble_shared_moments(np.array([[1,0]]),[0,1],[0,0],np.array([0.,1.]),np.zeros(2),mass,np.array([0]),np.zeros((1,5)))


if __name__=='__main__':unittest.main(verbosity=2)
