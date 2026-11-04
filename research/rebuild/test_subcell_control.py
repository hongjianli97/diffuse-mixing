"""Scientific checks of fixed partition, hidden state and coarse closure."""
import unittest
import numpy as np
from contact import Contact,mesh,METHODS,semidiscrete,laplacian
from contact_step_reference import StepReference
from subcell_control import split_faces_at_contact,aggregation_and_lift,compare


def problem(n=32,D=.02):
    f=Contact(D=D);r=StepReference(f);coarse=mesh(n,'cut');child=split_faces_at_contact(coarse)
    m=f.mass(coarse);ms=f.mass(child);c0=r.average(coarse,0.);z0=r.average(child,0.)
    A,B,parent=aggregation_and_lift(coarse,child,m,ms)
    return f,r,coarse,child,m,ms,c0,z0,A,B


class SubcellChecks(unittest.TestCase):
    def test_partition_preserves_exact_mass_initial_mean_and_unresolved_energy(self):
        for n in (32,64,128):
            f,r,e,z,m,ms,c0,z0,A,B=problem(n)
            self.assertEqual(len(ms),n+1)
            self.assertAlmostEqual(float(ms.sum()),3.,places=13)
            self.assertAlmostEqual(float(ms@z0),2.,places=13)
            np.testing.assert_allclose(A@z0,c0,atol=5e-15)
            np.testing.assert_array_equal(A@B,np.eye(n))
            np.testing.assert_array_equal(m@A,ms)
            delta=z0-B@c0
            energy_lost=.5*float(ms@(delta*delta))
            self.assertAlmostEqual(energy_lost,(2/n)/7,places=13)
            self.assertAlmostEqual(float(.5*ms@((z0-r.mean)**2)),1/3,places=13)

    def test_exact_reference_aggregation_is_independent_of_evolution(self):
        f,r,e,z,m,ms,c0,z0,A,B=problem(64)
        for t in (1e-4,.001,.1):
            np.testing.assert_allclose(A@r.average(z,t),r.average(e,t),atol=1e-13,rtol=0.)

    def test_pure_child_cells_make_all_backends_harmonic_and_equivalent(self):
        f,r,e,z,m,ms,c0,z0,A,B=problem()
        g=f.conductance(z,METHODS[0]);contact=np.flatnonzero(z==0)[0]-1;h=2/32
        expected=1/(h/(2*f.rho_left*f.D)+(h/3)/(2*f.rho_right*f.D))
        self.assertAlmostEqual(g[contact],expected,places=12)
        for method in METHODS:np.testing.assert_allclose(f.conductance(z,method),g,rtol=5e-13)

    def test_initial_right_inventory_is_identifiable_and_has_finite_discrete_rate(self):
        f,r,e,z,m,ms,c0,z0,A,B=problem();right=z[:-1]>=0
        self.assertEqual(float(ms[right]@z0[right]),0.)
        self.assertAlmostEqual(float(ms[right]@(B@c0)[right]),2*(2/32)/7,places=13)
        g=f.conductance(z,METHODS[0]);j=np.flatnonzero(z==0)[0]-1
        c,P,d=semidiscrete(z0,ms,g,1e-9)
        self.assertAlmostEqual(float(ms[right]@c[right])/1e-9,g[j],delta=2e-7)

    def test_retained_information_and_operator_change_sum_to_actual_change(self):
        f,r,e,z,m,ms,c0,z0,A,B=problem()
        fine,P,d=semidiscrete(z0,ms,f.conductance(z,METHODS[0]),.01)
        for method in METHODS:
            old,_,_=semidiscrete(c0,m,f.conductance(e,method),.01)
            row,arr=compare(A,B,P,z0,c0,old,m)
            self.assertLess(row['split_max_absolute_residual'],2e-15)
            np.testing.assert_allclose(arr['retained_information_term']+arr['operator_change_term'],A@fine-old,atol=2e-15)

    def test_same_parent_initial_state_has_different_future_due_to_hidden_subcell(self):
        f,r,e,z,m,ms,c0,z0,A,B=problem();delta=z0-B@c0
        np.testing.assert_allclose(A@delta,0.,atol=1e-15)
        g=f.conductance(z,METHODS[0]);Q=-laplacian(g)/ms[:,None]
        rate=A@(Q@delta);self.assertGreater(np.linalg.norm(rate),1.)
        _,P,_=semidiscrete(z0,ms,g,1e-8)
        observed=A@(P@delta)/1e-8
        np.testing.assert_allclose(observed,rate,rtol=1e-4,atol=1e-6)

    def test_positive_mass_preserving_coarse_maps_need_not_form_semigroup(self):
        f,r,e,z,m,ms,c0,z0,A,B=problem();g=f.conductance(z,METHODS[0])
        _,Ph,_=semidiscrete(z0,ms,g,.05);_,Pf,_=semidiscrete(z0,ms,g,.1)
        C=A@Pf@B;Ch=A@Ph@B;defect=C-Ch@Ch
        factor=A@Ph@(np.eye(len(ms))-B@A)@Ph@B
        np.testing.assert_allclose(defect,factor,atol=2e-15)
        self.assertGreater(np.max(abs(defect)),.01)
        np.testing.assert_allclose(C.sum(axis=1),1.,atol=2e-14)
        np.testing.assert_allclose(m@C,m,atol=2e-14)
        self.assertGreaterEqual(C.min(),0.)
        # Without aggregation, the child semigroup has no corresponding defect.
        np.testing.assert_allclose(Pf,Ph@Ph,atol=2e-14)

    def test_zero_diffusion_removes_exchange_and_projection_memory_defect(self):
        f,r,e,z,m,ms,c0,z0,A,B=problem(D=0.)
        c,P,d=semidiscrete(z0,ms,f.conductance(z,METHODS[0]),.1)
        np.testing.assert_array_equal(c,z0)
        np.testing.assert_array_equal(A@P@B,np.eye(len(m)))
        self.assertEqual(float(ms[z[:-1]>=0]@c[z[:-1]>=0]),0.)

    def test_refuse_invalid_partition_or_mass_modification(self):
        f,r,e,z,m,ms,c0,z0,A,B=problem()
        with self.assertRaises(ValueError):split_faces_at_contact(mesh(32,'aligned'))
        with self.assertRaises(ValueError):aggregation_and_lift(e,z,m,ms*1.1)
        with self.assertRaises(ValueError):aggregation_and_lift(e,z[1:],m,ms[1:])


if __name__=='__main__':unittest.main(verbosity=2)
