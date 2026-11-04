"""Physical and geometric checks of the a priori layer mesh."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal
from contact import Contact,mesh,semidiscrete,laplacian
from contact_step_reference import StepReference
from local_layer_mesh import locally_refined_faces
from subcell_control import aggregation_and_lift


class LocalLayerChecks(unittest.TestCase):
    def test_local_partition_preserves_parent_mass_and_sharp_initial_state(self):
        f=Contact();r=StepReference(f);parent=mesh(64,'cut');m=f.mass(parent);c0=r.average(parent,0.)
        for q in (2,4,8):
            faces=locally_refined_faces(parent,f.D,1e-4,q);ms=f.mass(faces);cs=r.average(faces,0.)
            A,B,index=aggregation_and_lift(parent,faces,m,ms)
            self.assertTrue(np.all(np.isin(parent,faces)))
            self.assertTrue(np.all(np.diff(faces)>0))
            self.assertFalse(np.any((faces[:-1]<0)&(faces[1:]>0)))
            self.assertAlmostEqual(float(ms.sum()),3.,places=13)
            self.assertAlmostEqual(float(ms@cs),2.,places=13)
            self.assertEqual(float(ms[faces[:-1]>=0]@cs[faces[:-1]>=0]),0.)
            np.testing.assert_allclose(A@cs,c0,atol=5e-15)

    def test_earliest_contact_width_criterion_and_controlled_growth_of_size(self):
        parent=mesh(64,'cut');last=0;ell=np.sqrt(.02*1e-4)
        for q in (2,4,8):
            faces=locally_refined_faces(parent,.02,1e-4,q)
            near=(faces[:-1]<=0)&(faces[1:]>=0)
            self.assertGreaterEqual(ell/np.max(np.diff(faces)[near]),q*(1-1e-12))
            self.assertGreater(len(faces),last);last=len(faces)

    def test_mesh_is_covariant_under_length_and_diffusivity_unit_scaling(self):
        parent=mesh(64,'cut')
        faces=locally_refined_faces(parent,.02,1e-4,4)
        scaled=locally_refined_faces(2*parent,4*.02,1e-4,4)
        np.testing.assert_allclose(scaled,2*faces,rtol=0.,atol=2e-14)

    def test_analytic_reference_aggregation_with_small_local_cells(self):
        f=Contact();r=StepReference(f);parent=mesh(64,'cut');faces=locally_refined_faces(parent,.02,1e-4,8)
        A,B,index=aggregation_and_lift(parent,faces,f.mass(parent),f.mass(faces))
        for t in (1e-4,.1):np.testing.assert_allclose(A@r.average(faces,t),r.average(parent,t),rtol=0,atol=2e-13)

    def test_uniform_heat_mode_obeys_analytic_defect_bound_and_independent_evolution(self):
        f=Contact(rho_left=1.,rho_right=1.);records=[];rate=.02*(np.pi/2)**2
        for q in (2,4,8):
            faces=locally_refined_faces(mesh(64,'cut'),.02,1e-4,q);m=f.mass(faces);h=np.diff(faces)
            c0=.5+.2*np.cos(np.pi*((faces[:-1]+faces[1:])/2+1)/2)*np.sinc(h/4)
            exact=.5+(c0-.5)*np.exp(-.02*(np.pi/2)**2*.1)
            g=f.conductance(faces,'ordinary_volume_center');diag=np.zeros(len(m));diag[:-1]+=g;diag[1:]+=g
            vals,U=eigh_tridiagonal(diag/m,-g/np.sqrt(m[:-1]*m[1:]))
            actual=(U@(np.exp(-.1*vals)*(U.T@(np.sqrt(m)*c0))))/np.sqrt(m)
            dense,P,d=semidiscrete(c0,m,g,.1)
            np.testing.assert_allclose(actual,dense,atol=1e-10,rtol=0.)
            error=float(np.sqrt(m@((actual-exact)**2)))
            # Exact continuous-mode defect and mass-contractivity Duhamel
            # bound. Arbitrary nonnested local refinement has no theorem
            # guaranteeing monotone global error for this different field.
            defect=-(laplacian(g)@(c0-.5))/m+rate*(c0-.5)
            bound=-np.expm1(-rate*.1)/rate*np.sqrt(m@(defect*defect))
            self.assertLessEqual(error,bound+1e-10)
            records.append({'resolution':q,'error':error,'analytic_defect_bound':float(bound),
                            'independent_evolution_difference':float(np.max(abs(actual-dense)))})
        type(self).heat_mode_records=records

    def test_true_split_resistance_integrals_at_contact_and_away(self):
        f=Contact();faces=locally_refined_faces(mesh(64,'cut'),.02,1e-4,8)
        centers=(faces[:-1]+faces[1:])/2;g=f.conductance(faces,'ordinary_volume_center')
        front=np.flatnonzero(faces==0.)[0]-1
        for i in (front-1,front,front+1,0,len(g)-1):
            a,b=centers[i:i+2];points=[a,0.,b] if a<0<b else [a,b];R=0.
            for l,r in zip(points[:-1],points[1:]):
                rho=f.rho_left if (l+r)/2<0 else f.rho_right
                R+=quad(lambda x:1/(rho*f.D),l,r)[0]
            self.assertAlmostEqual(g[i]*R,1.,places=12)

    def test_refuse_nonpositive_target_parameters_and_invalid_parent(self):
        parent=mesh(64,'cut')
        for D,t,q,beta in ((0.,1e-4,4,.2),(.02,0.,4,.2),(.02,1e-4,0.,.2),(.02,1e-4,4,0.)):
            with self.assertRaises(ValueError):locally_refined_faces(parent,D,t,q,beta)
        with self.assertRaises(ValueError):locally_refined_faces(mesh(64,'aligned'),.02,1e-4,4)


if __name__=='__main__':unittest.main(verbosity=2)
