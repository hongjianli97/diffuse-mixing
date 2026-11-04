"""Constitutive face closure, true averaged affine states and boundary balance."""
import unittest
import numpy as np
from becker import Becker,initial
from becker_polygon_benchmark import true_vertices,quad_connectivity
from nonorthogonal_patch_study import oracle_mass_centroids
from test_nonorthogonal_flux import fixture
from frozen_physical_faces import frozen_faces
from hybrid_material_fv import build_hybrid_fv,hybrid_action


def setup(nx=4,ny=3,shear=.6,rotation=0.,translation=(0.,0.)):
    history,p,centers,geometry,k=fixture(shear,nx,ny,rotation,translation)
    topology=frozen_faces(p[history.cell_vertex_indices],np.tile(np.arange(nx),ny),np.repeat(np.arange(ny),nx))
    operator=build_hybrid_fv(topology,history.mass,centers,.02)
    return history,p,centers,topology,operator


class HybridMaterialFVChecks(unittest.TestCase):
    def test_orthogonal_two_cell_limit_is_ordinary_central_resistance(self):
        h,p,b,t,op=setup(2,1,shear=0.);mass=np.array([2.,5.]);op=build_hybrid_fv(t,mass,b,.02)
        internal=np.flatnonzero(~t['boundary_mask']).item();n=op['normal'][internal];xf=op['face_center'][internal]
        d=abs((b-xf)@n);g=op['face_area'][internal]/np.sum(d/op['kappa'])
        expected=np.array([[g,-g],[-g,g]])
        np.testing.assert_allclose(op['L'],expected,atol=4e-16,rtol=2e-14)
        np.testing.assert_allclose(op['Q'],-expected/mass[:,None],atol=2e-16)

    def test_affine_patch_with_matching_physical_boundary_has_exact_native_flux(self):
        h,p,b,t,op=setup();D=.02;boundary=op['boundary_faces']
        for axis in range(2):
            state=.5+.1*b[:,axis];gradient=np.eye(2)[axis]*.1
            exact=-D*op['face_area']*(op['normal']@gradient)
            prescribed=exact[boundary]*op['boundary_sign'];answer=hybrid_action(op,state,prescribed)
            np.testing.assert_allclose(answer['trace'],.5+.1*op['face_center'][:,axis],atol=2e-14,rtol=0.)
            np.testing.assert_allclose(answer['face_flux'],exact,atol=3e-15,rtol=0.)
            np.testing.assert_allclose(answer['rate'],0.,atol=2e-13,rtol=0.)

    def test_local_affine_state_uses_true_becker_mass_centroid_and_mean_density(self):
        f=Becker();mu=f.mu(np.linspace(-1.,1.,9));eta=np.linspace(-.5,.5,5);time=.05
        p=true_vertices(f,mu,eta,np.array([time]))[0];cv=quad_connectivity(8,4);mass=np.tile(np.diff(mu)/4,4)
        b=oracle_mass_centroids(f,mu,eta,time);t=frozen_faces(p[cv],np.tile(np.arange(8),4),np.repeat(np.arange(4),8))
        op=build_hybrid_fv(t,mass,b,.02)
        self.assertGreater(float(np.max(np.linalg.norm(b-p[cv].mean(axis=1),axis=1))),1e-5)
        for cell in range(len(mass)):
            faces=op['cell_faces'][cell];u=np.r_[.5+.1*b[cell,0],.5+.1*op['face_center'][faces,0]]
            np.testing.assert_allclose(op['local_gradient'][cell]@u,[.1,0.],atol=2e-14)
            np.testing.assert_allclose(op['local_residual'][cell]@u,0.,atol=2e-14)
            flux=-op['local_energy'][cell,1:]@u
            exact=-op['kappa'][cell]*op['face_area'][faces]*t['incidence'][cell,faces]*op['normal'][faces,0]*.1
            np.testing.assert_allclose(flux,exact,atol=3e-15)

    def test_closed_energy_mass_and_constants_hold_without_post_symmetrization(self):
        h,p,b,t,op=setup();L=op['L'];Q=op['Q']
        self.assertLess(float(np.max(abs(L-L.T))),1e-14)
        self.assertLess(float(np.max(abs(h.mass@Q))),1e-14)
        self.assertLess(float(np.max(abs(Q.sum(axis=1)))),1e-12)
        self.assertGreater(float(np.linalg.eigvalsh((L+L.T)/2).min()),-1e-13)
        state=np.linspace(.2,.8,len(h.mass));self.assertLessEqual(float(state@(h.mass*(Q@state))),1e-13)

    def test_prescribed_boundary_and_native_face_divergence_have_global_balance(self):
        h,p,b,t,op=setup();state=np.linspace(.1,.9,len(h.mass));boundary=np.linspace(-.01,.02,len(op['boundary_faces']))
        answer=hybrid_action(op,state,boundary)
        np.testing.assert_allclose(answer['face_flux'][op['boundary_faces']]*op['boundary_sign'],boundary,atol=3e-15)
        np.testing.assert_allclose(answer['rate'],-(t['incidence']@answer['face_flux'])/h.mass,atol=3e-13)
        self.assertLess(abs(float(h.mass@answer['rate']+boundary.sum())),1e-14)

    def test_rigid_motion_preserves_condensed_constitutive_operator(self):
        h,p,b,t,op=setup();hr,pr,br,tr,rotated=setup(rotation=.41,translation=(.7,-.2))
        np.testing.assert_allclose(op['Q'],rotated['Q'],atol=8e-14,rtol=2e-12)
        np.testing.assert_allclose(op['face_flux_from_cells'],rotated['face_flux_from_cells'],atol=3e-15,rtol=2e-12)

    def test_cell_identity_permutation_preserves_real_face_closure(self):
        h,p,b,t,op=setup();perm=np.arange(len(h.mass))[::-1]
        tp=dict(t);tp['incidence']=t['incidence'][perm];tp['cell_vertices']=t['cell_vertices'][perm]
        other=build_hybrid_fv(tp,h.mass[perm],b[perm],.02)
        np.testing.assert_allclose(other['Q'],op['Q'][np.ix_(perm,perm)],atol=5e-14)
        np.testing.assert_allclose(other['face_flux_from_cells'],op['face_flux_from_cells'][:,perm],atol=3e-15)

    def test_physical_mass_D_scale_and_explicit_zero_D_limit(self):
        h,p,b,t,op=setup();scaled=build_hybrid_fv(t,7*h.mass,b,.02);double=build_hybrid_fv(t,h.mass,b,.04)
        np.testing.assert_allclose(scaled['Q'],op['Q'],atol=5e-14);np.testing.assert_allclose(double['Q'],2*op['Q'],atol=5e-14)
        zero=build_hybrid_fv(t,h.mass,b,0.);answer=hybrid_action(zero,np.linspace(.1,.9,len(h.mass)))
        np.testing.assert_array_equal(answer['rate'],0.);np.testing.assert_array_equal(answer['face_flux'],0.);self.assertIsNone(answer['trace'])
        with self.assertRaises(ValueError):hybrid_action(zero,np.ones(len(h.mass)),np.ones(len(zero['boundary_faces'])))

    def test_basis_initial_derivative_agrees_with_unaveraged_physical_face_flux(self):
        h,p,b,t,op=setup();Q=op['Q'];off=Q.copy();np.fill_diagonal(off,np.inf);i,j=np.unravel_index(np.argmin(off),off.shape)
        state=np.zeros(len(h.mass));state[j]=1.;answer=hybrid_action(op,state)
        self.assertAlmostEqual(answer['rate'][i],Q[i,j]);np.testing.assert_allclose(answer['rate'],-(t['incidence']@answer['face_flux'])/h.mass,atol=2e-13)

    def test_bad_mass_centroid_or_D_rejected_without_floors_or_fallback(self):
        h,p,b,t,op=setup()
        with self.assertRaises(ValueError):build_hybrid_fv(t,-h.mass,b,.02)
        with self.assertRaises(ValueError):build_hybrid_fv(t,h.mass,b,-.02)
        bad=b.copy();bad[0]+=10.
        with self.assertRaises(ValueError):build_hybrid_fv(t,h.mass,bad,.02)


if __name__=='__main__':unittest.main(verbosity=2)
