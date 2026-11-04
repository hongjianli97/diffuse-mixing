"""Independent face-gradient identities, lower bound and sign counterexample."""
import unittest
import numpy as np
from scipy.special import roots_legendre
from becker import Becker
from becker_polygon_benchmark import quad_connectivity,true_vertices,density_velocity
from material_polygon_history import MaterialPolygonHistory,POLYGON_UNITS,signed_area
from nonorthogonal_flux import internal_face_geometry,two_point_flux_matrix,scalar_reconstruction_reference,closed_operator_audit
from nonorthogonal_patch_study import oracle_mass_centroids


def fixture(shear=.6,nx=4,ny=3,rotation=0.,translation=(0.,0.)):
    p=np.array([[x+shear*y,y] for y in np.linspace(0.,1.,ny+1) for x in np.linspace(0.,1.,nx+1)])
    R=np.array([[np.cos(rotation),-np.sin(rotation)],[np.sin(rotation),np.cos(rotation)]])
    p=p@R.T+translation;cells=quad_connectivity(nx,ny);ids=np.array([f'v{i}' for i in range(len(p))])
    mass=signed_area(p[cells]);h=MaterialPolygonHistory([0.,1.],np.stack((p,p)),ids,
              np.array([f'c{i}' for i in range(len(cells))]),ids[cells],mass,np.full(len(cells),.5),
              POLYGON_UNITS,{'geometry_kind':'material_quadrilateral_cells'})
    center=p[cells].mean(axis=1);g=internal_face_geometry(h,p,center);coefficient=.02*g['length']
    return h,p,center,g,coefficient


class NonorthogonalFluxChecks(unittest.TestCase):
    def test_orthogonal_face_flux_and_positive_shared_generator(self):
        h,p,center,g,k=fixture(shear=0.);states=.5+.1*center;exact=-.1*k[:,None]*g['normal']
        for mode in ('normal_distance','center_distance'):
            F,weights=two_point_flux_matrix(g,k,len(h.mass),mode)
            np.testing.assert_allclose(F@states,exact,atol=2e-16,rtol=1e-13)
            audit,_=closed_operator_audit(F,g,h.mass)
            self.assertGreaterEqual(audit['minimum_offdiagonal_generator'],0.)
            self.assertLess(audit['generator_mass_residual'],1e-14)
            self.assertLess(audit['mass_weighted_symmetry_defect'],1e-14)

    def test_sheared_two_point_flux_has_unavoidable_vector_error(self):
        h,p,center,g,k=fixture();states=.5+.1*center;exact=-.1*k[:,None]*g['normal']
        F,_=two_point_flux_matrix(g,k,len(h.mass),'normal_distance')
        relative=np.linalg.norm(F@states-exact,axis=1)/(.1*k)
        np.testing.assert_allclose(relative,.6,atol=2e-14)
        np.testing.assert_allclose(g['any_nonnegative_scalar_relative_error_lower_bound'],.6/np.sqrt(1+.6**2),atol=2e-15)
        # Independent orthogonal projection identity gives the infimum over
        # scalar coefficients. It is a lower bound, not a backend fit.
        projection=np.sum(g['displacement']*g['normal'],axis=1)/np.sum(g['displacement']**2,axis=1)
        distance=np.linalg.norm(projection[:,None]*g['displacement']-g['normal'],axis=1)
        np.testing.assert_allclose(distance,g['any_nonnegative_scalar_relative_error_lower_bound'],atol=2e-15)

    def test_linear_reconstruction_exactness_does_not_imply_positivity(self):
        h,p,center,g,k=fixture();F,gradient,_=scalar_reconstruction_reference(g,k,center)
        states=.5+.1*center
        np.testing.assert_allclose(F@states,-.1*k[:,None]*g['normal'],atol=3e-16,rtol=1e-12)
        np.testing.assert_allclose(gradient@np.ones(len(h.mass)),0.,atol=2e-15)
        audit,arrays=closed_operator_audit(F,g,h.mass)
        self.assertLess(audit['minimum_offdiagonal_generator'],-1e-4)
        self.assertLess(audit['basis_counterexample']['value_at_probe_time'],0.)
        self.assertLess(audit['generator_mass_residual'],1e-14)

    def test_rigid_motion_preserves_geometric_flux_error_and_matrix(self):
        h,p,c,g,k=fixture();hr,pr,cr,gr,kr=fixture(rotation=.41,translation=(-2.,3.))
        F,_=two_point_flux_matrix(g,k,len(h.mass),'normal_distance')
        Fr,_=two_point_flux_matrix(gr,kr,len(hr.mass),'normal_distance')
        np.testing.assert_allclose(F,Fr,atol=5e-16,rtol=3e-14)
        np.testing.assert_allclose(g['angle_radians'],gr['angle_radians'],atol=5e-15)

    def test_physical_mass_scale_D_linearity_and_zero(self):
        h,p,c,g,k=fixture();F,_=two_point_flux_matrix(g,k,len(h.mass),'normal_distance')
        Fs,_=two_point_flux_matrix(g,13*k,len(h.mass),'normal_distance')
        a,aa=closed_operator_audit(F,g,h.mass);b,bb=closed_operator_audit(Fs,g,13*h.mass)
        np.testing.assert_allclose(aa['Q'],bb['Q'],atol=2e-15)
        zero,_=two_point_flux_matrix(g,np.zeros_like(k),len(h.mass),'normal_distance')
        np.testing.assert_array_equal(zero,0.)

    def test_oracle_mass_centroid_matches_independent_density_moment(self):
        f=Becker();edges=f.mu(np.linspace(-1.,1.,9));y=np.array([-.5,0.,.5]);t=.05
        center=oracle_mass_centroids(f,edges,y,t);cells=quad_connectivity(8,2);p=true_vertices(f,edges,y,np.array([t]))[0]
        cell=4;quad=p[cells[cell]];z,w=roots_legendre(64);u=(z+1)/2;weight=w/2
        q=quad[0]+u[:,None,None]*(quad[1]-quad[0])+u[None,:,None]*(quad[3]-quad[0])
        rho,_=density_velocity(f,q[None,...],np.array([t]));weights=rho[0]*weight[:,None]*weight[None,:]
        independent=np.sum(q*weights[...,None],axis=(0,1))/weights.sum()
        np.testing.assert_allclose(center[cell],independent,atol=1e-10,rtol=0.)
        self.assertGreater(float(np.linalg.norm(center[cell]-quad.mean(axis=0))),1e-5)

    def test_invalid_sides_rank_or_coefficients_rejected_without_repair(self):
        h,p,c,g,k=fixture();bad=c.copy();bad[[0,1]]=bad[[1,0]]
        with self.assertRaises(ValueError):internal_face_geometry(h,p,bad)
        with self.assertRaises(ValueError):two_point_flux_matrix(g,-k,len(h.mass),'normal_distance')
        with self.assertRaises(ValueError):scalar_reconstruction_reference(g,-k,c)
        h,p,c,g,k=fixture(ny=1)
        with self.assertRaises(ValueError):scalar_reconstruction_reference(g,k,c)

    def test_zero_linear_cell_divergence_can_hide_wrong_face_flux(self):
        h,p,c,g,k=fixture();states=.5+.1*c;F,_=two_point_flux_matrix(g,k,len(h.mass),'normal_distance')
        _,arrays=closed_operator_audit(F,g,h.mass)
        interior=np.array([5,6])
        np.testing.assert_allclose((arrays['Q']@states)[interior],0.,atol=2e-15)
        self.assertGreater(float(np.max(abs(F@states+.1*k[:,None]*g['normal']))),1e-4)


if __name__=='__main__':unittest.main(verbosity=2)
