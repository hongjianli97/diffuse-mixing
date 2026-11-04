"""Independent consistency, conservation and raw sign checks for one candidate."""
import unittest
import numpy as np
from becker import Becker
from becker_polygon_benchmark import true_vertices,quad_connectivity
from material_polygon_history import MaterialPolygonHistory,POLYGON_UNITS
from nonorthogonal_flux import two_point_flux_matrix,closed_operator_audit
from nonorthogonal_patch_study import oracle_mass_centroids
from test_nonorthogonal_flux import fixture
from vertex_mpfa import build_vertex_mpfa,internal_flux,continuity_points
from vertex_mpfa_study import oracle_half_coefficients,prescribed_linear_boundary
from stencil_moment_control import control,sheared_stencil_certificate


def setup(shear=.6,nx=4,ny=3,rotation=0.,translation=(0.,0.)):
    h,p,c,g,k=fixture(shear,nx,ny,rotation,translation)
    a,b=h.edge_vertex_indices.T;halves=np.repeat((.01*np.linalg.norm(p[b]-p[a],axis=1))[:,None],2,axis=1)
    return h,p,c,g,halves,build_vertex_mpfa(h,p,c,halves)


class VertexMPFAChecks(unittest.TestCase):
    def test_isotropic_quadratic_moments_and_nonnegative_support_obstruction(self):
        result,raw=control();certificate=result['certificate']
        np.testing.assert_allclose(raw['first_moment'],0.,atol=2e-15)
        np.testing.assert_allclose(raw['second_moment'],.02*np.eye(2),atol=3e-15)
        # Check actual quadratic cell averages, not just centroid point values.
        # Three-point tensor Gauss integrates these affine-cell polynomials exactly.
        z,w=np.polynomial.legendre.leggauss(3);u=(z+1)/2;weights=w/2
        quad=raw['positions'][quad_connectivity(5,5)]
        q=quad[:,0,None,None,:]+u[None,:,None,None]*(quad[:,1]-quad[:,0])[:,None,None,:]+u[None,None,:,None]*(quad[:,3]-quad[:,0])[:,None,None,:]
        fields=.5+.01*np.stack((q[...,0]**2,q[...,1]**2,q[...,0]*q[...,1]),axis=-1)
        means=np.einsum('a,b,cabf->cf',weights,weights,fields)
        np.testing.assert_allclose(raw['Q'][12]@means,[.0004,.0004,0.],atol=5e-15)
        self.assertLess(result['minimum_interior_offdiagonal_rate'],-.1)
        np.testing.assert_allclose(certificate['cross_to_transverse_second_moment_interval'],[.475,.725])
        self.assertFalse(certificate['zero_cross_moment_is_possible_by_this_necessary_condition'])
        self.assertEqual(certificate['necessary_column_reach_from_cross_moment'],5)
        self.assertTrue(result['all_cross_row_support_products_dx_dy_strictly_positive'])
        self.assertTrue(sheared_stencil_certificate(1/16,.5,.6,radius=5)['zero_cross_moment_is_possible_by_this_necessary_condition'])

    def test_linear_patch_uses_matching_boundary_and_retains_disabled_control(self):
        h,p,c,g,k,op=setup();state=.5+.1*c;data=prescribed_linear_boundary(op,k)
        expected=-.1*k[g['edge_indices']].sum(axis=1)[:,None]*g['normal']
        np.testing.assert_allclose(internal_flux(op,state,data),expected,atol=3e-16,rtol=1e-12)
        self.assertGreater(float(np.max(abs(internal_flux(op,state,np.zeros_like(data))-expected))),1e-4)
        self.assertLess(op['diagnostics']['maximum_local_elimination_residual'],1e-14)

    def test_orthogonal_control_reduces_to_two_point_operator(self):
        h,p,c,g,k,op=setup(shear=0.)
        F,_=two_point_flux_matrix(g,k[g['edge_indices']].sum(axis=1),len(h.mass),'normal_distance')
        np.testing.assert_allclose(op['F'],F,atol=3e-16,rtol=1e-13)
        audit,_=closed_operator_audit(op['F'],g,h.mass)
        # Raw roundoff is retained. The exact orthogonal formula has no negative rates.
        self.assertGreater(audit['minimum_offdiagonal_generator'],-1e-14)

    def test_closed_mass_constant_and_prescribed_global_boundary_balance(self):
        h,p,c,g,k,op=setup();audit,raw=closed_operator_audit(op['F'],g,h.mass)
        self.assertLess(audit['generator_constant_residual'],1e-13)
        self.assertLess(audit['generator_mass_residual'],1e-14)
        state=np.linspace(.1,.8,len(h.mass));data=np.linspace(-.01,.02,op['B'].shape[1])
        q=internal_flux(op,state,data);exterior=np.zeros(len(h.mass))
        cells=np.argmax(h.incidence[:,op['raw']['boundary_half_edges']]!=0,axis=0);np.add.at(exterior,cells,data)
        self.assertLess(abs(np.sum(g['incidence']@q+exterior)-data.sum()),1e-14)

    def test_strong_aspect_shear_retains_positivity_counterexample(self):
        h,p,c,g,k,op=setup(nx=32,ny=2);audit,raw=closed_operator_audit(op['F'],g,h.mass)
        self.assertLess(audit['minimum_offdiagonal_generator'],-.01)
        self.assertLess(audit['basis_counterexample']['value_at_probe_time'],-1e-6)
        self.assertLess(audit['generator_mass_residual'],1e-13)

    def test_rigid_motion_does_not_change_eliminated_flux_maps(self):
        h,p,c,g,k,op=setup();hr,pr,cr,gr,kr,rotated=setup(rotation=.43,translation=(1.,-2.))
        np.testing.assert_allclose(op['F'],rotated['F'],atol=2e-15,rtol=2e-12)
        np.testing.assert_allclose(op['B'],rotated['B'],atol=2e-14,rtol=2e-12)

    def test_cell_identity_permutation_preserves_maps(self):
        h,p,c,g,k,op=setup();perm=np.arange(len(h.mass))[::-1];ids=h.vertex_ids
        hp=MaterialPolygonHistory([0.,1.],np.stack((p,p)),ids,h.cell_ids[perm],ids[h.cell_vertex_indices[perm]],h.mass[perm],h.c0[perm],POLYGON_UNITS,{'geometry_kind':'material_quadrilateral_cells'})
        other=build_vertex_mpfa(hp,p,c[perm],k)
        np.testing.assert_allclose(other['F'],op['F'][:,perm],atol=3e-16)
        np.testing.assert_allclose(other['B'],op['B'],atol=3e-15)

    def test_mass_scale_D_linearity_and_exact_zero_diffusion(self):
        h,p,c,g,k,op=setup();scaled=build_vertex_mpfa(h,p,c,13*k)
        np.testing.assert_allclose(scaled['F'],13*op['F'],atol=5e-16,rtol=1e-13)
        np.testing.assert_allclose(scaled['B'],op['B'],atol=2e-15,rtol=1e-13)
        _,a=closed_operator_audit(op['F'],g,h.mass);_,b=closed_operator_audit(scaled['F'],g,13*h.mass)
        np.testing.assert_allclose(a['Q'],b['Q'],atol=3e-14)
        zero=build_vertex_mpfa(h,p,c,np.zeros_like(k));np.testing.assert_array_equal(internal_flux(zero,np.ones(len(c)),np.zeros(zero['B'].shape[1])),0.)
        with self.assertRaises(ValueError):internal_flux(zero,np.ones(len(c)),np.ones(zero['B'].shape[1]))

    def test_invalid_coefficients_and_exterior_continuity_point_rejected(self):
        h,p,c,g,k,op=setup()
        for bad in [-k,np.full_like(k,np.nan)]:
            with self.assertRaises(ValueError):build_vertex_mpfa(h,p,c,bad)
        mixed=k.copy();mixed[0,0]=0.
        with self.assertRaises(ValueError):build_vertex_mpfa(h,p,c,mixed)
        bad=c.copy();bad[0,1]+=10.
        with self.assertRaises(ValueError):continuity_points(h,p,bad)

    def test_nonconstant_density_half_integrals_match_independent_mass_primitive(self):
        f=Becker();mu=f.mu(np.linspace(-1.,1.,9));y=np.array([-.5,0.,.5]);t=.05
        p=true_vertices(f,mu,y,np.array([t]))[0];cells=quad_connectivity(8,2);ids=np.array([f'v{i}' for i in range(len(p))])
        h=MaterialPolygonHistory([0.,1.],np.stack((p,p)),ids,np.array([f'c{i}' for i in range(len(cells))]),ids[cells],np.tile(np.diff(mu)*.5,2),np.full(len(cells),.5),POLYGON_UNITS,{'geometry_kind':'material_quadrilateral_cells'})
        k=oracle_half_coefficients(h,f,p,t,.02,order=64);a,b=h.edge_vertex_indices.T
        for e in range(len(a)):
            i=a[e]%9
            if a[e]//9!=b[e]//9:
                expected=np.repeat(.02*.25*np.sqrt(1+.6**2)*f.rho_mu(mu[i],t),2)
            else:
                midpoint=(f.x_mu(mu[i],t)+f.x_mu(mu[i+1],t))/2
                z=f.z_at_xi(f.u1*t-midpoint);mu_mid=f.B*f.z_at_xi(1.)+f.j*t-f.B*z
                expected=.02*np.diff([mu[i],mu_mid,mu[i+1]])
            np.testing.assert_allclose(k[e],expected,rtol=2e-12,atol=2e-16)


if __name__=='__main__':unittest.main(verbosity=2)
