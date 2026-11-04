"""Independent scalar, density, material averages and face/volume controls."""
import unittest
import numpy as np
from numpy.polynomial import Polynomial
from becker import Becker
from becker_polygon_benchmark import true_vertices,quad_connectivity,to_world
from nonorthogonal_patch_study import oracle_mass_centroids
from noflux_probe import NoFluxProbe,bump,rho_log_stream_gradient,mass_coordinate_average,volume_reference
from frozen_physical_faces import frozen_faces,physical_scalar_face_flux


def fixture(nx=8,ny=4):
    f=Becker();t=.05;mu=f.mu(np.linspace(-1.,1.,nx+1));eta=np.linspace(-.5,.5,ny+1)
    p=true_vertices(f,mu,eta,np.array([t]))[0];quads=p[quad_connectivity(nx,ny)]
    columns=np.tile(np.arange(nx),ny);rows=np.repeat(np.arange(ny),nx);m=np.tile(np.diff(mu)/ny,ny)
    x=f.x_mu(mu,t);probe=NoFluxProbe((float(x[0]),float(x[-1])))
    return f,mu,eta,quads,columns,rows,m,probe


class NoFluxProbeChecks(unittest.TestCase):
    def test_polynomial_and_world_gradient_match_independent_derivatives(self):
        s=np.linspace(0.,1.,17);p=Polynomial([0.,0.,16.,-32.,16.]);f,fp,fpp=bump(s)
        np.testing.assert_allclose(f,p(s),atol=2e-14);np.testing.assert_allclose(fp,p.deriv()(s),atol=3e-14);np.testing.assert_allclose(fpp,p.deriv(2)(s),atol=6e-14)
        flow,mu,eta,q,c,r,m,probe=fixture();xi=sum(probe.xi_bounds)/2
        point=to_world(np.array([[[xi,.13]]]),np.array([.05]))[0];gradient=probe.world(point)[1]
        h=1e-6
        for k in range(2):
            delta=np.eye(2)[k]*h;difference=(probe.world(point+delta)[0]-probe.world(point-delta)[0])/(2*h)
            np.testing.assert_allclose(gradient[:,k],difference,atol=2e-9,rtol=0.)

    def test_scalar_bounds_and_zero_full_gradient_on_all_real_outer_faces(self):
        flow,mu,eta,q,c,r,m,probe=fixture();L,R=probe.xi_bounds;s=np.linspace(0.,1.,25)
        sides=np.concatenate([np.column_stack((np.full(len(s),x),-.5+s)) for x in (L,R)]+[np.column_stack((L+(R-L)*s,np.full(len(s),y))) for y in (-.5,.5)])
        world=to_world(sides[None,...],np.array([.05]))[0];value,gradient,*_=probe.world(world)
        np.testing.assert_allclose(value,.5,atol=2e-15);np.testing.assert_allclose(gradient,0.,atol=2e-13)
        X,Y=np.meshgrid(L+(R-L)*s,-.5+s);v=probe.canonical(X,Y)[0]
        self.assertGreaterEqual(float(v.min()),.5);self.assertLessEqual(float(v.max()),.75);self.assertAlmostEqual(float(v.max()),.75)

    def test_density_log_gradient_matches_independent_physical_profile_difference(self):
        f=Becker();t=.05;xi=np.array([f.u1*t-.02,f.u1*t,f.u1*t+.02]);h=1e-6
        rho,prime=rho_log_stream_gradient(f,xi,t)
        plus=f.j/f.u(f.z_at_xi(f.u1*t-(xi+h)));minus=f.j/f.u(f.z_at_xi(f.u1*t-(xi-h)))
        independent=(np.log(plus)-np.log(minus))/(2*h)
        np.testing.assert_allclose(prime,independent,atol=1e-6,rtol=1e-7);self.assertTrue(np.all(prime<0))

    def test_real_128_topology_has_shared_oriented_faces(self):
        f,mu,eta,q,c,r,m,probe=fixture(32,4);top=frozen_faces(q,c,r)
        self.assertEqual(len(top['positions']),165);self.assertEqual(len(top['edge_vertices']),292)
        self.assertEqual(int(top['boundary_mask'].sum()),72);self.assertEqual(int((~top['boundary_mask']).sum()),220)
        np.testing.assert_array_equal(top['incidence'][:,~top['boundary_mask']].sum(axis=0),0)
        self.assertEqual(top['interior_flux_cycle_dimension'],93)

    def test_face_flux_divergence_matches_independent_volume_operator_and_mass_balance(self):
        f,mu,eta,q,c,r,m,probe=fixture();top=frozen_faces(q,c,r);flux=physical_scalar_face_flux(top,probe,f,.02,64)
        face=-(top['incidence']@flux)/m;volume=volume_reference(probe,f,q,m,.02,64)
        np.testing.assert_allclose(face,volume['rate'],atol=3e-12,rtol=0.)
        self.assertLess(float(np.max(abs(flux[top['boundary_mask']]))),1e-13)
        self.assertLess(abs(float(m@face)),1e-13)

    def test_true_mass_average_matches_independent_density_quadrature_not_centroid_value(self):
        f,mu,eta,q,c,r,m,probe=fixture();average=mass_coordinate_average(probe,f,mu,eta,c,r,64)
        direct=volume_reference(probe,f,q,m,.02,64)
        np.testing.assert_allclose(average,direct['average'],atol=5e-14,rtol=0.)
        centers=oracle_mass_centroids(f,mu,eta,.05)
        self.assertGreater(float(np.max(abs(average-probe.world(centers)[0]))),1e-4)

    def test_physical_density_scale_preserves_averages_and_rates(self):
        f,mu,eta,q,c,r,m,probe=fixture();scaled=Becker(rho1=7.,p1=7.);top=frozen_faces(q,c,r)
        v1=volume_reference(probe,f,q,m,.02,32);v2=volume_reference(probe,scaled,q,7*m,.02,32)
        np.testing.assert_allclose(v1['average'],v2['average'],atol=5e-14);np.testing.assert_allclose(v1['rate'],v2['rate'],atol=5e-13)
        q1=physical_scalar_face_flux(top,probe,f,.02,32);q2=physical_scalar_face_flux(top,probe,scaled,.02,32)
        np.testing.assert_allclose(q2,7*q1,atol=5e-14)

    def test_constant_state_and_zero_D_have_exact_zero_reference_action(self):
        f,mu,eta,q,c,r,m,probe=fixture();top=frozen_faces(q,c,r)
        constant=NoFluxProbe(probe.xi_bounds,amplitude=0.)
        for state,D in ((constant,.02),(probe,0.)):
            np.testing.assert_array_equal(physical_scalar_face_flux(top,state,f,D,16),0.)
            np.testing.assert_array_equal(volume_reference(state,f,q,m,D,16)['rate'],0.)
        np.testing.assert_array_equal(mass_coordinate_average(constant,f,mu,eta,c,r),.5)

    def test_a_nonzero_internal_cycle_has_zero_cell_divergence(self):
        f,mu,eta,q,c,r,m,probe=fixture(2,2);top=frozen_faces(q,c,r);C=top['incidence'];cycle=np.zeros(C.shape[1])
        for i,j in zip([0,1,3,2],[1,3,2,0]):
            e=np.flatnonzero((C[i]!=0)&(C[j]!=0)).item();cycle[e]=C[i,e]
        self.assertTrue(np.any(cycle!=0));np.testing.assert_array_equal(C@cycle,0.)
        self.assertEqual(top['interior_flux_cycle_dimension'],1)

    def test_bad_scalar_domains_negative_D_and_inconsistent_shared_vertices_rejected(self):
        with self.assertRaises(ValueError):NoFluxProbe((1.,-1.))
        with self.assertRaises(ValueError):NoFluxProbe((-1.,1.),amplitude=.6)
        f,mu,eta,q,c,r,m,probe=fixture();bad=q.copy();bad[1,0,0]+=1e-6
        with self.assertRaises(ValueError):frozen_faces(bad,c,r)
        with self.assertRaises(ValueError):physical_scalar_face_flux(frozen_faces(q,c,r),probe,f,-.02)


if __name__=='__main__':unittest.main(verbosity=2)
