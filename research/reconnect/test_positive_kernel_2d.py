"""Normalization, point-input geometry and an independent Fourier action."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import roots_legendre
from trajectory_chain import TrajectorySamples,SI_UNITS,operator_audit
from positive_kernel_2d import C2,kernel_value,slope_over_radius,build_periodic_exchange
from kernel2d_study import fixture,continuum_decay


def altered(data,positions=None,mass=None,rho=None):
    return TrajectorySamples(data.time,data.positions if positions is None else positions,data.ids,
        data.mass if mass is None else mass,data.c0,SI_UNITS,rho=data.rho if rho is None else rho)


class Kernel2DChecks(unittest.TestCase):
    def test_independent_radial_normalization_moment_and_derivative(self):
        h=.09
        norm=quad(lambda r:2*np.pi*r*float(kernel_value(r,h)),0,2*h,epsabs=1e-12)[0]
        moment=quad(lambda r:np.pi*r**3*float(slope_over_radius(r,h)),0,2*h,epsabs=1e-12)[0]
        self.assertAlmostEqual(norm,1.,places=13);self.assertAlmostEqual(moment,1.,places=13)
        r=h*np.array([.17,.63,1.1,1.7]);eps=h*1e-6
        difference=-(kernel_value(r+eps,h)-kernel_value(r-eps,h))/(2*eps*r)
        np.testing.assert_allclose(difference,slope_over_radius(r,h),rtol=3e-8,atol=1e-8)
        self.assertEqual(float(slope_over_radius(0.,h)),(14*C2/3)/h**4)
        self.assertEqual(float(slope_over_radius(2*h,h)),0.)

    def test_periodic_support_enumeration_matches_independent_all_pairs(self):
        d,_,_=fixture(7);L,a,_=build_periodic_exchange(d,1,.02,.19,[1.,1.]);x=d.positions[1]
        i,j=np.triu_indices(len(d.mass),1);delta=x[j]-x[i];delta-=np.rint(delta);r=np.linalg.norm(delta,axis=1);mask=r<.38
        expected=set(zip(i[mask],j[mask]));actual=set(map(tuple,a['pairs'].tolist()));self.assertEqual(actual,expected)
        direct=np.zeros_like(L.toarray());weight=2*d.mass[i[mask]]*d.mass[j[mask]]*.02*slope_over_radius(r[mask],.19)
        direct[i[mask],j[mask]]=-weight;direct[j[mask],i[mask]]=-weight;np.fill_diagonal(direct,-direct.sum(axis=1))
        np.testing.assert_allclose(L.toarray(),direct,atol=3e-16)
        self.assertTrue(np.any(np.linalg.norm(x[a['pairs'][:,1]]-x[a['pairs'][:,0]],axis=1)>.5))

    def test_raw_action_is_independent_discrete_periodic_fourier_symbol(self):
        d,_,_=fixture(12);L,a,_=build_periodic_exchange(d,1,.02,.11,[1.,1.]);rate=-(L@d.c0)/d.mass
        i,j=a['pairs'].T;touch=(i==0)|(j==0);dx=a['minimum_image_displacement'][touch]
        symbol=np.sum(a['conductances'][touch]*(1-np.cos(2*np.pi*(dx[:,0]+dx[:,1]))))/d.mass[0]
        np.testing.assert_allclose(rate,-symbol*(d.c0-.5),atol=5e-14)

    def test_continuum_fourier_two_independent_forms_and_local_limit(self):
        k=2*np.pi*np.sqrt(2);D=.02
        a=continuum_decay(.09,D,k,64,'B');b=continuum_decay(.09,D,k,128,'W')
        self.assertAlmostEqual(a,b,places=12);self.assertGreater(a,0.);self.assertLess(a,D*k*k)
        near=continuum_decay(.0001,D,k,64,'W');self.assertLess(abs(near/(D*k*k)-1),1e-7)

    def test_material_mass_average_is_not_replaced_by_point_sample(self):
        d,centers,factor=fixture(8);z,w=roots_legendre(16);delta=z/(2*8)
        avgx=np.sum(np.cos(2*np.pi*(centers[:,0,None]+delta))*w/2,axis=1)
        avgy=np.sum(np.cos(2*np.pi*(centers[:,1,None]+delta))*w/2,axis=1)
        np.testing.assert_allclose(d.c0,.5+.2*avgx*avgy,atol=2e-16)
        point=.5+.2*np.cos(2*np.pi*centers[:,0])*np.cos(2*np.pi*centers[:,1]);self.assertGreater(np.max(abs(point-d.c0)),.001)

    def test_positive_shared_graph_conserves_constants_mass_and_energy(self):
        d,_,_=fixture(8);m=d.mass*np.linspace(.7,1.4,len(d.mass));rho=d.rho*np.linspace(.8,1.2,len(d.mass))[None,:]
        data=altered(d,mass=m,rho=rho);L,a,diag=build_periodic_exchange(data,1,.02,.19,[1.,1.]);A=L.toarray();Q=-A/m[:,None]
        self.assertTrue(operator_audit(m,Q)['graph_algebra_eligible']);self.assertFalse(diag['physical_acceptance'])
        self.assertTrue(np.all(a['conductances']>=0));np.testing.assert_allclose(A@np.ones(len(m)),0.,atol=3e-16)
        self.assertLess(float(d.c0@(m*(Q@d.c0))),0.);np.testing.assert_allclose(a['quadrature_area'],m/rho[1],atol=0.,rtol=0.)

    def test_physical_density_mass_scaling_D_linearity_and_zero_limit(self):
        d,_,_=fixture(8);L,a,_=build_periodic_exchange(d,1,.02,.19,[1.,1.])
        scaled=altered(d,mass=7*d.mass,rho=7*d.rho);Ls,aa,_=build_periodic_exchange(scaled,1,.02,.19,[1.,1.])
        np.testing.assert_allclose(Ls.toarray()/7,L.toarray(),atol=3e-16);np.testing.assert_allclose(aa['quadrature_area'],a['quadrature_area'],atol=0.)
        double,_,_=build_periodic_exchange(d,1,.04,.19,[1.,1.]);np.testing.assert_allclose(double.toarray(),2*L.toarray(),atol=0.,rtol=0.)
        zero,az,_=build_periodic_exchange(d,1,0.,.19,[1.,1.]);np.testing.assert_array_equal(zero.toarray(),0.);np.testing.assert_array_equal(az['conductances'],0.)

    def test_periodic_translation_rotation_and_material_permutation(self):
        d,_,_=fixture(8);L,a,_=build_periodic_exchange(d,1,.02,.19,[1.,1.])
        translated=altered(d,positions=np.mod(d.positions+np.array([.21,.37]),1.));Lt,_,_=build_periodic_exchange(translated,1,.02,.19,[1.,1.])
        rotated=altered(d,positions=np.mod(np.stack((d.positions[...,1],-d.positions[...,0]),axis=-1),1.));Lr,_,_=build_periodic_exchange(rotated,1,.02,.19,[1.,1.])
        np.testing.assert_allclose(Lt.toarray(),L.toarray(),atol=3e-16);np.testing.assert_allclose(Lr.toarray(),L.toarray(),atol=3e-16)
        p=np.arange(len(d.mass))[::-1];permuted=TrajectorySamples(d.time,d.positions[:,p],d.ids[p],d.mass[p],d.c0[p],SI_UNITS,rho=d.rho[:,p]);Lp,_,_=build_periodic_exchange(permuted,1,.02,.19,[1.,1.])
        np.testing.assert_allclose(Lp.toarray(),L.toarray()[np.ix_(p,p)],atol=3e-16)

    def test_existing_point_interface_needs_no_faces_and_probe_does_not_set_g(self):
        d,_,_=fixture(8);L,a,_=build_periodic_exchange(d,1,.02,.19,[1.,1.])
        changed=TrajectorySamples(d.time,d.positions,d.ids,d.mass,1-d.c0,SI_UNITS,rho=d.rho)
        Lc,ac,_=build_periodic_exchange(changed,1,.02,.19,[1.,1.]);np.testing.assert_array_equal(Lc.toarray(),L.toarray());np.testing.assert_array_equal(ac['conductances'],a['conductances'])
        no_rho=TrajectorySamples(d.time,d.positions,d.ids,d.mass,d.c0,SI_UNITS)
        with self.assertRaises(ValueError):build_periodic_exchange(no_rho,1,.02,.19,[1.,1.])

    def test_invalid_support_periods_frame_and_negative_D_are_rejected(self):
        d,_,_=fixture(8)
        for D,h,box,frame in [(-.02,.19,[1.,1.],1),(.02,.25,[1.,1.],1),(.02,.19,[1.,0.],1),(.02,.19,[1.,1.],2)]:
            with self.assertRaises(ValueError):build_periodic_exchange(d,frame,D,h,box)
        unwrapped=altered(d,positions=d.positions+1.)
        with self.assertRaises(ValueError):build_periodic_exchange(unwrapped,1,.02,.19,[1.,1.])
        with self.assertRaises(ValueError):build_periodic_exchange(d,1,.02,.19,[1.,1.],thickness=2.)


if __name__=='__main__':unittest.main(verbosity=2)
