"""Identity, immutable inputs, shared transport and sampling accuracy."""
from pathlib import Path
import tempfile,unittest
import numpy as np
from experiment import step
from material_face_history import MaterialFaceHistory,SI_UNITS


def inputs():
    t=np.array([0.,.13,.4]);a=np.array([-1.,-.4,.1,1.])
    f=np.array(['f0','f1','f2','f3']);c=np.array(['c0','c1','c2'])
    return {'time':t,'positions':(1-.3*t[:,None])*a+.2*t[:,None],
            'face_ids':f,'cell_ids':c,'cell_faces':np.column_stack((f[:-1],f[1:])),
            'mass':np.array([.7,.4,1.1]),'c0':np.array([.2,.5,.8]),
            'units':SI_UNITS,'metadata':{'geometry_kind':'material_faces'}}


class FaceHistoryChecks(unittest.TestCase):
    def test_affine_motion_nonuniform_times_gcl_and_density(self):
        h=MaterialFaceHistory(**inputs());q=np.array([0.,.04,.13,.27,.4]);v=h.evaluate(q)
        a=np.array([-1.,-.4,.1,1.])
        np.testing.assert_allclose(v['positions'],(1-.3*q[:,None])*a+.2*q[:,None],atol=2e-16)
        np.testing.assert_allclose(v['density']*v['volume'],np.broadcast_to(h.mass,v['volume'].shape),atol=2e-16)
        self.assertLess(h.geometric_conservation()['maximum_volume_increment_residual'],3e-16)
        self.assertFalse(h.geometric_conservation()['materiality_physically_verified_by_this_check'])

    def test_quadratic_sampling_error_is_second_order_and_widths_are_convex(self):
        a=np.array([-1.,-.4,.1,1.]);b=np.array([.05,.08,-.03,.02]);errors=[]
        for intervals in (2,4):
            kw=inputs();t=np.linspace(0.,1.,intervals+1);kw['time']=t;kw['positions']=a+b*t[:,None]**2
            h=MaterialFaceHistory(**kw);q=np.linspace(0.,1.,129);v=h.evaluate(q)
            errors.append(float(np.max(abs(v['positions']-(a+b*q[:,None]**2)))))
            self.assertGreaterEqual(v['volume'].min(),np.diff(kw['positions'],axis=1).min()-2e-16)
            self.assertLess(h.geometric_conservation()['maximum_volume_increment_residual'],3e-16)
        self.assertAlmostEqual(errors[0]/errors[1],4.,places=11)

    def test_changed_identity_connectivity_and_frame_order_are_rejected(self):
        kw=inputs();kw['face_ids']=np.array(['f0','f0','f2','f3'])
        with self.assertRaises(ValueError):MaterialFaceHistory(**kw)
        kw=inputs();kw['cell_faces'][1]=kw['cell_faces'][1,::-1]
        with self.assertRaises(ValueError):MaterialFaceHistory(**kw)
        kw=inputs();frame=np.tile(kw['face_ids'],(3,1));frame[1,[1,2]]=frame[1,[2,1]];kw['frame_face_ids']=frame
        with self.assertRaises(ValueError):MaterialFaceHistory(**kw)

    def test_time_units_boundary_and_extrapolation_are_rejected(self):
        for change in ({'time':np.array([0.,0.,.4])},{'units':{**SI_UNITS,'position':'cm'}},
                       {'boundary_mode':'periodic'},{'metadata':{'geometry_kind':'voronoi'}}):
            kw=inputs();kw.update(change)
            with self.assertRaises(ValueError):MaterialFaceHistory(**kw)
        h=MaterialFaceHistory(**inputs())
        for q in (-.001,.401,np.nan,np.array([[.1]])):
            with self.assertRaises(ValueError):h.evaluate(q)

    def test_invalid_mass_fraction_volume_and_missing_data_are_rejected(self):
        for key,value in [('mass',np.array([.7,0.,1.1])),('c0',np.array([.2,1.1,.8]))]:
            kw=inputs();kw[key]=value
            with self.assertRaises(ValueError):MaterialFaceHistory(**kw)
        for value in (np.nan,-1.1):
            kw=inputs();kw['positions'][1,1]=value
            with self.assertRaises(ValueError):MaterialFaceHistory(**kw)

    def test_immutable_snapshot_and_npz_roundtrip(self):
        kw=inputs();h=MaterialFaceHistory(**kw);kw['positions'][:]=0.;kw['mass'][:]=9.
        self.assertGreater(h.mass.min(),0.);self.assertLess(h.mass.max(),9.)
        with self.assertRaises(ValueError):h.positions[0,0]=0.
        with tempfile.TemporaryDirectory(prefix='ltn-face-test-',dir='/tmp') as tmp:
            p=Path(tmp)/'history.npz';h.save(p);other=MaterialFaceHistory.load(p)
            self.assertEqual(h.identity_digest,other.identity_digest)
            np.testing.assert_array_equal(h.positions,other.positions)
            np.testing.assert_array_equal(other.evaluate(other.time)['positions'],other.positions)
            with self.assertRaises(FileExistsError):h.save(p)

    def test_shared_face_velocity_has_declared_right_interval_knot_convention(self):
        kw=inputs();kw['time']=np.array([0.,.1,.4])
        kw['positions']=np.array([[-1.,-.4,.1,1.],[-.9,-.3,.2,1.1],[-.6,0.,.35,1.4]])
        h=MaterialFaceHistory(**kw)
        velocity=h.evaluate(np.array([0.,.1,.4]))['face_velocity']
        np.testing.assert_allclose(velocity[0],(h.positions[1]-h.positions[0])/.1,atol=1e-15)
        np.testing.assert_allclose(velocity[1],(h.positions[2]-h.positions[1])/.3,atol=1e-15)
        np.testing.assert_array_equal(velocity[1],velocity[2])

    def test_mass_scale_coefficients_and_constant_scalar_require_no_oracle(self):
        kw=inputs();h=MaterialFaceHistory(**kw);kw['mass']=11*kw['mass'];scaled=MaterialFaceHistory(**kw)
        q=np.array([0.,.05,.2,.4]);g=h.conductance(q,.02)
        np.testing.assert_allclose(scaled.conductance(q,.02),11*g,rtol=1e-14)
        np.testing.assert_array_equal(h.conductance(q,0.),np.zeros_like(g))
        c=np.full(3,.37)
        for weights in g:
            c=step(c,h.mass,weights,.01)
        np.testing.assert_allclose(c,.37,atol=2e-15)


if __name__=='__main__':unittest.main(verbosity=2)
