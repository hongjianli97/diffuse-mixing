"""Three independent small fixtures; no extra scientific flow/space cases."""
from fractions import Fraction as F
import argparse,copy,json,math,unittest
import numpy as np
from t1_uniform_core import pure_labels
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from contact_oracle_core import pure_contact,face_control,derive_material
from material_union import group_sum,observe_path
from contact_spatial_core import compare


def fixture():
    data={};meta={};runs={};paths={};controls={};cm={}
    for N in (8,16,32):
        a=pure_labels(N);rho=np.where(a[:,0]<.5,1.,4.);m=rho/N**2
        phi=.15*(math.sin(math.pi/N)/(math.pi/N))**2*np.cos(2*math.pi*a[:,0])*np.cos(2*math.pi*a[:,1]);c0=.5+phi
        ref=np.array([.5+phi*math.exp(-1.5*t) for t in (0.,.025,.05)])
        state=np.array([.5+phi*math.exp(-(1.5-1/N**2)*i/320) for i in range(17)])
        data[N]={'mass':m,'c0':c0,'reference':ref};paths[N]={'states':state,'independent_states':state.copy()};controls[N]={'states':state.copy()}
        cm[N]={'rows':[{'total_absolute':1e-14} for _ in range(17)]}
        meta[N]={'reference_evaluation_indicator_absolute':1e-14,'initial_serialization_bridge_absolute':1e-17,'float_serialization_mass_norms':[1e-17]*3,
          'tail_certificates_by_time':[None,{'material_mass_norm_tail_bound_exact':'1/1000000000000000000000000000000'},{'material_mass_norm_tail_bound_exact':'1/1000000000000000000000000000000'}]}
        runs[N]={'steps':[{'cumulative_machine_indicator_absolute':(i+1)*1e-15,'cumulative_tail_indicator_absolute':0.} for i in range(16)]}
    return data,meta,runs,paths,controls,cm


class Checks(unittest.TestCase):
    def test_multiN_point_face_laws_and_original_finite_cell_formula(self):
        case={'K':128,'requested_decimal_digits':90,'alpha':['0']*129,'beta':['0']*129};z,w=np.polynomial.legendre.leggauss(16)
        for N in (8,16):
            p=pure_contact(pure_labels(N),np.arange(N*N));packet=coefficient_packet(build_periodic_faces(p['ids'],p['positions'],p['rho'],.02))
            self.assertTrue(face_control(packet,p,N)['all_world_axis_faces_and_two_half_resistances_exact'])
            d,meta=derive_material(case,p,N);means=[]
            for x,y in p['positions']:
                a,b=(16/5,1/5) if x<.5 else (4/5,7/5);xx=x+z[:,None]/(2*N);yy=y+z[None,:]/(2*N)
                means.append(float(np.sum((.5+.2*np.cos(math.pi*(a*xx+b))*np.cos(2*math.pi*yy))*w[:,None]*w[None,:])/4))
            np.testing.assert_allclose(d['c0'],means,rtol=0,atol=3*np.finfo(float).eps)
            self.assertGreater(meta['initial_serialization_bridge_absolute'],0)
            bad=copy.deepcopy(packet);bad['conductances_exact'][0]=str(2*F(bad['conductances_exact'][0]))
            with self.assertRaises(AssertionError):face_control(bad,p,N)

    def test_fixed_parent_mass_component_integrals_and_nonzero_capacity_bridge(self):
        side=4;ratio=2;m=np.arange(1,65,dtype=float)/1024;c=np.arange(64,dtype=float)/128;mp=group_sum(m,side,ratio)
        actual=observe_path(c[None,:],m,mp,side,ratio)[0];expected=[]
        for y in range(side):
            for x in range(side):
                ids=[(2*y+dy)*8+2*x+dx for dy in (0,1) for dx in (0,1)]
                expected.append(float(sum((F(float(m[i]))*F(float(c[i])) for i in ids),F(0))/F(float(mp[y*side+x]))))
        np.testing.assert_array_equal(actual,expected)
        changed=mp*2;np.testing.assert_array_equal(observe_path(np.ones((1,64)),m,changed,side,ratio)[0],np.full(16,.5))
        args=fixture();args[0][16]['c0'][0]=math.nextafter(args[0][16]['c0'][0],math.inf)
        old=args[0][16]['c0'].tobytes();r,p=compare(*args,base=8);self.assertEqual(old,args[0][16]['c0'].tobytes())
        self.assertGreater(r['adjacent_spatial_comparisons'][0]['common_indicator_components_absolute']['n16_initial_projection_bridge'],0)

    def test_positive_budget_classification_and_native_hidden_error_rejection(self):
        args=fixture();values=[a for d in args[0].values() for a in d.values()]+[a for p in args[3].values() for a in p.values()]
        before=[a.tobytes() for a in values]
        for a in values:a.setflags(write=False)
        r,p=compare(*args,base=8);self.assertTrue(r['registered_three_space_level_contact_oracle_control_passed'])
        for row in r['adjacent_spatial_comparisons']:
            for key,field,den in [('native','native_indicator_components_absolute',r['common_initial_fluctuation_norm']),('common','common_indicator_components_absolute',r['common_initial_fluctuation_norm']),('native_H','native_H_indicator_components',1.),('common_H','common_H_indicator_components',1.)]:
                self.assertEqual(row['fixed_indicators'][key],sum(row[field].values())/den);self.assertGreaterEqual(min(row[field].values()),0)
        self.assertEqual(before,[a.tobytes() for a in values])
        badpaths={N:{k:v.copy() for k,v in p.items()} for N,p in args[3].items()};badpaths[32]['independent_states'][8,0]+=.01
        rejected,_=compare(args[0],args[1],args[2],badpaths,args[4],args[5],base=8);self.assertFalse(rejected['all_grid_action_reference_and_H_controls_passed'])
        bad=fixture();osc=np.tile([1.,-1.],512)
        for i in range(1,17):
            for field in ('states','independent_states'):bad[3][32][field][i]+=.01*i/16*osc
            bad[4][32]['states'][i]+=.01*i/16*osc
        r,p=compare(*bad,base=8);last=r['adjacent_spatial_comparisons'][-1]
        self.assertEqual(last['decisions']['common'],'fine_better_resolvably');self.assertEqual(last['decisions']['native'],'fine_worse_resolvably')
        self.assertFalse(r['registered_three_space_level_contact_oracle_control_passed'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
