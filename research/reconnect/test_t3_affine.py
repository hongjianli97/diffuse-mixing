"""Independent clock quadrature/directional geometry, raw laws and failed-model rejection."""
from fractions import Fraction as F
import argparse,copy,json,math,unittest
import numpy as np
from affine_compression_inputs import MODELS,COUNTS,PARAMETER,constitutive_values,pure_inputs,directional_certificate
from oriented_periodic_boundary import OrientedPeriodicRectangle,build_oriented_periodic_faces
from t3_affine_core import integrated_clocks,evaluate


def labels(N):
    axis=(np.arange(N)+.5)/N
    return np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)


def fixture():
    N=16;a=labels(N);m=np.full(N*N,1/(N*N));c0=.5+.2*(math.sin(math.pi/N)/(math.pi/N))**2*np.sin(2*math.pi*a[:,0])*np.cos(2*math.pi*a[:,1])
    world=pure_inputs(a,np.arange(N*N));runs={};paths={};res={};qh=4*N*N*math.sin(math.pi/N)**2
    for model in MODELS:
        for K in COUNTS:
            frames=world['models'][model][str(K)]['frames'];clock=F(0);states=[c0.copy()]
            for i in range(K):
                f=frames[2*i+1];scale=F(f['a_exact']);clock+=F(1,20*K)*F(f['Gamma_actual_exact'])*(scale+1/scale)
                states.append(.5+(c0-.5)*math.exp(-qh*float(clock)))
            states=np.array(states);paths[model,K]={'states':states,'independent_states':states.copy()};res[model,K]=[0.]*K
            runs[model,K]={'steps':[{'cumulative_machine_indicator_absolute':(i+1)*1e-15,'cumulative_tail_indicator_absolute':(i+1)*1e-18} for i in range(K)]}
    return {'mass':m,'c0':c0},world,runs,paths,res


class Checks(unittest.TestCase):
    def test_independent_clock_quadrature_and_directional_faces(self):
        x,w=np.polynomial.legendre.leggauss(32);p=float(PARAMETER)
        for model in MODELS:
            for t in (.025,.05):
                tt=(x+1)*t/2;a=1-5*tt;rho=1/a;D=np.full_like(a,p) if model=='constant_D' else p/rho
                ix=float(w@(D/(a*a))*t/2);iy=float(w@D*t/2)
                actual=integrated_clocks(model,t)
                self.assertAlmostEqual(actual[0],ix,places=16);self.assertAlmostEqual(actual[1],iy,places=16)
            a=labels(4);ids=np.arange(16);world=pure_inputs(a,ids);frame=world['models'][model]['4']['frames'][4]
            boundary=OrientedPeriodicRectangle.from_packet(frame['boundary'])
            local,wp,cert=build_oriented_periodic_faces(ids,frame['world_positions_exact'],frame['rho'],frame['D_float'],boundary)
            result=directional_certificate(local,frame,ids,4);self.assertTrue(result['all_directional_conductances_exact'])
            scale=F(7,8);gamma=F(frame['rho_float'])*F(frame['D_float'])
            self.assertEqual(set(map(F,local['conductances_exact'])),{gamma/scale,gamma*scale})

    def test_raw_constitutive_values_and_nonzero_rounding_defects_preserved(self):
        d=constitutive_values('constant_D',F(3,4));g=constitutive_values('constant_Gamma',F(3,4))
        self.assertEqual(d['rho_float'],g['rho_float']);self.assertEqual(d['rho_float'],float(F(4,3)))
        self.assertEqual(d['D_float'],.02);self.assertEqual(F(g['D_nominal_exact']),PARAMETER*F(3,4))
        self.assertEqual(F(d['Gamma_nominal_exact']),PARAMETER*F(4,3));self.assertEqual(F(g['Gamma_nominal_exact']),PARAMETER)
        self.assertNotEqual(F(g['Gamma_difference_exact']),0)
        self.assertEqual(F(g['Gamma_actual_exact']),F(g['rho_float'])*F(g['D_float']))
        ids=np.arange(16);world=pure_inputs(labels(4),ids);frame=world['models']['constant_Gamma']['4']['frames'][4]
        packet,wp,cert=build_oriented_periodic_faces(ids,frame['world_positions_exact'],frame['rho'],frame['D_float'],OrientedPeriodicRectangle.from_packet(frame['boundary']))
        bad=copy.deepcopy(frame);bad['rho']=[1.]*16
        self.assertFalse(directional_certificate(packet,bad,ids,4)['all_directional_conductances_exact'])
        with self.assertRaises(ValueError):constitutive_values('constant_unknown',F(3,4))

    def test_time_indices_model_response_and_bad_sequence_rejected(self):
        data,world,runs,paths,res=fixture();values=[data['mass'],data['c0']]+[v for p in paths.values() for v in p.values()]
        before=[v.tobytes() for v in values]
        for v in values:v.setflags(write=False)
        result,b=evaluate(data,world,runs,paths,res,N=16)
        self.assertTrue(result['all_frozen_symbol_and_sequence_controls_passed'])
        self.assertTrue(result['all_analytic_reference_error_decompositions_passed'])
        self.assertTrue(result['both_models_time_decreases_resolved'])
        self.assertFalse(result['T3_registered_affine_case_passed'])
        self.assertEqual(result['comparison_decision'],'T3_affine_model_response_inconclusive')
        self.assertTrue(result['model_comparisons'][-1]['model_state_response_resolved'])
        self.assertFalse(result['model_comparisons'][-1]['model_H_response_resolved'])
        t=.025;clock=.02*(t/(1-5*t)+t);reference=.5+(data['c0']-.5)*math.exp(-4*math.pi**2*clock)
        error=float(np.sqrt(data['mass']@((paths['constant_D',16]['states'][8]-reference)**2)))/result['initial_mass_fluctuation_norm']
        self.assertAlmostEqual(result['per_model_observations']['constant_D'][1]['reference_error_relative'],error,places=15)
        self.assertEqual(before,[v.tobytes() for v in values])
        bad={key:{k:v.copy() for k,v in p.items()} for key,p in paths.items()};bad['constant_D',16]['states'][8,0]+=.01
        rejected,_=evaluate(data,world,runs,bad,res,N=16);self.assertFalse(rejected['all_frozen_symbol_and_sequence_controls_passed']);self.assertFalse(rejected['T3_registered_affine_case_passed'])
        bad={key:{k:v.copy() for k,v in p.items()} for key,p in paths.items()}
        for K in COUNTS:
            for field in bad['constant_Gamma',K]:bad['constant_Gamma',K][field]=paths['constant_D',K][field].copy()
        rejected,_=evaluate(data,world,runs,bad,res,N=16);self.assertFalse(rejected['physical_model_response_resolved']);self.assertFalse(rejected['T3_registered_affine_case_passed'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',required=True);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    with open(a.json,'x') as f:json.dump({'successful':r.wasSuccessful(),'checks_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped)},f,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
