import euler2d as E, euler2d_visc as V, pickle, sys
dx = float(sys.argv[1]); E.Solver = V.ViscSolver
o = E.run(dx=dx, h_p=0.005, verbose=True); o["mu"] = V.MU
pickle.dump(o, open('sbv_dns_dx%g.pkl' % dx, 'wb')); print('ok', o['nstep'], flush=True)
