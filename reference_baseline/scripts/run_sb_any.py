import euler2d as E, pickle, sys
dx = float(sys.argv[1])
o = E.run(dx=dx, h_p=0.005, verbose=True)
pickle.dump(o, open('sb_dns_dx%g.pkl' % dx, 'wb')); print('ok', o['nstep'], flush=True)
