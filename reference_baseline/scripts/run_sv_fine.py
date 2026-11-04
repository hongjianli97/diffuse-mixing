import sv_case, pickle, sys
dx = float(sys.argv[1])
class SVf(sv_case.SV):
    pass
orig = sv_case.SV.__init__
def init(self, dx=dx, **k): orig(self, dx=dx, **k)
sv_case.SV.__init__ = init
o = sv_case.run(t_end=0.6)
pickle.dump(o, open('sv_dns_dx%g.pkl' % dx, 'wb')); print('ok', o['nstep'], flush=True)
