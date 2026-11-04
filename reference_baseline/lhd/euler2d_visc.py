"""Navier-Stokes extension of euler2d: constant mu, Pr=0.72, Sc=1 (rho*D = mu), conservative face fluxes."""
import numpy as np, euler2d as E
MU, PR, SC = 5e-4, 0.72, 1.0
class ViscSolver(E.Solver):
    def rhs(self, U):
        R = super().rhs(U)
        Ug = self.ghosts(U); W = E.prim(Ug); dx = self.dx; g = E.G
        r, u, v, p, Y1, Y2 = W; T = p/r; mu = MU; k = mu*g/((g-1)*PR); rD = mu/SC
        # x-faces for interior rows (rows 2:-2), faces between cols 1..nx+1 and 2..nx+2
        sl = slice(2, -2)
        def fx(a): return (a[sl, 2:-1]-a[sl, 1:-2])/dx
        def fy_at_xface(a): return 0.25*((a[3:-1, 1:-2]-a[1:-3, 1:-2])+(a[3:-1, 2:-1]-a[1:-3, 2:-1]))/dx
        def avx(a): return 0.5*(a[sl, 1:-2]+a[sl, 2:-1])
        ux, vx, uy, vy = fx(u), fx(v), fy_at_xface(u), fy_at_xface(v)
        txx = mu*(4/3*ux-2/3*vy); txy = mu*(uy+vx)
        Fx = np.stack([0*txx, txx, txy, avx(u)*txx+avx(v)*txy+k*fx(T), rD*fx(Y1), rD*fx(Y2)])
        # y-faces for interior cols
        def gy(a): return (a[2:-1, sl]-a[1:-2, sl])/dx
        def gx_at_yface(a): return 0.25*((a[1:-2, 3:-1]-a[1:-2, 1:-3])+(a[2:-1, 3:-1]-a[2:-1, 1:-3]))/dx
        def avy(a): return 0.5*(a[1:-2, sl]+a[2:-1, sl])
        uy2, vy2, ux2, vx2 = gy(u), gy(v), gx_at_yface(u), gx_at_yface(v)
        tyy = mu*(4/3*vy2-2/3*ux2); txy2 = mu*(uy2+vx2)
        Fy = np.stack([0*tyy, txy2, tyy, avy(u)*txy2+avy(v)*tyy+k*gy(T), rD*gy(Y1), rD*gy(Y2)])
        return R+(Fx[:, :, 1:]-Fx[:, :, :-1])/dx+(Fy[:, 1:]-Fy[:, :-1])/dx
