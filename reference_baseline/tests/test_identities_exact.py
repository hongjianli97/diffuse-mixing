"""Exact symbolic verification (sympy, no floating point) of every algebraic identity used in the paper.

Each test returns True only if the difference of the two sides simplifies to exactly zero.
Run:  python tests/test_identities_exact.py      (or pytest tests/)
"""
import sympy as sp

R90 = sp.Matrix([[0, -1], [1, 0]])


def cr(a, b):
    return a[0] * b[1] - a[1] * b[0]


def pts(prefix, n):
    return [sp.Matrix(sp.symbols(f"{prefix}x{k} {prefix}y{k}", real=True)) for k in range(n)]


def area(P):
    n = len(P)
    return sp.Rational(1, 2) * sum(cr(P[k], P[(k + 1) % n]) for k in range(n))


def pq(F):
    p = sp.Rational(1, 2) * ((F[0, 0] + F[1, 1]) + sp.I * (F[1, 0] - F[0, 1]))
    q = sp.Rational(1, 2) * ((F[0, 0] - F[1, 1]) + sp.I * (F[1, 0] + F[0, 1]))
    return p, q


def tri_F(X, Y):
    E0 = sp.Matrix.hstack(X[1] - X[0], X[2] - X[0])
    E1 = sp.Matrix.hstack(Y[1] - Y[0], Y[2] - Y[0])
    return E1 * E0.adjugate() / E0.det(), E0.det() / 2


def displacement_cochain_sum(X, Y):
    """sum over the closed polygon of Re and Im of conj(u_mid) q_mid"""
    n = len(X); re = 0; im = 0
    for k in range(n):
        a, b = k, (k + 1) % n
        um = (Y[a] - X[a] + Y[b] - X[b]) / 2
        qm = ((X[b] - X[a]) + (Y[b] - Y[a])) / 2
        re += um.dot(qm); im += cr(um, qm)
    return re, im


# ---------------------------------------------------------------- Theorem 1 (any closed polygon)
def test_thm1_polygon(nmax=5):
    ok = True
    for n in range(3, nmax + 1):
        X, Y = pts("a", n), pts("b", n)
        re, im = displacement_cochain_sum(X, Y)
        ok &= sp.expand(im - (area(Y) - area(X))) == 0                      # imaginary part = area change
        fan = 0
        for k in range(1, n - 1):                                           # any triangulation (here: a fan)
            F, A0 = tri_F([X[0], X[k], X[k + 1]], [Y[0], Y[k], Y[k + 1]])
            fan += 2 * A0 * sp.im(pq(F)[0])
        ok &= sp.simplify(sp.together(re - fan)) == 0                       # real part = sum 2 A Im p
    return ok


# ---------------------------------------------------------------- Theorem 2 (kinematic circulation identity)
def test_thm2_polygon(nmax=4):
    c = sp.symbols("c", real=True); ok = True
    for n in range(3, nmax + 1):
        Pm, Po, Pp = pts("m", n), pts("o", n), pts("p", n)
        lhs = displacement_cochain_sum(Po, Pp)[0] - displacement_cochain_sum(Pm, Po)[0]
        rhs = 0
        for k in range(n):
            a, b = k, (k + 1) % n
            d2a, d2b = Pp[a] - 2 * Po[a] + Pm[a], Pp[b] - 2 * Po[b] + Pm[b]
            q = Po[b] - Po[a]
            qt = q + c * ((Pp[b] - Pp[a]) - 2 * q + (Pm[b] - Pm[a]))
            rhs += ((d2a + d2b) / 2).dot(qt)
        ok &= sp.expand(lhs - rhs) == 0
    return ok


# ---------------------------------------------------------------- three-point map algebra
def generic_F(s):
    a, b, c, d = sp.symbols(f"{s}a {s}b {s}c {s}d", real=True)
    return sp.Matrix([[a, b], [c, d]])


def test_pq_algebra():
    F1, F2 = generic_F("f"), generic_F("g")
    p1, q1 = pq(F1); p2, q2 = pq(F2); p, q = pq(F2 * F1)
    ok = sp.expand(p - (p2 * p1 + q2 * sp.conjugate(q1))) == 0
    ok &= sp.expand(q - (p2 * q1 + q2 * sp.conjugate(p1))) == 0
    ok &= sp.expand(F1.det() - (p1 * sp.conjugate(p1) - q1 * sp.conjugate(q1))) == 0      # J = |p|^2 - |q|^2
    # action on a complex number: F zeta = p zeta + q conj(zeta)
    x, y = sp.symbols("x y", real=True); z = x + sp.I * y; v = F1 * sp.Matrix([x, y])
    ok &= sp.expand(v[0] + sp.I * v[1] - (p1 * z + q1 * sp.conjugate(z))) == 0
    return bool(ok)


def test_frame_rule():
    F = generic_F("f"); b0, b1 = sp.symbols("b0 b1", real=True)
    Q = lambda b: sp.Matrix([[sp.cos(b), -sp.sin(b)], [sp.sin(b), sp.cos(b)]])
    p, q = pq(F); pn, qn = pq(Q(b1).T * F * Q(b0))
    ok = sp.simplify(sp.expand_complex(pn - sp.exp(-sp.I * (b1 - b0)) * p)) == 0
    ok &= sp.simplify(sp.expand_complex(qn - sp.exp(-sp.I * (b1 + b0)) * q)) == 0
    return bool(ok)


# ---------------------------------------------------------------- uniaxial signature is a characterisation
def test_uniaxial_iff():
    """On the branch arg p = 0 (F symmetric, positive trace) with eigenvalues l1,l2 > 0:
    |mu| = tanh(|ln J|/2)  <=>  (l1-1)(l2-1) = 0.  We verify the squared, rationalised condition factorises."""
    l1, l2 = sp.symbols("l1 l2", positive=True)
    mu2 = (l1 - l2) ** 2 / (l1 + l2) ** 2
    th2 = (1 - l1 * l2) ** 2 / (1 + l1 * l2) ** 2                          # tanh^2(|ln J|/2), J = l1 l2
    num = sp.factor(sp.numer(sp.together(mu2 - th2)))
    target = sp.factor(-4 * l1 * l2 * (l1 - 1) * (l1 + 1) * (l2 - 1) * (l2 + 1))
    ok = sp.simplify(num - target) == 0 or sp.simplify(num + target) == 0
    # and the planar-shock map F = I - (1 - 1/eta) n n^T has eigenvalues (1, 1/eta) and real p
    eta, th = sp.symbols("eta theta", positive=True); n = sp.Matrix([sp.cos(th), sp.sin(th)])
    F = sp.eye(2) - (1 - 1 / eta) * n * n.T; p, q = pq(F)
    ok &= sp.simplify(sp.im(p)) == 0
    mod2 = lambda z: sp.expand_complex(z * sp.conjugate(z))
    ok &= sp.simplify(mod2(q) / mod2(p) - ((eta - 1) / (eta + 1)) ** 2) == 0
    return bool(ok)


# ---------------------------------------------------------------- Nanson in 2D
def test_nanson_2d():
    F = generic_F("f"); th = sp.symbols("th", real=True)
    N = sp.Matrix([sp.cos(th), sp.sin(th)]); T = R90.T * N
    lhs = (F * T).dot(F * T)
    rhs = F.det() ** 2 * (F.inv().T * N).dot(F.inv().T * N)
    return sp.simplify(lhs - rhs) == 0


# ---------------------------------------------------------------- reduced ring and polygon perturbation
def test_reduced_ring_affine():
    """any vertex subset (>=3, original order) of a ring moved affinely has area ratio det F"""
    F = generic_F("f"); cvec = sp.Matrix(sp.symbols("c1 c2", real=True)); X = pts("r", 6)
    Y = [F * x + cvec for x in X]; ok = True
    for sub in ([0, 1, 2, 3, 4, 5], [0, 2, 3, 5], [1, 3, 4], [0, 1, 4, 5]):
        ok &= sp.expand(area([Y[k] for k in sub]) - F.det() * area([X[k] for k in sub])) == 0
    return bool(ok)


def test_polygon_area_expansion():
    """A(x+e) = A(x) + sum_k dA/dx_k . e_k + A(e), dA/dx_k = 1/2 R(-90)(x_{k+1}-x_{k-1}): exact (area is quadratic)"""
    n = 6; Xs, Es = pts("x", n), pts("e", n)
    lin = sum(sp.Rational(1, 2) * cr(Es[k], Xs[(k + 1) % n] - Xs[(k - 1) % n]) for k in range(n))
    full = area([Xs[k] + Es[k] for k in range(n)])
    return sp.expand(full - (area(Xs) + lin + area(Es))) == 0


def test_noise_constant_hexagon():
    pts6 = [sp.Matrix([sp.cos(k * sp.pi / 3), sp.sin(k * sp.pi / 3)]) for k in range(6)]
    S = sum((pts6[(k + 1) % 6] - pts6[(k - 1) % 6]).dot(pts6[(k + 1) % 6] - pts6[(k - 1) % 6]) for k in range(6))
    cstar = sp.sqrt(S) / (2 * area(pts6))
    return sp.simplify(cstar - sp.sqrt(sp.Rational(2, 3))) == 0           # = 0.8165


TESTS = [test_thm1_polygon, test_thm2_polygon, test_pq_algebra, test_frame_rule, test_uniaxial_iff,
         test_nanson_2d, test_reduced_ring_affine, test_polygon_area_expansion, test_noise_constant_hexagon]

if __name__ == "__main__":
    import json, time
    res = {}
    for f in TESTS:
        t0 = time.time(); r = bool(f()); res[f.__name__] = r
        print("%-32s %s  (%.1fs)" % (f.__name__, "PASS" if r else "FAIL", time.time() - t0), flush=True)
    json.dump(res, open("results/res_t_identities_exact.json", "w"), indent=1)
