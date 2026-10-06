import math, sys, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import hcss_mc as mc
LS = 2*math.cos(math.radians(15))
F = mc.dodecagon_fillings()

def cell(fill_idx, L=LS, nc=1):
    """nc x nc periodic 3.12.12 cell (one 12-gon per primitive cell), fillings given by fill_idx list."""
    D = L/math.tan(math.radians(15))
    a1 = np.array([D, 0.0]); a2 = np.array([D/2, D*math.sqrt(3)/2])
    V = mc._dodecagon_vertices(L)
    M = np.column_stack([nc*a1, nc*a2])
    Pv, Pi = [], []
    k = 0
    for i in range(nc):
        for j in range(nc):
            c = i*a1 + j*a2
            Pv.append(V + c); Pi.append(F[fill_idx[k % len(fill_idx)]]*L + c); k += 1
    Pv = mc._dedupe_periodic(np.vstack(Pv), M)
    return np.vstack([Pv, np.vstack(Pi)]), M

def pairs_with_images(P, M, rcut):
    inv = np.linalg.inv(M); out = []
    N = len(P)
    for i in range(N):
        for j in range(i, N):
            for n1 in (-1, 0, 1):
                for n2 in (-1, 0, 1):
                    if i == j and (n1, n2) <= (0, 0):
                        continue
                    d = P[j] + n1*M[:, 0] + n2*M[:, 1] - P[i]
                    r = np.linalg.norm(d)
                    if r < rcut:
                        out.append((i, j, n1, n2, r))
    return out

def jam(fill_idx, lam, nc=1, rcut_extra=0.6, verbose=False):
    P0, M0 = cell(fill_idx, LS*1.0005, nc)
    N = len(P0)
    pr = pairs_with_images(P0, M0, LS + rcut_extra)
    I = np.array([p[0] for p in pr]); J = np.array([p[1] for p in pr])
    N1 = np.array([p[2] for p in pr]); N2 = np.array([p[3] for p in pr])
    r0 = np.array([p[4] for p in pr])
    dmin = np.where(r0 < 1.2, 1.0, lam)          # S pairs (core contacts, r ~ 1) keep >= 1; all others >= lam
    # variables: positions of particles 1..N-1 (particle 0 fixed at origin-shift), cell (a, b, c)
    a0, b0, c0 = M0[0, 0], M0[0, 1], M0[1, 1]
    X0 = np.concatenate([(P0[1:] - P0[0]).ravel(), [a0, b0, c0]])
    def unpack(X):
        P = np.vstack([[0.0, 0.0], X[:-3].reshape(N-1, 2)])
        a, b, c = X[-3:]
        return P, np.array([[a, b], [0.0, c]])
    def cons(X):
        P, M = unpack(X)
        d = P[J] + N1[:, None]*M[:, 0] + N2[:, None]*M[:, 1] - P[I]
        return (d**2).sum(1) - dmin**2
    def cons_jac(X):
        P, M = unpack(X)
        d = P[J] + N1[:, None]*M[:, 0] + N2[:, None]*M[:, 1] - P[I]
        G = np.zeros((len(I), len(X)))
        for k in range(len(I)):
            for (p, sgn) in ((J[k], 1.0), (I[k], -1.0)):
                if p > 0:
                    G[k, 2*(p-1):2*(p-1)+2] += 2*sgn*d[k]
            G[k, -3] += 2*d[k, 0]*N1[k]            # da: d_x += n1*a
            G[k, -2] += 2*d[k, 0]*N2[k]            # db: d_x += n2*b
            G[k, -1] += 2*d[k, 1]*N2[k]            # dc: d_y += n2*c
        return G
    obj = lambda X: X[-3]*X[-1] / N
    def obj_grad(X):
        g = np.zeros_like(X); g[-3] = X[-1]/N; g[-1] = X[-3]/N; return g
    res = minimize(obj, X0, jac=obj_grad, method='SLSQP',
                   constraints=[{'type': 'ineq', 'fun': cons, 'jac': cons_jac}],
                   options={'maxiter': 2000, 'ftol': 1e-12})
    P, M = unpack(res.x)
    # verify: no pair (any, recomputed with wider search) closer than allowed
    pr2 = pairs_with_images(P, M, lam - 1e-6)
    newS = sum(1 for p in pr2 if p[4] < lam - 1e-6) - int((r0 < 1.2).sum())
    act = cons(res.x)
    nact = int((np.abs(act) < 1e-6).sum())
    return dict(v=res.x[-3]*res.x[-1]/N, v_ideal=a0*b0*0 + a0*c0/N, success=res.success, newS=newS,
                n_active=nact, n_cons=len(I), N=N)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    idxs = rng.integers(len(F), size=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
    lams = [LS, 1.92, 1.91, 1.90, 1.88, 1.85, 1.82, 1.80]
    vt_star = LS**2*(14*math.sqrt(3)+24)/76
    print('v_t(lam*) ideal =', round(vt_star, 5))
    for lam in lams:
        vs = []; ok = 0; nS = 0; na = []
        for k in idxs:
            r = jam([k], lam)
            vs.append(r['v']); ok += r['success']; nS += max(r['newS'], 0); na.append(r['n_active'])
        vs = np.array(vs)
        print(f"lam={lam:.4f}  v_jam mean {vs.mean():.5f}  sd {vs.std():.5f}  (dv = {vt_star - vs.mean():+.5f}, "
              f"dv/(lam*-lam) = {(vt_star - vs.mean())/max(LS-lam,1e-9):.3f})  converged {ok}/{len(idxs)}  "
              f"active constraints {np.mean(na):.1f}/{r['n_cons']}  new cost pairs {nS}")
