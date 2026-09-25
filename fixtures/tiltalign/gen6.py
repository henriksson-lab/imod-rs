# Deterministic synthetic fiducial models for the wave-6 tiltalign differential.
# Writes point2model text (obj cont x y z) plus a .rawtlt of nominal tilt angles.
import math, random, sys

def project(r, P, v, nf, t0, dt, g):
    x, y, z = P
    tilt = t0 + dt * v + g['terr'][v]
    alf = math.radians(g['alpha'] if v >= g['alfview'] else 0.)
    y1 = y * math.cos(alf) - z * math.sin(alf)
    z1 = y * math.sin(alf) + z * math.cos(alf)
    b = math.radians(tilt)
    xp = x * math.cos(b) + z1 * math.sin(b) * g['comp']
    yp = y1
    xp = (1 + g['dmag']) * xp + g['skew'] * yp
    m = g['mag'][v]
    xp *= m; yp *= m
    psi = math.radians(g['rot'] + g['rdrift'] * v)
    X = xp * math.cos(psi) - yp * math.sin(psi) + g['sx'][v] + r.gauss(0, g['noise'])
    Y = xp * math.sin(psi) + yp * math.cos(psi) + g['sy'][v] + r.gauss(0, g['noise'])
    return X, Y

def geom(r, nf, rot=-12., noise=0.3, comp=1.0, dmag=0., skew=0., alpha=0., alfview=10**9,
         rdrift=0.01):
    g = dict(rot=rot, noise=noise, comp=comp, dmag=dmag, skew=skew, alpha=alpha,
             alfview=alfview, rdrift=rdrift)
    g['terr'] = [r.gauss(0, 0.15) for v in range(nf)]
    mag, g['mag'] = 1.0, []
    for v in range(nf):
        mag += r.gauss(0, 0.0015); g['mag'].append(mag)
    g['sx'] = [r.uniform(-25, 25) for v in range(nf)]
    g['sy'] = [r.uniform(-25, 25) for v in range(nf)]
    return g

def beads(name, nf, nreal, seed, nx=1024, ny=1024, t0=-60., dt=3., nobj=1, miss=0.05,
          thick=120., gaps=True, **kw):
    r = random.Random(seed)
    g = geom(r, nf, **kw)
    cnum = [0] * (nobj + 1)
    with open(name + '.txt', 'w') as f:
        for j in range(nreal):
            x = r.uniform(-0.42 * nx, 0.42 * nx); y = r.uniform(-0.42 * ny, 0.42 * ny)
            z = (thick / 2 if r.random() < 0.5 else -thick / 2) + r.gauss(0, 4)
            o = 1 + j % nobj; cnum[o] += 1
            # some beads only on part of the series
            vlo, vhi = 0, nf - 1
            if gaps and j % 7 == 3: vlo = r.randint(1, nf // 4)
            if gaps and j % 11 == 5: vhi = r.randint(3 * nf // 4, nf - 2)
            for v in range(vlo, vhi + 1):
                if j >= 4 and r.random() < miss: continue
                X, Y = project(r, (x, y, z), v, nf, t0, dt, g)
                f.write(f"{o} {cnum[o]} {X + nx / 2:.3f} {Y + ny / 2:.3f} {v}\n")
    with open(name + '.rawtlt', 'w') as f:
        for v in range(nf): f.write(f"{t0 + dt * v:7.2f}\n")

def patches(name, nf, ntrack, seed, nx=1024, ny=1024, t0=-60., dt=3., seglo=4, seghi=12, **kw):
    r = random.Random(seed)
    g = geom(r, nf, **kw)
    with open(name + '.txt', 'w') as f:
        c = 0
        for k in range(ntrack):
            x = r.uniform(-0.45 * nx, 0.45 * nx); y = r.uniform(-0.45 * ny, 0.45 * ny)
            z = r.uniform(-60, 60)
            start = r.randint(0, nf // 3)
            end = r.randint(2 * nf // 3, nf - 1) if r.random() < 0.7 else r.randint(min(start + 4, nf - 1), nf - 1)
            pts = {v: project(r, (x, y, z), v, nf, t0, dt, g) for v in range(start, end + 1)}
            v = start
            while v < end:
                ve = min(end, v + r.randint(seglo, seghi))
                c += 1
                for u in range(v, ve + 1):
                    X, Y = pts[u]
                    f.write(f"1 {c} {X + nx / 2:.3f} {Y + ny / 2:.3f} {u}\n")
                v = ve
    with open(name + '.rawtlt', 'w') as f:
        for v in range(nf): f.write(f"{t0 + dt * v:7.2f}\n")

beads('g1', 41, 30, 1)
beads('g2', 61, 70, 2, nx=2048, ny=2048, t0=-60., dt=2., nobj=3, rot=78.)
beads('g3', 41, 24, 3, comp=0.97, dmag=0.004, skew=0.003, alpha=0.6, alfview=20, rot=5.)
beads('g4', 31, 12, 4, t0=-45., dt=3., miss=0.0, gaps=False, rot=-85.)
beads('g5', 81, 230, 5, nx=2048, ny=1536, t0=-60., dt=1.5, noise=0.5, rot=-3.)
beads('g6', 41, 40, 6, nx=1536, ny=1536, noise=2.0, rot=15.)
patches('p1', 41, 60, 11)
patches('p2', 61, 250, 12, nx=2048, ny=2048, dt=2.)
