"""Plane-stress FE (scikit-fem, gmsh mesh) of the toothed shank plate and of the 3-tooth pawl under the lock load.
Load case: knee external flexion moment M applied through the engaged pawl -> tangential force F = M / R_c on the
three contact faces, distributed 50/30/20 % (tolerance-driven unequal sharing).  Plate thickness t = 5 mm, steel.
"""
import numpy as np, gmsh, meshio, json, math, sys, os
sys.path.append(os.getcwd())
import cad_knee
from skfem import MeshTri, ElementTriP2, ElementVector, Basis, FacetBasis, asm, solve, condense
from skfem.models.elasticity import linear_elasticity, lame_parameters
from skfem.helpers import sym_grad, ddot, trace, eye

P = cad_knee.P
E, nu, t = 200e3, 0.3, P["t_steel"]            # MPa, -, mm
Rt, Rr = P["R_tip"], P["R_tip"] - P["tooth_depth"]
R_c = 0.5 * (Rt + Rr)
pol = cad_knee.pol

def mesh_polygon(outer, holes, lc_fine, lc_coarse, fine_pts, name):
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(name)
    def loop(pts, lc):
        tags = [gmsh.model.geo.addPoint(x, y, 0, lc) for x, y in pts]
        lines = [gmsh.model.geo.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
        return gmsh.model.geo.addCurveLoop(lines)
    lo = loop(outer, lc_coarse)
    hs = [loop(h, lc_coarse) for h in holes]
    s = gmsh.model.geo.addPlaneSurface([lo] + hs)
    gmsh.model.geo.synchronize()
    # refinement near given points
    f = gmsh.model.mesh.field.add("Distance")
    pts = [gmsh.model.geo.addPoint(x, y, 0, lc_fine) for x, y in fine_pts]
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.field.setNumbers(f, "PointsList", pts)
    th = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(th, "InField", f)
    gmsh.model.mesh.field.setNumber(th, "SizeMin", lc_fine); gmsh.model.mesh.field.setNumber(th, "SizeMax", lc_coarse)
    gmsh.model.mesh.field.setNumber(th, "DistMin", 1.5); gmsh.model.mesh.field.setNumber(th, "DistMax", 10.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(th)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.model.mesh.generate(2)
    gmsh.write(f"{name}.msh"); gmsh.finalize()
    m = meshio.read(f"{name}.msh")
    tri = m.cells_dict["triangle"]
    used = np.unique(tri); remap = -np.ones(len(m.points), int); remap[used] = np.arange(len(used))
    return MeshTri(m.points[used, :2].T.copy(), remap[tri].T.copy())

def solve_plane_stress(mesh, fixed_fn, tractions):
    """tractions: list of (facet_selector_fn(x,y)->bool, (tx,ty) MPa). Returns basis, u, von Mises per element."""
    e = ElementVector(ElementTriP2()); basis = Basis(mesh, e)
    lam, mu = lame_parameters(E, nu)
    lam_ps = 2 * lam * mu / (lam + 2 * mu)       # plane-stress correction
    K = asm(linear_elasticity(lam_ps, mu), basis)
    f = np.zeros(basis.N)
    for sel, (tx, ty) in tractions:
        facets = mesh.facets_satisfying(lambda x: sel(x[0], x[1]), boundaries_only=True)
        if len(facets) == 0: continue
        fb = FacetBasis(mesh, e, facets=facets)
        from skfem import LinearForm
        @LinearForm
        def l(v, w): return tx * v[0] + ty * v[1]
        f += asm(l, fb)
    D = basis.get_dofs(lambda x: fixed_fn(x[0], x[1]))
    u = solve(*condense(K, f, D=D))
    # stresses at element centroids via P0 projection of strain
    from skfem import ElementTriP0, ElementTriP1
    b0 = basis.with_element(ElementVector(ElementTriP0()))
    ug = basis.interpolate(u)
    # evaluate strain at quadrature pts then average per element
    eps = sym_grad(ug)
    exx, eyy, exy = eps[0, 0].mean(axis=1), eps[1, 1].mean(axis=1), eps[0, 1].mean(axis=1)
    sxx = lam_ps * (exx + eyy) + 2 * mu * exx; syy = lam_ps * (exx + eyy) + 2 * mu * eyy; sxy = 2 * mu * exy
    vm = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
    return basis, u, vm, (sxx, syy, sxy)

def face_selector(a_deg, r0, r1, tol=0.35):
    """Select points on the segment from pol(r0,a) to pol(r1,a-0.6) (the tooth load face)."""
    x0, y0 = pol(r0, a_deg); x1, y1 = pol(r1, a_deg - 0.6)
    L = math.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / L, (y1 - y0) / L
    def sel(x, y):
        dx, dy = x - x0, y - y0
        s = dx * ux + dy * uy; d = abs(-dx * uy + dy * ux)
        return (s > -tol) & (s < L + tol) & (d < tol)
    return sel

def run(M_Nm, share=(0.5, 0.3, 0.2), tag=""):
    F = M_Nm / (R_c / 1000.0)                      # N total tangential force
    faces = [P["pawl_tip_deg"][0] - j * P["pitch_deg"] for j in range(3)]      # 115, 107.5, 100 deg
    face_len = math.hypot(*(np.array(pol(Rt, faces[0] - 0.6)) - np.array(pol(Rr, faces[0]))))
    # ---- shank plate sector (inner plate): hub profile with bore hole, fixed at the bore (bushing seat)
    outer = cad_knee.ratchet_profile(P)
    bore = [pol(P["pin_d"] / 2 + 1.5, a) for a in np.linspace(0, 360, 48, endpoint=False)]
    fine = [pol(Rr, a) for a in faces] + [pol(Rt, a - 0.6) for a in faces]
    mesh = mesh_polygon(outer, [bore], 0.18, 2.5, fine, f"sector{tag}")
    tr = []
    for a, s in zip(faces, share):
        p = F * s / (face_len * t)                 # MPa on the face
        # traction direction: tangential CW (toward decreasing angle), i.e. the pawl pushes the tooth face CW
        ang = math.radians(a - 90.0)
        tr.append((face_selector(a, Rr, Rt), (p * math.cos(ang), p * math.sin(ang))))
    r_fix = P["pin_d"] / 2 + 1.5 + 0.3
    print("fixed dofs:", len(Basis(mesh, ElementVector(ElementTriP2())).get_dofs(lambda x: np.hypot(x[0], x[1]) < r_fix)))
    b1, u1, vm1, _ = solve_plane_stress(mesh, lambda x, y: np.hypot(x, y) < r_fix, tr)
    # ---- pawl: explicit outline (annular body + 3 serrations), fixed at pin bore, loaded with the reaction
    ro, ri = P["pawl_r_out"], P["pawl_r_in"]; a_piv, a_end = P["pawl_pivot_deg"], faces[0] + 1.0
    pv = np.array(pol(P["pawl_pivot_r"], P["pawl_pivot_deg"])); rb = 5.0
    outer_p = [pol(ro, a) for a in np.linspace(a_piv, a_end, 30)] + [pol(ri, a_end)]
    for a in faces:   # going from high angle to low angle along the inner edge
        outer_p += [pol(ri, a + 0.6), pol(Rt + 0.01, a - 0.5), pol(Rr + 0.3, a), pol(Rr + 0.3, a - 1.0), pol(Rt, a - 6.0), pol(ri, a - 6.4)]
    a_in_end = P["pawl_pivot_deg"] + 5.0
    outer_p += [pol(ri, a) for a in np.linspace(faces[-1] - 7.5, a_in_end, 10)]
    # pivot boss: circle of radius rb around pv from the inner-edge end point, clockwise, to the outer-arc start point
    p_in = np.array(pol(ri, a_in_end)); p_out = np.array(pol(ro, a_piv))
    phi1 = math.atan2(*(p_in - pv)[::-1]); phi2 = math.atan2(*(p_out - pv)[::-1])
    while phi2 > phi1: phi2 -= 2 * math.pi
    outer_p += [(pv[0] + rb * math.cos(q), pv[1] + rb * math.sin(q)) for q in np.linspace(phi1, phi2, 16)[1:-1]]
    pv = np.array(pol(P["pawl_pivot_r"], P["pawl_pivot_deg"]))
    holes_p = [[(pv[0] + (P["pawl_pin_d"] / 2) * math.cos(q), pv[1] + (P["pawl_pin_d"] / 2) * math.sin(q)) for q in np.linspace(0, 2 * np.pi, 36, endpoint=False)]]
    fine_p = [pol(Rr + 0.3, a) for a in faces] + [pol(Rt, a - 0.5) for a in faces]
    mesh2 = mesh_polygon(outer_p, holes_p, 0.18, 0.7, fine_p, f"pawl{tag}")
    tr2 = []
    for a, s in zip(faces, share):
        p = F * s / (face_len * t); ang = math.radians(a + 90.0)       # reaction on the pawl: CCW tangential
        tr2.append((face_selector(a, Rr + 0.3, Rt + 0.01), (p * math.cos(ang), p * math.sin(ang))))
    b2, u2, vm2, _s = solve_plane_stress(mesh2, lambda x, y: np.hypot(x - pv[0], y - pv[1]) < P["pawl_pin_d"] / 2 + 0.3, tr2)
    return dict(M=M_Nm, F=F, face_len=face_len, sector=(mesh, b1, u1, vm1), pawl=(mesh2, b2, u2, vm2))


def run_sector_only(M_Nm, share=(0.5, 0.3, 0.2), tag=""):
    F = M_Nm / (R_c / 1000.0)
    faces = [P["pawl_tip_deg"][0] - j * P["pitch_deg"] for j in range(3)]
    face_len = math.hypot(*(np.array(pol(Rt, faces[0] - 0.6)) - np.array(pol(Rr, faces[0]))))
    outer = cad_knee.ratchet_profile(P)
    bore = [pol(P["pin_d"] / 2 + 1.5, a) for a in np.linspace(0, 360, 48, endpoint=False)]
    fine = [pol(Rr, a) for a in faces] + [pol(Rt, a - 0.6) for a in faces]
    mesh = mesh_polygon(outer, [bore], 0.15, 2.0, fine, f"sector{tag}")
    tr = []
    for a, s in zip(faces, share):
        p = F * s / (face_len * t); ang = math.radians(a - 90.0)
        tr.append((face_selector(a, Rr, Rt), (p * math.cos(ang), p * math.sin(ang))))
    r_fix = P["pin_d"] / 2 + 1.5 + 0.3
    return solve_plane_stress(mesh, lambda x, y: np.hypot(x, y) < r_fix, tr)

if __name__ == "__main__":
    out = {}
    for M in (36.0, 70.0, 150.0):
        r = run(M, tag=f"_{int(M)}")
        out[M] = dict(F_N=r["F"], vm_sector_max=float(np.percentile(r["sector"][3], 99.5)), vm_sector_absmax=float(r["sector"][3].max()),
                      vm_pawl_max=float(np.percentile(r["pawl"][3], 99.5)), vm_pawl_absmax=float(r["pawl"][3].max()),
                      n_el_sector=r["sector"][0].t.shape[1], n_el_pawl=r["pawl"][0].t.shape[1])
        print(M, {k: round(v, 1) if isinstance(v, float) else v for k, v in out[M].items()})
    json.dump(out, open("fea_results.json", "w"), indent=1)
