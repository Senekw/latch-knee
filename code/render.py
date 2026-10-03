"""Minimal numpy z-buffer renderer for STL part sets (no OpenGL needed)."""
import numpy as np, json, struct, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:5] == b"solid" and b"facet" in data[:300]:
        # ascii
        tris = []
        for line in data.decode(errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("vertex"):
                tris.append([float(v) for v in line.split()[1:4]])
        return np.array(tris).reshape(-1, 3, 3)
    n = struct.unpack("<I", data[80:84])[0]
    arr = np.frombuffer(data[84:84+n*50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return arr["v"].astype(float)

def view_matrix(azim_deg, elev_deg):
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    # camera looks from direction d toward origin; build right/up/forward
    d = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    up = np.array([0, 0, 1.0])
    right = np.cross(up, d); right /= np.linalg.norm(right)
    up2 = np.cross(d, right)
    return np.stack([right, up2, d])   # rows: screen x, screen y, depth (toward camera)

def render(parts, out, azim=-55, elev=25, W=1800, H=1300, explode=None, title=None, light=(0.4, -0.6, 0.7),
           bg=(1, 1, 1), margin=0.06, section_y=None, hide=()):
    """parts: list of dict(name, tris (N,3,3), color). explode: dict name->(dx,dy,dz)."""
    R = view_matrix(azim, elev)
    L = np.array(light, float); L /= np.linalg.norm(L)
    allv = []
    P = []
    for p in parts:
        if p["name"] in hide: continue
        t = p["tris"].copy()
        if explode and p["name"] in explode:
            t = t + np.array(explode[p["name"]], float)
        if section_y is not None:
            # drop triangles entirely on the removed side (y > section_y) -- simple cutaway
            keep = (t[:, :, 1].min(axis=1) <= section_y)
            t = t[keep]
        P.append((t, np.array(p["color"], float)))
        allv.append(t.reshape(-1, 3))
    allv = np.concatenate(allv)
    cam = allv @ R.T
    xmin, ymin = cam[:, 0].min(), cam[:, 1].min(); xmax, ymax = cam[:, 0].max(), cam[:, 1].max()
    span = max(xmax-xmin, (ymax-ymin)*W/H)
    scale = W*(1-2*margin)/span
    cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
    img = np.ones((H, W, 3))*np.array(bg)
    zbuf = np.full((H, W), -np.inf)
    for t, col in P:
        n = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
        nn = np.linalg.norm(n, axis=1); ok = nn > 1e-12
        t, n = t[ok], n[ok]/nn[ok, None]
        shade = 0.35 + 0.65*np.clip(n @ L, 0, 1) + 0.15*np.clip(n @ np.array([0, 0, 1.0]), 0, 1)
        c = t @ R.T
        sx = (c[:, :, 0]-cx)*scale + W/2
        sy = H/2 - (c[:, :, 1]-cy)*scale
        sz = c[:, :, 2]
        for i in range(len(t)):
            x0, x1 = int(max(0, np.floor(sx[i].min()))), int(min(W-1, np.ceil(sx[i].max())))
            y0, y1 = int(max(0, np.floor(sy[i].min()))), int(min(H-1, np.ceil(sy[i].max())))
            if x1 < x0 or y1 < y0: continue
            xs, ys = np.meshgrid(np.arange(x0, x1+1)+0.5, np.arange(y0, y1+1)+0.5)
            ax, ay = sx[i, 0], sy[i, 0]; bx, by = sx[i, 1], sy[i, 1]; qx, qy = sx[i, 2], sy[i, 2]
            det = (bx-ax)*(qy-ay) - (qx-ax)*(by-ay)
            if abs(det) < 1e-9: continue
            l1 = ((bx-xs)*(qy-ys) - (qx-xs)*(by-ys))/det
            l2 = ((qx-xs)*(ay-ys) - (ax-xs)*(qy-ys))/det
            l3 = 1 - l1 - l2
            inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
            if not inside.any(): continue
            z = l1*sz[i, 0] + l2*sz[i, 1] + l3*sz[i, 2]
            sub = zbuf[y0:y1+1, x0:x1+1]
            upd = inside & (z > sub)
            sub[upd] = z[upd]
            img[y0:y1+1, x0:x1+1][upd] = np.clip(col*shade[i], 0, 1)
    fig = plt.figure(figsize=(W/150, H/150), dpi=150)
    ax_ = fig.add_axes([0, 0, 1, 1]); ax_.imshow(img); ax_.axis("off")
    if title: ax_.text(0.02, 0.97, title, transform=ax_.transAxes, fontsize=14, va="top", weight="bold")
    fig.savefig(out, dpi=150); plt.close(fig)
    return img

def crease_edges(tris, angle_deg=28.0):
    """Return (N,2,3) array of edges whose adjacent-face normals differ by > angle, plus boundary edges."""
    v = tris.reshape(-1, 3)
    key = np.round(v, 3)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1, 3)
    n = np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0]); nn = np.linalg.norm(n, axis=1); n = n/np.maximum(nn, 1e-12)[:, None]
    e = np.concatenate([inv[:, [0, 1]], inv[:, [1, 2]], inv[:, [2, 0]]])
    f = np.concatenate([np.arange(len(tris))]*3)
    e_s = np.sort(e, axis=1)
    order = np.lexsort((e_s[:, 1], e_s[:, 0])); e_s, f = e_s[order], f[order]
    same = np.all(e_s[1:] == e_s[:-1], axis=1)
    out = []
    cosang = np.cos(np.radians(angle_deg))
    i = 0
    starts = np.flatnonzero(np.r_[True, ~same])
    ends = np.r_[starts[1:], len(e_s)]
    for s_, t_ in zip(starts, ends):
        faces = f[s_:t_]
        if len(faces) == 1 or (len(faces) == 2 and np.dot(n[faces[0]], n[faces[1]]) < cosang):
            a, b = e_s[s_]
            # recover coordinates from first occurrence
            pa = v[np.argmax(inv.reshape(-1) == a)]; pb = v[np.argmax(inv.reshape(-1) == b)]
            out.append((pa, pb))
    return np.array(out) if out else np.zeros((0, 2, 3))

def render_lines(img, zbuf, segs, R, scale, cx, cy, W, H, color=(0.12, 0.12, 0.14), ztol=0.6, lw=1):
    """Rasterize 3D segments as dark lines with a depth test against zbuf (tolerance ztol in model units)."""
    if len(segs) == 0: return
    c = segs @ R.T
    sx = (c[:, :, 0]-cx)*scale + W/2; sy = H/2 - (c[:, :, 1]-cy)*scale; sz = c[:, :, 2]
    for i in range(len(segs)):
        n = int(max(abs(sx[i, 1]-sx[i, 0]), abs(sy[i, 1]-sy[i, 0]))) + 1
        t = np.linspace(0, 1, n)
        xs = (sx[i, 0] + t*(sx[i, 1]-sx[i, 0])).astype(int); ys = (sy[i, 0] + t*(sy[i, 1]-sy[i, 0])).astype(int)
        zs = sz[i, 0] + t*(sz[i, 1]-sz[i, 0])
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xs, ys, zs = xs[ok], ys[ok], zs[ok]
        vis = zs >= zbuf[ys, xs] - ztol
        for dx in range(lw):
            for dy in range(lw):
                xx = np.clip(xs[vis]+dx, 0, W-1); yy = np.clip(ys[vis]+dy, 0, H-1)
                img[yy, xx] = np.array(color)

def render_cad(parts, out, azim=-55, elev=25, W=1800, H=1300, explode=None, hide=(), section_y=None, title=None,
               edge_angle=28.0, bg=(1, 1, 1), lw=1, light=(0.35, -0.55, 0.75)):
    """CAD-style render: flat shading + crease/silhouette edge lines.  Returns (img, proj) where proj maps 3D->pixel."""
    R = view_matrix(azim, elev)
    L = np.array(light, float); L /= np.linalg.norm(L)
    P = []
    for p in parts:
        if p["name"] in hide: continue
        t = p["tris"].copy()
        if explode and p["name"] in explode: t = t + np.array(explode[p["name"]], float)
        if section_y is not None: t = t[t[:, :, 1].min(axis=1) <= section_y]
        if len(t): P.append((t, np.array(p["color"], float)))
    allv = np.concatenate([t.reshape(-1, 3) for t, _ in P])
    cam = allv @ R.T
    xmin, ymin = cam[:, 0].min(), cam[:, 1].min(); xmax, ymax = cam[:, 0].max(), cam[:, 1].max()
    span = max(xmax-xmin, (ymax-ymin)*W/H); scale = W*0.88/span
    cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
    img = np.ones((H, W, 3))*np.array(bg); zbuf = np.full((H, W), -np.inf)
    for t, col in P:
        n = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]); nn = np.linalg.norm(n, axis=1); ok = nn > 1e-12
        t, n = t[ok], n[ok]/nn[ok, None]
        diff = np.clip(n @ L, 0, 1); spec = np.clip(n @ np.array([0, 0, 1.0]), 0, 1)**8
        shade = 0.42 + 0.55*diff + 0.12*spec
        c = t @ R.T
        sx = (c[:, :, 0]-cx)*scale + W/2; sy = H/2 - (c[:, :, 1]-cy)*scale; sz = c[:, :, 2]
        for i in range(len(t)):
            x0, x1 = int(max(0, np.floor(sx[i].min()))), int(min(W-1, np.ceil(sx[i].max())))
            y0, y1 = int(max(0, np.floor(sy[i].min()))), int(min(H-1, np.ceil(sy[i].max())))
            if x1 < x0 or y1 < y0: continue
            xs, ys = np.meshgrid(np.arange(x0, x1+1)+0.5, np.arange(y0, y1+1)+0.5)
            ax, ay = sx[i, 0], sy[i, 0]; bx, by = sx[i, 1], sy[i, 1]; qx, qy = sx[i, 2], sy[i, 2]
            det = (bx-ax)*(qy-ay) - (qx-ax)*(by-ay)
            if abs(det) < 1e-9: continue
            l1 = ((bx-xs)*(qy-ys) - (qx-xs)*(by-ys))/det; l2 = ((qx-xs)*(ay-ys) - (ax-xs)*(qy-ys))/det; l3 = 1-l1-l2
            inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
            if not inside.any(): continue
            z = l1*sz[i, 0] + l2*sz[i, 1] + l3*sz[i, 2]
            sub = zbuf[y0:y1+1, x0:x1+1]; upd = inside & (z > sub); sub[upd] = z[upd]
            img[y0:y1+1, x0:x1+1][upd] = np.clip(col*shade[i], 0, 1)
    for t, col in P:
        segs = crease_edges(t, edge_angle)
        render_lines(img, zbuf, segs, R, scale, cx, cy, W, H, color=tuple(0.25*c for c in col), lw=lw)
    fig = plt.figure(figsize=(W/150, H/150), dpi=150)
    ax_ = fig.add_axes([0, 0, 1, 1]); ax_.imshow(img); ax_.axis("off")
    if title: ax_.text(0.02, 0.97, title, transform=ax_.transAxes, fontsize=14, va="top", weight="bold")
    fig.savefig(out, dpi=150); plt.close(fig)
    def proj(pt):
        c = np.array(pt, float) @ R.T
        return ((c[0]-cx)*scale + W/2, H/2 - (c[1]-cy)*scale)
    return img, proj

def load_parts(stl_dir="stl", meta="parts.json"):
    meta = json.load(open(meta))
    parts = []
    for m in meta:
        parts.append(dict(name=m["name"], color=m["color"], tris=load_stl(os.path.join(stl_dir, m["name"]+".stl"))))
    return parts
