"""LATCH knee - Load-Actuated, Thigh-Cable-free, any-angle stance-Control Hinge.
Parametric CadQuery model.  Frame: knee axis = Z (lateral = +Z), anterior = +X, up = +Y.
Right-leg lateral joint.  Knee flexion angle k rotates the SHANK side clockwise (about +Z) by k.

Layers along Z (mm):
  z  -4..0   shank plate INNER  (steel 4 mm, ratchet sector + stop lug + hub)
  z   0..6   thigh plate         (steel 6 mm, hub + upper bar stub)        <- pawl pins pressed here
  z   6..10  shank plate OUTER   (steel 4 mm, clevis, smooth hub)          <- pawls run in a window cut in this plate? no:
             pawls live at z 6..10 in the pocket of the OUTER plate (which is cut away around the hub) -> see below
  z  10..13  printed control deck (lift yoke, levers, direction ring tab)
  z  13..26  printed housing / cover with dampers and mode selector
"""
import cadquery as cq, numpy as np, json, os, math

# ---------------- parameters ----------------
P = dict(
    R_hub=38.0,          # shank hub outer radius (free rim) [mm]
    R_tip=34.0,          # ratchet tooth tip radius
    tooth_depth=3.5,     # radial tooth depth
    pitch_deg=7.5,       # tooth pitch
    teeth_arc=(62.5, 122.5),   # toothed arc in THIGH frame at k=0 [deg, CCW from +X]
    k_range=(-5.0, 30.0),      # knee angles over which teeth are engaged
    pawl_tip_deg=(115.0, 115.0),   # same angle, one pawl per clevis plate; plates phased by half a pitch  # pawl tip angular positions (thigh frame) -> half-pitch phased
    pawl_pivot_r=41.0, pawl_pivot_deg=84.0, pawl_r_out=46.0, pawl_r_in=38.0, pawl_n_teeth=3, bridge_r=42.0, bridge_deg=100.0,   # pivot radius and CCW offset from tip
    t_steel=5.0, t_thigh=6.0,
    bar_w=16.0, bar_t=4.0, bar_len=70.0,
    pin_d=12.0, pawl_pin_d=6.0,
    stop_lug_deg=-15.0,   # stop lug angle on shank hub at k=0
    R_ring=41.0,         # direction ring mean radius (runs on the outer shank plate boss)
)
G = 9.81

def deg(a): return math.radians(a)
def pol(r, a): return (r*math.cos(deg(a)), r*math.sin(deg(a)))

# ---------------- ratchet sector profile ----------------
def ratchet_profile(P, phase=0.0):
    """2D closed polyline of the shank INNER plate hub with teeth on the toothed arc (thigh-frame angles at k=0)."""
    a0, a1 = P["teeth_arc"]; a0 += phase; a1 += phase; Rt, Rr = P["R_tip"], P["R_tip"]-P["tooth_depth"]; p = P["pitch_deg"]
    pts = []
    # smooth hub from a1 around (CCW) to a0+360 at R_hub-? : the free rim must be BELOW root radius so lifted pawls float: use R_free
    Rf = Rt            # free rim at tooth-tip radius: lifted/dropped pawls ride onto the teeth without a step
    n = int(round((a1 - a0) / p))
    # teeth: pawl blocks CW rotation of sector -> load face is the face pointing CW (toward decreasing angle). Make the
    # load face radial with a 6 deg undercut (self-energising), the back face a long ramp.
    for i in range(n):
        ta = a0 + i * p                      # tooth start (CW end of tooth = load face)
        under = 6.0
        pts.append(pol(Rr, ta))             # root at load-face bottom
        pts.append(pol(Rt, ta - under*(Rt-Rr)/Rt*0.5 + 0.0 + under*0.0 + (under/ (2*math.pi*Rt/360)) * 0 ))  # placeholder replaced below
    # simpler explicit construction:
    pts = []
    for i in range(n):
        ta = a0 + i * p
        # load face: from root (Rr, ta) up to tip (Rt, ta + u) where u gives the undercut so face leans toward CW side
        u = -0.6   # deg: tip displaced CW relative to root => face overhangs (undercut) -> pulls pawl in under load
        pts.append(pol(Rr, ta - 0.9))          # root flat before the load face
        pts.append(pol(Rr, ta))
        pts.append(pol(Rt, ta + u))
        pts.append(pol(Rt, ta + u + 0.8))     # short flat tip land
        # ramp back down to next root at ta + p
    pts.append(pol(Rr, a0 + n * p))
    # continue the hub: from end of teeth go CCW along free rim to a0+360 (-> back to start)
    free = [pol(Rf, a) for a in np.linspace(a1 + 2.0, a0 + 360.0 - 2.0, 40)]
    return pts + free

def shank_inner_plate(P):
    prof = ratchet_profile(P)
    hub = cq.Workplane("XY").polyline(prof).close().extrude(P["t_steel"])
    # bar stub downward (-Y)
    bar = (cq.Workplane("XY").center(0, -(P["R_hub"] + P["bar_len"]/2 - 8)).rect(P["bar_w"], P["bar_len"]+16).extrude(P["t_steel"]))
    # stop lug: protrusion on the hub rim at stop_lug_deg, projecting radially to R_hub+10
    lug = (cq.Workplane("XY").polyline([pol(P["R_tip"]-4, P["stop_lug_deg"]-7), pol(P["R_hub"]+9, P["stop_lug_deg"]-4),
                                        pol(P["R_hub"]+9, P["stop_lug_deg"]+4), pol(P["R_tip"]-4, P["stop_lug_deg"]+7)]).close().extrude(P["t_steel"]))
    plate = hub.union(bar).union(lug).translate((0, 0, -P["t_steel"]))
    plate = plate.faces("<Z").workplane().hole(P["pin_d"])
    # rivet holes in bar
    for y in (-(P["R_hub"] + 20), -(P["R_hub"] + 40), -(P["R_hub"] + 60)):
        plate = plate.cut(cq.Workplane("XY").center(0, y).circle(2.6).extrude(20).translate((0, 0, -10)))
    return plate

def shank_outer_plate(P):
    """Clevis outer plate: ring hub (free rim radius R_hub-? ) with a window cut around the pawl region so pawls (z 6..10)
    can reach the inner plate's teeth... Pawls are at z -4..0 (same layer as the inner plate teeth) instead -> see pawl().
    So the outer plate is a plain hub + bar, and carries the direction-ring boss."""
    hub = cq.Workplane("XY").polyline(ratchet_profile(P, phase=P["pitch_deg"]/2)).close().extrude(P["t_steel"])
    bar = (cq.Workplane("XY").center(0, -(P["R_hub"] + P["bar_len"]/2 - 8)).rect(P["bar_w"], P["bar_len"]+16).extrude(P["t_steel"]))
    plate = hub.union(bar).translate((0, 0, P["t_thigh"]))
    plate = plate.cut(cq.Workplane("XY").circle(P["pin_d"]/2).extrude(40).translate((0, 0, -10)))
    for y in (-(P["R_hub"] + 20), -(P["R_hub"] + 40), -(P["R_hub"] + 60)):
        plate = plate.cut(cq.Workplane("XY").center(0, y).circle(2.6).extrude(40).translate((0, 0, -10)))
    # direction-ring boss (cylinder) on the lateral face
    boss = cq.Workplane("XY").circle(P["R_ring"]-2.0).circle(P["pin_d"]/2+4).extrude(3.0).translate((0, 0, P["t_thigh"]+P["t_steel"]))
    return plate.union(boss)

def thigh_plate(P):
    """Thigh plate between the clevis plates: hub (radius R_hub+?) with cut-outs so it does not collide with the ratchet
    sector/stop lug region of the inner plate (different z, so no collision) ; carries pawl pins, stop-lever pivot and upper bar."""
    Rh = P["R_hub"] + 14.0
    hub = cq.Workplane("XY").circle(Rh).extrude(P["t_thigh"])
    bar = cq.Workplane("XY").center(0, (Rh + P["bar_len"]/2 - 8)).rect(P["bar_w"], P["bar_len"]+16).extrude(P["t_thigh"])
    plate = hub.union(bar)
    # lighten: cut a crescent on the anterior-inferior side (no function there)
    for (cx_, cy_, rr) in ((-18, -30, 13), (22, -30, 13), (0, -40, 9)):
        plate = plate.cut(cq.Workplane("XY").center(cx_, cy_).circle(rr).extrude(P["t_thigh"]))
    plate = plate.cut(cq.Workplane("XY").circle(P["pin_d"]/2).extrude(40).translate((0, 0, -10)))
    for y in ((Rh + 20), (Rh + 40), (Rh + 60)):
        plate = plate.cut(cq.Workplane("XY").center(0, y).circle(2.6).extrude(40).translate((0, 0, -10)))
    # pawl pin holes
    x, y = pol(P["pawl_pivot_r"], P["pawl_pivot_deg"])
    plate = plate.cut(cq.Workplane("XY").center(x, y).circle(P["pawl_pin_d"]/2).extrude(40).translate((0, 0, -10)))
    # arc slot for the pawl bridge pin (pawl rotates up to -9 deg about its pivot)
    pv = (x, y); bx, by = pol(P["bridge_r"], P["bridge_deg"])
    for a in np.linspace(0, -12.0, 9):
        c = math.cos(deg(a)), math.sin(deg(a)); dx, dy = bx - pv[0], by - pv[1]
        sx_, sy_ = pv[0] + c[0]*dx - c[1]*dy, pv[1] + c[1]*dx + c[0]*dy
        plate = plate.cut(cq.Workplane("XY").center(sx_, sy_).circle(2.1).extrude(40).translate((0, 0, -10)))
    # stop-lever pivot hole and lift-yoke pivot hole
    sx, sy = pol(P["R_hub"] + 6, P["stop_lug_deg"] + 22)
    plate = plate.cut(cq.Workplane("XY").center(sx, sy).circle(2.0).extrude(40).translate((0, 0, -10)))
    return plate

def annular_sector(r_in, r_out, a0, a1, h, n=16):
    pts = [pol(r_out, a) for a in np.linspace(a0, a1, n)] + [pol(r_in, a) for a in np.linspace(a1, a0, n)]
    return cq.Workplane("XY").polyline(pts).close().extrude(h)

def pawl(P, idx, lifted=False):
    """Hook-shaped tension pawl. idx 0 -> inner layer (z -4..0), idx 1 -> outer layer (z 6..10).
    Tip at (R_root+0.3, 115 deg). Pivot at (41 mm, 84 deg) lies ~10 deg inside the tooth-force line -> load pulls the pawl in."""
    tip_a = P["pawl_tip_deg"][idx]
    Rr = P["R_tip"] - P["tooth_depth"]
    ro, ri = P["pawl_r_out"], P["pawl_r_in"]
    pv = pol(P["pawl_pivot_r"], P["pawl_pivot_deg"])
    h = P["t_steel"]
    body = annular_sector(ri, ro, P["pawl_pivot_deg"] - 4, tip_a - 2.0, h)
    # three serrations matching the ratchet pitch; contact faces at tip_a - j*pitch
    tip = None
    for j in range(P["pawl_n_teeth"]):
        a = tip_a - j * P["pitch_deg"]
        w = cq.Workplane("XY").polyline([pol(ri + 0.01, a + 0.8), pol(P["R_tip"] + 0.01, a - 0.5), pol(Rr + 0.3, a),
                                         pol(Rr + 0.3, a - 1.0), pol(P["R_tip"], a - 6.0), pol(ri + 0.01, a - 7.0)]).close().extrude(h)
        tip = w if tip is None else tip.union(w)
    body = body.union(tip).union(cq.Workplane("XY").center(*pv).circle(5.0).extrude(h))
    body = body.cut(cq.Workplane("XY").center(*pv).circle(P["pawl_pin_d"]/2 + 0.05).extrude(20).translate((0, 0, -10)))
    body = body.cut(cq.Workplane("XY").center(*pol(P["bridge_r"], P["bridge_deg"])).circle(1.8).extrude(20).translate((0, 0, -10)))
    z0 = -P["t_steel"] if idx == 0 else P["t_thigh"]
    body = body.translate((0, 0, z0))
    rot = -11.0 if lifted else 0.0         # CW about pivot lifts the teeth clear (3.5 mm depth)
    return body.rotate((pv[0], pv[1], 0), (pv[0], pv[1], 1), rot)

def pawl_bridge_pin(P, lifted=False):
    """Pin linking the two pawls through an arc slot in the thigh plate; the lift yoke acts on it."""
    pv = pol(P["pawl_pivot_r"], P["pawl_pivot_deg"]); bx, by = pol(P["bridge_r"], P["bridge_deg"])
    pin = cq.Workplane("XY").center(bx, by).circle(1.75).extrude(P["t_steel"]*2 + P["t_thigh"] + 3.5).translate((0, 0, -P["t_steel"]))
    rot = -11.0 if lifted else 0.0
    return pin.rotate((pv[0], pv[1], 0), (pv[0], pv[1], 1), rot)

def knee_pin(P):
    return cq.Workplane("XY").circle(P["pin_d"]/2).extrude(P["t_steel"]*2 + P["t_thigh"] + 3 + 6).translate((0, 0, -P["t_steel"]-3)) \
        .union(cq.Workplane("XY").circle(P["pin_d"]/2 + 3).extrude(2.5).translate((0, 0, -P["t_steel"]-3)))

def bushing(P, z0):
    return cq.Workplane("XY").circle(P["pin_d"]/2 + 1.5).circle(P["pin_d"]/2 + 0.05).extrude(P["t_steel"]).translate((0, 0, z0))

def pawl_pin(P):
    px, py = pol(P["pawl_pivot_r"], P["pawl_pivot_deg"])
    return cq.Workplane("XY").circle(P["pawl_pin_d"]/2).extrude(P["t_steel"]*2 + P["t_thigh"] + 4).translate((px, py, -P["t_steel"]-2))

def stop_lever(P, deflected=False):
    """Sprung extension stop: a lever pivoted on the thigh plate whose pad is hit by the shank stop lug at k = -5 deg.
    Lives in the inner-plate layer (z -4..0) next to the lug; 1 mm travel against the detector spring, then hard stop."""
    sx, sy = pol(P["R_hub"] + 6, P["stop_lug_deg"] + 22)
    # pad position: where the lug front face sits at k=-5 : lug at stop_lug_deg+5 (CCW by 5)
    lx, ly = pol(P["R_hub"] + 5, P["stop_lug_deg"] + 5 + 4.5)
    L = math.hypot(lx-sx, ly-sy); ang = math.degrees(math.atan2(ly-sy, lx-sx))
    body = (cq.Workplane("XY").polyline([(-4, -3.5), (L+3, -3.5), (L+3, 3.5), (-4, 3.5)]).close().extrude(P["t_steel"]).translate((0, 0, -P["t_steel"])))
    body = body.cut(cq.Workplane("XY").circle(2.05).extrude(20).translate((0, 0, -10)))
    return body.rotate((0, 0, 0), (0, 0, 1), ang + (-2.0 if deflected else 0.0)).translate((sx, sy, 0))

def lift_yoke(P, lifted=False):
    """Printed yoke (z 10..13): pivot at posterior-superior hub edge, a fork that straddles the pawl bridge pin."""
    yx, yy = pol(P["R_hub"] + 10, 150.0)
    bx, by = pol(P["bridge_r"], P["bridge_deg"])
    L = math.hypot(bx-yx, by-yy); ang = math.degrees(math.atan2(by-yy, bx-yx))
    body = cq.Workplane("XY").polyline([(-4, -3), (L-4, -3), (L+4, -3), (L+4, -0.3), (L-1.8, -0.3), (L-1.8, 2.2), (L+4, 2.2), (L+4, 3.5), (-4, 3.5)]).close().extrude(3.0)
    # fork is open radially; add an arm toward the direction-ring tab (extension) : tab pushes this arm
    arm = cq.Workplane("XY").polyline([(2, 3.5), (14, 3.5), (14, 9), (2, 9)]).close().extrude(3.0)
    body = body.union(arm).cut(cq.Workplane("XY").circle(2.0).extrude(10).translate((0, 0, -3)))
    body = body.rotate((0, 0, 0), (0, 0, 1), ang).translate((yx, yy, P["t_thigh"] + P["t_steel"] + 0.2))
    rot = 3.0 if lifted else 0.0
    return body.rotate((yx, yy, 0), (yx, yy, 1), rot)

def direction_ring(P, k=0.0):
    """Friction drag ring on the outer shank boss (z 10..13): split steel ring with a radial tab that sits between two stops
    on the lift yoke. Rotates with the shank until the tab hits a stop (deadband = 15 deg)."""
    ring = cq.Workplane("XY").circle(P["R_ring"]).circle(P["R_ring"]-2.6).extrude(3.0)
    gap = cq.Workplane("XY").polyline([(0, 0), pol(60, 268), pol(60, 272)]).close().extrude(10).translate((0, 0, -3))
    ring = ring.cut(gap)
    tab = cq.Workplane("XY").polyline([pol(P["R_ring"]-2.6, 148), pol(P["R_ring"]+9, 149.5), pol(P["R_ring"]+9, 152.5), pol(P["R_ring"]-2.6, 154)]).close().extrude(3.0)
    return ring.union(tab).translate((0, 0, P["t_thigh"] + P["t_steel"] + 0.2))

def housing(P):
    """Printed lateral housing/cover (z 10..26) over the mechanism deck; pockets for two rotary dampers, timer dashpot,
    latch lever, springs and the 3-position mode selector. Modelled as a shell with pockets."""
    Rh = P["R_hub"] + 14.0
    z0 = P["t_thigh"] + P["t_steel"] + 3.2
    shell = cq.Workplane("XY").circle(Rh).extrude(12.0).translate((0, 0, z0))
    inner = cq.Workplane("XY").circle(Rh - 2.5).extrude(9.5).translate((0, 0, z0))
    shell = shell.cut(inner)
    # anterior-inferior relief matching the thigh plate crescent (clearance for shank bar motion is below anyway)
    # damper pockets (two rotary dampers d=15) and timer dashpot pocket
    for (r, a) in ((Rh-12, 170), (Rh-12, 205)):
        x, y = pol(r, a)
        shell = shell.cut(cq.Workplane("XY").center(x, y).circle(7.7).extrude(10).translate((0, 0, z0+9.5)))
    tx, ty = pol(Rh-12, 100)
    shell = shell.cut(cq.Workplane("XY").center(tx, ty).rect(26, 9).extrude(10).translate((0, 0, z0+9.5)))
    # mode selector slot (posterior edge)
    shell = shell.cut(cq.Workplane("XY").center(-Rh+9, 0).rect(6, 22).extrude(10).translate((0, 0, z0+9.5)))
    # screw bosses
    for a in (30, 140, 250):
        x, y = pol(Rh-6, a)
        shell = shell.union(cq.Workplane("XY").center(x, y).circle(3.5).extrude(9.5).translate((0, 0, z0)))
        shell = shell.cut(cq.Workplane("XY").center(x, y).circle(1.6).extrude(40).translate((0, 0, z0-5)))
    # knee-pin clearance / central boss
    shell = shell.cut(cq.Workplane("XY").circle(P["pin_d"]/2 + 3.6).extrude(40).translate((0, 0, z0-5)))
    return shell

def rotary_damper(P, r, a, z):
    x, y = pol(r, a)
    return cq.Workplane("XY").center(x, y).circle(7.5).extrude(9.0).translate((0, 0, z)).union(
        cq.Workplane("XY").center(x, y).rect(3, 3).extrude(4).translate((0, 0, z - 4)))

def timer_dashpot(P, z):
    Rh = P["R_hub"] + 14.0; tx, ty = pol(Rh-12, 100)
    return cq.Workplane("XY").center(tx, ty).rect(24, 8).extrude(8).translate((0, 0, z)).union(
        cq.Workplane("XY").center(tx-15, ty).circle(1.5).extrude(6).translate((0, 0, z+4)).rotate((tx, ty, z), (tx, ty, z+1), 0))

def latch_lever(P, z, set_=False):
    Rh = P["R_hub"] + 14.0
    x, y = pol(Rh-12, 205)
    body = cq.Workplane("XY").polyline([(-3, -2.5), (22, -2.5), (22, 2.5), (-3, 2.5)]).close().extrude(3.0)
    body = body.cut(cq.Workplane("XY").circle(1.6).extrude(10).translate((0, 0, -3)))
    return body.rotate((0, 0, 0), (0, 0, 1), 240 + (10 if set_ else 0)).translate((x, y, z))

def mode_selector(P, z):
    Rh = P["R_hub"] + 14.0
    return cq.Workplane("XY").center(-Rh+9, 0).rect(5, 8).extrude(6).translate((0, 0, z)).union(
        cq.Workplane("XY").center(-Rh+9, 0).rect(5, 30).extrude(2).translate((0, 0, z-2)))

def screw(P, a, z):
    Rh = P["R_hub"] + 14.0; x, y = pol(Rh-6, a)
    return cq.Workplane("XY").center(x, y).circle(1.5).extrude(12).translate((0, 0, z-12)).union(
        cq.Workplane("XY").center(x, y).circle(2.7).extrude(2.0).translate((0, 0, z)))

# ---------------- assembly ----------------
COL = dict(steel=(0.70, 0.72, 0.76), steel_dark=(0.40, 0.43, 0.50), steel_mid=(0.55, 0.60, 0.68), ring=(0.35, 0.55, 0.60), print=(0.20, 0.45, 0.80), print2=(0.85, 0.45, 0.15),
           bronze=(0.72, 0.52, 0.30), damper=(0.15, 0.15, 0.17), spring=(0.75, 0.75, 0.30))

def build(P, k=0.0, lifted=False, stop_deflected=False, latch_set=False):
    """Return list of (name, solid, color, material, printed, qty-group). Shank-side parts rotated by -k about Z (flexion)."""
    parts = []
    def shank(sol): return sol.rotate((0, 0, 0), (0, 0, 1), -k)
    parts.append(("shank_plate_inner_ratchet", shank(shank_inner_plate(P)), COL["steel_dark"], "steel 5 mm laser-cut, hardened", False))
    parts.append(("shank_plate_outer_ratchet", shank(shank_outer_plate(P)), COL["steel"], "steel 5 mm laser-cut, hardened (teeth phased +3.75 deg)", False))
    parts.append(("thigh_plate", thigh_plate(P), COL["steel_mid"], "steel 6 mm laser-cut", False))
    for i in range(2):
        parts.append((f"pawl_{'AB'[i]}", pawl(P, i, lifted), COL["steel_dark"], "steel 5 mm laser-cut, hardened", False))
    parts.append(("pawl_pin", pawl_pin(P), COL["steel"], "dowel pin 6 mm", False))
    parts.append(("pawl_bridge_pin", pawl_bridge_pin(P, lifted), COL["steel"], "dowel pin 3.5 mm", False))
    parts.append(("knee_pin", knee_pin(P), COL["steel"], "shoulder bolt 12 mm", False))
    parts.append(("bushing_inner", bushing(P, -P["t_steel"]), COL["bronze"], "flanged bronze bushing", False))
    parts.append(("bushing_outer", bushing(P, P["t_thigh"]), COL["bronze"], "flanged bronze bushing", False))
    parts.append(("stop_lever", stop_lever(P, stop_deflected), COL["print2"], "steel 5 mm laser-cut", False))
    parts.append(("lift_yoke", lift_yoke(P, lifted), COL["print2"], "PA12-CF printed", True))
    parts.append(("direction_ring", direction_ring(P), COL["ring"], "spring-steel split ring", False))
    parts.append(("housing", housing(P), COL["print"], "PA12-CF printed", True))
    z0 = P["t_thigh"] + P["t_steel"] + 3.2
    Rh = P["R_hub"] + 14.0
    parts.append(("damper_lift", rotary_damper(P, Rh-12, 170, z0+0.6), COL["damper"], "rotary damper 15 mm", False))
    parts.append(("damper_latch", rotary_damper(P, Rh-12, 205, z0+0.6), COL["damper"], "rotary damper 15 mm", False))
    parts.append(("timer_dashpot", timer_dashpot(P, z0+1.0), COL["damper"], "miniature dashpot", False))
    parts.append(("latch_lever", latch_lever(P, z0+0.5, latch_set), COL["print2"], "PA12-CF printed", True))
    parts.append(("mode_selector", mode_selector(P, z0+9.5), COL["print2"], "PA12-CF printed", True))
    for a in (30, 140, 250):
        parts.append((f"screw_M3_{a}", screw(P, a, z0+12.0), COL["steel"], "M3x12 screw", False))
    return parts

def export(parts, stl_dir="stl", step_path="latch_knee_assembly.step"):
    os.makedirs(stl_dir, exist_ok=True)
    meta = []
    asm = cq.Assembly()
    for name, sol, col, mat, printed in parts:
        cq.exporters.export(sol, f"{stl_dir}/{name}.stl", tolerance=0.02, angularTolerance=0.1)
        vol = sol.val().Volume()
        meta.append(dict(name=name, color=col, material=mat, printed=printed, volume_mm3=round(vol, 1)))
        asm.add(sol, name=name, color=cq.Color(*col))
    asm.save(step_path)
    json.dump(meta, open(f"{stl_dir}/parts.json", "w"), indent=1)
    return meta

if __name__ == "__main__":
    parts = build(P, k=0.0)
    meta = export(parts)
    print(len(parts), "parts;", sum(m["volume_mm3"] for m in meta)/1000, "cm3")
