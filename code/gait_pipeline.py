"""Preprocess WBDS (Fukuchi et al. 2018) treadmill trials into the sagittal-plane
quantities needed to simulate a thigh-mounted pendulum stance-control trigger.

Per trial we produce (right leg, lateral side):
  t            time [s] (150 Hz)
  theta        thigh absolute inclination from vertical [rad], + = knee anterior of hip
  knee         knee flexion angle [rad] from markers (thigh vs shank segments)
  pivot[s]     position (x fwd, y up) [m] of a point on the thigh upright at height s above the knee
  acc[s]       its acceleration (x fwd, y up) [m/s^2]
  hs, to       right heel-strike and toe-off times [s] from the right-foot plate
  knee_mom     knee moment curve (N.m/kg, 0..100 %GC, + = internal extensor = external flexion load)
"""
import numpy as np, pandas as pd, os, glob, json
from scipy.signal import butter, filtfilt

D = "data/51subjs"
FS_MK, FS_GRF = 150.0, 300.0
HEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20]          # pendulum mount height above knee axis [m]

def lp(x, fc, fs, order=4):
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, x, axis=0)

def events_from_plate(fz, fs, thr=30.0, min_gap=0.25):
    on = fz > thr
    d = np.diff(on.astype(int))
    hs = np.where(d == 1)[0] + 1
    to = np.where(d == -1)[0] + 1
    hs_t, to_t = hs / fs, to / fs
    # de-bounce
    def deb(ts):
        out = []
        for t in ts:
            if not out or t - out[-1] > min_gap:
                out.append(t)
        return np.array(out)
    return deb(hs_t), deb(to_t)


def contacts_from_plate(gr, pl, mk, fs=FS_GRF, thr=20.0, min_dur=0.3, tol=0.18):
    """Return (hs_times, to_times) of RIGHT-foot contacts on plate `pl`, validated by COP vs right-foot markers."""
    fz = gr[f"Fy{pl}"].values; on = fz > thr
    d = np.diff(np.concatenate([[0], on.astype(int), [0]]))
    starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
    hs, to = [], []
    rhx = mk["R.HeelX"].values; raz = mk["R.AnkleZ"].values
    for s, e in zip(starts, ends):
        if (e - s) / fs < min_dur: continue
        k = min(s + 6, e - 1); j = min(int(k / fs * FS_MK), len(mk) - 1)
        cx, cz = gr[f"COPx{pl}"].values[k], gr[f"COPz{pl}"].values[k]
        if abs(cx - rhx[j]) / 1000 > tol or abs(cz - raz[j]) / 1000 > tol: continue
        hs.append(s / fs); to.append(e / fs)
    return np.array(hs), np.array(to)

def process_trial(subj, sp):
    base = f"{D}/WBDS{subj:02d}walkT{sp:02d}"
    if not os.path.exists(base + "mkr.txt"):
        return None
    mk = pd.read_csv(base + "mkr.txt", sep="\t")
    gr = pd.read_csv(base + "grf.txt", sep="\t")
    knt = pd.read_csv(base + "knt.txt", sep="\t")
    t = np.arange(len(mk)) / FS_MK
    # facing direction (+1 if walking toward +X)
    face = np.sign(np.nanmean(mk["R.MT1X"] - mk["R.HeelX"]))
    def P(name):
        arr = mk[[name + "X", name + "Y"]].values.astype(float)
        arr[:, 0] *= face
        # fill short NaN gaps
        df = pd.DataFrame(arr).interpolate(limit_direction="both")
        return df.values / 1000.0
    gtr, knee, ank = P("R.GTR"), P("R.Knee"), P("R.Ankle")
    if np.isnan(gtr).any() or np.isnan(knee).any() or np.isnan(ank).any():
        return None
    gtr, knee, ank = lp(gtr, 8, FS_MK), lp(knee, 8, FS_MK), lp(ank, 8, FS_MK)
    th = gtr - knee                      # thigh vector knee->hip
    sh = knee - ank                      # shank vector ankle->knee
    theta = np.arctan2(-th[:, 0], th[:, 1])     # + when knee anterior of hip
    theta_sh = np.arctan2(-sh[:, 0], sh[:, 1])
    kflex = theta - theta_sh             # knee flexion (+) when shank inclined back relative to thigh
    u = th / np.linalg.norm(th, axis=1)[:, None]
    pivots, accs = {}, {}
    for s in HEIGHTS:
        p = knee + s * u
        v = np.gradient(p, 1 / FS_MK, axis=0)
        a = np.gradient(v, 1 / FS_MK, axis=0)
        pivots[s] = p; accs[s] = a
    # events: right-foot contacts, validated by COP location, from whichever plate(s) the right foot hits
    hs_all, to_all = [], []
    for pl in (1, 2):
        h, o = contacts_from_plate(gr, pl, mk)
        hs_all += list(h); to_all += list(o)
    if len(hs_all) < 5: return None
    order = np.argsort(hs_all); hs = np.array(hs_all)[order]; to = np.array(to_all)[order]
    mass = float(pd.read_csv("data/WBDSinfo.csv").query("Subject==@subj")["Mass"].iloc[0])
    return dict(subj=subj, sp=sp, t=t, theta=theta, knee=kflex, pivots=pivots, accs=accs,
                hs=hs, to=to, knee_mom=knt["RKneeMomentZ"].values, mass=mass,
                thigh_len=float(np.mean(np.linalg.norm(th, axis=1))))

def speed_lookup():
    info = pd.read_csv("data/WBDSinfo.csv")
    info = info[info.FileName.str.contains("walkT") & info.FileName.str.endswith("mkr.txt")]
    out = {}
    for _, r in info.iterrows():
        sp = int(r.FileName.split("walkT")[1][:2])
        try: out[(int(r.Subject), sp)] = float(r["GaitSpeed(m/s)"])
        except: pass
    return out

if __name__ == "__main__":
    import pickle
    speeds = speed_lookup()
    trials = []
    for subj in range(1, 52):
        for sp in range(1, 9):
            tr = process_trial(subj, sp)
            if tr is None: continue
            tr["speed"] = speeds.get((subj, sp), np.nan)
            trials.append(tr)
    pickle.dump(trials, open("data/trials.pkl", "wb"))
    print(len(trials), "trials;", len(set(t["subj"] for t in trials)), "subjects")
