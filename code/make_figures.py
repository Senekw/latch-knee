import numpy as np, pandas as pd, pickle, json, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
exec(open("sim_lock.py").read())
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
trials = pickle.load(open("data/trials.pkl", "rb"))
df = pd.read_csv("strides_design.csv")
DESIGN = json.load(open("lock_results.json"))["design"]
os.makedirs("fig", exist_ok=True)

# ---------- Fig A: why any-angle + why not a pendulum ----------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].hist(df.knee_hs, bins=np.arange(-10, 36, 1), color="#4C72B0", alpha=0.9)
for v, lab in ((5, "5°"), (10, "10°")):
    ax[0].axvline(v, color="k", ls="--", lw=0.8); ax[0].text(v + 0.3, ax[0].get_ylim()[1] * 0.9, f"{(df.knee_hs > v).mean()*100:.0f}% > {lab}", fontsize=8)
ax[0].set_xlabel("Knee flexion at heel strike (deg)"); ax[0].set_ylabel("strides"); ax[0].set_title(f"A  Knee angle at heel strike\n{len(df)} strides, 51 subjects, 0.3–2.2 m/s", fontsize=9)
# thigh angle and pivot acceleration for one typical stride (knee-mounted pendulum)
tr = [t for t in trials if t["subj"] == 1 and t["sp"] == 5][0]
t = tr["t"]; hs = tr["hs"]; k = 8; t0, t1 = hs[k], hs[k + 1]; j0, j1 = np.searchsorted(t, t0), np.searchsorted(t, t1)
pct = (t[j0:j1] - t0) / (t1 - t0) * 100
th = np.degrees(tr["theta"][j0:j1]); ax0 = tr["accs"][0.0][j0:j1, 0]; ax2 = tr["accs"][0.2][j0:j1, 0]
tilt0 = np.degrees(np.arctan2(ax0, 9.81)); tilt2 = np.degrees(np.arctan2(ax2, 9.81))
ax[1].plot(pct, th, "k", lw=1.6, label="thigh inclination θ (what a pendulum should read)")
ax[1].plot(pct, th + tilt0, color="#C44E52", lw=1.2, label="apparent θ, pendulum at knee (g − a tilt)")
ax[1].plot(pct, th + tilt2, color="#DD8452", lw=1.2, label="apparent θ, pendulum 20 cm up the thigh")
ax[1].axhline(0, color="grey", lw=0.6); tto = tr["to"][(tr["to"] > t0) & (tr["to"] < t1)][0]
ax[1].axvline((tto - t0) / (t1 - t0) * 100, color="grey", ls=":", lw=0.8); ax[1].text((tto - t0) / (t1 - t0) * 100 + 1, -38, "toe-off", fontsize=7, color="grey")
ax[1].set_xlabel("gait cycle (%)"); ax[1].set_ylabel("deg (+ = thigh forward)"); ax[1].legend(fontsize=7, loc="upper left"); ax[1].set_title("B  Gravity-referenced sensing is corrupted by\nthe knee's own accelerations (subject 1, 1.24 m/s)", fontsize=9)
# pendulum sweep summary (from sweep1.csv): best achievable locked-at-HS and release-in-time
sw = pd.read_csv("sweep1.csv")
piv = sw[sw.zeta == 0.7].pivot(index="L_eff", columns="s", values="locked_at_hs")
piv2 = sw[sw.zeta == 0.7].pivot(index="L_eff", columns="s", values="released_in_time")
for s, c in zip(piv.columns, ("#4C72B0", "#55A868", "#8172B2")):
    ax[2].plot(piv.index * 100, piv[s] * 100, "-o", ms=3, color=c, label=f"locked at HS, mount {s*100:.0f} cm")
    ax[2].plot(piv2.index * 100, piv2[s] * 100, "--s", ms=3, color=c, alpha=0.6, label=f"released in time, mount {s*100:.0f} cm")
ax[2].set_xscale("log"); ax[2].set_xlabel("pendulum effective length (cm)"); ax[2].set_ylabel("% of strides"); ax[2].set_ylim(0, 102)
ax[2].legend(fontsize=6.5, ncol=1, loc="lower left"); ax[2].set_title("C  Pendulum-triggered lock, all strides (ζ = 0.7):\nno geometry achieves both functions", fontsize=9)
fig.tight_layout(); fig.savefig("fig/fig_rationale.png", dpi=170); plt.close(fig)

# ---------- Fig B: state-machine timeline for one stride ----------
def trace(tr, **kw):
    p = dict(DESIGN); p.update(kw)
    rows, ENG, LATCH = simulate_seq([tr], **p)
    return ENG[0], LATCH[0]
tr = [t for t in trials if t["subj"] == 1 and t["sp"] == 4][0]
ENG, LATCH = trace(tr)
t = tr["t"]; hs = tr["hs"]; k = 10; t0, t1 = hs[k], hs[k + 1]; j0, j1 = np.searchsorted(t, t0), np.searchsorted(t, t1) + 3
pct = (t[j0:j1] - t0) / (t1 - t0) * 100
mom = np.interp(np.clip(pct, 0, 100), np.linspace(0, 100, 101), np.nan_to_num(tr["knee_mom"]))
tto = tr["to"][(tr["to"] > t0) & (tr["to"] < t1)][0]; pto = (tto - t0) / (t1 - t0) * 100
fig, ax = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1.2, 0.9]))
kn = np.degrees(tr["knee"][j0:j1]); eng = ENG[j0:j1]
# virtual-lock knee: clamp while engaged
kv = kn.copy(); klock = np.inf
for i in range(len(kv)):
    if eng[i]: klock = min(klock, kn[i]); kv[i] = min(kn[i], klock)
    else: klock = np.inf
ax[0].plot(pct, kn, color="grey", lw=1, ls="--", label="donor (healthy) knee angle")
ax[0].plot(pct, kv, color="#4C72B0", lw=1.8, label="knee angle with LATCH joint (virtual lock)")
ax[0].fill_between(pct, 0, 70, where=eng, color="#C44E52", alpha=0.12, label="pawls engaged (flexion blocked)")
ax[0].set_ylabel("knee flexion (deg)"); ax[0].legend(fontsize=7, loc="upper left"); ax[0].set_ylim(-5, 70)
ax[1].fill_between(pct, 0, np.clip(mom, 0, None), color="#C44E52", alpha=0.5, label="external FLEXION moment → carried by pawls")
ax[1].fill_between(pct, 0, -np.clip(-mom, 0, None), color="#55A868", alpha=0.5, label="external EXTENSION moment → carried by sprung stop")
ax[1].axhline(DESIGN["M_pawl"], color="#C44E52", ls=":", lw=0.8); ax[1].text(1, DESIGN["M_pawl"] + 0.02, f"M_pawl = {DESIGN['M_pawl']} N·m/kg (sets LATCH)", fontsize=7, color="#C44E52")
ax[1].axhline(-DESIGN["M_thr"], color="#55A868", ls=":", lw=0.8); ax[1].text(1, -DESIGN["M_thr"] - 0.09, f"M_thr = {DESIGN['M_thr']} N·m/kg (stop loaded)", fontsize=7, color="#55A868")
ax[1].set_ylabel("knee moment (N·m/kg)"); ax[1].legend(fontsize=7, loc="lower left")
lat = LATCH[j0:j1]
ax[2].step(pct, eng.astype(int) + 2.2, where="post", color="#C44E52", lw=1.5); ax[2].text(101, 2.2, "ENGAGED", fontsize=7, va="bottom")
ax[2].step(pct, lat.astype(int) + 1.1, where="post", color="#8172B2", lw=1.5); ax[2].text(101, 1.1, "LATCH", fontsize=7, va="bottom")
ax[2].step(pct, (mom < -DESIGN["M_thr"]).astype(int), where="post", color="#55A868", lw=1.5); ax[2].text(101, 0, "STOP LOADED", fontsize=7, va="bottom")
ax[2].set_yticks([]); ax[2].set_xlabel("gait cycle (% , heel strike → heel strike)"); ax[2].set_xlim(0, 112)
for a in ax: a.axvline(pto, color="grey", ls=":", lw=0.8); a.axvline(100, color="k", lw=0.6)
ax[0].text(pto + 0.5, 62, "toe-off", fontsize=7, color="grey"); ax[0].text(100.5, 62, "next HS", fontsize=7)
ax[0].set_title(f"Load-sequence stance control on a real stride (subject {tr['subj']}, {tr['speed']:.2f} m/s)", fontsize=10)
fig.tight_layout(); fig.savefig("fig/fig_timeline.png", dpi=170); plt.close(fig)

# ---------- Fig C: performance by speed + sensitivity ----------
by = df.groupby("sp").agg(speed=("speed", "mean"), locked=("locked_at_hs", "mean"), exposure=("exposure", "mean"), false_lock=("false_lock", "mean"),
                          rel=("released_in_time", "mean"), stiff=("stiff", "mean"), n=("subj", "size"))
sens = json.load(open("lock_sensitivity.json"))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
ax[0].plot(by.speed, by.locked * 100, "-o", color="#4C72B0", label="locked at heel strike")
ax[0].plot(by.speed, by.rel * 100, "-s", color="#55A868", label="released before pre-swing flexion")
ax[0].plot(by.speed, (1 - by.stiff) * 100, "-^", color="#8172B2", label="released before toe-off (not stiff)")
ax[0].axvspan(0.3, 1.5, color="grey", alpha=0.08); ax[0].text(0.35, 20, "KAFO-user\nspeed envelope", fontsize=7, color="grey")
ax[0].set_xlabel("walking speed (m/s)"); ax[0].set_ylabel("% of strides"); ax[0].set_ylim(0, 102); ax[0].legend(fontsize=7, loc="lower left"); ax[0].set_title("A  Function vs speed (design point)", fontsize=9)
ax[1].plot(by.speed, by.exposure * 100, "-o", color="#C44E52", label="buckling exposure (unlocked while flexion load > 0.05 N·m/kg)")
ax[1].plot(by.speed, by.false_lock * 100, "-s", color="#DD8452", label="false lock during swing flexion")
ax[1].axvspan(0.3, 1.5, color="grey", alpha=0.08); ax[1].set_xlabel("walking speed (m/s)"); ax[1].set_ylabel("% of strides"); ax[1].legend(fontsize=7); ax[1].set_title("B  Failure modes vs speed", fontsize=9)
names = ["design", "impact_gate_0.5s", "preload_-30%", "preload_+30%", "terminal_swing_impulse_0.3", "delta_10", "delta_20", "t_drop_0.06", "t_drop_0.15", "no_timer"]
labels = ["design point", "+ heel-strike impact gate", "preload −30 %", "preload +30 %", "stop impulse 0.3 N·m/kg\nat terminal swing", "δ = 10°", "δ = 20°", "t_drop = 60 ms", "t_drop = 150 ms", "no stop timer"]
y = np.arange(len(names))
ax[2].barh(y - 0.2, [sens[n]["env_locked"] * 100 for n in names], height=0.38, color="#4C72B0", label="locked at HS (≤1.5 m/s)")
ax[2].barh(y + 0.2, [sens[n]["env_released_in_time"] * 100 for n in names], height=0.38, color="#55A868", label="released in time (≤1.5 m/s)")
for i, n in enumerate(names): ax[2].text(101, i - 0.3, f"exp {sens[n]['env_exposure']*100:.1f}%", fontsize=6.5, color="#C44E52")
ax[2].set_yticks(y); ax[2].set_yticklabels(labels, fontsize=7); ax[2].invert_yaxis(); ax[2].set_xlim(50, 118); ax[2].set_xlabel("% of strides"); ax[2].legend(fontsize=7, loc="lower right"); ax[2].set_title("C  Sensitivity (speed ≤ 1.5 m/s)", fontsize=9)
fig.tight_layout(); fig.savefig("fig/fig_performance.png", dpi=170); plt.close(fig)
print("figures done", by.round(3).to_dict())
