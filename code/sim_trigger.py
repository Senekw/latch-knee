"""Dynamic simulation of a thigh-mounted pendulum trigger driving an any-angle ratchet lock.

Mechanism model
---------------
* Pendulum pivot sits on the lateral thigh upright, height s above the knee axis.
* psi = pendulum angle relative to the thigh upright (+ = pendulum ahead of the upright).
  Absolute angle phi = psi + theta (theta = thigh inclination, + knee anterior of hip).
* I*(psi'' + theta'') = -m*l*[(g + a_y) sin(phi) + a_x cos(phi)] - c*psi' + stop torques
  with I = m*l*L_eff  (L_eff = effective pendulum length = I/(m l)).
* Travel stops at +-psi_stop (stiff spring-damper).
* Bistable toggle (mechanical Schmitt trigger): ARM when psi <= -theta_L, DISARM when psi >= +theta_U.
  Statically psi = -theta, so ARM <=> thigh forward by theta_L, DISARM <=> thigh back by theta_U.
* Ratchet: while ARMED the pawls engage -> knee cannot flex (extension free).  On DISARM the pawls
  lift only when the external knee flexion load is <= M_rel (teeth are self-holding under load).
"""
import numpy as np, pickle

G = 9.81

def simulate(trials, L_eff=0.04, zeta=0.5, s=0.0, theta_L=10.0, theta_U=10.0, psi_stop=35.0,
             substeps=10, M_rel=0.0, knee_free_thr=5.0, spl_tol=5.0, delta_deg=12.0, ideal=False):
    """Vectorised over trials (all WBDS treadmill trials have 4500 frames at 150 Hz)."""
    n = len(trials)
    N = min(len(tr["t"]) for tr in trials)
    dt = (1 / 150.0) / substeps
    theta = np.stack([tr["theta"][:N] for tr in trials])          # (n, N)
    knee = np.stack([tr["knee"][:N] for tr in trials])
    ax = np.stack([tr["accs"][s][:N, 0] for tr in trials])
    ay = np.stack([tr["accs"][s][:N, 1] for tr in trials])
    thdd = np.gradient(np.gradient(theta, 1 / 150.0, axis=1), 1 / 150.0, axis=1)
    wn = np.sqrt(G / L_eff)
    c_over_I = 2 * zeta * wn                 # c/I
    gl_over_I = G / L_eff                    # m g l / I
    thL, thU, pst = np.radians(theta_L), np.radians(theta_U), np.radians(psi_stop)
    k_stop, c_stop = (50 * wn) ** 2, 2 * 1.0 * (50 * wn)   # stiff stop normalised by I
    psi = -theta[:, 0].copy(); psid = np.zeros(n)
    if ideal:
        ax = ax * 0; ay = ay * 0; thdd = thdd * 0; c_over_I = 2 * 1.0 * wn * 0 + 2 * 50 * wn; gl_over_I = (50 * wn) ** 2  # near-ideal inclinometer
    armed = np.zeros(n, bool)
    armed[:] = psi <= -thL
    PSI = np.zeros((n, N)); ARMED = np.zeros((n, N), bool)

    def acc(psi, psid, j):
        phi = psi + theta[:, j]
        a = -gl_over_I * ((1 + ay[:, j] / G) * np.sin(phi) + (ax[:, j] / G) * np.cos(phi)) \
            - c_over_I * psid - thdd[:, j]
        over = np.clip(np.abs(psi) - pst, 0, None) * np.sign(psi)
        a += -k_stop * over - c_stop * psid * (np.abs(psi) > pst)
        return a

    for j in range(N):
        for _ in range(substeps):
            # RK2 (midpoint)
            a1 = acc(psi, psid, j)
            psi_m, psid_m = psi + 0.5 * dt * psid, psid + 0.5 * dt * a1
            a2 = acc(psi_m, psid_m, j)
            psi = psi + dt * psid_m; psid = psid + dt * a2
        armed = np.where(psi <= -thL, True, np.where(psi >= thU, False, armed))
        PSI[:, j] = psi; ARMED[:, j] = armed
    # --- causal lock-state machine per trial.
    # Direction sensor = friction-drag backlash element on the knee axle (deadband delta): it reports LIFT (knee flexing)
    # or DROP (knee extending). Pawls engage when ARMED and not LIFT. Engaged pawls are self-holding: they disengage only
    # when the external knee flexion load <= M_rel. While engaged the knee cannot flex (virtual lock): the direction sensor
    # therefore sees k_eff = min(k, k_lock), i.e. the loading-response flexion of the healthy donor gait is suppressed.
    delta = np.radians(delta_deg)
    t = trials[0]["t"][:N]
    ENG = np.zeros((n, N), bool); LIFT = np.zeros((n, N), bool); LOAD = np.zeros((n, N))
    for i, tr in enumerate(trials):
        hs = tr["hs"]; mom = tr["knee_mom"]
        load = np.zeros(N)
        for k in range(len(hs) - 1):
            j0, j1 = np.searchsorted(t, hs[k]), min(np.searchsorted(t, hs[k + 1]), N)
            if j1 <= j0: continue
            pct = (t[j0:j1] - hs[k]) / (hs[k + 1] - hs[k]) * 100
            load[j0:j1] = np.interp(pct, np.linspace(0, 100, len(mom)), mom)
        LOAD[i] = load
        K = knee[i]; A = ARMED[i]
        ring = K[0]; lift = False; eng = False; k_lock = np.inf
        for j in range(N):
            k = K[j]
            if eng:
                k_lock = min(k_lock, k)          # ratchet: extension allowed, flexion blocked
                k_eff = min(k, k_lock)
            else:
                k_lock = np.inf; k_eff = k
            if k_eff > ring + delta / 2: ring = k_eff - delta / 2; lift = True
            elif k_eff < ring - delta / 2: ring = k_eff + delta / 2; lift = False
            want = A[j] and not lift
            if want: eng = True
            elif load[j] <= M_rel: eng = False
            ENG[i, j] = eng; LIFT[i, j] = lift
    rows = []
    for i, tr in enumerate(trials):
        hs, to = tr["hs"], tr["to"]
        for k in range(len(hs) - 1):
            t0, t1 = hs[k], hs[k + 1]
            tos = to[(to > t0) & (to < t1)]
            if len(tos) != 1 or t1 - t0 > 2.5: continue
            tto = tos[0]
            j0, j1, jt = np.searchsorted(t, t0), np.searchsorted(t, t1), np.searchsorted(t, tto)
            if j1 >= N or j0 < 1: continue
            E = ENG[i, j0:j1]; K = knee[i, j0:j1]; TT = t[j0:j1]; A = ARMED[i, j0:j1]
            pct = lambda tt: (tt - t0) / (t1 - t0) * 100
            nS = len(E)
            locked_at_hs0 = bool(E[0])
            # stance: pendulum disarm and pawl release after 10 %GC (ignore the HS transient)
            j10 = int(0.10 * nS)
            dis = np.where(~A[j10: jt - j0])[0]; t_dis = TT[j10 + dis[0]] if len(dis) else np.nan
            rel = np.where(~E[j10:])[0]; t_rel = TT[j10 + rel[0]] if len(rel) else np.nan
            # when does the knee NEED to flex? midstance minimum between 25 and 50 %GC, then +5 deg
            a25, a50 = int(0.25 * nS), int(0.50 * nS)
            jm = a25 + int(np.argmin(K[a25:a50])); kmin = K[jm]
            need = np.where(K[jm:] > kmin + np.radians(knee_free_thr))[0]
            t_need = TT[jm + need[0]] if len(need) else tto
            early_rel = (not np.isnan(t_rel)) and pct(t_rel) < 25.0
            # swing: false lock = engaged while knee is flexing (before peak swing flexion) beyond the lock angle
            sw = slice(jt - j0, nS); Ksw, Esw, Tsw = K[sw], E[sw], TT[sw]; jpk = int(np.argmax(Ksw))
            false_lock = bool(np.any(Esw[:jpk])) and (not np.isnan(t_rel)) and (t_rel <= tto)
            held_through_swing = bool(np.all(E[j10:]))       # never released: stiff-knee stride (safe fail)
            eng_after = np.where(Esw[jpk:])[0]
            t_eng = Tsw[jpk + eng_after[0]] if len(eng_after) else np.nan
            knee_at_eng = np.degrees(Ksw[jpk + eng_after[0]]) if len(eng_after) else np.nan
            knee_hs_next = np.degrees(knee[i, j1])
            locked_at_hs = bool(ENG[i, j1])
            rows.append(dict(subj=tr["subj"], sp=tr["sp"], speed=tr["speed"], stride=t1 - t0, mass=tr["mass"],
                             t_dis=pct(t_dis) if not np.isnan(t_dis) else np.nan,
                             t_rel=pct(t_rel) if not np.isnan(t_rel) else np.nan,
                             t_need=pct(t_need), t_to=pct(tto),
                             released=not np.isnan(t_rel), released_in_time=(not np.isnan(t_rel)) and t_rel <= t_need,
                             early_release=early_rel, false_lock=false_lock, held_through_swing=held_through_swing,
                             t_eng=pct(t_eng) if not np.isnan(t_eng) else np.nan,
                             lock_margin_ms=(t1 - t_eng) * 1000 if not np.isnan(t_eng) else np.nan,
                             locked_at_hs=locked_at_hs, armed_at_hs=bool(ARMED[i, j1]),
                             knee_at_eng=knee_at_eng, knee_hs=knee_hs_next,
                             clicks_deg=max(knee_at_eng - knee_hs_next, 0) if not np.isnan(knee_at_eng) else np.nan,
                             spl_would_lock=(knee_hs_next <= spl_tol),
                             theta_max=np.degrees(theta[i, j0:j1].max()), theta_min=np.degrees(theta[i, j0:j1].min()),
                             load_peak=LOAD[i, j0:j1].max()))
    return rows, PSI, ARMED, ENG

def summarize(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    out = dict(n_strides=len(df), n_subj=df.subj.nunique(),
               locked_at_hs=df.locked_at_hs.mean(), false_lock=df.false_lock.mean(),
               released=df.released.mean(), released_in_time=df.released_in_time.mean(),
               early_release=df.early_release.mean(), held_through_swing=df.held_through_swing.mean(),
               lock_margin_p5_ms=np.nanpercentile(df.lock_margin_ms, 5), lock_margin_med_ms=np.nanmedian(df.lock_margin_ms),
               clicks_deg_med=np.nanmedian(df.clicks_deg), spl_would_lock=df.spl_would_lock.mean())
    return out, df
