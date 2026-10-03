"""Load-sequence stance-control lock logic (final model used in the paper).
Signals available to a purely mechanical knee joint on a KAFO:
  * external knee FLEXION moment  -> carried by the engaged ratchet pawls  (flex_load, N.m/kg)
  * external knee EXTENSION moment -> carried by the sprung hyperextension stop (ext_load, N.m/kg)
  * knee rotation direction -> friction-drag backlash ring (deadband delta_deg)
State machine (all thresholds per kg body mass; the stop/pawl springs are preloaded per patient):
  LATCH   : set while pawls carry flex_load > M_pawl; expires T_latch after the pawl unloads.
  TIMER   : dur = continuous time the stop has been loaded (> M_thr).
  RELEASE : when stop loaded AND (LATCH or dur > tau_long) -> rel_active; persists while the stop stays loaded.
  LIFT    : pawls lift (t_lift) when rel_active OR knee flexing (direction ring); drop after t_drop when neither.
  Self-holding teeth: an engaged pawl under flex_load cannot be lifted.
  Virtual lock: while engaged the knee cannot flex beyond the lock angle (donor kinematics are clamped).
"""
import numpy as np, pandas as pd
def simulate_seq(trials, impact_gate=0.0, T_latch=0.35, rel_vel_gate=0.0, vel_gate=0.0, M_thr=0.05, M_pawl=0.03, tau_f=0.02, tau_long=0.25, M_thr_long=0.05, delta_deg=12.0, t_drop=0.12, t_lift=0.03, imp_amp=0.0, imp_dur=0.025,
                 knee_free_thr=5.0, spl_tol=5.0, teeth_max_deg=30.0, M_rel=0.0, mass_scale=1.0):
    """Load-sequence stance control: pawl flexion load sets LATCH; extension-stop load with LATCH set lifts pawls (and resets LATCH).
    Direction sensor (backlash delta) holds pawls lifted while knee flexes; pawls drop (delay t_drop) when knee extends.
    Loads are per-kg (N.m/kg) from the donor stride's inverse dynamics; thresholds are per-kg too."""
    n=len(trials); N=min(len(tr["t"]) for tr in trials); dt=1/150.0; t=trials[0]["t"][:N]; delta=np.radians(delta_deg)
    rows=[]; ENG=np.zeros((n,N),bool); LATCH=np.zeros((n,N),bool)
    for i,tr in enumerate(trials):
        hs,to=tr["hs"],tr["to"]; mom=np.nan_to_num(tr["knee_mom"])*mass_scale; K=tr["knee"][:N]
        load=np.zeros(N)
        for k in range(len(hs)-1):
            j0,j1=np.searchsorted(t,hs[k]),min(np.searchsorted(t,hs[k+1]),N)
            if j1<=j0: continue
            load[j0:j1]=np.interp((t[j0:j1]-hs[k])/(hs[k+1]-hs[k])*100,np.linspace(0,100,len(mom)),mom)
        flex_load=np.clip(load,0,None); ext_load=np.clip(-load,0,None)
        if imp_amp>0:
            for k in range(len(hs)-1):
                tos=to[(to>hs[k])&(to<hs[k+1])]
                if len(tos)!=1: continue
                a,b=np.searchsorted(t,tos[0]),min(np.searchsorted(t,hs[k+1]),N)
                if b-a<5: continue
                seg=K[a:b]; jp=np.argmax(seg); jm=a+jp+int(np.argmin(seg[jp:])); ext_load[jm:min(jm+int(imp_dur/dt)+1,N)]+=imp_amp
        f=0.0; dur=0.0; rel_active=False; latch_t=-1e9; ke_prev=K[0]; ring=K[0]; lift_dir=False; p=0.0; eng=True; k_lock=np.inf; latch=False; E=np.zeros(N,bool); LT=np.zeros(N,bool)
        for j in range(N):
            f+=dt/tau_f*(ext_load[j]-f); k=K[j]
            moving = False
            dur = dur+dt if (f>M_thr and not moving) else 0.0
            if eng: k_lock=min(k_lock,k); ke=min(k,k_lock)
            else: k_lock=np.inf; ke=k
            if ke>ring+delta/2: ring=ke-delta/2; lift_dir=True
            elif ke<ring-delta/2: ring=ke+delta/2; lift_dir=False
            recent_hs = True if impact_gate<=0 else (len(hs)>0 and (j*dt - hs[np.searchsorted(hs, j*dt, side='right')-1] if np.searchsorted(hs, j*dt, side='right')>0 else 1e9) <= impact_gate)
            if eng and flex_load[j]>M_pawl and recent_hs: latch_t=j*dt
            latch = (j*dt-latch_t) <= T_latch
            still = not (rel_vel_gate>0 and abs(ke-ke_prev)/dt > np.radians(rel_vel_gate)); ke_prev=ke
            if f>M_thr and still and (latch or (tau_long>0 and dur>tau_long)): rel_active=True
            if f<=M_thr and rel_active: rel_active=False; latch_t=-1e9
            cmd = rel_active or lift_dir
            target=1.0 if cmd else 0.0
            p = min(1.0,p+dt/t_lift) if target>p else max(0.0,p-dt/t_drop)
            if eng and flex_load[j]>M_rel: p=0.0
            eng=(p<0.5) and (ke<np.radians(teeth_max_deg)); E[j]=eng; LT[j]=latch
        ENG[i]=E; LATCH[i]=LT
        for k in range(len(hs)-1):
            t0,t1=hs[k],hs[k+1]; tos=to[(to>t0)&(to<t1)]
            if len(tos)!=1 or t1-t0>2.5: continue
            tto=tos[0]; j0,j1,jt=np.searchsorted(t,t0),np.searchsorted(t,t1),np.searchsorted(t,tto)
            if j1>=N or j0<1: continue
            Es=E[j0:j1]; Ks=K[j0:j1]; TT=t[j0:j1]; nS=len(Es); pct=lambda tt:(tt-t0)/(t1-t0)*100
            j10=int(0.10*nS); rel=np.where(~Es[j10:])[0]; t_rel=TT[j10+rel[0]] if len(rel) else np.nan
            rel0=np.where(~Es[:jt-j0])[0]; t_rel0=TT[rel0[0]] if len(rel0) else np.nan     # any unlock in stance (incl. HS)
            a25,a50=int(0.25*nS),int(0.50*nS); jm=a25+int(np.argmin(Ks[a25:a50])); kmin=Ks[jm]
            need=np.where(Ks[jm:]>kmin+np.radians(knee_free_thr))[0]; t_need=TT[jm+need[0]] if len(need) else tto
            sw=slice(jt-j0,nS); Ksw,Esw,Tsw=Ks[sw],Es[sw],TT[sw]; jpk=int(np.argmax(Ksw))
            false_lock=bool(np.any(Esw[:jpk]&(Ksw[:jpk]<np.radians(teeth_max_deg)))) and (not np.isnan(t_rel0)) and (t_rel0<=tto)
            ea=np.where(Esw[jpk:])[0]; t_eng=Tsw[jpk+ea[0]] if len(ea) else np.nan
            knee_at_eng=np.degrees(Ksw[jpk+ea[0]]) if len(ea) else np.nan; knee_hs_next=np.degrees(K[j1])
            stiff=np.isnan(t_rel0) or t_rel0>tto
            jn=np.searchsorted(TT,t_need); FL=flex_load[j0:j0+jn]; EXP=(~Es[:jn]) & (FL>0.05)
            exposure=bool(EXP.any()); exp_ms=EXP.sum()*dt*1000
            # knee flexion the healthy donor performed while exposed (upper bound of uncaught flexion)
            exp_deg=float(np.degrees(np.max(Ks[:jn][EXP])-np.min(Ks[:jn][EXP]))) if exposure else 0.0
            rows.append(dict(subj=tr["subj"],sp=tr["sp"],speed=tr["speed"],stride=t1-t0,mass=tr["mass"],t0=t0,
                t_rel=pct(t_rel0) if not np.isnan(t_rel0) else np.nan,t_need=pct(t_need),t_to=pct(tto),
                released=not np.isnan(t_rel0),released_in_time=(not np.isnan(t_rel0)) and t_rel0<=t_need,
                early_release=(not np.isnan(t_rel0)) and pct(t_rel0)<25.0,false_lock=false_lock,stiff=stiff,
                t_eng=pct(t_eng) if not np.isnan(t_eng) else np.nan,lock_margin_ms=(t1-t_eng)*1000 if not np.isnan(t_eng) else np.nan,
                locked_at_hs=bool(E[j1]) and bool(E[min(j1+3,N-1)]),knee_at_eng=knee_at_eng,knee_hs=knee_hs_next,
                clicks_deg=max(knee_at_eng-knee_hs_next,0) if not np.isnan(knee_at_eng) else np.nan,
                spl_would_lock=knee_hs_next<=spl_tol,ext_peak=ext_load[j0:jt].max(),flex_peak=flex_load[j0:jt].max(),exposure=exposure,exp_ms=exp_ms,exp_deg=exp_deg))
    return rows,ENG,LATCH

def summ_s(rows, valid_only=True):
    df=pd.DataFrame(rows)
    if valid_only:
        df=df[(df.t_to>=50)&(df.t_to<=72)&(df.stride>0.6)&(df.stride<2.4)]
    out=dict(n=len(df),n_subj=df.subj.nunique(),locked_at_hs=df.locked_at_hs.mean(),exposure=df.exposure.mean(),exp_ms_p95=np.percentile(df.exp_ms,95),exp_deg_p99=np.percentile(df.exp_deg,99),
             false_lock=df.false_lock.mean(),released_in_time=df.released_in_time.mean(),stiff=df.stiff.mean(),
             t_rel_med=np.nanmedian(df.t_rel),t_rel_p5=np.nanpercentile(df.t_rel,5),t_rel_p95=np.nanpercentile(df.t_rel,95),clicks_med=np.nanmedian(df.clicks_deg),spl_would_lock=df.spl_would_lock.mean())
    return {k:round(float(v),3) for k,v in out.items()},df
