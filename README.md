# LATCH knee — load-sequence-sensing, any-angle stance-control KAFO knee joint

**L**oad-**A**ctuated, **T**high-cable-free, any-angle stance-**C**ontrol **H**inge. A knee joint for knee–ankle–foot orthoses that decides stance vs. swing from the loads it already carries — no pendulum, no ankle cable, no battery — and locks against flexion at whatever angle the knee reaches at heel strike (3.75° resolution).

Everything needed to understand, reproduce, modify, cut, print and test the design.

## Headline numbers (from 9,781 measured strides, 51 subjects, 0.34–2.23 m/s)
| Metric (speeds ≤ 1.5 m/s, n = 7,278) | Value |
|---|---|
| Locked at heel strike | 99.2 % |
| Unlocked while a buckling (flexion) load was present | 0.27 % of strides |
| False lock during swing flexion | 0.23 % |
| Released before pre-swing flexion / before toe-off | 89.6 % / 96.4 % |
| Median release instant | 28 % of gait cycle |
| Knee angle at heel strike in the data | 7.4 ± 6.0° (64 % > 5°) — why a full-extension notch is not enough |
| Best pendulum trigger (any length / height / damping) | 99.7 % locked but 2.5 % released, or 84 % released but 53 % false locks |
| Tooth-root stress (FE) at 70 N·m design / 150 N·m proof | 250–280 / 540–600 MPa (17-4PH H900 yield 1,170 MPa) |
| Parts cost (BOM) | US$74 |
| Modelled mass (un-pocketed plates) | 963 g |

## Deliverables
| File | What it is |
|---|---|
| `latch_knee_paper.html` | Research report, self-contained HTML with embedded figures (also `latch_knee_paper.docx`, `latch_knee_paper.md`): problem, data, pendulum negative result, load-sequence logic, per-speed performance, sensitivity, CAD, FE, validation protocol, 30 verified references. **No PDF was produced**: the Tectonic LaTeX engine could not fetch its package bundle through the sandbox proxy. |
| `latch_knee_paper_latex.zip` | LaTeX source + figures; compiles to PDF on Overleaf or with `tectonic latch_knee_paper.tex` on a machine with internet access |
| `latch_knee_assembly.step` | Full assembly, 22 components, design pose (knee 0°, locked). Opens in FreeCAD / Fusion / Onshape / SolidWorks |
| `latch_knee_stl_parts.zip` | One STL per component + `parts.json` (name, colour, material, printed flag, volume) |
| `latch_knee_bom.csv` | Bill of materials, 23 lines, unit and extended cost, sources |
| `latch_knee_isef_abstract_and_assembly.md` | ISEF abstract (≤ 250 words) and step-by-step assembly / fitting / tuning instructions |
| `render_*.png` | CAD renders: closed, open, exploded, inner load-path layer (locked / released / 30° flexion), control deck, medial view, 60° flexion |
| `fig_rationale.png` | Knee angle at heel strike; why gravity pendulums at the knee fail; pendulum sweep |
| `fig_timeline.png` | The state machine running on a real stride |
| `fig_performance.png` | Performance by speed, failure modes, sensitivity |
| `fig_fea_vm.png` | Plane-stress FE of the toothed plate at 70 and 150 N·m |
| `references.csv`, `evidence_summary.md` | Literature base: 137 rows (118 CrossRef-resolved DOIs + 19 dated web sources), topic-organised synthesis with numbers and uncertainty flags |

## Code (Python, environment `mech-cad`: cadquery 2.8, scikit-fem 12, gmsh, trimesh, scipy, pandas)
| File | Purpose |
|---|---|
| `gait_pipeline.py` | WBDS treadmill trials → thigh angle, knee angle, upright-point accelerations, COP-validated heel strikes / toe-offs → `data/trials.pkl` |
| `sim_trigger.py` | Pendulum trigger dynamics (vectorised over trials) + early lock-logic variants |
| `sim_lock.py` | **Final load-sequence state machine** (`simulate_seq`) and summary (`summ_s`); virtual lock, self-holding pawls, latch memory, dashpot timer, drag ring, pawl delays, optional heel-strike gate |
| `cad_knee.py` | Parametric CadQuery model → STEP + STLs; `build(P, k, lifted, stop_deflected, latch_set)` poses the mechanism |
| `fea_ratchet.py` | gmsh + scikit-fem plane-stress FE of the toothed plate (`run_sector_only`) |
| `make_figures.py` | Analysis figures |
| `render.py` | Dependency-free numpy z-buffer STL renderer with feature edges (`render_cad`) |
| `strides_design.csv` | Per-stride outcomes at the design point (9,781 valid + flagged strides) |
| `sweep1.csv`, `sweep2.csv`, `sweep_seq.csv` | Pendulum sweep (L_eff × mount height × ζ), pendulum+stop-gate sweep, load-sequence threshold grid |
| `lock_results.json`, `lock_sensitivity.json`, `fea_results.json`, `moment_stats.json`, `parts_meta.json` | Numbers quoted in the report |

## Reproduce
```
python gait_pipeline.py          # needs data/51subjs/*T0?{mkr,grf,knt}.txt from WBDSascii.zip (figshare 5722711)
python -c "exec(open('sim_lock.py').read()); import pickle; tr=pickle.load(open('data/trials.pkl','rb')); r,_,_=simulate_seq(tr,M_thr=0.05,M_pawl=0.08,T_latch=0.15,tau_long=0.3,delta_deg=15,t_drop=0.10); print(summ_s(r)[0])"
python cad_knee.py               # STEP + STL
python fea_plot.py               # FE figure + fea_results.json
python make_figures.py
```

## Design parameters (cad_knee.py `P`)
Tooth tip radius 34 mm, depth 3.5 mm, pitch 7.5°, toothed arc 62.5–122.5° (knee −5…+30°), outer plate phased +3.75°; pawl pivot (41 mm, 84°), 3 serrations, lift 11°; plates 5 / 6 / 5 mm; knee pin 12 mm; pawl pin 6 mm; bridge pin 3.5 mm; stop lug at −15°; drag ring radius 41 mm, dead band 15°. Thresholds: M_thr 0.05 N·m/kg (stop), M_pawl 0.08 N·m/kg (latch), T_latch 0.15 s, T_long 0.30 s, t_drop 0.10 s, t_lift 0.03 s.

## Honest limitations
Simulation replays unimpaired gait with a virtual lock; no physical prototype has been built; FE is 2-D and excludes the pawl and contact; springs/dampers are specified, not detailed; mass needs pocketing. Above 1.5 m/s the design degrades (89–93 % locked) through a pre-swing-load chain that a heel-strike inertia gate removes (simulated: 98.2 %). See the report's Discussion and validation protocol.

## Repository layout

| Path | Contents |
|---|---|
| `cad/` | `latch_knee_assembly.step` (full assembly), `stl/` (22 individual parts), `latch_knee_bom.csv` |
| `code/` | `gait_pipeline.py` (WBDS parsing → strides), `sim_trigger.py` (pendulum sweeps), `sim_lock.py` (load-sequence state machine), `cad_knee.py` (parametric CadQuery model), `fea_ratchet.py` / `fea_plot.py` (plane-stress FE), `make_figures.py`, `render.py` |
| `results/` | `lock_results.json`, `lock_sensitivity.json`, `fea_results.json`, `moment_stats.json`, `parts_meta.json`, `strides_design.csv`, pendulum sweeps `sweep1.csv` / `sweep2.csv` / `sweep_seq.csv` |
| `figures/`, `renders/` | Report figures and CAD renders |
| `paper/` | Report as HTML / DOCX / Markdown; `latex_src/` holds the LaTeX source and figures (compile on Overleaf) |
| `references/` | `references.csv` (137 rows, 118 CrossRef-verified DOIs) and `evidence_summary.md` |
| `docs/` | ISEF abstract and assembly notes |

The gait dataset (Fukuchi et al. 2018, figshare 5722711, ~690 MB) is not included; `code/gait_pipeline.py` expects it unpacked under `data/51subjs/`.

Environment: Python 3 with `cadquery`, `scikit-fem`, `gmsh`, `meshio`, `trimesh`, `scipy`, `numpy`, `matplotlib`.
