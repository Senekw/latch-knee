# ISEF abstract (Engineering Mechanics / Biomedical Engineering) — 249 words

**A knee joint that knows when you are standing on it: load-sequence stance control for knee–ankle–foot orthoses, validated on 9,781 real strides**

Twelve to twenty million polio survivors and millions with spinal-cord injury or stroke walk in knee–ankle–foot orthoses whose knee is locked straight all the time, forcing hip hiking and circumduction. Stance-control joints that free the knee in swing cost US$1,700–3,500 and need either an ankle cable or a gravity pendulum that locks only at full extension. I asked whether a joint can detect stance using only the loads it already carries. Using a public dataset of 51 adults walking at eight speeds, I first showed that any pendulum trigger fails: the knee's own accelerations (up to 14 m/s²) tilt apparent gravity by 40–55°, so no pendulum length, mount height or damping both locked at heel strike and released before swing (best cases: 99.7 % locked but 2.5 % released, or 84 % released with 53 % false locks). I then designed a mechanism in which a fine-pitch ratchet (3.75°, any angle) is controlled by the sign sequence of the knee moment: a flexion load sets a mechanical latch, a later extension load on a sprung stop releases the pawls, a friction ring tracks knee direction in swing, and two rotary dampers and a dashpot give the timing. Simulated causally on 9,781 strides, the joint locked at heel strike in 99.2 % of strides at ≤ 1.5 m/s, exposed the knee to a buckling load in 0.27 %, and released before swing in 89.6 %. A full parametric CAD (22 parts, laser-cut steel load path, printed control deck), finite-element sizing (250–280 MPa at 70 N·m) and a US$74 bill of materials are provided for prototyping.

---

# Assembly, fitting and tuning instructions

## Parts (see `latch_knee_bom.csv` and `latch_knee_stl_parts.zip`)
Laser-cut steel: inner shank plate (toothed, with stop lug), outer shank plate (toothed, phase +3.75°, with ring boss), thigh plate (6 mm), two identical 3-tooth pawls, stop lever. Hardware: 12 mm shoulder bolt + nylock nut, two flanged bronze bushings, 6 mm pawl pin, 3.5 mm bridge pin, three M3×12 screws with heat-set inserts, two M4 preload screws. Printed (PA12-CF, 0.2 mm layers, 100 % infill, lie flat): housing/cover, lift yoke, latch lever, mode selector. Commercial: two 15 mm rotary dampers, one miniature dashpot (3 mm stroke), die spring (stop), Belleville stack (pawl-pin seat), two pawl torsion springs, two return springs, spring-steel split direction ring.

## Assembly (lateral joint of a right KAFO; mirror for left)
1. Press the two bronze bushings into the inner and outer shank plates. Deburr all laser-cut edges; the tooth flanks and pawl serrations should be stoned lightly so they seat over their full 5 mm width.
2. Fit the stop lever on its 4 mm pivot in the thigh plate with the die spring behind it and the M4 preload screw backed fully out. Check that the lever has 1.0 mm of travel before the hard stop.
3. Place the inner pawl in the inner layer (medial side of the thigh plate), align its pivot bore with the thigh-plate pawl hole, and insert the 6 mm pawl pin from the medial side. Fit the Belleville stack between the pawl pin seat and the thigh plate (this is the pawl-load sensor); set its preload screw fully out.
4. Fit the outer pawl on the lateral end of the same pin; push the 3.5 mm bridge pin through both pawls and the arc slot in the thigh plate. Both pawls must rotate together freely through 12°.
5. Hook the two pawl torsion springs so that the pawls are pushed into the teeth.
6. Slide the inner shank plate over the knee bolt from the medial side, then the thigh plate assembly, then the outer shank plate; the two toothed sectors must point proximally-posteriorly and the bar stubs distally. Tighten the nylock nut until there is no lateral play but the knee swings freely; the clevis plates are riveted to the KAFO shank upright through the three 5.2 mm holes, the thigh plate to the thigh upright.
7. Clamp the split direction ring on the outer plate boss with its tab pointing proximally-posteriorly; it should need 2–4 N at the tab to slip.
8. Fit the lift yoke on its pivot so that its fork straddles the bridge pin and its arm lies between the two faces of the ring tab (15° of free travel). Fit the yoke rotary damper (drop delay).
9. Fit the latch lever and its rotary damper (latch memory); its nose must be pushed by the Belleville-sprung pawl-pin seat when the pawls are loaded, and must block/unblock the stop-lever push-rod.
10. Connect the stop-lever push-rod to the dashpot (timer) and to the yoke through the latch gate.
11. Fit the mode selector into the cover, close the cover with the three M3 screws.

## Bench checks before fitting to a user
- LOCK mode: knee must not flex under a 70 N·m hand lever load at 0°, 10°, 20° and 30° flexion; extension must always be free (ratcheting).
- FREE mode: knee swings freely from 0° to 110°.
- AUTO, knee at 0°, load the shank in extension (stop loaded) with no prior flexion load: pawls must stay engaged. Load in flexion to > M_pawl (about 6 N·m for a 70 kg user), remove the load, then load in extension: pawls must lift within 50 ms and stay up while the extension load is held; on removing the load they must drop after about 0.1 s.
- AUTO, with the knee at 0° and the stop loaded continuously: pawls must lift after about 0.3 s (timer path).
- Flex the knee quickly from 0° to 60° with the pawls lifted: they must stay up; extend slowly: they must drop after about 15° of extension and ratchet to 0°.

## Fitting and tuning
- Set the two preload screws from body mass: stop threshold 0.05 N·m/kg (die spring force = 0.05·m·9.81/0.043 m), pawl threshold 0.08 N·m/kg (Belleville preload = 0.08·m·9.81/0.032 m). Err high (Section 3 of the report): too-low thresholds reduce heel-strike locking, too-high thresholds only make some strides stiff.
- Align the KAFO so that the user's heel-strike knee angle falls inside the toothed sector (−5° to +30°).
- Start in LOCK mode in parallel bars; switch to AUTO; confirm locking at every heel strike before unsupported walking. Stairs down and prolonged standing with weight shifting: LOCK. Sitting and cycling: FREE.
- Expected behaviour: six to eight audible clicks per stride during terminal swing; release felt at mid-stance as the knee becomes free while the orthosis rests against the extension stop.
