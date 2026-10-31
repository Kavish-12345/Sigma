"""
Design layer: material selection in the design process (Ashby).

    1. Translation  - function, constraints, objective, free variable
    2. Screening    - design calculation -> required Sy / Su / E (hard limits)
    3. Ranking      - material index (e.g. Sy^(2/3)/rho) as a TOPSIS criterion
    4. Supporting   - design check (actual factor of safety) on every result

A part is designed as a standard machine element (its function), or as a general
part: the loads on its critical section, combined with von Mises. The formulas are the standard textbook ones (Bhandari / Shigley); assumptions are
stated in the calculation steps shown to the user. No LLM is involved here.
"""
from __future__ import annotations

import math

# Fatigue: endurance limit Se = 0.5 * Su * K, with K = combined surface and size
# factor (machined surface, small part). A deliberately simple, stated assumption.
SE_RATIO, MARIN_K = 0.5, 0.75


class MissingInput(ValueError):
    """A value the calculation needs was left blank (as opposed to a value that is wrong)."""

COMMON_INPUTS = [
    {"key": "fos", "label": "Factor of safety", "unit": "", "hint": "e.g. 2", "default": 2.0},
    {"key": "loading", "label": "Type of load", "type": "select", "default": "steady",
     "options": [{"value": "steady", "label": "Steady (static)"},
                 {"value": "cyclic", "label": "Fluctuating / repeated (fatigue)"}]},
]

# index: M = (strength)^n / rho  (the Ashby minimum-mass index for that function,
# section size free). For the cost objective rho is replaced by price * rho.
FUNCTIONS = {
    "tie": {
        "label": "Tie rod / bolt / link (axial tension)",
        "description": "Carries a pulling or pushing force along its length",
        "examples": "tie rod, bolt, chain link, hanger, connecting link",
        "load_mode": "Axial tension",
        "classes": ["Structural"],
        "inputs": [
            {"key": "F_kN", "label": "Axial force", "unit": "kN", "hint": "e.g. 20"},
            {"key": "d_mm", "label": "Rod diameter", "unit": "mm", "hint": "e.g. 16"},
        ],
        "free_variable": "Cross-section area",
        "index": {"n": 1.0, "prop": "Sy", "text": "σy / ρ"},
    },
    "beam": {
        "label": "Beam / bracket / lever (bending)",
        "description": "A load acts across its length and bends it",
        "examples": "bracket, lever, holder, arm, frame member, shelf support",
        "load_mode": "Bending",
        "classes": ["Structural"],
        "inputs": [
            {"key": "support", "label": "Support", "type": "select", "default": "cantilever",
             "options": [{"value": "cantilever", "label": "Cantilever (fixed at one end, load at free end)"},
                         {"value": "simply_supported", "label": "Simply supported (load at centre)"}]},
            {"key": "F_N", "label": "Load", "unit": "N", "hint": "e.g. 500 (50 kg ≈ 490 N)"},
            {"key": "L_mm", "label": "Length / span", "unit": "mm", "hint": "e.g. 200"},
            {"key": "b_mm", "label": "Section width b", "unit": "mm", "hint": "e.g. 20"},
            {"key": "h_mm", "label": "Section depth h", "unit": "mm", "hint": "e.g. 30"},
            {"key": "defl_mm", "label": "Max deflection allowed", "unit": "mm", "hint": "optional, e.g. 1",
             "optional": True},
        ],
        "free_variable": "Section depth",
        "index": {"n": 2 / 3, "prop": "Sy", "text": "σy^(2/3) / ρ"},
    },
    "shaft": {
        "label": "Shaft / axle (torsion + bending)",
        "description": "Rotates and transmits power; may also carry a pulley or gear load",
        "examples": "transmission shaft, axle, spindle, gear shaft, motor shaft",
        "load_mode": "Torsion + bending",
        "classes": ["Heat-treatable", "Structural"],
        "inputs": [
            {"key": "P_kW", "label": "Power", "unit": "kW", "hint": "e.g. 10"},
            {"key": "N_rpm", "label": "Speed", "unit": "rpm", "hint": "e.g. 960"},
            {"key": "M_Nm", "label": "Bending moment", "unit": "N·m", "hint": "0 if none, e.g. 150",
             "optional": True},
            {"key": "d_mm", "label": "Shaft diameter", "unit": "mm", "hint": "e.g. 30"},
        ],
        "free_variable": "Shaft diameter",
        "index": {"n": 2 / 3, "prop": "Sy", "text": "σy^(2/3) / ρ"},
    },
    "pressure_vessel": {
        "label": "Pressure vessel / pipe / cylinder (internal pressure)",
        "description": "Thin-walled shell holding a fluid under pressure",
        "examples": "pipe, tank, hydraulic cylinder, boiler shell, gas cylinder",
        "load_mode": "Internal pressure (hoop stress)",
        "classes": ["Structural"],
        "inputs": [
            {"key": "p_MPa", "label": "Internal pressure", "unit": "MPa", "hint": "e.g. 2 (= 20 bar)"},
            {"key": "D_mm", "label": "Inner diameter", "unit": "mm", "hint": "e.g. 300"},
            {"key": "t_mm", "label": "Wall thickness", "unit": "mm", "hint": "e.g. 5"},
        ],
        "free_variable": "Wall thickness",
        "index": {"n": 1.0, "prop": "Sy", "text": "σy / ρ"},
    },
    "spring": {
        "label": "Helical spring (torsion of the wire)",
        "description": "Coil spring that stores energy when compressed or stretched",
        "examples": "compression spring, valve spring, return spring",
        "load_mode": "Torsion of the wire",
        "classes": ["Spring"],
        "inputs": [
            {"key": "F_N", "label": "Max spring force", "unit": "N", "hint": "e.g. 300"},
            {"key": "Dm_mm", "label": "Mean coil diameter", "unit": "mm", "hint": "e.g. 30"},
            {"key": "dw_mm", "label": "Wire diameter", "unit": "mm", "hint": "e.g. 4"},
        ],
        "free_variable": "Wire diameter",
        "index": {"n": 2.0, "prop": "Su", "text": "σ² / (E·ρ)", "use_E": True},
    },
    "gear": {
        "label": "Spur gear (tooth bending, Lewis equation)",
        "description": "Toothed wheel transmitting power; teeth bend and rub",
        "examples": "spur gear, pinion, gearbox gear",
        "load_mode": "Tooth bending + contact wear (repeated)",
        "classes": ["Wear-resistant", "Heat-treatable"],
        "inputs": [
            {"key": "P_kW", "label": "Power", "unit": "kW", "hint": "e.g. 5"},
            {"key": "N_rpm", "label": "Speed", "unit": "rpm", "hint": "e.g. 1440"},
            {"key": "m_mm", "label": "Module", "unit": "mm", "hint": "e.g. 3"},
            {"key": "z", "label": "Number of teeth", "unit": "", "hint": "e.g. 20"},
            {"key": "b_mm", "label": "Face width", "unit": "mm", "hint": "e.g. 30 (≈10 × module)"},
        ],
        "free_variable": "Face width",
        "index": {"n": 1.0, "prop": "Sy", "text": "σy / ρ"},
        "always_cyclic": True,
    },
    "general": {
        "label": "General part (any shape, combined loads)",
        "description": "Anything else: describe its most stressed section and the loads on it, or enter a known stress",
        "examples": "connecting rod, hook, clevis, crank arm, housing, mounting plate",
        "load_mode": "Combined loads (von Mises)",
        "classes": ["Structural"],
        # show_if: the field applies only to these sections; need_one: at least one of these is required
        "inputs": [
            {"key": "section", "label": "Most stressed section", "type": "select", "default": "solid_round",
             "options": [{"value": "solid_round", "label": "Solid round"},
                         {"value": "hollow_round", "label": "Hollow round / tube"},
                         {"value": "rectangle", "label": "Rectangle / plate"},
                         {"value": "known_stress", "label": "I already know the peak stress (hand calculation or FEA)"}]},
            {"key": "d_mm", "label": "Outer diameter", "unit": "mm", "hint": "e.g. 25",
             "show_if": {"key": "section", "in": ["solid_round", "hollow_round"]}},
            {"key": "di_mm", "label": "Inner diameter", "unit": "mm", "hint": "e.g. 15",
             "show_if": {"key": "section", "in": ["hollow_round"]}},
            {"key": "b_mm", "label": "Section width b", "unit": "mm", "hint": "e.g. 20",
             "show_if": {"key": "section", "in": ["rectangle"]}},
            {"key": "h_mm", "label": "Section depth h (in the bending direction)", "unit": "mm", "hint": "e.g. 30",
             "show_if": {"key": "section", "in": ["rectangle"]}},
            {"key": "F_kN", "label": "Axial force", "unit": "kN", "hint": "0 if none, e.g. 20", "optional": True,
             "need_one": True, "show_if": {"key": "section", "in": ["solid_round", "hollow_round", "rectangle"]}},
            {"key": "V_kN", "label": "Shear force", "unit": "kN", "hint": "0 if none, e.g. 5", "optional": True,
             "need_one": True, "show_if": {"key": "section", "in": ["solid_round", "hollow_round", "rectangle"]}},
            {"key": "M_Nm", "label": "Bending moment", "unit": "N·m", "hint": "0 if none, e.g. 150", "optional": True,
             "need_one": True, "show_if": {"key": "section", "in": ["solid_round", "hollow_round", "rectangle"]}},
            {"key": "T_Nm", "label": "Torque", "unit": "N·m", "hint": "0 if none, e.g. 80", "optional": True,
             "need_one": True, "show_if": {"key": "section", "in": ["solid_round", "hollow_round", "rectangle"]}},
            {"key": "Kt", "label": "Stress concentration factor Kt", "unit": "", "default": 1.0, "optional": True,
             "hint": "1 if none; holes and sharp fillets ≈ 2-3 (fatigue only)",
             "show_if": {"key": "section", "in": ["solid_round", "hollow_round", "rectangle"]}},
            {"key": "stress_MPa", "label": "Peak von Mises stress", "unit": "MPa", "hint": "e.g. 180",
             "show_if": {"key": "section", "in": ["known_stress"]}},
        ],
        "free_variable": "Critical section size",
        "index": {"n": 2 / 3, "prop": "Sy", "text": "σy^(2/3) / ρ"},
    },
}


def function_options() -> list[dict]:
    """Options for the 'function' question (and the input fields each one needs)."""
    return [{"value": k, "label": f["label"], "description": f["description"], "examples": f["examples"],
             "inputs": f["inputs"] + COMMON_INPUTS} for k, f in FUNCTIONS.items()]


def required_inputs(function: str) -> list[str]:
    f = FUNCTIONS.get(function)
    return [i["key"] for i in f["inputs"]
            if not i.get("optional") and i.get("type") != "select" and "show_if" not in i] if f else []


def _shown(spec: dict, values: dict) -> bool:
    """Does this input apply, given the select values chosen so far?"""
    cond = spec.get("show_if")
    return not cond or values.get(cond["key"]) in cond["in"]


def clean_inputs(function: str, raw: dict) -> dict:
    """Validate a design-input dict. Raises ValueError with a clear message."""
    f = FUNCTIONS[function]
    out = {}
    for spec in f["inputs"] + COMMON_INPUTS:
        if not _shown(spec, out):
            continue
        k = spec["key"]
        v = raw.get(k)
        if spec.get("type") == "select":
            vals = [o["value"] for o in spec["options"]]
            out[k] = v if v in vals else spec["default"]
            continue
        if v in (None, ""):
            if "default" in spec:
                out[k] = spec["default"]
            elif not spec.get("optional"):
                raise MissingInput(f"{spec['label']} ({spec['unit']}) is missing")
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{spec['label']} must be a number")
        if x < 0 or (x == 0 and not spec.get("optional")):
            raise ValueError(f"{spec['label']} must be positive")
        out[k] = x
    if out["fos"] < 1:
        raise ValueError("Factor of safety must be at least 1")
    loads = [s for s in f["inputs"] if s.get("need_one") and _shown(s, out)]
    if loads and not any(out.get(s["key"]) for s in loads):
        raise MissingInput("Enter at least one load (" + ", ".join(s["label"].lower() for s in loads) + ")")
    if out.get("di_mm", 0) >= out.get("d_mm", math.inf):
        raise ValueError("Inner diameter must be smaller than the outer diameter")
    if out.get("Kt", 1) < 1:
        raise ValueError("Stress concentration factor must be at least 1")
    return out


def _r(x: float, d: int = 1) -> float:
    return round(float(x), d)


def _fatigue(sigma_a: float, fos: float) -> float:
    """Su needed so that Se = 0.5*K*Su >= FoS * alternating stress."""
    return fos * sigma_a / (SE_RATIO * MARIN_K)


def _section(x: dict) -> tuple[float, float, float, float, str]:
    """Area, bending modulus Z, torsion modulus Zt (τ = T/Zt), peak shear factor, formula text."""
    if x["section"] == "rectangle":
        b, h = x["b_mm"], x["h_mm"]
        w, t = max(b, h), min(b, h)
        return (b * h, b * h ** 2 / 6, w * t ** 2 / (3 + 1.8 * t / w), 1.5,
                f"Rectangle {b:g} × {h:g} mm: A = bh, Z = bh²/6, Zt = wt²/(3 + 1.8t/w)")
    d = x["d_mm"]
    di = x.get("di_mm", 0.0) if x["section"] == "hollow_round" else 0.0
    Z = math.pi * (d ** 4 - di ** 4) / (32 * d)
    if di:
        return (math.pi * (d ** 2 - di ** 2) / 4, Z, 2 * Z, 2.0,
                f"Tube Ø{d:g}/Ø{di:g} mm: A = π(d²−di²)/4, Z = π(d⁴−di⁴)/(32d), Zt = 2Z")
    return math.pi * d ** 2 / 4, Z, 2 * Z, 4 / 3, f"Solid round Ø{d:g} mm: A = πd²/4, Z = πd³/32, Zt = 2Z"


def calculate(function: str, inputs: dict) -> dict:
    """
    Run the design calculation. Returns
      steps:        human-readable calculation lines (shown in the UI)
      requirements: {"Sy": MPa, "Su": MPa, "E": MPa}  -> used as hard screening limits
      checks:       [{name, prop, required}]          -> per-material actual FoS
    """
    x = clean_inputs(function, inputs)
    fos = x["fos"]
    cyclic = x["loading"] == "cyclic" or FUNCTIONS[function].get("always_cyclic", False)
    steps, req, checks = [], {}, []

    def need(prop, value, name):
        req[prop] = max(req.get(prop, 0.0), value)
        checks.append({"name": name, "prop": prop, "required": _r(value)})

    if function == "tie":
        A = math.pi * x["d_mm"] ** 2 / 4
        s = x["F_kN"] * 1000 / A
        steps += [f"Area A = πd²/4 = π·{x['d_mm']:g}²/4 = {A:.1f} mm²",
                  f"Axial stress σ = F/A = {x['F_kN'] * 1000:g}/{A:.1f} = {s:.1f} MPa"]
        need("Sy", fos * s, "Yielding (static)")
        steps.append(f"Static: σ ≤ Sy/FoS  →  Sy ≥ {fos:g} × {s:.1f} = {fos * s:.0f} MPa")
        if cyclic:
            su = _fatigue(s, fos)
            need("Su", su, "Fatigue")
            steps.append(f"Fatigue: Se = {SE_RATIO}·{MARIN_K}·Su ≥ FoS·σ  →  Su ≥ {su:.0f} MPa")

    elif function == "beam":
        cant = x["support"] == "cantilever"
        F, L, b, h = x["F_N"], x["L_mm"], x["b_mm"], x["h_mm"]
        M = F * L if cant else F * L / 4
        Z = b * h ** 2 / 6
        s = M / Z
        steps += [f"Bending moment M = {'F·L' if cant else 'F·L/4'} = {M / 1000:.1f} N·m",
                  f"Section modulus Z = bh²/6 = {b:g}·{h:g}²/6 = {Z:.0f} mm³",
                  f"Bending stress σ = M/Z = {M:.0f}/{Z:.0f} = {s:.1f} MPa"]
        need("Sy", fos * s, "Yielding (static)")
        steps.append(f"Static: σ ≤ Sy/FoS  →  Sy ≥ {fos:g} × {s:.1f} = {fos * s:.0f} MPa")
        if cyclic:
            su = _fatigue(s, fos)
            need("Su", su, "Fatigue")
            steps.append(f"Fatigue: Se = {SE_RATIO}·{MARIN_K}·Su ≥ FoS·σ  →  Su ≥ {su:.0f} MPa")
        if x.get("defl_mm"):
            I = b * h ** 3 / 12
            c = 3 if cant else 48
            E = F * L ** 3 / (c * I * x["defl_mm"])
            steps += [f"Moment of inertia I = bh³/12 = {I:.0f} mm⁴",
                      f"Deflection δ = FL³/({c}EI) ≤ {x['defl_mm']:g} mm  →  E ≥ {E:.0f} MPa"]
            req["E"] = max(req.get("E", 0.0), E)
            checks.append({"name": "Stiffness (deflection)", "prop": "E", "required": _r(E, 0), "no_fos": True})

    elif function == "shaft":
        T = 9550 * x["P_kW"] / x["N_rpm"]           # N·m
        M = x.get("M_Nm", 0.0) or 0.0
        d = x["d_mm"]
        k = 16 / (math.pi * d ** 3)
        tau = k * math.sqrt(M ** 2 + T ** 2) * 1000   # MPa
        steps += [f"Torque T = 9550·P/N = 9550·{x['P_kW']:g}/{x['N_rpm']:g} = {T:.1f} N·m",
                  f"Max shear stress (max shear stress theory) τ = 16/(πd³)·√(M²+T²) = {tau:.1f} MPa"]
        need("Sy", 2 * fos * tau, "Yielding (static)")
        steps.append(f"Static: τ ≤ 0.5·Sy/FoS  →  Sy ≥ 2 × {fos:g} × {tau:.1f} = {2 * fos * tau:.0f} MPa")
        if M > 0:  # a rotating shaft sees fully reversed bending every revolution
            sb = 32 * M * 1000 / (math.pi * d ** 3)
            su = _fatigue(sb, fos)
            steps += [f"Rotating bending stress σb = 32M/(πd³) = {sb:.1f} MPa (fully reversed)",
                      f"Fatigue: Se = {SE_RATIO}·{MARIN_K}·Su ≥ FoS·σb  →  Su ≥ {su:.0f} MPa"]
            need("Su", su, "Fatigue (rotating bending)")
        elif cyclic:
            su = _fatigue(tau * 2, fos)
            need("Su", su, "Fatigue")
            steps.append(f"Fatigue (fluctuating torque): shear endurance Sse ≈ 0.5·Se  →  Su ≥ {su:.0f} MPa")

    elif function == "pressure_vessel":
        s = x["p_MPa"] * x["D_mm"] / (2 * x["t_mm"])
        ratio = x["D_mm"] / x["t_mm"]
        steps += [f"Hoop stress σh = pD/(2t) = {x['p_MPa']:g}·{x['D_mm']:g}/(2·{x['t_mm']:g}) = {s:.1f} MPa",
                  f"Longitudinal stress σl = pD/(4t) = {s / 2:.1f} MPa (hoop stress governs)"]
        if ratio < 20:
            steps.append(f"Note: D/t = {ratio:.0f} < 20, so thin-cylinder theory is approximate")
        need("Sy", fos * s, "Yielding (static)")
        steps.append(f"Static: σh ≤ Sy/FoS  →  Sy ≥ {fos:g} × {s:.1f} = {fos * s:.0f} MPa")
        if cyclic:
            su = _fatigue(s, fos)
            need("Su", su, "Fatigue (pressure cycles)")
            steps.append(f"Fatigue: Su ≥ {su:.0f} MPa")

    elif function == "spring":
        C = x["Dm_mm"] / x["dw_mm"]
        Kw = (4 * C - 1) / (4 * C - 4) + 0.615 / C
        tau = Kw * 8 * x["F_N"] * x["Dm_mm"] / (math.pi * x["dw_mm"] ** 3)
        ratio = 0.30 if cyclic else 0.45   # allowable shear as a fraction of Su (spring wire)
        su = fos * tau / ratio
        steps += [f"Spring index C = D/d = {C:.1f}" + ("  (recommended 4–12)" if not 4 <= C <= 12 else ""),
                  f"Wahl factor K = (4C−1)/(4C−4) + 0.615/C = {Kw:.3f}",
                  f"Shear stress τ = K·8FD/(πd³) = {tau:.0f} MPa",
                  f"Allowable τ ≈ {ratio}·Su ({'fatigue' if cyclic else 'static'} spring wire)  →  "
                  f"Su ≥ {fos:g} × {tau:.0f}/{ratio} = {su:.0f} MPa"]
        need("Su", su, "Wire shear stress")

    elif function == "gear":
        T = 9550 * x["P_kW"] / x["N_rpm"]
        d = x["m_mm"] * x["z"]
        Ft = 2 * T * 1000 / d
        v = math.pi * d * x["N_rpm"] / 60000
        Cv = 3 / (3 + v)
        Y = 0.484 - 2.87 / x["z"]
        s = Ft / (Cv * x["b_mm"] * x["m_mm"] * Y)
        steps += [f"Pitch diameter d = m·z = {x['m_mm']:g}·{x['z']:g} = {d:.0f} mm",
                  f"Torque T = 9550·P/N = {T:.1f} N·m;  tangential force Ft = 2T/d = {Ft:.0f} N",
                  f"Pitch line velocity v = πdN/60000 = {v:.2f} m/s;  velocity factor Cv = 3/(3+v) = {Cv:.2f}",
                  f"Lewis form factor Y = 0.484 − 2.87/z = {Y:.3f} (20° full depth)",
                  f"Tooth bending stress σ = Ft/(Cv·b·m·Y) = {s:.1f} MPa"]
        if x["z"] < 17:
            steps.append("Note: fewer than 17 teeth (20° full depth) risks interference/undercut")
        need("Sy", fos * s, "Tooth bending (static)")
        su = _fatigue(s, fos)
        need("Su", su, "Tooth bending fatigue")
        steps += [f"Static: Sy ≥ {fos:g} × {s:.1f} = {fos * s:.0f} MPa",
                  f"Teeth are loaded once per revolution (fatigue): Su ≥ {su:.0f} MPa",
                  "Teeth also rub → Wear-resistant / surface-hardenable materials preferred"]

    elif function == "general":
        Kt = x.get("Kt", 1.0)
        axial = x.get("F_kN", 0.0) or 0.0
        if x["section"] == "known_stress":
            s = x["stress_MPa"]
            steps.append(f"Peak von Mises stress entered directly: σ = {s:.1f} MPa")
        else:
            A, Z, Zt, kv, text = _section(x)
            sn = axial * 1000 / A + (x.get("M_Nm", 0.0) or 0.0) * 1000 / Z
            tau = (x.get("T_Nm", 0.0) or 0.0) * 1000 / Zt + kv * (x.get("V_kN", 0.0) or 0.0) * 1000 / A
            s = math.sqrt(sn ** 2 + 3 * tau ** 2)
            steps += [text + f"  →  A = {A:.1f} mm², Z = {Z:.0f} mm³, Zt = {Zt:.0f} mm³",
                      f"Normal stress σ = F/A + M/Z = {sn:.1f} MPa",
                      f"Shear stress τ = T/Zt + {kv:.2g}·V/A = {tau:.1f} MPa (peaks added, conservative)",
                      f"Von Mises stress σv = √(σ² + 3τ²) = {s:.1f} MPa"]
        need("Sy", fos * s, "Yielding (static)")
        steps.append(f"Static: σ ≤ Sy/FoS  →  Sy ≥ {fos:g} × {s:.1f} = {fos * s:.0f} MPa")
        if cyclic:
            su = _fatigue(Kt * s, fos)
            need("Su", su, "Fatigue")
            steps.append(f"Fatigue (treated as fully reversed, conservative): Se = {SE_RATIO}·{MARIN_K}·Su ≥ "
                         f"FoS·Kt·σ  →  Su ≥ {su:.0f} MPa")
        if axial:
            steps.append("Note: if the axial force is compressive on a slender part, check buckling separately")
        steps.append("First estimate: valid only if these values describe the most stressed section. "
                     "Confirm complex shapes with FEA or testing")

    if function == "shaft" and (x.get("M_Nm") or 0) > 0:
        cyclic = True   # rotating bending is always fully reversed
    if cyclic:
        x["loading"] = "cyclic"
    if cyclic and function != "spring":
        steps.append(f"Assumed endurance limit Se ≈ {SE_RATIO}·Su × {MARIN_K} (surface + size factor)")
    return {"inputs": x, "fos": fos, "cyclic": cyclic, "steps": steps,
            "requirements": {k: _r(v, 0) for k, v in req.items()}, "checks": checks}


def translation(function: str, inputs: dict, objective: str, environment_label: str) -> dict:
    """Ashby step 1: function, constraints, objective, free variable."""
    f = FUNCTIONS[function]
    labels = {i["key"]: i for i in f["inputs"] + COMMON_INPUTS}
    cons = []
    for k, v in inputs.items():
        spec = labels.get(k, {})
        if spec.get("type") == "select":
            v = next((o["label"] for o in spec["options"] if o["value"] == v), v)
            cons.append(f"{spec['label']}: {v}")
        elif v not in (None, 0, 0.0) or not spec.get("optional"):
            cons.append(f"{spec.get('label', k)} = {v:g} {spec.get('unit', '')}".strip())
    cons.append(f"Environment: {environment_label}")
    return {"function": f"{f['label'].split(' (')[0]}: {f['load_mode']}",
            "constraints": cons, "objective": objective, "free_variable": f["free_variable"]}


def material_index(function: str, objective: str) -> dict:
    """Ashby step 3: the index used to rank (larger = better)."""
    ix = FUNCTIONS[function]["index"]
    base = ix["text"]
    text = base.replace("ρ", "(Cost·ρ)") if objective == "min_cost" else base
    return {"n": ix["n"], "prop": ix["prop"], "use_E": ix.get("use_E", False), "mass_text": base,
            "text": text, "objective": objective}
