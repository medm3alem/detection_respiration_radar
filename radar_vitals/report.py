"""Affichage console, evaluation vs verite (simulation) et figure de synthese."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEATS = {0: "Conducteur", 1: "Passager AV", 2: "Arriere G", 3: "Arriere milieu", 4: "Arriere D"}
STATE_COLORS = {"INIT": "#d0d0d0", "VIDE": "#e8e8e8", "RESPIRE": "#2e9e5b", "MOUVEMENT": "#3b7dd8",
                "SURVEILLANCE": "#f0a020", "ALERTE": "#d63a3a", "INCERTAIN": "#9c8ac9"}


def seat_name(z: int) -> str:
    return SEATS.get(z, f"Zone {z}")


def format_states(t: float, results: dict) -> str:
    cells = []
    for z, r in sorted(results.items()):
        d = r.as_dict() if hasattr(r, "as_dict") else r
        txt = d["state"]
        if d["state"] == "RESPIRE" and d["breath_bpm"]:
            txt += f" {d['breath_bpm']:.0f}/min"
        elif d["state"] in ("SURVEILLANCE", "ALERTE", "INCERTAIN") and d["no_breath_s"]:
            txt += f" ({d['no_breath_s']:.0f}s)"
        cells.append(f"{seat_name(z)[:14]:>14}: {txt:<22}")
    return f"t={t:6.1f}s | " + " | ".join(cells)


def expected_state(zone_truth: dict | None, t: float, onset: float | None, alert_after: float) -> str:
    """Etat attendu a partir de la verite terrain (avec la latence d'alerte autorisee)."""
    if zone_truth is None:
        return "VIDE"
    if zone_truth["moving"]:
        return "MOUVEMENT"
    if zone_truth["breathing"]:
        return "RESPIRE"
    if onset is not None and t - onset < alert_after:
        return "TRANSITION"
    return "ALERTE"


def evaluate(timeline: list[dict], truth: dict, alert_after: float = 10.0, latency_s: float = 12.0) -> dict:
    """Compare les etats estimes a la verite simulee. Marge de latence apres chaque changement."""
    fr_rate = truth["frame_rate_hz"]
    frames = truth["frames"]
    zones = sorted({int(z) for ent in timeline for z in ent["zones"]})
    stats = {z: {"ok": 0, "n": 0, "rate_err": [], "alert_t": None, "onset": None, "false_breath": 0,
                 "false_alert": 0} for z in zones}
    # instants de debut d'arret respiratoire
    for z in zones:
        prev = None
        for fr in frames:
            zt = fr["zones"].get(str(z))
            b = None if zt is None else zt["breathing"]
            if prev is True and b is False and stats[z]["onset"] is None:
                stats[z]["onset"] = fr["t"]
            if prev is None and b is False and stats[z]["onset"] is None:
                stats[z]["onset"] = fr["t"]
            prev = b
    changes = {z: [] for z in zones}
    for z in zones:
        prev = None
        for fr in frames:
            zt = fr["zones"].get(str(z))
            key = None if zt is None else (zt["breathing"], zt["moving"])
            if key != prev:
                changes[z].append(fr["t"])
            prev = key
    for ent in timeline:
        t = ent["t"]
        k = min(int(t * fr_rate) - 1, len(frames) - 1)
        if t < 12:
            continue
        for z in zones:
            est = ent["zones"][z] if z in ent["zones"] else ent["zones"][str(z)]
            zt = frames[k]["zones"].get(str(z))
            exp = expected_state(zt, t, stats[z]["onset"], alert_after)
            onset = stats[z]["onset"]
            if est["state"] == "ALERTE" and stats[z]["alert_t"] is None and onset is not None and t >= onset:
                stats[z]["alert_t"] = t
            if est["state"] == "ALERTE" and (onset is None or t < onset):
                stats[z]["false_alert"] += 1
            if any(0 <= t - c < latency_s for c in changes[z]) or exp == "TRANSITION":
                continue  # periode de transition : non comptee
            stats[z]["n"] += 1
            ok = est["state"] == exp or (exp == "MOUVEMENT" and est["state"] == "RESPIRE")
            stats[z]["ok"] += int(ok)
            if exp in ("ALERTE", "VIDE") and est["state"] == "RESPIRE":
                stats[z]["false_breath"] += 1
            if exp == "RESPIRE" and est["state"] == "RESPIRE" and est["breath_bpm"]:
                stats[z]["rate_err"].append(est["breath_bpm"] - zt["breath_bpm"])
    out = {}
    for z, s in stats.items():
        out[z] = {
            "correct_pct": round(100 * s["ok"] / s["n"], 1) if s["n"] else None,
            "rate_err_median": round(float(np.median(np.abs(s["rate_err"]))), 2) if s["rate_err"] else None,
            "breath_onset_stop_s": s["onset"],
            "first_alert_s": s["alert_t"],
            "alert_delay_s": (round(s["alert_t"] - s["onset"], 1) if s["alert_t"] is not None and s["onset"] is not None else None),
            "false_breathing_updates": s["false_breath"],
            "false_alert_updates": s["false_alert"],
        }
    return out


def print_evaluation(ev: dict, truth: dict):
    occ = {o["zone"]: o for o in truth["occupants"]}
    print("\nEvaluation (simulation) :")
    for z, e in ev.items():
        o = occ.get(z)
        desc = "vide" if o is None else f"{o['kind']} {o['breath_bpm']:.0f}/min" + (
            f", arret respi a {e['breath_onset_stop_s']:.0f}s" if e["breath_onset_stop_s"] is not None else "")
        line = f"  {seat_name(z):<15} ({desc}) : etats corrects {e['correct_pct']}%"
        if e["rate_err_median"] is not None:
            line += f", erreur rythme mediane {e['rate_err_median']}/min"
        if e["alert_delay_s"] is not None:
            line += f", ALERTE {e['alert_delay_s']}s apres l'arret"
        elif e["breath_onset_stop_s"] is not None:
            line += ", PAS D'ALERTE"
        if e["false_breathing_updates"]:
            line += f", ATTENTION {e['false_breathing_updates']} s 'respire' a tort"
        if e["false_alert_updates"]:
            line += f", {e['false_alert_updates']} s de fausse alerte"
        print(line)


def plot_summary(timeline: list[dict], traces: dict, fs: float, out_png: str | Path,
                 truth: dict | None = None, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    zones = sorted(int(z) for z in timeline[-1]["zones"])
    fig, axes = plt.subplots(len(zones), 2, figsize=(15, 2.3 * len(zones)), sharex="col",
                             gridspec_kw={"width_ratios": [3, 2]}, squeeze=False)
    times = np.array([e["t"] for e in timeline])
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    occ = {o["zone"]: o for o in truth["occupants"]} if truth else {}
    for row, z in enumerate(zones):
        ax, axr = axes[row]
        tr = np.array(traces.get(z, []), float)
        t_tr = np.arange(len(tr)) / fs
        ax.plot(t_tr, tr, lw=0.8, color="#333")
        ax.set_ylabel("depl. (mm)")
        ax.set_title(f"{seat_name(z)} - deplacement respiratoire mesure", fontsize=9, loc="left")
        ax.grid(alpha=0.3)
        # bande d'etats
        for e in timeline:
            d = e["zones"][z] if z in e["zones"] else e["zones"][str(z)]
            axr.axvspan(e["t"] - dt, e["t"], ymin=0, ymax=0.18, color=STATE_COLORS.get(d["state"], "#fff"), lw=0)
        rates = [((e["zones"][z] if z in e["zones"] else e["zones"][str(z)])["breath_bpm"] or np.nan) for e in timeline]
        axr.plot(times, rates, ".", ms=3, color="#2e9e5b", label="rythme mesure")
        if z in occ and truth:
            tt = np.array([f["t"] for f in truth["frames"]])
            ref = [occ[z]["breath_bpm"] if f["zones"][str(z)]["breathing"] else np.nan for f in truth["frames"]]
            axr.plot(tt, ref, "--", lw=1, color="#555", label="rythme reel")
        axr.set_ylim(-9, 50)
        axr.set_ylabel("/min")
        axr.set_title(f"{seat_name(z)} - etat et rythme", fontsize=9, loc="left")
        axr.grid(alpha=0.3)
        if row == 0:
            axr.legend(fontsize=7, loc="upper right")
    axes[-1][0].set_xlabel("temps (s)")
    axes[-1][1].set_xlabel("temps (s)")
    handles = [Patch(color=c, label=s) for s, c in STATE_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def save_timeline(timeline: list[dict], path: str | Path):
    Path(path).write_text(json.dumps(timeline, indent=1, default=float))
