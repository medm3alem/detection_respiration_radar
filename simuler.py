"""Simulation complete sans carte : habitacle vide (reference) + scenario d'accident, puis traitement.

    python simuler.py                      # scenario par defaut, 120 s
    python simuler.py --duree 60 --sortie resultats_sim

Produit dans le dossier de sortie :
    vide.bin, scenario.bin   enregistrements au format UART de la carte (rejouables avec traiter.py)
    verite.json              ce qui se passe vraiment dans la scene
    reference_vide.npz       energie statique de chaque siege, habitacle vide
    etats.json               etats estimes chaque seconde
    synthese.png             figure (deplacement, etats, rythme mesure vs reel)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from radar_vitals.cfg_parser import parse_cfg
from radar_vitals.simulator import CabinSimulator, Occupant, Scene
from radar_vitals.processing import Params, VitalsProcessor, build_reference, save_reference
from radar_vitals import tlv, report

HERE = Path(__file__).resolve().parent


def default_scene(duration: float, seed: int = 7) -> Scene:
    """Scenario d'accident de demonstration (modifiable librement)."""
    return Scene(duration_s=duration, seed=seed, occupants=[
        # conducteur : respire puis arret respiratoire a mi-parcours (le coeur continue)
        Occupant(zone=0, breath_bpm=14, breath_mm=5.0, heart_bpm=72, apnea=[(duration / 2, None)]),
        # passager avant : conscient, respire, bouge entre 30 et 36 s
        Occupant(zone=1, breath_bpm=18, breath_mm=6.0, heart_bpm=80, moving=[(30.0, 36.0)]),
        # arriere gauche : vide
        # arriere milieu : enfant, respiration rapide et faible
        Occupant(zone=3, kind="enfant", breath_bpm=26, breath_mm=2.5, heart_bpm=110, rcs=0.5),
        # arriere droit : inconscient, ne respire pas des le debut, pas de battement visible
        Occupant(zone=4, breath_bpm=15, breath_mm=5.0, heart_bpm=0, heart_mm=0.0,
                 apnea=[(0.0, None)], heart_stops_with_breath=True),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", default=str(HERE / "cfg" / "cpd_vitals.cfg"))
    ap.add_argument("--duree", type=float, default=120.0, help="duree du scenario (s)")
    ap.add_argument("--sortie", default=str(HERE / "resultats_sim"))
    ap.add_argument("--bruit", type=float, default=6.0, help="bruit apres FFT (LSB)")
    ap.add_argument("--graine", type=int, default=7, help="graine aleatoire du scenario")
    ap.add_argument("--pas-de-figure", action="store_true")
    args = ap.parse_args()

    out = Path(args.sortie)
    out.mkdir(parents=True, exist_ok=True)
    cfg = parse_cfg(args.cfg)
    print("Config :", cfg.summary())

    t0 = time.time()
    print("\n1) Habitacle vide (reference, 20 s)")
    CabinSimulator(cfg, Scene(duration_s=20, occupants=[], noise_counts=args.bruit, seed=3)).run(out / "vide.bin", progress=False)
    with open(out / "vide.bin", "rb") as fh:
        ref = build_reference(cfg, tlv.iter_frames(fh))
    save_reference(ref, out / "reference_vide.npz")

    print(f"2) Scenario d'accident ({args.duree:.0f} s)")
    scene = default_scene(args.duree, args.graine)
    scene.noise_counts = args.bruit
    CabinSimulator(cfg, scene).run(out / "scenario.bin", out / "verite.json", progress=False)
    print(f"   simulation : {time.time() - t0:.1f} s, fichier {(out / 'scenario.bin').stat().st_size / 1e6:.1f} Mo")

    print("3) Traitement (comme en direct, mise a jour chaque seconde)\n")
    params = Params()
    proc = VitalsProcessor(cfg, params, reference=ref)
    t1 = time.time()
    with open(out / "scenario.bin", "rb") as fh:
        for fr in tlv.iter_frames(fh):
            res = proc.push(fr)
            if res is not None and int(round(proc.n / proc.fs)) % 10 == 0:
                print(report.format_states(proc.n / proc.fs, res))
    dt = time.time() - t1
    print(f"\nTraitement : {proc.n} images en {dt:.1f} s ({proc.n / max(dt, 1e-9):.0f} images/s, "
          f"temps reel = {cfg.frame_rate_hz:.0f} images/s)")

    report.save_timeline(proc.timeline, out / "etats.json")
    truth = json.loads((out / "verite.json").read_text())
    ev = report.evaluate(json.loads((out / "etats.json").read_text()), truth, params.alert_after_s)
    report.print_evaluation(ev, truth)
    (out / "evaluation.json").write_text(json.dumps(ev, indent=1))
    if not args.pas_de_figure:
        report.plot_summary(proc.timeline, proc.traces, proc.fs, out / "synthese.png", truth,
                            "Simulation AWRL6844 - scenario post-accident")
        print(f"\nFigure : {out / 'synthese.png'}")


if __name__ == "__main__":
    main()
