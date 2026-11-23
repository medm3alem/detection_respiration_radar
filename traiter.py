"""Traite un enregistrement (simule ou capture sur la carte) et affiche l'etat de chaque siege.

    python traiter.py essai1.bin --reference vide.bin
    python traiter.py resultats_sim/scenario.bin --reference resultats_sim/reference_vide.npz \\
                      --verite resultats_sim/verite.json

Sorties (dans --sortie, par defaut a cote de l'enregistrement) : etats.json, synthese.png.
Sans --reference, la presence statique est inconnue : un siege sans respiration est INCERTAIN.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from radar_vitals.cfg_parser import parse_cfg
from radar_vitals.processing import Params, VitalsProcessor, build_reference, load_reference, save_reference
from radar_vitals import tlv, report

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("enregistrement", help="fichier .bin (flux UART brut)")
    ap.add_argument("--cfg", help="cfg utilise pour l'enregistrement (defaut : <enregistrement>.cfg "
                                  "s'il existe, sinon cfg/cpd_vitals.cfg)")
    ap.add_argument("--inverser-azimut", action="store_true", help="si gauche/droite sont inverses")
    ap.add_argument("--inverser-elevation", action="store_true", help="si avant/arriere sont inverses")
    ap.add_argument("--reference", help="habitacle vide : .npz ou .bin")
    ap.add_argument("--verite", help="verite.json (simulation) pour evaluer")
    ap.add_argument("--sortie", help="dossier de sortie")
    ap.add_argument("--chaque", type=float, default=5.0, help="afficher les etats toutes les N s")
    args = ap.parse_args()

    rec = Path(args.enregistrement)
    cfg_path = Path(args.cfg) if args.cfg else (rec.with_suffix(".cfg") if rec.with_suffix(".cfg").exists()
                                                 else HERE / "cfg" / "cpd_vitals.cfg")
    cfg = parse_cfg(cfg_path)
    print(f"Config ({cfg_path.name}) :", cfg.summary())
    out = Path(args.sortie) if args.sortie else rec.parent / (rec.stem + "_resultats")
    out.mkdir(parents=True, exist_ok=True)

    ref = None
    if args.reference:
        if args.reference.endswith(".npz"):
            ref = load_reference(args.reference)
        else:
            with open(args.reference, "rb") as fh:
                ref = build_reference(cfg, tlv.iter_frames(fh, cfg.num_virt_ant))
            save_reference(ref, out / "reference_vide.npz")
        print(f"Reference habitacle vide : {ref['num_frames']} images")

    params = Params(flip_azimuth=args.inverser_azimut, flip_elevation=args.inverser_elevation)
    proc = VitalsProcessor(cfg, params, reference=ref)
    n_frames = n_cube = 0
    every = max(1, int(round(args.chaque)))
    with open(rec, "rb") as fh:
        for fr in tlv.iter_frames(fh, cfg.num_virt_ant):
            n_frames += 1
            n_cube += fr.radar_cube is not None
            res = proc.push(fr)
            if res is not None and int(round(proc.n / proc.fs)) % every == 0:
                print(report.format_states(proc.n / proc.fs, res))
    print(f"\n{n_frames} trames lues, {n_cube} avec radar cube, {proc.dropped} trames perdues")
    if n_cube == 0:
        print("Aucun radar cube (TLV 2007) : le cfg contient-il dbgGuiMonitor ... 1 et exportRadCubeChunk ?")
        return
    report.save_timeline(proc.timeline, out / "etats.json")
    truth = None
    if args.verite:
        truth = json.loads(Path(args.verite).read_text())
        ev = report.evaluate(json.loads((out / "etats.json").read_text()), truth, params.alert_after_s)
        report.print_evaluation(ev, truth)
    report.plot_summary(proc.timeline, proc.traces, proc.fs, out / "synthese.png", truth, rec.name)
    print(f"Resultats : {out}")


if __name__ == "__main__":
    main()
