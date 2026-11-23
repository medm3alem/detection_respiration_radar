"""Capture avec la carte AWRL6844 (firmware in-cabin TI precompile, mode CPD).

PAS ENCORE TESTE SUR CARTE : ecrit d'apres les sources du firmware et du visualiseur TI.

    python capture.py --lister-ports
    python capture.py --duree 60 --sortie vide.bin                       # habitacle vide (reference)
    python capture.py --duree 120 --sortie essai1.bin --direct --reference vide.bin

Ce que fait le script :
1. ouvre le port de commandes (115200 bauds) et envoie le cfg ligne par ligne
   (les lignes dbgGuiMonitor / exportRadCubeChunk sont ajoutees si absentes) ;
2. lit le port de donnees (1 250 000 bauds) et enregistre TOUT le flux brut dans le .bin
   (rejouable ensuite avec traiter.py) ;
3. option --direct : traite les trames au fil de l'eau et affiche l'etat des sieges.

Ports de l'EVM (XDS110) sous Windows, gestionnaire de peripheriques :
    "XDS110 Class Application/User UART (COMx)"  -> commandes (--port-cli)
    "XDS110 Class Auxiliary Data Port (COMy)"    -> donnees (--port-data)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


from radar_vitals.serie import list_ports, guess_ports, send_cfg, cfg_lines  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", default=str(HERE / "cfg" / "cpd_vitals.cfg"))
    ap.add_argument("--port-cli", help="ex. COM5 (detecte automatiquement sinon)")
    ap.add_argument("--port-data", help="ex. COM6 (detecte automatiquement sinon)")
    ap.add_argument("--baud-data", type=int, default=1_250_000)
    ap.add_argument("--duree", type=float, default=60.0, help="duree d'enregistrement (s)")
    ap.add_argument("--sortie", default="capture.bin")
    ap.add_argument("--direct", action="store_true", help="traitement en direct + affichage des etats")
    ap.add_argument("--inverser-azimut", action="store_true", help="si gauche/droite sont inverses")
    ap.add_argument("--inverser-elevation", action="store_true", help="si avant/arriere sont inverses")
    ap.add_argument("--reference", help="habitacle vide : .npz (traiter.py) ou .bin (capture.py)")
    ap.add_argument("--sans-cfg", action="store_true", help="ne pas envoyer le cfg (carte deja lancee)")
    ap.add_argument("--lister-ports", action="store_true")
    args = ap.parse_args()

    try:
        import serial
    except ImportError:
        sys.exit("pyserial manquant : pip install pyserial")

    if args.lister_ports:
        list_ports()
        return

    from radar_vitals.cfg_parser import parse_cfg, with_radcube_export
    from radar_vitals import tlv, report
    from radar_vitals.processing import Params, VitalsProcessor, build_reference, load_reference

    cli, data = args.port_cli, args.port_data
    if not cli or not data:
        g_cli, g_data = guess_ports()
        cli, data = cli or g_cli, data or g_data
    if not data or (not cli and not args.sans_cfg):
        list_ports()
        sys.exit("Ports non trouves : precise --port-cli et --port-data")

    cfg_text = Path(args.cfg).read_text(encoding="utf-8", errors="replace").replace("\r", "")
    if "exportRadCubeChunk" not in cfg_text:
        cfg_text = with_radcube_export(cfg_text)
        print("(lignes dbgGuiMonitor / exportRadCubeChunk ajoutees au cfg)")
    tmp_cfg = Path(args.sortie).with_suffix(".cfg")
    tmp_cfg.write_text(cfg_text)          # le cfg reellement envoye, garde avec l'enregistrement
    cfg = parse_cfg(tmp_cfg)
    print("Config :", cfg.summary())

    ser = serial.Serial(data, args.baud_data, timeout=0.2)
    ser.reset_input_buffer()

    if not args.sans_cfg:
        print(f"Envoi du cfg sur {cli} :")
        if not send_cfg(cli, cfg_lines(cfg_text)):
            print("Des commandes ont ete refusees : verifier le firmware flashe et le cfg.")

    proc = None
    if args.direct:
        ref = None
        if args.reference:
            if args.reference.endswith(".npz"):
                ref = load_reference(args.reference)
            else:
                with open(args.reference, "rb") as fh:
                    ref = build_reference(cfg, tlv.iter_frames(fh, cfg.num_virt_ant))
        proc = VitalsProcessor(cfg, Params(flip_azimuth=args.inverser_azimut,
                                           flip_elevation=args.inverser_elevation), reference=ref)

    pb = tlv.PacketBuffer()
    n_bytes = n_frames = 0
    t0 = time.time()
    print(f"Enregistrement {args.duree:.0f} s sur {data} -> {args.sortie}  (Ctrl+C pour arreter)")
    try:
        with open(args.sortie, "wb") as fh:
            while time.time() - t0 < args.duree:
                chunk = ser.read(8192)
                if not chunk:
                    continue
                fh.write(chunk)
                n_bytes += len(chunk)
                for pkt in pb.feed(chunk):
                    n_frames += 1
                    if proc is not None:
                        fr = tlv.decode_packet(pkt, cfg.num_virt_ant)
                        if fr.radar_cube is None and n_frames == 5:
                            print("ATTENTION : pas de TLV 2007 (radar cube) dans les trames.")
                        res = proc.push(fr)
                        if res is not None:
                            print(report.format_states(proc.n / proc.fs, res))
                if proc is None and n_frames and n_frames % 25 == 0:
                    el = time.time() - t0
                    print(f"  {el:5.0f} s : {n_frames} trames, {n_bytes / max(el, 1e-3) / 1e3:.0f} ko/s")
    except KeyboardInterrupt:
        print("\nArret demande.")
    finally:
        ser.close()
        if not args.sans_cfg and cli:
            try:
                send_cfg(cli, ["sensorStop 0"])
            except Exception as exc:  # carte deja arretee, port occupe...
                print(f"(sensorStop non envoye : {exc})")
    el = time.time() - t0
    print(f"Termine : {n_frames} trames ({n_frames / max(el, 1e-3):.1f} /s), {n_bytes / 1e6:.1f} Mo -> {args.sortie}")
    if n_frames == 0:
        print("Aucune trame : mauvais port de donnees, baud, ou carte pas en mode fonctionnel (switches + NRST).")
    if proc is not None and proc.dropped:
        print(f"Trames perdues : {proc.dropped} (UART sature ? reduire exportRadCubeChunk)")


if __name__ == "__main__":
    main()
