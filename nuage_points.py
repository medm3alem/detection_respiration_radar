"""Nuage de points de la carte AWRL6844 (firmware in-cabin TI), en direct ou depuis un enregistrement.
Remplace le visualiseur MATLAB de TI pour voir ce que detecte le radar.

    python nuage_points.py --lister-ports
    python nuage_points.py                          # direct : envoie cfg/cpd_ti.cfg, affiche et enregistre
    python nuage_points.py --cfg cfg/sbr_ti.cfg     # mode SBR (rappel de ceinture) au lieu de CPD
    python nuage_points.py --repere monde           # repere voiture, avec les zones des sieges
    python nuage_points.py --fichier enregistrements/nuage_xxx.bin       # relecture (cfg : le .cfg a cote)
    python nuage_points.py --fichier resultats_sim/scenario.bin --repere monde   # simulation (simuler.py)

Repere capteur (par defaut, pour un essai sur une table) : origine = radar,
    Y = devant les antennes (axe de visee), X = lateral, Z = vertical (carte posee debout, antennes vers toi).
Repere monde : celui des zones du cfg (sensorPosition + cuboidDef), comme le visualiseur MATLAB TI.
Fermer la fenetre (ou Ctrl+C avec --sans-affichage) arrete le radar et termine l'enregistrement.
"""
from __future__ import annotations

import argparse
import queue
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from radar_vitals import tlv
from radar_vitals.cfg_parser import RadarConfig, parse_cfg
from radar_vitals.nuage import (CUBOID_EDGES, ZONE_HORS, count_per_zone, cuboid_corners,
                                spherical_to_sensor, zone_of_points)
from radar_vitals.report import SEATS
from radar_vitals.lecture import Lecteur, LecteurFichier, LecteurSerie
from radar_vitals.serie import BAUD_DATA, cfg_lines, guess_ports, list_ports, send_cfg

HERE = Path(__file__).resolve().parent
ZONE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b", "#e377c2", "#17becf"]
MODES = {0: "detection d'intrusion", 1: "SBR (ceinture)", 2: "CPD (presence enfant)"}


# --------------------------------------------------------------------------- etat affiche
class Etat:
    def __init__(self, cfg: RadarConfig, persistance: int, flip_az: bool, flip_el: bool):
        self.cfg, self.flip_az, self.flip_el = cfg, flip_az, flip_el
        self.hist: deque = deque(maxlen=max(1, persistance))
        self.n_frames = 0
        self.n_with_points = 0
        self.total_points = 0
        self.zone_total = np.zeros(max(cfg.num_zones, 1), dtype=int)
        self.last_frame_number: int | None = None
        self.lost = 0
        self.tlv_seen: set[int] = set()
        self.occupancy: np.ndarray | None = None
        self.n_last = 0
        self.t_recv: deque = deque(maxlen=25)

    @property
    def rate(self) -> float:
        if len(self.t_recv) < 2:
            return 0.0
        return (len(self.t_recv) - 1) / max(self.t_recv[-1] - self.t_recv[0], 1e-3)

    def ajouter(self, fr: tlv.Frame):
        self.t_recv.append(getattr(fr, "t_recv", time.time()))
        if self.last_frame_number is not None and fr.frame_number > self.last_frame_number + 1:
            self.lost += fr.frame_number - self.last_frame_number - 1
        self.last_frame_number = fr.frame_number
        self.n_frames += 1
        self.tlv_seen.update(fr.tlv_types)
        if fr.occupancy is not None and np.ndim(fr.occupancy) == 2:
            self.occupancy = fr.occupancy
        pts = fr.points if fr.points is not None else np.zeros((0, 5))
        xyz_s = spherical_to_sensor(pts, self.flip_az, self.flip_el)
        xyz_w = self.cfg.sensor_to_world(xyz_s) if len(pts) else np.zeros((0, 3))
        zones = zone_of_points(self.cfg, xyz_w)
        self.hist.append({"s": xyz_s, "w": xyz_w, "snr": pts[:, 4], "dop": pts[:, 3], "zone": zones})
        self.n_last = len(pts)
        self.total_points += len(pts)
        self.n_with_points += bool(len(pts))
        if self.cfg.num_zones:
            self.zone_total += count_per_zone(zones, self.cfg.num_zones)

    def affiches(self) -> dict[str, np.ndarray]:
        keys = ("s", "w", "snr", "dop", "zone")
        if not self.hist:
            return {"s": np.zeros((0, 3)), "w": np.zeros((0, 3)), "snr": np.zeros(0), "dop": np.zeros(0),
                    "zone": np.zeros(0, int)}
        return {k: np.concatenate([h[k] for h in self.hist]) for k in keys}

    def ligne_console(self) -> str:
        d = self.affiches()
        zc = count_per_zone(d["zone"], self.cfg.num_zones) if self.cfg.num_zones else []
        zs = " ".join(f"{z}:{c}" for z, c in enumerate(zc))
        occ = ""
        if self.occupancy is not None:
            p_occ = 1 - self.occupancy[0]
            occ = " | firmware occupe " + " ".join(f"{z}:{p:.2f}" for z, p in enumerate(p_occ))
        return (f"trame {self.last_frame_number} ({self.rate:4.1f}/s) | {self.n_last:3d} pts "
                f"| zones {zs}{occ}")


# --------------------------------------------------------------------------- affichage
def _limits(cfg: RadarConfig, repere: str) -> np.ndarray:
    if repere == "capteur":
        r = cfg.num_range_bins * cfg.range_step_m
        return np.array([[-r / 2, r / 2], [0.0, r], [-r / 2, r / 2]])
    pts = [cfg.sensor_pos]
    for boxes in cfg.cuboids.values():
        for b in boxes:
            pts.extend(cuboid_corners(b))
    pts = np.array(pts)
    lo, hi = pts.min(axis=0) - 0.3, pts.max(axis=0) + 0.3
    lo[2] = min(lo[2], 0.0)
    return np.column_stack([lo, hi])


class Affichage:
    def __init__(self, cfg: RadarConfig, repere: str, couleur: str, avec_3d: bool, titre: str):
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

        self.plt, self.cfg, self.repere, self.couleur = plt, cfg, repere, couleur
        self.lim = _limits(cfg, repere)
        nz = max(cfg.num_zones, 1)
        if couleur == "snr":
            self.cmap, self.norm, label = "viridis", Normalize(5, 30), "SNR (dB)"
        elif couleur == "doppler":
            self.cmap, self.norm, label = "coolwarm", Normalize(-0.5, 0.5), "vitesse radiale (m/s)"
        elif couleur == "hauteur":
            self.cmap, self.norm, label = "plasma", Normalize(*self.lim[2]), "Z (m)"
        else:
            self.cmap = ListedColormap(["#9aa0a6"] + ZONE_COLORS[:nz])
            self.norm, label = BoundaryNorm(np.arange(-1.5, nz), nz + 1), None

        self.fig = plt.figure(figsize=(15, 8.2) if avec_3d else (11, 8.2))
        self.fig.canvas.manager.set_window_title(titre) if self.fig.canvas.manager else None
        if avec_3d:
            gs = self.fig.add_gridspec(2, 3, width_ratios=[1.55, 1.0, 0.85], wspace=0.28, hspace=0.25)
            self.ax3 = self.fig.add_subplot(gs[:, 0], projection="3d")
            self.axt, self.axs = self.fig.add_subplot(gs[0, 1]), self.fig.add_subplot(gs[1, 1])
            self.axi = self.fig.add_subplot(gs[:, 2])
        else:
            gs = self.fig.add_gridspec(2, 2, width_ratios=[1.0, 0.8], wspace=0.25, hspace=0.25)
            self.ax3 = None
            self.axt, self.axs = self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[1, 0])
            self.axi = self.fig.add_subplot(gs[:, 1])
        self.axi.axis("off")
        self.fig.suptitle(titre, fontsize=11)

        kw = dict(c=np.zeros(0), cmap=self.cmap, norm=self.norm, s=22, edgecolors="none")
        self.sct = self.axt.scatter([], [], **kw)
        self.scs = self.axs.scatter([], [], **kw)
        self.sc3 = self.ax3.scatter([], [], [], depthshade=False, **kw) if self.ax3 else None

        (xl, yl, zl) = self.lim
        self.axt.set(xlim=xl, ylim=yl, xlabel="X lateral (m)", ylabel="Y profondeur (m)", title="Vue de dessus")
        self.axs.set(xlim=yl, ylim=zl, xlabel="Y profondeur (m)", ylabel="Z hauteur (m)", title="Vue de cote")
        for ax in (self.axt, self.axs):
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.3)
        if repere == "monde":   # vue de dessus : avant de la voiture en haut, conducteur a gauche
            self.axt.invert_xaxis()
            self.axt.invert_yaxis()
            self.axt.set_title("Vue de dessus (avant en haut)")
        if self.ax3:
            self.ax3.set(xlim=xl, ylim=yl, zlim=zl, xlabel="X", ylabel="Y", zlabel="Z")
            self.ax3.set_box_aspect(tuple(np.diff(self.lim, axis=1).ravel()))
            self.ax3.view_init(elev=22, azim=-125 if repere == "monde" else -70)
            self.ax3.set_title("Nuage de points 3D")
        if label:
            self.fig.colorbar(self.sct, ax=[self.axt, self.axs], shrink=0.6, label=label, pad=0.02)
        self._decor()
        self.txt = self.axi.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=9,
                                 transform=self.axi.transAxes)

    def _decor(self):
        cfg = self.cfg
        if self.repere == "capteur":
            origin, bore = np.zeros(3), np.array([0, 0.35, 0])
        else:
            origin, bore = cfg.sensor_pos, cfg.rot_tw @ np.array([0, 0.35, 0])
        tip = origin + bore
        for ax, (i, j) in ((self.axt, (0, 1)), (self.axs, (1, 2))):
            ax.plot(origin[i], origin[j], "k^", ms=9, zorder=5)
            ax.annotate("", xy=(tip[i], tip[j]), xytext=(origin[i], origin[j]),
                        arrowprops=dict(arrowstyle="->", color="k"))
        if self.ax3:
            self.ax3.scatter(*origin, marker="^", s=60, c="k")
            self.ax3.plot(*np.vstack([origin, tip]).T, "k-")
        if self.repere == "capteur":
            r = self.lim[1, 1]
            for rr in np.arange(0.5, r + 1e-6, 0.5):          # cercles de distance (vue de dessus)
                a = np.linspace(-np.pi / 2, np.pi / 2, 90)
                self.axt.plot(rr * np.sin(a), rr * np.cos(a), color="0.85", lw=0.7, zorder=0)
            return
        for zone, boxes in sorted(cfg.cuboids.items()):
            col = ZONE_COLORS[zone % len(ZONE_COLORS)]
            for b in boxes:
                c = cuboid_corners(b)
                for i, j in CUBOID_EDGES:
                    seg = c[[i, j]]
                    self.axt.plot(seg[:, 0], seg[:, 1], color=col, lw=0.8, alpha=0.6)
                    self.axs.plot(seg[:, 1], seg[:, 2], color=col, lw=0.8, alpha=0.4)
                    if self.ax3:
                        self.ax3.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=col, lw=0.6, alpha=0.5)
            x1, x2, y1, y2, _, _ = boxes[0]
            self.axt.text((x1 + x2) / 2, (y1 + y2) / 2, SEATS.get(zone, f"Zone {zone}").replace(" ", "\n"),
                          ha="center", va="center", fontsize=7, color=col)

    def maj(self, etat: Etat, infos: str):
        d = etat.affiches()
        xyz = d["s"] if self.repere == "capteur" else d["w"]
        vals = {"snr": d["snr"], "doppler": d["dop"], "hauteur": xyz[:, 2], "zone": d["zone"]}[self.couleur]
        self.sct.set_offsets(xyz[:, [0, 1]] if len(xyz) else np.zeros((0, 2)))
        self.scs.set_offsets(xyz[:, [1, 2]] if len(xyz) else np.zeros((0, 2)))
        for sc in (self.sct, self.scs):
            sc.set_array(np.asarray(vals, float))
        if self.sc3 is not None:
            self.sc3._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
            self.sc3.set_array(np.asarray(vals, float))
        self.txt.set_text(infos)


def texte_infos(etat: Etat, cfg: RadarConfig, repere: str, source: str, persistance: int, conseil: str) -> str:
    lines = [source, ""]
    lines.append(f"Trame     {etat.last_frame_number if etat.last_frame_number is not None else '-'}")
    lines.append(f"Debit     {etat.rate:4.1f} trames/s (prevu {cfg.frame_rate_hz:.0f})")
    lines.append(f"Recues    {etat.n_frames}   perdues {etat.lost}")
    lines.append(f"Points    {etat.n_last} (affiches : {persistance} trame(s))")
    if etat.n_frames:
        lines.append(f"Moyenne   {etat.total_points / etat.n_frames:.1f} pts/trame, "
                     f"{100 * etat.n_with_points / etat.n_frames:.0f} % avec points")
    lines.append(f"TLV       {' '.join(str(t) for t in sorted(etat.tlv_seen)) or '-'}")
    lines.append(f"Mode      {MODES.get(cfg.running_mode, cfg.running_mode)}")
    lines.append("")
    if cfg.num_zones:
        d = etat.affiches()
        zc = count_per_zone(d["zone"], cfg.num_zones)
        occ = etat.occupancy
        head = "Siege            pts" + ("   firmware" if occ is not None else "")
        lines.append(head)
        for z in range(cfg.num_zones):
            row = f"{SEATS.get(z, f'Zone {z}'):15s} {zc[z]:4d}"
            if occ is not None and z < occ.shape[1]:
                p = 1 - occ[0, z]
                row += f"   {'occupe' if p > 0.5 else 'vide  '} {p:.2f}"
            lines.append(row)
        lines.append(f"{'Hors zones':15s} {(d['zone'] == ZONE_HORS).sum():4d}")
        if occ is not None and cfg.running_mode == 2:
            p_life = 1 - occ[0].min()
            lines.append(f"Vie detectee (LPD) : {'OUI' if p_life > 0.5 else 'non'} {p_life:.2f}")
        if repere == "capteur":
            lines += ["", "(zones calculees avec sensorPosition", " du cfg : utiles seulement si le",
                      " radar est monte dans la voiture)"]
    if conseil:
        lines += ["", conseil]
    return "\n".join(lines)


# --------------------------------------------------------------------------- programme
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", help="cfg a envoyer / du fichier (defaut : cfg/cpd_ti.cfg, ou le .cfg du fichier)")
    ap.add_argument("--fichier", help="relire un enregistrement .bin au lieu de la carte")
    ap.add_argument("--vitesse", type=float, default=1.0, help="relecture : 1 = temps reel, 0 = au plus vite")
    ap.add_argument("--port-cli", help="port de commandes, ex. COM5 (detecte sinon)")
    ap.add_argument("--port-data", help="port de donnees, ex. COM6 (detecte sinon)")
    ap.add_argument("--baud-data", type=int, default=BAUD_DATA)
    ap.add_argument("--sans-cfg", action="store_true", help="ne pas envoyer le cfg (radar deja lance)")
    ap.add_argument("--sortie", help="fichier d'enregistrement (defaut : enregistrements/nuage_<date>.bin)")
    ap.add_argument("--pas-d-enregistrement", action="store_true")
    ap.add_argument("--duree", type=float, default=0.0, help="arret automatique apres N secondes (0 = jamais)")
    ap.add_argument("--repere", choices=["capteur", "monde"], default="capteur")
    ap.add_argument("--couleur", choices=["snr", "doppler", "zone", "hauteur"], default="snr")
    ap.add_argument("--persistance", type=int, default=3, help="nombre de trames superposees a l'ecran")
    ap.add_argument("--sans-3d", action="store_true", help="vues 2D seulement (plus fluide)")
    ap.add_argument("--sans-affichage", action="store_true", help="console seulement")
    ap.add_argument("--inverser-azimut", action="store_true", help="si gauche et droite sont inverses")
    ap.add_argument("--inverser-elevation", action="store_true", help="si haut et bas sont inverses")
    ap.add_argument("--lister-ports", action="store_true")
    args = ap.parse_args()

    if args.lister_ports:
        list_ports()
        return

    # ---- configuration
    if args.cfg:
        cfg_path = Path(args.cfg)
    elif args.fichier and Path(args.fichier).with_suffix(".cfg").exists():
        cfg_path = Path(args.fichier).with_suffix(".cfg")
    else:
        cfg_path = HERE / "cfg" / "cpd_ti.cfg"
    cfg_text = cfg_path.read_text(encoding="utf-8", errors="replace").replace("\r", "")
    cfg = parse_cfg(cfg_path)
    print(f"Config {cfg_path.name} : {cfg.summary()}")

    # ---- source des trames
    cli = None
    sortie = None
    if args.fichier:
        lecteur: Lecteur = LecteurFichier(Path(args.fichier), cfg.frame_rate_hz, args.vitesse, cfg.num_virt_ant)
        source = f"Fichier {Path(args.fichier).name}"
    else:
        try:
            import serial  # noqa: F401
        except ImportError:
            sys.exit("pyserial manquant : python -m pip install pyserial")
        cli, data = args.port_cli, args.port_data
        if not cli or not data:
            g_cli, g_data = guess_ports()
            cli, data = cli or g_cli, data or g_data
        if not data or (not cli and not args.sans_cfg):
            list_ports()
            sys.exit("Ports XDS110 non trouves : carte branchee sur l'USB XDS110 ? sinon precise --port-cli et --port-data")
        if not args.pas_d_enregistrement:
            sortie = Path(args.sortie) if args.sortie else \
                HERE / "enregistrements" / f"nuage_{datetime.now():%Y%m%d_%H%M%S}.bin"
            sortie.parent.mkdir(parents=True, exist_ok=True)
            sortie.with_suffix(".cfg").write_text(cfg_text)   # le cfg envoye, garde avec l'enregistrement
        lecteur = LecteurSerie(data, args.baud_data, sortie, cfg.num_virt_ant)
        source = f"Carte : commandes {cli or '-'}, donnees {data}"
    print(source + (f"  -> enregistrement {sortie}" if sortie else ""))

    lecteur.start()
    lecteur.pret.wait(5)
    if lecteur.erreur:
        sys.exit(f"Impossible d'ouvrir la source : {lecteur.erreur}")
    cfg_envoye = False
    if not args.fichier and not args.sans_cfg:
        print(f"Envoi du cfg sur {cli} :")
        if not send_cfg(cli, cfg_lines(cfg_text)):
            print("!! Des commandes ont ete refusees ou sont restees sans reponse (voir ci-dessus).")
        cfg_envoye = True

    etat = Etat(cfg, args.persistance, args.inverser_azimut, args.inverser_elevation)
    t_start = time.time()
    t_print = [t_start]
    conseil = [""]
    conseil_affiche = [""]

    def pomper() -> bool:
        """Vide la file des trames. Renvoie False quand il faut s'arreter."""
        while True:
            try:
                etat.ajouter(lecteur.q.get_nowait())
            except queue.Empty:
                break
        now = time.time()
        if now - t_print[0] >= 1.0 and etat.n_frames:
            print("  " + etat.ligne_console())
            t_print[0] = now
        el = now - t_start
        if not args.fichier:
            if lecteur.n_bytes == 0 and el > 4:
                conseil[0] = ("Aucun octet recu sur le port de donnees :\nverifier S1.4 = ON et S1.5 = ON\n"
                              "(UART-A vers l'USB XDS110), puis RESET.")
            elif etat.n_frames >= 15 and etat.total_points == 0:
                conseil[0] = ("Trames recues mais aucun point :\nplace-toi a 0,5-1,5 m devant les\n"
                              "antennes et bouge un bras.")
            else:
                conseil[0] = ""
        if conseil[0] and conseil[0] != conseil_affiche[0]:
            print("  !! " + conseil[0].replace("\n", " "))
        conseil_affiche[0] = conseil[0]
        if lecteur.erreur:
            print(f"Erreur de lecture : {lecteur.erreur}")
            return False
        if args.duree and el >= args.duree:
            return False
        return not (lecteur.fini.is_set() and lecteur.q.empty() and args.fichier is None)

    try:
        if args.sans_affichage:
            while pomper() and not (args.fichier and lecteur.fini.is_set() and lecteur.q.empty()):
                time.sleep(0.05)
        else:
            import matplotlib
            if matplotlib.get_backend().lower() == "agg":
                print("(pas d'ecran disponible : utilise --sans-affichage)")
            from matplotlib.animation import FuncAnimation
            titre = f"AWRL6844 - nuage de points ({'repere capteur' if args.repere == 'capteur' else 'repere voiture'})"
            aff = Affichage(cfg, args.repere, args.couleur, not args.sans_3d, titre)

            def update(_):
                if not pomper():
                    aff.plt.close(aff.fig)
                    return []
                info = texte_infos(etat, cfg, args.repere, source, args.persistance, conseil[0])
                if args.fichier and lecteur.fini.is_set() and lecteur.q.empty():
                    info += "\n\nFin du fichier."
                aff.maj(etat, info)
                return []

            aff.fig.canvas.mpl_connect("close_event", lambda _e: lecteur.stop_evt.set())
            anim = FuncAnimation(aff.fig, update, interval=100, cache_frame_data=False)  # noqa: F841
            aff.plt.show()
    except KeyboardInterrupt:
        print("\nArret demande.")
    finally:
        lecteur.stop_evt.set()
        lecteur.join(timeout=2)
        if cfg_envoye and cli:
            try:
                send_cfg(cli, ["sensorStop 0"], verbose=False)
            except Exception as exc:
                print(f"(sensorStop non envoye : {exc})")

    el = max(time.time() - t_start, 1e-3)
    print(f"\nTermine : {etat.n_frames} trames en {el:.0f} s, {etat.lost} perdues, {lecteur.n_bad} illisibles, "
          f"{lecteur.n_bytes} octets recus.")
    if etat.n_frames:
        print(f"Points : {etat.total_points / etat.n_frames:.1f} par trame en moyenne, "
              f"{100 * etat.n_with_points / etat.n_frames:.0f} % des trames avec au moins un point.")
        if cfg.num_zones:
            print("Points par siege (repere voiture) : " +
                  ", ".join(f"{SEATS.get(z, z)} {n}" for z, n in enumerate(etat.zone_total)))
    elif not args.fichier and lecteur.n_bytes == 0:
        print("Rien sur le port de donnees. Le firmware envoie les trames sur l'UART-A (1,25 Mbaud) :\n"
              "  switches S1.4 = ON et S1.5 = ON pour la router vers l'USB XDS110, puis RESET et relancer.")
    elif not args.fichier:
        print("Des octets arrivent mais aucune trame valide : mauvais debit (--baud-data) ou mauvais port.")
    if sortie:
        print(f"Enregistrement : {sortie} (+ {sortie.with_suffix('.cfg').name})")


if __name__ == "__main__":
    main()
