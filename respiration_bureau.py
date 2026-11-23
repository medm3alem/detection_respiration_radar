"""Essai sur table : carte de l'environnement + rythme respiratoire des personnes immobiles devant le radar.

Plusieurs personnes (jusqu'a 4, a plus de 50 cm l'une de l'autre) sont suivies separement : P1, P2...

    python respiration_bureau.py                     # direct : envoie cfg/bureau_vitals.cfg, affiche, enregistre
    python respiration_bureau.py --fichier enregistrements/bureau_xxx.bin      # relecture
    python respiration_bureau.py --fichier resultats_sim/scenario.bin --cfg cfg/cpd_vitals.cfg   # simulation

Installation : carte debout sur la table, antennes vers la personne, assise a 0,5-1,5 m,
immobile, qui respire normalement. Il faut environ 10 s avant le premier rythme (fenetre de 20 s).

Fenetre :
    carte statique        ce qui renvoie l'onde (murs, meubles, personne), vue de dessus, repere capteur
    micro-mouvements      ce qui bouge dans la bande respiratoire ; le cercle = la cible suivie
    deplacement           mouvement du thorax de la cible (mm), filtre 6-45 /min
    spectre               le pic donne le rythme respiratoire
"""
from __future__ import annotations

import argparse
import queue
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from radar_vitals.bureau import CarteRespiration, ParamsBureau
from radar_vitals.cfg_parser import parse_cfg
from radar_vitals.lecture import Lecteur, LecteurFichier, LecteurSerie
from radar_vitals.nuage import spherical_to_sensor
from radar_vitals.serie import BAUD_DATA, cfg_lines, guess_ports, list_ports, send_cfg

HERE = Path(__file__).resolve().parent


COULEURS = ["#00c853", "#00b0ff", "#ff9100", "#d500f9"]           # P1, P2, P3, P4


def couleur(c: dict) -> str:
    return COULEURS[(c["num"] - 1) % len(COULEURS)] if c.get("num") else "#9e9e9e"


def etat_txt(c: dict) -> str:
    etat = c.get("etat")
    if etat == "apnee":
        return f"APNEE {c['pause_s']:.0f} s" + (" (mvt)" if c.get("mouvement") else "")
    if etat == "pause":
        return f"pause {c['pause_s']:.0f} s"
    if etat == "reprise":
        return "reprise"
    if etat == "faible":
        return f"{c['bpm']:.1f} /min (faible)"
    return f"{c['bpm']:.1f} /min" if c.get("respire") else "--"


def ligne(res: dict | None) -> str:
    if res is None:
        return "accumulation..."
    c = res.get("cible")
    if c is None:
        return f"fenetre {res['duree_s']:.0f}/{res['fenetre_s']:.0f} s | aucune cible"
    tete = f"fenetre {res['duree_s']:4.0f}/{res['fenetre_s']:.0f} s"
    cibles = res.get("cibles") or []
    if cibles:                                   # une ou plusieurs personnes confirmees
        morceaux = [f"{k['nom']} {k['cible']['distance_m']:.2f} m az {k['cible']['azimut_deg']:+3.0f} "
                    f"el {k['cible']['elevation_deg']:+3.0f} : {etat_txt(k):>14s} ({k['amplitude_mm']:.1f} mm)"
                    for k in cibles]
        return tete + " | " + " | ".join(morceaux)
    return (f"{tete} | recherche : cible {c['distance_m']:.2f} m "
            f"az {c['azimut_deg']:+3.0f} el {c['elevation_deg']:+3.0f} | respiration {etat_txt(res):>10s} "
            f"(pic {res['bpm_brut'] or 0:.1f}, q {res['q']:.0f}, bande {res['frac']:.2f}, "
            f"prom {res['prom']:.1f}) | amplitude {res['amplitude_mm']:.1f} mm")


class Affichage:
    def __init__(self, carte: CarteRespiration, r_aff: float, titre: str):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.carte = carte
        self.fig = plt.figure(figsize=(15, 9))
        if self.fig.canvas.manager:
            self.fig.canvas.manager.set_window_title(titre)
        self.fig.suptitle(titre, fontsize=11)
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1.5, 1], width_ratios=[1, 1, 0.75], hspace=0.28, wspace=0.22,
                                   left=0.05, right=0.98, top=0.93, bottom=0.07)
        self.ax_s, self.ax_d, self.ax_i = (self.fig.add_subplot(gs[0, i]) for i in range(3))
        self.ax_w = self.fig.add_subplot(gs[1, :2])
        self.ax_f = self.fig.add_subplot(gs[1, 2])
        self.ax_i.axis("off")

        keep = carte.ranges <= r_aff
        self.keep = keep
        rr, uu = np.meshgrid(carte.ranges[keep], carte.u, indexing="ij")
        X, Y = rr * uu, rr * np.sqrt(1 - uu ** 2)
        z = np.full(X.shape, -40.0)
        self.mesh = {}
        for ax, key, ttl in ((self.ax_s, "carte_stat", "Carte statique + nuage de points TI (bleu)"),
                             (self.ax_d, "carte_dyn", "Micro-mouvements (bande respiratoire)")):
            self.mesh[key] = ax.pcolormesh(X, Y, z, shading="gouraud", cmap="magma", vmin=-40, vmax=0)
            ax.set(xlim=(-0.9 * r_aff, 0.9 * r_aff), ylim=(0, r_aff), xlabel="X lateral (m)", ylabel="Y devant le radar (m)",
                   title=ttl)
            ax.set_aspect("equal")
            ax.plot(0, 0, "w^", ms=9, mec="k")
            for r in np.arange(0.5, r_aff + 1e-6, 0.5):
                a = np.linspace(-np.pi / 2, np.pi / 2, 90)
                ax.plot(r * np.sin(a), r * np.cos(a), color="w", lw=0.4, alpha=0.35)
        self.fig.colorbar(self.mesh["carte_dyn"], ax=[self.ax_s, self.ax_d], shrink=0.8, pad=0.015, label="dB (relatif)")
        self.pts = self.ax_s.scatter([], [], s=4, c="#4fc3f7", alpha=0.6)
        n_max = carte.p.max_cibles
        # un cercle + une etiquette par personne, sur les deux cartes
        self.marks = [[ax.plot([], [], "o", mfc="none", mew=2, ms=16)[0] for _ in range(n_max + 1)]
                      for ax in (self.ax_s, self.ax_d)]
        self.labels = [[ax.text(0, 0, "", fontsize=9, fontweight="bold", color="w") for _ in range(n_max + 1)]
                       for ax in (self.ax_s, self.ax_d)]

        self.l_raw, = self.ax_w.plot([], [], color="0.75", lw=1, label="brut, cible principale")
        self.l_fs = [self.ax_w.plot([], [], lw=1.8)[0] for _ in range(n_max + 1)]
        self.ax_w.set(xlabel="temps (s)", ylabel="deplacement du thorax (mm)",
                      title="Mouvement de chaque personne (filtre 6-45 /min)")
        self.ax_w.grid(alpha=0.3)
        self.ax_w.legend(loc="upper left", fontsize=8)
        self.l_sp, = self.ax_f.plot([], [], color="#1565c0")
        self.v_pk = self.ax_f.axvline(np.nan, color="#d84315", ls="--")
        self.ax_f.axvspan(*carte.p.bande_bpm, color="#c8e6c9", alpha=0.35, lw=0)
        self.ax_f.set(xlim=(0, 60), ylim=(0, 1.05), xlabel="rythme (/min)", ylabel="spectre (norm.)", title="Spectre")
        self.ax_f.grid(alpha=0.3)

        # une ligne d'etat par personne (P1 15.2 /min, P2 APNEE 12 s...)
        self.bigs = [self.ax_i.text(0.0, 1.02 - 0.085 * k, "", fontsize=19, fontweight="bold", va="top",
                                    color="0.5", transform=self.ax_i.transAxes) for k in range(n_max)]
        self.txt = self.ax_i.text(0.0, 1.0 - 0.085 * n_max - 0.02, "", fontsize=8, family="monospace", va="top",
                                  transform=self.ax_i.transAxes)

    COULEUR_ETAT = {"apnee": "#c62828", "pause": "#ef6c00", "faible": "#9e9d24", "reprise": "#2e7d32",
                    "respire": "#2e7d32"}

    def maj(self, res: dict | None, pts_xy: np.ndarray | None, infos: str):
        if res is not None:
            for key in ("carte_stat", "carte_dyn"):
                m = res[key][self.keep]
                db = 10 * np.log10(m / (m.max() + 1e-30) + 1e-12)
                self.mesh[key].set_array(np.clip(db, -40, 0).ravel())
            cibles = list(res.get("cibles") or [])
            if not cibles and res.get("cible"):
                cibles = [dict(res, num=None, nom="?")]            # candidat pas encore confirme
            for ax_k in range(2):
                for k, (mk, lb) in enumerate(zip(self.marks[ax_k], self.labels[ax_k])):
                    if k < len(cibles):
                        c = cibles[k]
                        x, y = c["cible"]["xyz"][0], c["cible"]["xyz"][1]
                        mk.set_data([x], [y])
                        mk.set_markeredgecolor(couleur(c))
                        lb.set_position((x + 0.08, y + 0.08))
                        lb.set_text(c["nom"])
                        lb.set_color(couleur(c))
                    else:
                        mk.set_data([], [])
                        lb.set_text("")
            if res.get("cible") and "deplacement_mm" in res:
                t = res["t"]
                self.l_raw.set_data(t, res["deplacement_brut_mm"])
                lim = 1.0
                for k, lf in enumerate(self.l_fs):
                    if k < len(cibles):
                        lf.set_data(t, cibles[k]["deplacement_mm"])
                        lf.set_color(couleur(cibles[k]))
                        lim = max(lim, np.abs(cibles[k]["deplacement_mm"]).max() * 1.1)
                    else:
                        lf.set_data([], [])
                lim = max(lim, np.abs(res["deplacement_brut_mm"]).max() * 1.1)
                self.ax_w.set(xlim=(min(t[0], -5), 0), ylim=(-lim, lim))
                self.l_sp.set_data(res["spectre_bpm"], res["spectre"])
                self.v_pk.set_xdata([res["bpm_brut"] or np.nan] * 2)
            for k, big in enumerate(self.bigs):
                if k < len(cibles) and cibles[k].get("num"):
                    c = cibles[k]
                    big.set_text(f"{c['nom']}  {etat_txt(c)}")
                    big.set_color(self.COULEUR_ETAT.get(c["etat"], "0.5"))
                elif k == 0:
                    big.set_text("recherche..." if res.get("cible") else "--")
                    big.set_color("0.5")
                else:
                    big.set_text("")
        if pts_xy is not None:
            self.pts.set_offsets(pts_xy if len(pts_xy) else np.zeros((0, 2)))
        self.txt.set_text(infos)


def texte_infos(res: dict | None, carte: CarteRespiration, source: str, n_trames: int, n_cube: int) -> str:
    out = [source, ""]
    out.append(f"Trames recues   {n_trames} ({n_cube} avec cube)")
    out.append(f"Trames comblees {carte.perdues}")
    if res is None:
        out.append("Accumulation des donnees...")
        if n_trames > 10 and n_cube == 0:
            out += ["", "Pas de radar cube dans les trames :", "le cfg doit contenir dbgGuiMonitor",
                    "... 1 et exportRadCubeChunk", "(cfg/bureau_vitals.cfg)."]
        return "\n".join(out)
    out.append(f"Fenetre         {res['duree_s']:.0f} / {res['fenetre_s']:.0f} s")
    c = res.get("cible")
    if c is None:
        out.append("Aucune cible (rien ne bouge ?)")
        return "\n".join(out)
    cibles = res.get("cibles") or []
    if cibles:
        out.append(f"Personnes       {len(cibles)}")
        for k in cibles:
            cc = k["cible"]
            out += ["", f"{k['nom']}  {cc['distance_m']:.2f} m  az {cc['azimut_deg']:+.0f}  el {cc['elevation_deg']:+.0f}"
                        f"  ({k['contraste_db']:.0f} dB)",
                    f"   pic {k['bpm_brut'] or 0:.1f}/min q {k['q']:.0f} prom {k['prom']:.1f}"
                    f"  ampl. {k['amplitude_mm']:.1f} mm"]
            if k["etat"] == "apnee":
                out.append(f"   ALERTE : pas de respiration depuis {k['pause_s']:.0f} s"
                           + ("  (mouvement : reprise ?)" if k.get("mouvement") else ""))
            elif k["etat"] == "pause":
                out.append(f"   pause depuis {k['pause_s']:.0f} s (alerte a {carte.p.apnee_s:.0f} s)")
            elif k["etat"] == "reprise":
                out.append("   respiration reprise, rythme en calcul")
            elif k["etat"] == "faible":
                out.append("   respiration plus faible, meme rythme")
        return "\n".join(out)
    # rien de confirme : detail du meilleur candidat
    out += ["", "Recherche d'une respiration",
            f"  distance      {c['distance_m']:.2f} m",
            f"  azimut        {c['azimut_deg']:+.0f} deg",
            f"  elevation     {c['elevation_deg']:+.0f} deg",
            f"  contraste     {res.get('contraste_db', 0):.0f} dB",
            f"  pic           {res['bpm_brut'] or 0:.1f} /min",
            f"  qualite q     {res['q']:.0f}  (>= {carte.p.q_min:.0f})",
            f"  part bande    {res['frac']:.2f}  (>= {carte.p.frac_min:.2f})",
            f"  proeminence   {res['prom']:.1f}  (>= {carte.p.prom_min:.1f})",
            f"  amplitude     {res['amplitude_mm']:.1f} mm (>= {carte.p.amplitude_min_mm:.1f})"]
    if res["duree_s"] < carte.p.min_s:
        out += ["", "Encore quelques secondes..."]
    else:
        out += ["", "Pas de rythme net : rester immobile,", "respirer normalement, face au radar."]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", help="defaut : cfg/bureau_vitals.cfg (ou le .cfg du fichier relu)")
    ap.add_argument("--fichier", help="relire un enregistrement .bin")
    ap.add_argument("--vitesse", type=float, default=1.0, help="relecture : 1 = temps reel, 0 = au plus vite")
    ap.add_argument("--port-cli")
    ap.add_argument("--port-data")
    ap.add_argument("--baud-data", type=int, default=BAUD_DATA)
    ap.add_argument("--sans-cfg", action="store_true")
    ap.add_argument("--sortie", help="defaut : enregistrements/bureau_<date>.bin")
    ap.add_argument("--pas-d-enregistrement", action="store_true")
    ap.add_argument("--duree", type=float, default=0.0, help="arret automatique apres N secondes")
    ap.add_argument("--fenetre", type=float, default=20.0, help="fenetre d'analyse (s)")
    ap.add_argument("--distance-min", type=float, default=0.3, help="zone de recherche de la personne (m)")
    ap.add_argument("--distance-max", type=float, default=2.5)
    ap.add_argument("--amplitude-min", type=float, default=0.3,
                    help="mouvement crete a crete minimal pour annoncer une respiration (mm)")
    ap.add_argument("--cibles", type=int, default=4, help="nombre maximal de personnes suivies (1 = une seule)")
    ap.add_argument("--separation", type=float, default=0.5, help="distance minimale entre deux personnes (m)")
    ap.add_argument("--apnee", type=float, default=10.0, help="duree de pause declenchant l'alerte (s)")
    ap.add_argument("--seuil-pause", type=float, default=0.3,
                    help="pause si l'amplitude sur 6 s tombe sous cette fraction de l'amplitude habituelle")
    ap.add_argument("--sans-points", action="store_true", help="ne pas superposer le nuage de points TI")
    ap.add_argument("--sans-affichage", action="store_true")
    ap.add_argument("--garder-fenetre", action="store_true",
                    help="relecture : laisser la fenetre ouverte a la fin (sinon fermee apres 3 s)")
    ap.add_argument("--inverser-azimut", action="store_true")
    ap.add_argument("--inverser-elevation", action="store_true")
    ap.add_argument("--lister-ports", action="store_true")
    args = ap.parse_args()

    if args.lister_ports:
        list_ports()
        return

    if args.cfg:
        cfg_path = Path(args.cfg)
    elif args.fichier and Path(args.fichier).with_suffix(".cfg").exists():
        cfg_path = Path(args.fichier).with_suffix(".cfg")
    else:
        cfg_path = HERE / "cfg" / "bureau_vitals.cfg"
    cfg_text = cfg_path.read_text(encoding="utf-8", errors="replace").replace("\r", "")
    cfg = parse_cfg(cfg_path)
    print(f"Config {cfg_path.name} : {cfg.summary()}")
    params = ParamsBureau(fenetre_s=args.fenetre, amplitude_min_mm=args.amplitude_min, apnee_s=args.apnee,
                          pause_ratio=args.seuil_pause, max_cibles=max(1, args.cibles),
                          separation_m=args.separation, r_min=args.distance_min, r_max=args.distance_max,
                          flip_azimuth=args.inverser_azimut, flip_elevation=args.inverser_elevation)
    try:
        carte = CarteRespiration(cfg, params)
    except ValueError as exc:
        sys.exit(str(exc))

    cli = sortie = None
    if args.fichier:
        lecteur: Lecteur = LecteurFichier(Path(args.fichier), cfg.frame_rate_hz, args.vitesse, cfg.num_virt_ant)
        source = f"Fichier {Path(args.fichier).name}"
    else:
        cli, data = args.port_cli, args.port_data
        if not cli or not data:
            g_cli, g_data = guess_ports()
            cli, data = cli or g_cli, data or g_data
        if not data or (not cli and not args.sans_cfg):
            list_ports()
            sys.exit("Ports XDS110 non trouves : preciser --port-cli et --port-data")
        if not args.pas_d_enregistrement:
            sortie = Path(args.sortie) if args.sortie else \
                HERE / "enregistrements" / f"bureau_{datetime.now():%Y%m%d_%H%M%S}.bin"
            sortie.parent.mkdir(parents=True, exist_ok=True)
            sortie.with_suffix(".cfg").write_text(cfg_text)
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

    etat = {"res": None, "n": 0, "n_cube": 0, "pts": None, "t_print": time.time(), "t_an": 0.0, "new": 0,
            "historique": {}, "pauses": {}, "vus": set()}
    t_start = time.time()

    def analyser():
        etat["res"] = carte.analyser()
        etat["new"] = 0
        r = etat["res"]
        if r is None:
            return
        for c in r.get("cibles") or []:
            nom = c["nom"]
            if c.get("respire"):
                etat["historique"].setdefault(nom, []).append(c["bpm"])
            if c.get("etat") in ("pause", "apnee"):
                debut = c["t_flux_s"] - c["pause_s"]
                j = etat["pauses"].setdefault(nom, [])
                if j and c["t_flux_s"] - j[-1]["vu"] < 3.0:       # meme pause (vue sans interruption)
                    j[-1]["debut"] = min(j[-1]["debut"], debut)
                else:
                    j.append({"debut": debut, "alerte": None})
                j[-1]["vu"] = c["t_flux_s"]
                j[-1]["fin"] = c["fin_pause_estimee_s"]
                if c["etat"] == "apnee" and j[-1]["alerte"] is None:
                    j[-1]["alerte"] = c["t_flux_s"]
                    print(f"  !! ALERTE {nom} ({c['cible']['distance_m']:.2f} m) : pas de respiration depuis "
                          f"{c['pause_s']:.0f} s (instant {c['t_flux_s']:.0f} s du flux)")
            if nom not in etat["vus"]:
                etat["vus"].add(nom)
                print(f"  ++ {nom} : respiration trouvee a {c['cible']['distance_m']:.2f} m, "
                      f"az {c['cible']['azimut_deg']:+.0f}, el {c['cible']['elevation_deg']:+.0f}")

    def pomper() -> bool:
        while True:
            try:
                fr = lecteur.q.get_nowait()
            except queue.Empty:
                break
            etat["n"] += 1
            if carte.push(fr):
                etat["n_cube"] += 1
                etat["new"] += 1
            elif fr.radar_cube is not None:
                etat["forme"] = fr.radar_cube.shape
            if fr.points is not None and not args.sans_points:
                xyz = spherical_to_sensor(fr.points, args.inverser_azimut, args.inverser_elevation)
                etat["pts"] = xyz[:, :2]
            if etat["new"] >= 2:                       # une analyse toutes les 2 images (0,4 s)
                analyser()
        if etat["new"] and args.fichier and lecteur.fini.is_set():
            analyser()
        now = time.time()
        fin_fichier = bool(args.fichier) and lecteur.fini.is_set() and lecteur.q.empty()
        if fin_fichier and not etat.get("fin_annoncee"):
            # relecture terminee : on donne la derniere ligne une fois, la fenetre reste ouverte pour la figure
            print("  " + ligne(etat["res"]))
            if not args.sans_affichage:
                print("  -- Fin du fichier." + ("  Fermez la fenetre pour terminer." if args.garder_fenetre
                                                 else "  Fenetre fermee dans 3 s (figure enregistree en .png)."))
            etat["fin_annoncee"] = True
        elif not fin_fichier and now - etat["t_print"] >= 2.0 and etat["n"]:
            print("  " + ligne(etat["res"]))
            etat["t_print"] = now
            if etat["n"] > 10 and etat["n_cube"] == 0:
                if etat.get("forme"):
                    print(f"  !! Radar cube recu mais de forme {etat['forme']} au lieu de "
                          f"({carte.nc}, {carte.R}, {carte.A}) [bursts, cases distance, antennes]")
                else:
                    print("  !! Les trames n'ont pas de radar cube : utiliser cfg/bureau_vitals.cfg")
            if lecteur.n_bytes == 0 and now - t_start > 4 and not args.fichier:
                print("  !! Rien sur le port de donnees (S1.4 = ON, S1.5 = ON, puis RESET)")
        if lecteur.erreur:
            print(f"Erreur de lecture : {lecteur.erreur}")
            return False
        if args.duree and now - t_start >= args.duree:
            return False
        return not (lecteur.fini.is_set() and lecteur.q.empty() and args.fichier is None)

    aff = None
    try:
        if args.sans_affichage:
            while pomper() and not (args.fichier and lecteur.fini.is_set() and lecteur.q.empty()):
                time.sleep(0.05)
            pomper()
        else:
            from matplotlib.animation import FuncAnimation
            aff = Affichage(carte, min(3.0, carte.ranges[-1]), "AWRL6844 - carte et respiration (essai sur table)")

            def update(_):
                if not pomper():
                    aff.plt.close(aff.fig)
                    return []
                if etat.get("fin_annoncee") and not args.garder_fenetre:
                    etat.setdefault("t_fin", time.time())
                    if time.time() - etat["t_fin"] > 3.0:       # la figure est enregistree en .png
                        aff.plt.close(aff.fig)
                        return []
                info = texte_infos(etat["res"], carte, source, etat["n"], etat["n_cube"])
                if args.fichier and lecteur.fini.is_set() and lecteur.q.empty():
                    info += "\n\nFin du fichier."
                aff.maj(etat["res"], etat["pts"], info)
                return []

            aff.fig.canvas.mpl_connect("close_event", lambda _e: lecteur.stop_evt.set())
            anim = FuncAnimation(aff.fig, update, interval=200, cache_frame_data=False)  # noqa: F841
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

    print(f"\nTermine : {etat['n']} trames ({etat['n_cube']} avec radar cube), {carte.perdues} comblees, "
          f"{carte.redemarrages} redemarrage(s) de la fenetre, "
          f"{carte.gigue} numero(s) de trame decale(s), {lecteur.n_bytes} octets.")
    noms = sorted(set(etat["historique"]) | set(etat["pauses"]), key=lambda n: int(n[1:]))
    if not noms:
        print("Aucun rythme net. Personne immobile a 0,5-1,5 m, face au radar, pendant au moins 20 s ?")
    for nom in noms:
        h = etat["historique"].get(nom, [])
        if h:
            bpm = np.array(h)
            print(f"{nom} : rythme median {np.median(bpm):.1f} /min (min {bpm.min():.1f}, max {bpm.max():.1f}) "
                  f"sur {len(h)} analyses")
        else:
            print(f"{nom} : aucun rythme net")
        for k, ev in enumerate(etat["pauses"].get(nom, []), 1):
            al = f", alerte a {ev['alerte']:.0f} s" if ev["alerte"] is not None else ""
            print(f"   pause {k} : debut estime a {ev['debut']:.0f} s, fin a {ev['fin']:.0f} s "
                  f"(duree environ {ev['fin'] - ev['debut']:.0f} s{al}) [temps depuis la 1re trame]")
    if etat["res"] is not None and (sortie or args.fichier):
        import matplotlib
        if aff is None:
            matplotlib.use("Agg")
            aff = Affichage(carte, min(3.0, carte.ranges[-1]), "AWRL6844 - carte et respiration (essai sur table)")
        aff.maj(etat["res"], etat["pts"], texte_infos(etat["res"], carte, source, etat["n"], etat["n_cube"]))
        png = (sortie or Path(args.fichier)).with_suffix(".png")
        try:
            aff.fig.savefig(png, dpi=110)
            print(f"Figure : {png}")
        except OSError as exc:
            print(f"(figure non enregistree : {exc})")
    if sortie:
        print(f"Enregistrement : {sortie} (+ {sortie.with_suffix('.cfg').name})")


if __name__ == "__main__":
    main()