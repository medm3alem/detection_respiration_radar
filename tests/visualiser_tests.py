"""Une figure par test automatique, pour voir de quoi il s'agit.

    python tests/visualiser_tests.py            -> tests/figures/01_cfg.png ... 11_deux_personnes.png

Chaque figure rejoue la situation fabriquee par le test (memes donnees synthetiques) et montre
ce que le programme en tire, avec en titre ce que le test exige.
"""
import struct
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent))
sys.path.insert(0, str(ICI))
from radar_vitals import tlv  # noqa: E402
from radar_vitals.bureau import CarteRespiration  # noqa: E402
from radar_vitals.cfg_parser import parse_cfg  # noqa: E402
import test_base as tb  # noqa: E402

SORTIE = ICI / "figures"
CFG_CPD = ICI.parent / "cfg" / "cpd_vitals.cfg"
CFG_BUREAU = ICI.parent / "cfg" / "bureau_vitals.cfg"

COUL_ETAT = {"respire": "#2e7d32", "faible": "#9e9d24", "reprise": "#81c784", "incertain": "#bdbdbd",
             "pause": "#ef6c00", "apnee": "#c62828"}
NOM_ETAT = {"respire": "respire", "faible": "respiration faible", "reprise": "reprise",
            "incertain": "pas encore de rythme net", "pause": "pause (< 10 s)", "apnee": "APNEE (alerte)"}


def enregistrer(fig, nom):
    SORTIE.mkdir(exist_ok=True)
    fig.savefig(SORTIE / nom, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("  ", SORTIE / nom)


def bande_etats(ax, journal, y0=0.0, h=1.0):
    """Bande coloree : l'etat annonce par le programme a chaque instant."""
    for (t, e), (t2, _) in zip(journal, journal[1:] + [(journal[-1][0] + 0.4, None)]):
        ax.add_patch(Rectangle((t, y0), t2 - t, h, color=COUL_ETAT.get(e, "#bdbdbd"), lw=0))
    ax.set_ylim(y0, y0 + h)
    ax.set_yticks([])


def dessiner_carte(ax, carte, dyn, r_max=1.8):
    """Carte des micro-mouvements vue de dessus (max sur l'elevation), comme dans respiration_bureau.py."""
    keep = carte.ranges <= r_max
    rr, uu = np.meshgrid(carte.ranges[keep], carte.u, indexing="ij")
    db = 10 * np.log10(dyn[keep] / dyn.max() + 1e-12)
    ax.pcolormesh(rr * uu, rr * np.sqrt(1 - uu ** 2), np.clip(db, -40, 0), cmap="magma", shading="auto")
    ax.set_facecolor("k")


def legende_etats(ax, etats):
    ax.legend(handles=[Patch(color=COUL_ETAT[e], label=NOM_ETAT[e]) for e in etats if e in COUL_ETAT],
              loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=6, fontsize=8, frameon=False)


# ---------------------------------------------------------------------------------------------- 1
def fig_cfg():
    cfg = parse_cfg(CFG_CPD)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
    a1.scatter(cfg.ant_geom0, cfg.ant_geom1, s=160, c="#1565c0")
    for k, (x, y) in enumerate(zip(cfg.ant_geom0, cfg.ant_geom1)):
        a1.text(x, y, str(k), color="w", ha="center", va="center", fontsize=8, fontweight="bold")
    a1.set(title="Les 16 antennes virtuelles (antGeometry0/1)", xlabel="position horizontale (λ/2)",
           ylabel="position verticale (λ/2)")
    a1.set_aspect("equal")
    a1.margins(0.25)
    a1.grid(alpha=0.3)
    texte = (f"Ce que le programme lit dans cpd_vitals.cfg :\n\n"
             f"  antennes virtuelles  {cfg.num_virt_ant}  ({cfg.num_tx} TX x {cfg.num_rx} RX)    test : 16\n"
             f"  cases de distance    {cfg.num_range_bins}                     test : 64\n"
             f"  taille d'une case    {cfg.range_step_m * 100:.2f} cm                test : 4,59 cm\n"
             f"  portee               {cfg.num_range_bins * cfg.range_step_m:.2f} m\n"
             f"  longueur d'onde      {cfg.wavelength_m * 1e3:.2f} mm\n"
             f"  images par seconde   {cfg.frame_rate_hz:.1f}\n"
             f"  export radar cube    {'oui' if cfg.radcube_export_enabled else 'non'}, rafales "
             f"{cfg.exported_chirp_indices()}   test : 3 rafales\n"
             f"  zones de sieges      {cfg.num_zones}                      test : 5")
    a2.axis("off")
    a2.text(0, 0.95, texte, family="monospace", fontsize=10, va="top")
    fig.suptitle("Test 1 - test_cfg : la configuration de la carte est bien comprise", fontweight="bold")
    enregistrer(fig, "01_cfg.png")


# ---------------------------------------------------------------------------------------------- 2
def fig_tlv():
    rng = np.random.default_rng(0)
    cube = (rng.normal(0, 500, (3, 64, 16)) + 1j * rng.normal(0, 500, (3, 64, 16))).round()
    probs = np.zeros((3, 5))
    probs[0] = 1.0
    probs[:, 2] = [0.1, 0.8, 0.1]
    pkt = tlv.encode_packet(42, [(tlv.TLV_RADAR_CUBE, tlv.encode_radar_cube(cube)),
                                 (tlv.TLV_OCC_CLASS, tlv.encode_occupancy(probs))])
    l_cube = 8 + 12 + 3 * 64 * 16 * 4
    l_occ = 8 + 15
    l_pad = len(pkt) - 40 - l_cube - l_occ
    # schema (largeurs non proportionnelles : le radar cube ferait sinon 99 % de la figure)
    segs = [("octets\nparasites", "8 o", 1.0, "#9e9e9e")]
    for k in range(2):
        segs += [(f"en-tete\ntrame {k + 1}", "40 o", 1.3, "#1565c0"),
                 ("TLV 2007\nradar cube", f"{l_cube} o", 4.0, "#90caf9"),
                 ("TLV 1041\nsieges", f"{l_occ} o", 1.3, "#ffb74d"),
                 ("bourrage", f"{l_pad} o", 1.0, "#e0e0e0")]
        if k == 0:
            segs.append(("octets\nparasites", "2 o", 1.0, "#9e9e9e"))
    fig, ax = plt.subplots(figsize=(13, 3.4))
    x = 0.0
    for nom, taille, w, c in segs:
        ax.add_patch(Rectangle((x, 0), w, 1, color=c, ec="k", lw=0.8))
        ax.text(x + w / 2, 0.55, nom, ha="center", va="center", fontsize=8,
                color="w" if c == "#1565c0" else "k")
        ax.text(x + w / 2, -0.15, taille, ha="center", va="top", fontsize=7, color="0.3")
        x += w
    for k in np.arange(0.7, x, 0.9):
        ax.plot([k, k], [1.05, 1.25], color="#c62828", lw=1)
    ax.text(x / 2, 1.35, "le flux arrive par morceaux de 37 octets (traits rouges), coupes n'importe ou",
            ha="center", fontsize=8, color="#c62828")
    ax.set(xlim=(-0.2, x + 0.2), ylim=(-1.2, 1.6))
    ax.axis("off")
    ax.text(0, -0.55, f"Chaque trame fait {len(pkt)} octets (multiple de 32, comme sur la carte). "
                      "Le decodeur cherche le mot magique de debut de trame, lit la longueur annoncee et "
                      "reassemble les morceaux.\nResultat exige : 2 trames retrouvees, numero 42, TLV [2007, 1041], "
                      "radar cube et probabilites des sieges identiques a l'original.", fontsize=9, va="top")
    fig.suptitle("Test 2 - test_tlv_roundtrip : retrouver les trames dans un flux serie decoupe et bruite "
                 "(schema, largeurs non proportionnelles)", fontweight="bold")
    enregistrer(fig, "02_tlv_roundtrip.png")


# ---------------------------------------------------------------------------------------------- 3 et 4
def fig_zones_et_geometrie():
    from radar_vitals.nuage import sensor_to_spherical, spherical_to_sensor, zone_of_points
    cfg = parse_cfg(CFG_CPD)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    noms = ["conducteur", "passager avant", "arriere gauche", "arriere milieu", "arriere droit"]
    coul = plt.cm.tab10(np.arange(5))
    for zone, boxes in cfg.cuboids.items():
        for i, (x1, x2, y1, y2, z1, z2) in enumerate(boxes):
            a1.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fc=coul[zone], alpha=0.15 if i else 0.3,
                                   ec=coul[zone]))
        x1, x2, y1, y2, z1, z2 = boxes[0]
        c = np.array([[(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2]])
        back = cfg.sensor_to_world(spherical_to_sensor(sensor_to_spherical(cfg.world_to_sensor(c))))
        z_trouvee = zone_of_points(cfg, back)[0]
        a1.plot(c[0, 0], c[0, 1], "o", color=coul[zone], ms=10, mec="k")
        dy = -0.2 if zone >= 2 else 0.1
        a1.text(c[0, 0], c[0, 1] + dy, f"{noms[zone]}\nzone {zone} -> {z_trouvee}", ha="center", fontsize=7.5,
                bbox={"fc": "w", "ec": "none", "alpha": 0.7, "pad": 1})
    a1.plot(cfg.sensor_pos[0], cfg.sensor_pos[1], "k^", ms=12)
    a1.text(cfg.sensor_pos[0], cfg.sensor_pos[1] - 0.12, "radar", ha="center", fontsize=9)
    a1.set(title="Test 3 - test_point_cloud : le centre de chaque siege\n"
                 "repasse par le repere radar et retombe dans sa zone",
           xlabel="x lateral (m)", ylabel="y vers l'arriere (m)", xlim=(-1, 1), ylim=(-0.4, 2.2))
    a1.set_aspect("equal")
    a1.grid(alpha=0.3)

    p = np.array([[0.4, 0.8, 0.9], [-0.3, 1.6, 0.7]])
    ps = cfg.world_to_sensor(p)
    pr = cfg.sensor_to_world(ps)
    a2.axis("off")
    lignes = ["Deux points dans la voiture (x, y, z en m) :", ""]
    for k in range(2):
        lignes += [f"  point {k + 1}  voiture   ({p[k, 0]:+.3f}, {p[k, 1]:+.3f}, {p[k, 2]:+.3f})",
                   f"           -> radar   ({ps[k, 0]:+.3f}, {ps[k, 1]:+.3f}, {ps[k, 2]:+.3f})",
                   f"           -> voiture ({pr[k, 0]:+.3f}, {pr[k, 1]:+.3f}, {pr[k, 2]:+.3f})", ""]
    lignes += [f"Ecart maximal apres l'aller-retour : {np.abs(pr - p).max():.1e} m",
               "", f"Radar en ({cfg.sensor_pos[0]:.2f}, {cfg.sensor_pos[1]:.2f}, {cfg.sensor_pos[2]:.2f}) m,",
               f"incline de {cfg.elev_tilt_deg:.0f} deg en elevation et {cfg.azim_tilt_deg:.0f} deg en azimut",
               "(ligne sensorPosition du cfg)."]
    a2.text(0, 0.95, "\n".join(lignes), family="monospace", fontsize=9.5, va="top")
    a2.set_title("Test 4 - test_geometry_round_trip : voiture -> radar -> voiture\nredonne exactement le point",
                 fontsize=11)
    enregistrer(fig, "03_04_zones_et_geometrie.png")


# ---------------------------------------------------------------------------------------------- 5
def fig_entete_radar_cube():
    n_ch, n_rng, n_ant = 3, 64, 16
    cube = np.zeros((n_ch, n_rng, n_ant), complex)
    cube[:, 20, :] = 3000                      # une cible a la case 20, sur toutes les antennes
    cube[:, 45, ::2] = 1500                    # une autre a la case 45, une antenne sur deux
    re, im = cube.real.astype("<i2"), cube.imag.astype("<i2")
    inter = np.stack([im, re], axis=-1).astype("<i2").tobytes()
    bon = tlv._decode_radar_cube(struct.pack("<3I", n_ant, n_rng, n_ch) + inter, 16)
    data = np.frombuffer(inter, "<i2")
    cplx = data[1::2] + 1j * data[0::2]
    # ancien code : 1er nombre de l'en-tete (16) lu comme nombre de cases -> 12 « rafales » de 16 « cases »
    faux = cplx.reshape(12, 16, 16)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    axs[0].imshow(np.abs(bon[0]).T, aspect="auto", origin="lower", cmap="magma")
    axs[0].set(title="Lecture correcte : en-tete [16 antennes, 64 cases, 3 rafales]\n"
                     "forme (3, 64, 16) : les cibles sont aux cases 20 et 45",
               xlabel="case de distance", ylabel="antenne virtuelle")
    axs[1].imshow(np.abs(faux.reshape(-1, 16)).T, aspect="auto", origin="lower", cmap="magma")
    for k in range(1, 12):
        axs[1].axvline(16 * k - 0.5, color="w", lw=0.4)
    axs[1].set(title="Ancien code (ordre du commentaire TI) : 12 « rafales » de 16 « cases »\n"
                     "forme (12, 16, 16) au lieu de (3, 64, 16) -> trame rejetee (« pas de radar cube »)",
               xlabel="« cases » lues, rafale par rafale", ylabel="« antenne »")
    fig.suptitle("Test 5 - test_radar_cube_entete_firmware : l'ordre reel des dimensions ecrit par la carte",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    enregistrer(fig, "05_entete_radar_cube.png")


# ---------------------------------------------------------------------------------------------- 6
def fig_gigue():
    cfg = parse_cfg(CFG_BUREAU)
    nums = [10, 11, 13, 13, 14, 16, 16, 17, 18, 19, 21, 21, 40]
    carte = CarteRespiration(cfg)
    cube = np.zeros((carte.nc, carte.R, carte.A), np.complex64)
    n, red = [], []
    for k in nums:
        carte.push(tlv.Frame(frame_number=k, num_detected_obj=0, radar_cube=cube))
        n.append(carte.n)
        red.append(carte.redemarrages)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    a1.plot(range(len(nums)), nums, "o-", color="#1565c0")
    for i, k in enumerate(nums):
        a1.text(i, k + 0.8, str(k), ha="center", fontsize=8)
    a1.annotate("vraie coupure\n(19 trames manquent)", (12, 40), (8.5, 34), arrowprops={"arrowstyle": "->"})
    a1.set(title="Numeros de trame recus : +2 puis repetition\n(lecture tardive du compteur par la carte)",
           xlabel="trame recue", ylabel="numero dans l'en-tete")
    a2.step(range(len(nums)), n, where="post", color="#2e7d32", label="images dans la fenetre d'analyse")
    a2.step(range(len(nums)), red, where="post", color="#c62828", label="redemarrages")
    a2.set(title="La fenetre grandit sans redemarrer ; elle ne repart de zero\nqu'a la vraie coupure",
           xlabel="trame recue")
    a2.legend()
    fig.suptitle("Test 6 - test_gigue_numero_trame : numeros decales sans perte reelle", fontweight="bold", y=1.04)
    fig.tight_layout()
    enregistrer(fig, "06_gigue_numero_trame.png")


# ---------------------------------------------------------------------------------------------- 7
def fig_respiration():
    cfg = parse_cfg(CFG_BUREAU)
    rng = np.random.default_rng(0)
    u0, r0 = np.sin(np.radians(20)), 0.9
    rb = int(round(r0 / cfg.range_step_m))
    a = np.conj(cfg.steering(u0, 0.0))[0] / cfg.rx_comp
    carte = CarteRespiration(cfg)
    t_burst = np.array(cfg.exported_chirp_indices()) * cfg.burst_period_us * 1e-6
    for k in range(int(25 * cfg.frame_rate_hz)):
        t = k / cfg.frame_rate_hz + t_burst
        d = 4e-3 * np.sin(2 * np.pi * 15 / 60 * t)
        cube = (rng.normal(size=(3, cfg.num_range_bins, cfg.num_virt_ant))
                + 1j * rng.normal(size=(3, cfg.num_range_bins, cfg.num_virt_ant))) * 2
        cube[:, 12, :] += 300 * np.conj(cfg.steering(-0.5, 0.1))[0] / cfg.rx_comp
        cube[:, rb, :] += (200 * np.exp(1j * 4 * np.pi * d / cfg.wavelength_m))[:, None] * a[None, :]
        carte.push(tlv.Frame(frame_number=k, num_detected_obj=0, radar_cube=cube.astype(np.complex64)))
        if k % 2 == 1:
            res = carte.analyser()
    fig = plt.figure(figsize=(14, 4.6))
    a1, a2, a3 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)
    dessiner_carte(a1, carte, res["carte_dyn"])
    a1.plot(r0 * u0, r0 * np.cos(np.radians(20)), "c+", ms=18, mew=2, label="vraie position")
    c = res["cible"]
    a1.plot(c["xyz"][0], c["xyz"][1], "o", mfc="none", mec="#00e676", ms=16, mew=2, label="cible trouvee")
    rm = 12 * cfg.range_step_m
    xm, ym = rm * -0.5, rm * np.sqrt(1 - 0.25 - 0.01)
    a1.plot(xm, ym, "wx", ms=8)
    a1.text(xm - 0.05, ym - 0.13, "mur fixe", color="w", fontsize=8, ha="right")
    a1.set(title="Carte des micro-mouvements", xlabel="x (m)", ylabel="y (m)", xlim=(-1.5, 1.5), ylim=(0, 1.8))
    a1.set_facecolor("k")
    a1.legend(fontsize=8, loc="upper left")
    a2.plot(res["t"], res["deplacement_mm"], color="#1565c0")
    a2.set(title="Mouvement mesure a la cible (thorax ±4 mm simule)", xlabel="temps (s)", ylabel="mm")
    a3.plot(res["spectre_bpm"], res["spectre"], color="#1565c0")
    a3.axvline(15, color="c", ls="--", label="15 /min simule")
    a3.axvline(res["bpm"], color="#2e7d32", label=f"{res['bpm']:.1f} /min mesure")
    a3.axvspan(6, 45, color="#2e7d32", alpha=0.08)
    a3.set(title="Spectre : un seul pic", xlabel="rythme (/min)", xlim=(0, 60))
    a3.legend(fontsize=8)
    fig.suptitle(f"Test 7 - test_bureau_respiration : personne simulee a 0,90 m, +20 deg, 15 /min  ->  "
                 f"trouvee a {c['distance_m']:.2f} m, {c['azimut_deg']:+.0f} deg, {res['bpm']:.1f} /min, "
                 f"{res['amplitude_mm']:.1f} mm (exige : 15 ±1, ±10 cm, ±6 deg)", fontweight="bold", fontsize=10)
    enregistrer(fig, "07_respiration.png")


# ---------------------------------------------------------------------------------------------- 8, 9, 10
def chronologie(nom_fichier, titre, scenario_kw, verite, annotations, exige):
    cfg = parse_cfg(CFG_BUREAU)
    carte = CarteRespiration(cfg)
    journal, rythmes, pauses = [], [], []
    for k, cube in tb._scenario_apnee(cfg, np.random.default_rng(scenario_kw.pop("graine", 3)), **scenario_kw):
        carte.push(tlv.Frame(frame_number=k, num_detected_obj=0, radar_cube=cube))
        if k % 2 == 1 and k > 20:
            r = tb._p1(carte.analyser())
            t = k / cfg.frame_rate_hz
            journal.append((t, r["etat"]))
            rythmes.append((t, r["bpm"]))
            pauses.append((t, r["pause_s"]))
    fig, axs = plt.subplots(4, 1, figsize=(13, 8.2), sharex=True,
                            gridspec_kw={"height_ratios": [2.2, 0.5, 1.6, 1.4]})
    tt = np.arange(0, scenario_kw.get("total", 80.0), 0.05)
    axs[0].plot(tt, verite(tt) * 1e3, color="0.3")
    axs[0].set(ylabel="mm", title="Ce qu'on simule : mouvement du thorax")
    for x0, x1, txt, c in annotations:
        axs[0].axvspan(x0, x1, color=c, alpha=0.15)
        axs[0].text((x0 + x1) / 2, axs[0].get_ylim()[1] * 0.85, txt, ha="center", fontsize=8)
    bande_etats(axs[1], journal)
    axs[1].set_title("Ce que le programme annonce", fontsize=10)
    t_r = [t for t, b in rythmes if b is not None]
    axs[2].plot(t_r, [b for _, b in rythmes if b is not None], ".", color="#2e7d32", ms=4)
    axs[2].axhline(15, color="c", ls="--", lw=1)
    axs[2].set(ylabel="/min", ylim=(0, 30), title="Rythme annonce (rien pendant pause, apnee, reprise)")
    axs[3].plot([t for t, _ in pauses], [p for _, p in pauses], color="#c62828")
    axs[3].axhline(10, color="#c62828", ls=":", lw=1)
    axs[3].text(1, 11, "seuil d'alerte : 10 s", fontsize=8, color="#c62828")
    axs[3].set(ylabel="s", xlabel="temps (s)", title="Duree de la pause comptee par le programme")
    alerte = next((t for t, e in journal if e == "apnee"), None)
    if alerte is not None:
        for ax in axs:
            ax.axvline(alerte, color="#c62828", lw=1)
        axs[0].text(alerte + 0.3, axs[0].get_ylim()[0] * 0.9, f"alerte a {alerte:.0f} s", color="#c62828", fontsize=8)
    legende_etats(axs[1], ["respire", "faible", "incertain", "pause", "apnee", "reprise"])
    fig.suptitle(f"{titre}\n{exige}", fontweight="bold", fontsize=10)
    fig.tight_layout()
    enregistrer(fig, nom_fichier)


def fig_apnee():
    verite = lambda t: np.where((t >= 30) & (t < 55), 0, 3e-3 * np.sin(2 * np.pi * 15 / 60 * t))  # noqa: E731
    chronologie("08_apnee.png", "Test 8 - test_apnee : respiration, apnee de 30 a 55 s, reprise "
                "(la tete bouge un peu ailleurs pendant l'apnee)", {"graine": 3}, verite,
                [(30, 55, "apnee simulee", "#c62828")],
                "Exige : 15 /min avant ; alerte entre 38 et 44 s ; aucun rythme annonce de 37 a 55 s ; "
                "la cible reste sur la personne ; reprise vue avant 65 s ; 15 /min ensuite\n"
                "(la reprise n'est confirmee qu'apres 7 s de mouvement : d'ou le compteur qui monte jusqu'a ~62 s)")


def fig_grandes_inspirations():
    prof = lambda t: np.where((t >= 55) & (t < 65), 6e-3, np.where(t >= 65, 0.8e-3, 3e-3))  # noqa: E731
    verite = lambda t: np.where((t >= 30) & (t < 55), 0, prof(t) * np.sin(2 * np.pi * 15 / 60 * t))  # noqa: E731
    chronologie("09_grandes_inspirations.png", "Test 9 - test_grandes_inspirations_puis_normal : apres l'apnee, "
                "10 s de grandes inspirations (±6 mm), puis respiration faible (±0,8 mm)",
                {"graine": 7, "total": 110.0, "profil": prof}, verite,
                [(30, 55, "apnee", "#c62828"), (55, 65, "grandes\ninspirations", "#1565c0"),
                 (65, 110, "respiration faible (13 % des grandes)", "#9e9d24")],
                "Exige : une alerte pendant l'apnee, puis plus aucune apnee apres 64 s ; "
                "a la fin, « respire » ou « respiration faible »")


def fig_sursaut():
    def verite(t):
        d = np.where((t >= 30) & (t < 55), 0, 3e-3 * np.sin(2 * np.pi * 15 / 60 * t))
        return d + np.where((t >= 46) & (t < 47), 3e-3 * np.sin(np.pi * (t - 46)), 0)
    chronologie("10_sursaut_pendant_apnee.png", "Test 10 - test_sursaut_pendant_apnee : un mouvement d'une "
                "seconde a 46 s, en pleine apnee", {"graine": 5, "sursaut": 46.0}, verite,
                [(30, 55, "apnee simulee", "#c62828"), (46, 47, "sursaut", "#1565c0")],
                "Exige : APNEE en continu de 41 a 55 s, et une duree de pause qui suit le vrai temps "
                "d'apnee (pas de remise a zero au sursaut)")


# ---------------------------------------------------------------------------------------------- 11
def fig_deux_personnes():
    cfg = parse_cfg(CFG_BUREAU)
    carte = CarteRespiration(cfg)
    j = {0.6: [], 1.2: []}
    dernier = None
    for k, cube in tb._scenario_deux(cfg, np.random.default_rng(11)):
        carte.push(tlv.Frame(frame_number=k, num_detected_obj=0, radar_cube=cube))
        if k % 2 == 1 and k > 20:
            r = carte.analyser()
            t = k / cfg.frame_rate_hz
            for dist in j:
                c = next((c for c in r.get("cibles", []) if abs(c["cible"]["distance_m"] - dist) < 0.15), None)
                j[dist].append((t, c["etat"] if c else "incertain", c["bpm"] if c else None,
                                c["nom"] if c else "-"))
            if abs(t - 50) < 0.21:
                dernier = r
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 2.2], height_ratios=[1, 1, 1])
    a0 = fig.add_subplot(gs[:, 0])
    dessiner_carte(a0, carte, dernier["carte_dyn"])
    for (r0, az, w0, *_), lab in zip(tb.PERSONNES_DEUX, ["P1 simulee", "P2 simulee"]):
        u = np.sin(np.radians(az))
        x, y = r0 * u, r0 * np.sqrt(1 - u ** 2 - w0 ** 2)
        a0.plot(x, y, "c+", ms=16, mew=2)
        a0.text(x, y + 0.12, lab, color="c", ha="center", fontsize=8)
    for c in dernier["cibles"]:
        a0.plot(c["cible"]["xyz"][0], c["cible"]["xyz"][1], "o", mfc="none", mec="#00e676", ms=16, mew=2)
        a0.text(c["cible"]["xyz"][0] + 0.12, c["cible"]["xyz"][1] - 0.15, c["nom"], color="#00e676", fontsize=9)
    a0.set(title="Micro-mouvements a t = 50 s\n(fenetre de 30 a 50 s : P2 y respire encore 5 s)",
           xlabel="x (m)", ylabel="y (m)",
           xlim=(-1.5, 1.5), ylim=(0, 1.8))
    a0.set_facecolor("k")
    for ligne, (dist, bpm_vrai, txt) in enumerate([(0.6, 15, "P1 : 0,6 m, -20 deg, 15 /min, respire tout le temps"),
                                                   (1.2, 22, "P2 : 1,2 m, +25 deg, 22 /min, apnee de 35 a 60 s")]):
        a = fig.add_subplot(gs[ligne, 1])
        jj = j[dist]
        bande_etats(a, [(t, e) for t, e, _, _ in jj], 0, 1)
        a2 = a.twinx()
        a2.plot([t for t, _, b, _ in jj if b], [b for _, _, b, _ in jj if b], "k.", ms=3)
        a2.axhline(bpm_vrai, color="c", ls="--", lw=1)
        a2.set(ylim=(0, 30), ylabel="/min")
        if dist == 1.2:
            a.axvspan(35, 60, ymin=0, ymax=0.08, color="k")
        a.set_title(txt, fontsize=9)
        a.set_xlim(0, 80)
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.axis("off")
    ax3.text(0, 0.8, "Bande de couleur = etat annonce pour chaque personne ; points noirs = rythme annonce ; "
                     "tirets = rythme simule.\nTrait noir sous la bande de P2 = apnee simulee.", fontsize=9, va="top")
    legende_etats(ax3, ["respire", "incertain", "pause", "apnee", "reprise"])
    fig.suptitle("Test 11 - test_deux_personnes : deux personnes suivies separement\n"
                 "Exige : les deux rythmes a ±1 /min de 25 a 35 s ; de 50 a 60 s, P2 en APNEE et P1 respire "
                 "toujours ; l'alerte de P2 passe en premier a l'affichage", fontweight="bold", fontsize=10)
    fig.tight_layout()
    enregistrer(fig, "11_deux_personnes.png")


if __name__ == "__main__":
    print("Figures des tests :")
    fig_cfg()
    fig_tlv()
    fig_zones_et_geometrie()
    fig_entete_radar_cube()
    fig_gigue()
    fig_respiration()
    fig_apnee()
    fig_grandes_inspirations()
    fig_sursaut()
    fig_deux_personnes()
