"""Tests rapides (sans carte) : python -m pytest tests  ou  python tests/test_base.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_vitals.cfg_parser import parse_cfg  # noqa: E402
from radar_vitals import tlv  # noqa: E402

CFG = Path(__file__).resolve().parents[1] / "cfg" / "cpd_vitals.cfg"


def test_cfg():
    cfg = parse_cfg(CFG)
    assert cfg.num_virt_ant == 16 and cfg.num_range_bins == 64
    assert abs(cfg.range_step_m - 0.0459) < 1e-3
    assert cfg.radcube_export_enabled and cfg.radcube_num_chirps == 3
    assert cfg.num_zones == 5


def test_tlv_roundtrip():
    rng = np.random.default_rng(0)
    cube = (rng.normal(0, 500, (3, 64, 16)) + 1j * rng.normal(0, 500, (3, 64, 16))).round()
    probs = np.zeros((3, 5))
    probs[0] = 1.0
    probs[:, 2] = [0.1, 0.8, 0.1]          # zone 2 occupee
    occ = tlv.encode_occupancy(probs)
    pkt = tlv.encode_packet(42, [(tlv.TLV_RADAR_CUBE, tlv.encode_radar_cube(cube)), (tlv.TLV_OCC_CLASS, occ)])
    assert len(pkt) % 32 == 0
    # flux avec des octets parasites avant / entre les trames
    stream = b"\x00garbage" + pkt + b"\x01\x02" + pkt
    frames = []
    pb = tlv.PacketBuffer()
    for i in range(0, len(stream), 37):
        frames += [tlv.decode_packet(p) for p in pb.feed(stream[i:i + 37])]
    assert len(frames) == 2
    fr = frames[0]
    assert fr.frame_number == 42 and fr.tlv_types == [2007, 1041]
    assert np.allclose(fr.radar_cube, cube)
    assert fr.occupancy.shape == (3, 5)
    assert np.allclose(fr.occupancy, probs, atol=1 / 128)
    # ordre firmware : [zone][classe] -> les 3 premiers octets sont la zone 0
    assert list(occ[:3]) == [128, 0, 0] and list(occ[6:9]) == [13, 102, 13]


def test_bureau_respiration():
    """Cible synthetique qui respire a 15 /min a 0,9 m, azimut +20 deg : la carte la retrouve."""
    from radar_vitals.bureau import CarteRespiration
    cfg = parse_cfg(Path(CFG).parent / "bureau_vitals.cfg")
    rng = np.random.default_rng(0)
    u0, w0, r0 = np.sin(np.radians(20)), 0.0, 0.9
    rb = int(round(r0 / cfg.range_step_m))
    a = np.conj(cfg.steering(u0, w0))[0] / cfg.rx_comp          # reponse du reseau, avant compensation
    carte = CarteRespiration(cfg)
    t_burst = np.array(cfg.exported_chirp_indices()) * cfg.burst_period_us * 1e-6
    for k in range(int(25 * cfg.frame_rate_hz)):
        t = k / cfg.frame_rate_hz + t_burst
        d = 4e-3 * np.sin(2 * np.pi * 15 / 60 * t)               # thorax +-4 mm a 15 /min
        cube = (rng.normal(size=(3, cfg.num_range_bins, cfg.num_virt_ant))
                + 1j * rng.normal(size=(3, cfg.num_range_bins, cfg.num_virt_ant))) * 2
        cube[:, 12, :] += 300 * np.conj(cfg.steering(-0.5, 0.1))[0] / cfg.rx_comp   # mur fixe
        cube[:, rb, :] += (200 * np.exp(1j * 4 * np.pi * d / cfg.wavelength_m))[:, None] * a[None, :]
        carte.push(tlv.Frame(frame_number=k, num_detected_obj=0, radar_cube=cube.astype(np.complex64)))
        if k % 2 == 1:
            res = carte.analyser()          # comme le programme : une analyse toutes les 2 images
    assert res["respire"] and abs(res["bpm"] - 15) < 1.0, res["bpm_brut"]
    assert abs(res["cible"]["distance_m"] - r0) < 0.1 and abs(res["cible"]["azimut_deg"] - 20) < 6
    assert 6.0 < res["amplitude_mm"] < 10.0      # crete a crete d'une sinusoide +-4 mm = 8 mm


def test_point_cloud():
    from radar_vitals.nuage import spherical_to_sensor, sensor_to_spherical, zone_of_points
    pts = np.array([[0.8, 0.3, -0.2, 0.05, 15.0], [1.5, -0.6, 0.4, -0.3, 9.5]])
    pkt = tlv.encode_packet(7, [(tlv.TLV_POINT_CLOUD, tlv.encode_point_cloud(pts))], num_obj=2)
    fr = tlv.decode_packet(pkt)
    assert fr.num_detected_obj == 2 and fr.points.shape == (2, 5)
    tol = [0.00025, np.pi / 254, np.pi / 254, 0.00028, 1 / 256]
    assert np.all(np.abs(fr.points - pts) <= np.array(tol) * 0.51 + 1e-9)
    xyz = spherical_to_sensor(pts)
    assert np.allclose(sensor_to_spherical(xyz), pts[:, :3])
    assert np.isclose(xyz[0, 1], 0.8 * np.cos(-0.2) * np.cos(0.3))   # y = axe de visee
    # un point place au centre d'un siege tombe dans la bonne zone
    cfg = parse_cfg(CFG)
    for zone, boxes in cfg.cuboids.items():
        x1, x2, y1, y2, z1, z2 = boxes[0]
        c = np.array([[(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2]])
        p_s = cfg.world_to_sensor(c)
        back = cfg.sensor_to_world(spherical_to_sensor(sensor_to_spherical(p_s)))
        assert zone_of_points(cfg, back)[0] == zone


def test_geometry_round_trip():
    cfg = parse_cfg(CFG)
    p = np.array([[0.4, 0.8, 0.9], [-0.3, 1.6, 0.7]])
    assert np.allclose(cfg.sensor_to_world(cfg.world_to_sensor(p)), p)


if __name__ == "__main__":
    test_cfg(); test_tlv_roundtrip(); test_point_cloud(); test_bureau_respiration(); test_geometry_round_trip()
    print("OK")


def test_radar_cube_entete_firmware():
    """Charge utile ecrite comme mmwave_demo_mss.c : dims [numAnt, numRangeBins, numChirps], puis (imag, reel)."""
    import struct
    n_ch, n_rng, n_ant = 3, 64, 16
    rng = np.random.default_rng(0)
    re = rng.integers(-500, 500, (n_ch, n_rng, n_ant)).astype("<i2")
    im = rng.integers(-500, 500, (n_ch, n_rng, n_ant)).astype("<i2")
    inter = np.stack([im, re], axis=-1).astype("<i2")
    payload = struct.pack("<3I", n_ant, n_rng, n_ch) + inter.tobytes()
    cube = tlv._decode_radar_cube(payload, 16)
    assert cube.shape == (n_ch, n_rng, n_ant)
    assert np.allclose(cube.real, re) and np.allclose(cube.imag, im)
    # ancien format du simulateur toujours lisible
    old = struct.pack("<3I", n_rng, 4, 4 * n_ch) + inter.tobytes()
    assert tlv._decode_radar_cube(old, 16).shape == (n_ch, n_rng, n_ant)


def test_gigue_numero_trame():
    """Numeros qui sautent de 2 puis se repetent (lecture tardive du compteur) : pas de redemarrage."""
    from radar_vitals.bureau import CarteRespiration
    from radar_vitals.cfg_parser import parse_cfg
    cfg = parse_cfg(CFG.parent / "bureau_vitals.cfg")
    carte = CarteRespiration(cfg)
    cube = np.zeros((carte.nc, carte.R, carte.A), np.complex64)
    nums = [10, 11, 13, 13, 14, 16, 16, 17, 18, 19, 21, 21]
    for k in nums:
        assert carte.push(tlv.Frame(frame_number=k, num_detected_obj=0, radar_cube=cube))
    assert carte.redemarrages == 0 and carte.perdues == 0 and carte.n == len(nums)
    assert carte.push(tlv.Frame(frame_number=40, num_detected_obj=0, radar_cube=cube))
    assert carte.redemarrages == 1


def _scenario_apnee(cfg, rng, debut=30.0, fin=55.0, total=80.0, bpm=15.0, profil=None, sursaut=None):
    """Personne a 0,6 m qui respire, s'arrete de debut a fin, puis reprend ; pendant l'apnee, la tete
    (0,75 m, plus haut) bouge un peu, de facon irreguliere."""
    u0, r0 = np.sin(np.radians(-5)), 0.6
    rb, rh = int(round(r0 / cfg.range_step_m)), int(round(0.75 / cfg.range_step_m))
    a = np.conj(cfg.steering(u0, -0.5))[0] / cfg.rx_comp
    ah = np.conj(cfg.steering(0.0, 0.3))[0] / cfg.rx_comp
    t_burst = np.array(cfg.exported_chirp_indices()) * cfg.burst_period_us * 1e-6
    n_tr = int(total * cfg.frame_rate_hz)
    tete = np.convolve(np.random.default_rng(123).normal(size=n_tr), np.ones(10) / 10, "same")
    tete *= 3e-4 / tete.std()                                         # petits mouvements irreguliers (0,3 mm)
    for k in range(n_tr):
        t = k / cfg.frame_rate_hz + t_burst
        amp = 3e-3 if profil is None else profil(t)
        d = amp * np.sin(2 * np.pi * bpm / 60 * t)
        d = np.where((t >= debut) & (t < fin), 0.0, d)
        if sursaut is not None:                                         # mouvement isole de 1 s, 3 mm
            d = d + np.where((t >= sursaut) & (t < sursaut + 1), 3e-3 * np.sin(np.pi * (t - sursaut)), 0.0)
        dh = np.where((t >= debut) & (t < fin), tete[k], 0.0)           # la tete bouge un peu pendant l'apnee
        cube = (rng.normal(size=(3, cfg.num_range_bins, cfg.num_virt_ant))
                + 1j * rng.normal(size=(3, cfg.num_range_bins, cfg.num_virt_ant))) * 2
        cube[:, 12, :] += 300 * np.conj(cfg.steering(-0.5, 0.1))[0] / cfg.rx_comp
        cube[:, rb, :] += (200 * np.exp(1j * 4 * np.pi * d / cfg.wavelength_m))[:, None] * a[None, :]
        cube[:, rh, :] += (250 * np.exp(1j * 4 * np.pi * dh / cfg.wavelength_m))[:, None] * ah[None, :]
        yield k, cube.astype(np.complex64)


def _p1(r):
    """Resultat de la piste P1 (la personne), ou du candidat principal tant que rien n'est confirme."""
    r = next((c for c in r.get("cibles", []) if c["num"] == 1), r)
    if r.get("cible") is None:                     # moins de 8 s de donnees : pas encore de cible
        return {"etat": "incertain", "bpm": None, "pause_s": 0.0, "cible": {"distance_m": np.nan}}
    return r


def test_apnee():
    from radar_vitals.bureau import CarteRespiration
    cfg = parse_cfg(Path(CFG).parent / "bureau_vitals.cfg")
    carte = CarteRespiration(cfg)
    rng = np.random.default_rng(3)
    journal = []
    for k, cube in _scenario_apnee(cfg, rng):
        carte.push(tlv.Frame(frame_number=k, num_detected_obj=0, radar_cube=cube))
        if k % 2 == 1 and k > 20:
            r = _p1(carte.analyser())
            journal.append((k / cfg.frame_rate_hz, r["etat"], r["bpm"], r["pause_s"], r["cible"]["distance_m"]))
    avant = [j for j in journal if 22 <= j[0] < 30]
    assert all(e == "respire" and abs(b - 15) < 1.0 for _, e, b, _, _ in avant), avant
    alerte = next(t for t, e, *_ in journal if e == "apnee")
    assert 38 <= alerte <= 44, alerte                         # apnee a 30 s, alerte vers 10-14 s apres
    faux = [j for j in journal if 37 <= j[0] < 55 and j[1] == "respire"]
    assert not faux, faux                                     # plus de rythme annonce 7 s apres l'arret
    pendant = [j for j in journal if 45 <= j[0] < 55]
    assert all(e == "apnee" for _, e, *_ in pendant), pendant
    assert all(abs(dist - 0.6) < 0.1 for *_, dist in pendant)  # la cible reste sur la personne
    reprise = next(t for t, e, *_ in journal if t > 55 and e != "apnee" and e != "pause")
    assert reprise < 65, reprise                              # reprise confirmee apres 7 s de mouvement
    apres = [(t, b) for t, e, b, *_ in journal if t > 55 and e == "respire"]
    assert apres and all(abs(b - 15) < 1.5 for _, b in apres), apres   # pas de rythme "melange" apres la reprise


def test_grandes_inspirations_puis_normal():
    """Apres l'apnee : 10 s de grandes inspirations (+-6 mm), puis respiration calme et faible (+-0,8 mm,
    moins de 30 % des grandes). Elle ne doit pas etre prise pour une apnee."""
    from radar_vitals.bureau import CarteRespiration
    cfg = parse_cfg(Path(CFG).parent / "bureau_vitals.cfg")
    carte = CarteRespiration(cfg)
    profil = lambda t: np.where((t >= 55) & (t < 65), 6e-3, np.where(t >= 65, 0.8e-3, 3e-3))  # noqa: E731
    journal = []
    for k, cube in _scenario_apnee(cfg, np.random.default_rng(7), total=110.0, profil=profil):
        carte.push(tlv.Frame(frame_number=k, num_detected_obj=0, radar_cube=cube))
        if k % 2 == 1 and k > 20:
            r = _p1(carte.analyser())
            journal.append((k / cfg.frame_rate_hz, r["etat"]))
    assert any(e == "apnee" for t, e in journal if 40 <= t < 55)
    faux = [(t, e) for t, e in journal if t >= 64 and e == "apnee"]   # reprise confirmee en 7 s
    assert not faux, faux
    fin = [e for t, e in journal if t >= 90]
    assert all(e in ("respire", "faible") for e in fin), fin


def test_sursaut_pendant_apnee():
    """Un mouvement isole (1 s) au milieu de l'apnee ne doit pas l'interrompre ni remettre le compteur a zero."""
    from radar_vitals.bureau import CarteRespiration
    cfg = parse_cfg(Path(CFG).parent / "bureau_vitals.cfg")
    carte = CarteRespiration(cfg)
    journal = []
    for k, cube in _scenario_apnee(cfg, np.random.default_rng(5), sursaut=46.0):
        carte.push(tlv.Frame(frame_number=k, num_detected_obj=0, radar_cube=cube))
        if k % 2 == 1 and k > 20:
            r = _p1(carte.analyser())
            journal.append((k / cfg.frame_rate_hz, r["etat"], r["pause_s"]))
    pendant = [(t, e, ps) for t, e, ps in journal if 41 <= t < 55]
    assert all(e == "apnee" for _, e, _ in pendant), pendant
    assert all(abs(ps - (t - 30)) < 4 for t, _, ps in pendant), pendant


def test_deux_personnes():
    """P1 a 0,6 m (azimut -20 deg) respire a 15 /min ; P2 a 1,2 m (azimut +25 deg) respire a 22 /min et
    fait une apnee de 35 a 60 s. Les deux rythmes sont trouves ; seule P2 passe en apnee."""
    from radar_vitals.bureau import CarteRespiration
    cfg = parse_cfg(Path(CFG).parent / "bureau_vitals.cfg")
    carte = CarteRespiration(cfg)
    rng = np.random.default_rng(11)
    pers = [  # (distance, azimut deg, elevation sin, rythme /min, amplitude m, apnee)
        (0.6, -20, -0.3, 15.0, 3e-3, None),
        (1.2, 25, -0.1, 22.0, 2.5e-3, (35.0, 60.0)),
    ]
    t_burst = np.array(cfg.exported_chirp_indices()) * cfg.burst_period_us * 1e-6
    journal = []
    for k in range(int(80 * cfg.frame_rate_hz)):
        t = k / cfg.frame_rate_hz + t_burst
        cube = (rng.normal(size=(3, cfg.num_range_bins, cfg.num_virt_ant))
                + 1j * rng.normal(size=(3, cfg.num_range_bins, cfg.num_virt_ant))) * 2
        cube[:, 12, :] += 300 * np.conj(cfg.steering(-0.5, 0.1))[0] / cfg.rx_comp      # mur fixe
        for r0, az, w0, bpm, amp, ap in pers:
            d = amp * np.sin(2 * np.pi * bpm / 60 * t + r0)
            if ap is not None:
                d = np.where((t >= ap[0]) & (t < ap[1]), 0.0, d)
            a = np.conj(cfg.steering(np.sin(np.radians(az)), w0))[0] / cfg.rx_comp
            rb = int(round(r0 / cfg.range_step_m))
            cube[:, rb, :] += (200 * np.exp(1j * 4 * np.pi * d / cfg.wavelength_m))[:, None] * a[None, :]
        carte.push(tlv.Frame(frame_number=k, num_detected_obj=0, radar_cube=cube.astype(np.complex64)))
        if k % 2 == 1 and k > 20:
            journal.append((k / cfg.frame_rate_hz, carte.analyser()))

    def piste(r, dist):
        return next((c for c in r.get("cibles", []) if abs(c["cible"]["distance_m"] - dist) < 0.15), None)

    # les deux personnes sont trouvees, aux bons rythmes, avant l'apnee
    for t, r in journal:
        if 25 <= t < 35:
            a, b = piste(r, 0.6), piste(r, 1.2)
            assert a and b, (t, [(c["nom"], c["cible"]["distance_m"]) for c in r.get("cibles", [])])
            assert a["etat"] == "respire" and abs(a["bpm"] - 15) < 1.0, (t, a["etat"], a["bpm"])
            assert b["etat"] == "respire" and abs(b["bpm"] - 22) < 1.0, (t, b["etat"], b["bpm"])
    # P2 en apnee, P1 respire toujours
    for t, r in journal:
        if 50 <= t < 60:
            a, b = piste(r, 0.6), piste(r, 1.2)
            assert b["etat"] == "apnee", (t, b["etat"])
            assert a["etat"] == "respire" and abs(a["bpm"] - 15) < 1.0, (t, a["etat"])
            assert r["principale"] is b                       # l'alerte passe en premier a l'affichage
    assert len(journal[-1][1]["cibles"]) == 2
