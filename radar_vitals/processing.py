"""Traitement PC : radar cube (TLV 2007) -> etat de chaque siege.

Pour chaque zone (cuboides du cfg) :
1. formation de voie classique (Bartlett) vers chaque voxel (case distance x direction) de la zone
2. presence statique : ce qui a change par rapport a l'habitacle vide (reference complexe)
3. respiration : phase des voxels les plus "vivants" -> deplacement -> spectre 6-45 /min.
   Les sieges sont traites du plus "vivant" au moins vivant ; chacun est analyse avec un zero
   dans la direction des sources reelles deja trouvees (supprime les fuites entre sieges)
4. apnee : plus de mouvement au-dessus du bruit sur les 8 dernieres secondes
5. coeur (experimental) : pic 48-132 /min apres retrait de la respiration

Etats : INIT, VIDE, RESPIRE, MOUVEMENT, SURVEILLANCE (pas de respiration < seuil d'alerte),
        ALERTE (occupant immobile sans respiration detectee), INCERTAIN (pas de reference).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cfg_parser import RadarConfig
from .tlv import Frame


@dataclass
class Params:
    window_s: float = 25.6          # fenetre d'analyse du rythme respiratoire
    short_s: float = 8.0            # fenetre courte pour l'apnee / le mouvement
    min_data_s: float = 10.0        # donnees minimales avant de conclure
    update_s: float = 1.0           # periode de mise a jour des etats
    alert_after_s: float = 10.0     # duree sans respiration avant ALERTE
    confirm_s: float = 3.0          # respiration continue necessaire pour lever une alerte
    breath_band_bpm: tuple = (6.0, 45.0)
    heart_band_bpm: tuple = (48.0, 132.0)
    dyn_snr_min_db: float = 6.0     # voxel "vivant" si mouvement > bruit + 6 dB
    num_candidates: int = 8         # voxels candidats par zone (les plus dynamiques)
    null_min_gain: float = 0.15     # voxel ignore si l'annulation des voisins lui laisse < 15 % de gain
    footprint_floor_db: float = 30.0  # empreinte d'une source : cases distance a moins de 30 dB du pic
    leak_drop_db: float = 6.0       # residu = fuite si l'annulation des voisins a retire > 6 dB,
    leak_drop_same_rate_db: float = 3.0  # ... ou > 3 dB avec le meme rythme qu'un siege reconnu
    leak_rate_tol_bpm: float = 1.5
    breath_prom_min: float = 2.0    # pic >= 2x le spectre a +-3 /min (rejette le bruit "rouge")
    apnea_snr_db: float = 4.0       # sous ce niveau sur la fenetre courte -> plus aucun mouvement
    apnea_amp_mm: float = 0.3       # amplitude respiratoire (8 s) sous laquelle on parle d'apnee
    apnea_rel: float = 0.2          # ... ou < 20 % de l'amplitude du debut de fenetre
    breath_q_min: float = 4.0       # pic respiratoire / mediane de la bande
    breath_frac_min: float = 0.35   # part d'energie dans la bande respiratoire
    move_mm: float = 8.0            # ecart-type du deplacement au-dela duquel = gros mouvement
    presence_db: float = 20.0       # changement statique (CLEAN) >= bruit + 20 dB ...            [a calibrer]
    presence_rel_db: float = -32.0  # ... et >= (siege qui change le plus) - 32 dB              [a calibrer]
    clean_max_iter: int = 120
    clean_stop_db: float = 10.0     # arret quand le changement restant < bruit + 10 dB
    clean_rel_stop_db: float = 50.0 # ... ou < changement le plus fort - 50 dB
    angle_step: float = 0.08        # pas de la grille u/w (cosinus directeurs)
    flip_azimuth: bool = False      # a activer si gauche/droite sont inverses sur la vraie carte
    flip_elevation: bool = False    # idem avant/arriere (haut/bas du capteur)
    min_range_bin: int = 3
    max_dir_cos: float = 0.93       # directions retenues : sqrt(u^2+w^2) <= 0.93 (~68 deg)


@dataclass
class ZoneResult:
    zone: int
    state: str = "INIT"
    breath_bpm: float | None = None
    heart_bpm: float | None = None
    breath_quality: float = 0.0
    breath_amp_mm: float = 0.0
    presence_db: float | None = None
    no_breath_s: float = 0.0
    firmware_occ: float | None = None
    note: str = ""

    def as_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class ZoneMap:
    """Voxels (case distance, u, w) dont la position monde tombe dans les cuboides de chaque zone."""

    def __init__(self, cfg: RadarConfig, p: Params):
        self.cfg = cfg
        self.su = -1.0 if p.flip_azimuth else 1.0
        self.sw = -1.0 if p.flip_elevation else 1.0
        grid = np.arange(-1.0, 1.0 + 1e-9, p.angle_step)
        uu, ww = np.meshgrid(grid, grid, indexing="ij")
        ok = uu ** 2 + ww ** 2 <= p.max_dir_cos ** 2
        uu, ww = uu[ok], ww[ok]
        self.zones: dict[int, dict[str, np.ndarray]] = {}
        rbins = np.arange(p.min_range_bin, cfg.num_range_bins)
        R = rbins * cfg.range_step_m - cfg.range_bias_m
        for z, boxes in cfg.cuboids.items():
            rb_l, u_l, w_l = [], [], []
            for rb, r in zip(rbins, R):
                if r <= 0.05:
                    continue
                x, zz = r * uu, r * ww
                y = np.sqrt(np.maximum(r ** 2 - x ** 2 - zz ** 2, 0))
                world = cfg.sensor_to_world(np.stack([x, y, zz], axis=1))
                inside = np.zeros(len(uu), bool)
                for x1, x2, y1, y2, z1, z2 in boxes:
                    inside |= ((world[:, 0] > x1) & (world[:, 0] < x2) & (world[:, 1] > y1) & (world[:, 1] < y2)
                               & (world[:, 2] > z1) & (world[:, 2] < z2))
                rb_l.append(np.full(inside.sum(), rb))
                u_l.append(uu[inside])
                w_l.append(ww[inside])
            rb_a, u_a, w_a = np.concatenate(rb_l), np.concatenate(u_l), np.concatenate(w_l)
            # reponse du reseau a(u,w) = exp(+j*pi*(m*u + n*w)) = conjugue du vecteur directeur TI
            resp = np.conj(cfg.steering(self.su * u_a, self.sw * w_a)).astype(np.complex64)
            rb_a = rb_a.astype(int)
            groups = {int(r): np.flatnonzero(rb_a == r) for r in np.unique(rb_a)}
            self.zones[z] = {"rbin": rb_a, "u": u_a, "w": w_a, "resp": resp, "groups": groups}
        zs = sorted(self.zones)
        self.all_rbin = np.concatenate([self.zones[z]["rbin"] for z in zs])
        self.all_resp = np.concatenate([self.zones[z]["resp"] for z in zs]).astype(np.complex128)
        self.all_zone = np.concatenate([np.full(len(self.zones[z]["rbin"]), z) for z in zs])
        self.all_u = np.concatenate([self.zones[z]["u"] for z in zs])
        self.all_w = np.concatenate([self.zones[z]["w"] for z in zs])

    def beamform(self, x_ra: np.ndarray) -> dict[int, np.ndarray]:
        """Formation de voie classique. x_ra : [range, ant] compense -> {zone: y[voxel]}."""
        n = self.cfg.num_virt_ant
        return {z: np.sum(x_ra[d["rbin"], :] * d["resp"].conj(), axis=1) / n for z, d in self.zones.items()}

    def beamform_series(self, x_tra: np.ndarray) -> dict[int, np.ndarray]:
        """Series temporelles par voxel (formation de voie classique). x_tra : [temps, range, ant]."""
        n = self.cfg.num_virt_ant
        out = {}
        for z, d in self.zones.items():
            y = np.zeros((x_tra.shape[0], len(d["rbin"])), np.complex64)
            for r, idx in d["groups"].items():
                y[:, idx] = x_tra[:, r, :] @ (d["resp"][idx].conj().T / n)
            out[z] = y
        return out

    def describe(self) -> str:
        out = []
        for z, d in self.zones.items():
            rb = d["rbin"]
            out.append(f"zone {z}: {len(rb)} voxels, cases distance {rb.min() if len(rb) else '-'}-"
                       f"{rb.max() if len(rb) else '-'}")
        return " | ".join(out)


def _band_peak(sig: np.ndarray, fs: float, band_bpm: tuple, nfft: int = 2048, with_prominence: bool = False):
    """Pic spectral dans une bande : (frequence /min, qualite pic/mediane, part d'energie bande[, prominence])."""
    n = len(sig)
    s = (sig - sig.mean()) * np.hanning(n)
    spec = np.abs(np.fft.rfft(s, nfft)) ** 2
    f_bpm = np.fft.rfftfreq(nfft, 1 / fs) * 60
    band = (f_bpm >= band_bpm[0]) & (f_bpm <= band_bpm[1])
    total = (f_bpm >= 3) & (f_bpm <= fs * 30 * 0.96)
    if not band.any() or spec[total].sum() <= 0:
        return (None, 0.0, 0.0, 0.0) if with_prominence else (None, 0.0, 0.0)
    idx = np.flatnonzero(band)
    k = idx[np.argmax(spec[idx])]
    if 0 < k < len(spec) - 1:  # interpolation parabolique
        a, b, c = np.log(spec[k - 1:k + 2] + 1e-30)
        delta = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    else:
        delta = 0.0
    f_peak = (k + np.clip(delta, -0.5, 0.5)) * (f_bpm[1] - f_bpm[0])
    q = spec[k] / (np.median(spec[band]) + 1e-30)
    frac = spec[band].sum() / spec[total].sum()
    if not with_prominence:
        return float(f_peak), float(q), float(frac)
    df = f_bpm[1] - f_bpm[0]
    side = [spec[int(round((f_peak + s) / df))] for s in (-3.0, 3.0) if 0 <= int(round((f_peak + s) / df)) < len(spec)]
    prom = spec[k] / (max(side) + 1e-30) if side else 0.0
    return float(f_peak), float(q), float(frac), float(prom)


def _fft_filter(sig: np.ndarray, fs: float, lo_hz: float, hi_hz: float) -> np.ndarray:
    """Passe-bande par FFT avec remplissage de zeros (limite le repliement circulaire)."""
    n = len(sig)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(sig - sig.mean(), nfft)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    spec[(f < lo_hz) | (f > hi_hz)] = 0
    return np.fft.irfft(spec, nfft)[:n]


class VitalsProcessor:
    def __init__(self, cfg: RadarConfig, params: Params | None = None,
                 reference: dict | None = None):
        self.cfg, self.p = cfg, params or Params()
        self.zmap = ZoneMap(cfg, self.p)
        self.fs = cfg.frame_rate_hz
        self.W = int(round(self.p.window_s * self.fs))
        self.Ws = int(round(self.p.short_s * self.fs))
        self.update_every = max(1, int(round(self.p.update_s * self.fs)))
        self.lam = cfg.wavelength_m
        self.reference = reference
        self.nc = max(1, cfg.radcube_num_chirps)       # bursts exportes par image
        self.xbuf = np.zeros((self.W, self.nc, cfg.num_range_bins, cfg.num_virt_ant), np.complex64)
        self.n = 0
        self.last_frame_number = None
        self.dropped = 0
        self.results = {z: ZoneResult(z) for z in self.zmap.zones}
        self.timeline: list[dict] = []
        self.traces: dict[int, list[float]] = {z: [] for z in self.zmap.zones}   # deplacement (mm), pour l'affichage
        self.fw_occ: np.ndarray | None = None
        self._streak: dict[int, int] = {}

    # ------------------------------------------------------------ entree
    def push(self, fr: Frame) -> dict[int, ZoneResult] | None:
        """Ajoute une trame ; renvoie les etats a chaque mise a jour (sinon None)."""
        if fr.radar_cube is None:
            return None
        if self.last_frame_number is not None and fr.frame_number > self.last_frame_number + 1:
            self.dropped += fr.frame_number - self.last_frame_number - 1
        self.last_frame_number = fr.frame_number
        cube = fr.radar_cube
        if cube.shape[0] != self.nc:   # nombre de bursts inattendu : on moyenne / on duplique
            cube = np.repeat(cube.mean(axis=0, keepdims=True), self.nc, axis=0)
        self.xbuf[self.n % self.W] = cube * self.cfg.rx_comp[None, None, :].astype(np.complex64)
        if fr.occupancy is not None:
            self.fw_occ = fr.occupancy
        self.n += 1
        if self.n % self.update_every == 0:
            return self._analyze()
        return None

    def _window(self, length: int) -> np.ndarray:
        length = min(length, self.n, self.W)
        idx = np.arange(self.n - length, self.n) % self.W
        return self.xbuf[idx]

    # ------------------------------------------------------------ analyse
    def _analyze(self) -> dict[int, ZoneResult]:
        p, t_now = self.p, self.n / self.fs
        if t_now < p.min_data_s:
            for z in self.traces:
                self.traces[z].extend([np.nan] * self.update_every)
            self.timeline.append({"t": round(t_now, 2), "zones": {z: r.as_dict() for z, r in self.results.items()}})
            return self.results
        X = self._window(self.W)                              # (T, bursts, range, ant)
        mean = X.mean(axis=(0, 1))
        T = X.shape[0]
        Xd = (X - mean[None, None]).reshape(T * self.nc, *mean.shape).astype(np.complex64)
        series = self.zmap.beamform_series(Xd)
        dyn_all = np.concatenate([np.mean(np.abs(Y) ** 2, axis=0) for Y in series.values()])
        noise = float(np.percentile(dyn_all, 20)) + 1e-12
        presence = self._presence(mean, noise)
        known = [v for v in presence.values() if v is not None]
        p_max = max(known) if known else None
        # 1er passage sans annulation, pour ordonner les sieges du plus "vivant" au moins vivant
        first = {z: self._analyze_zone(Y, noise) for z, Y in series.items()}
        order = sorted(series, key=lambda z: first[z]["dyn_best"], reverse=True)
        # 2e passage : chaque siege est analyse avec un zero vers les sources reelles deja trouvees
        genuine: list[tuple[int, dict[int, np.ndarray]]] = []
        per_zone = {}
        for z in order:
            if genuine:
                Y = self._nulled_series(z, genuine, Xd)
                a = self._analyze_zone(Y, noise)
                if a["breath"]:
                    # residu de fuite ? (meme rythme qu'un siege reel + l'annulation a retire l'essentiel)
                    drop_db = 10 * np.log10((first[z]["dyn_best"] + 1e-30) / (a["dyn_best"] + 1e-30))
                    same = [zz for zz, _ in genuine if per_zone[zz]["f_peak"] is not None
                            and abs(per_zone[zz]["f_peak"] - a["f_peak"]) <= p.leak_rate_tol_bpm]
                    if (same and drop_db > p.leak_drop_same_rate_db) or drop_db > p.leak_drop_db:
                        a.update(breath=False, bpm=None, heart=None, leak_from=same[0] if same else genuine[0][0])
                if first[z]["breath"] and not a["breath"] and a.get("leak_from") is None:
                    a["leak_from"] = genuine[0][0]
            else:
                Y, a = series[z], first[z]
            if a["breath"] or a["moving"]:
                genuine.append((z, self._footprint(Y[:, a["v"]], Xd)))
            a["static"] = presence[z]
            a["static_rel"] = None if presence[z] is None else presence[z] - p_max
            per_zone[z] = a
        for z in sorted(per_zone):
            a = per_zone[z]
            self._decide(z, a)
            d = a["disp"]
            self.traces[z].extend(list(d[-self.update_every:] * 1e3) if d is not None else [np.nan] * self.update_every)
        self.timeline.append({"t": round(t_now, 2), "zones": {z: r.as_dict() for z, r in self.results.items()}})
        return self.results

    def _footprint(self, s: np.ndarray, Xd: np.ndarray) -> dict[int, np.ndarray]:
        """Empreinte d'une source : composante de chaque (case distance, antenne) correlee a son
        signal s(t). Couvre tous ses echos (thorax, abdomen, tete) a chaque distance."""
        h = np.einsum("t,tra->ra", s.conj(), Xd) / (float(np.real(np.vdot(s, s))) + 1e-12)
        pw = np.sum(np.abs(h) ** 2, axis=1)
        keep = pw >= pw.max() * 10 ** (-self.p.footprint_floor_db / 10)
        return {r: h[r].astype(np.complex128) for r in np.flatnonzero(keep)}

    def _nulled_series(self, z: int, genuine: list, Xd: np.ndarray) -> np.ndarray:
        """Formation de voie vers les voxels de z avec un zero vers l'empreinte des sources reelles des
        autres sieges a la meme case distance (projection orthogonale), gain unite vers le voxel vise."""
        d = self.zmap.zones[z]
        n = self.cfg.num_virt_ant
        y = np.zeros((Xd.shape[0], len(d["rbin"])), np.complex64)
        for r, idx in d["groups"].items():
            a = d["resp"][idx].astype(np.complex128)                       # (V_r, n)
            interf = [fp[r] for zz, fp in genuine if zz != z and r in fp and np.linalg.norm(fp[r]) > 0]
            if interf:
                A = np.stack(interf, axis=1)                               # (n, k)
                P = np.eye(n) - A @ np.linalg.pinv(A)
                w = a @ P.T                                                # P a pour chaque voxel
                g = np.einsum("vi,vi->v", w.conj(), a)                     # gain vers le voxel
                ok = np.abs(g) > self.p.null_min_gain * n
                w = np.where(ok[:, None], w / np.where(ok, g, 1)[:, None].conj(), 0)
            else:
                w = a / n
            y[:, idx] = (Xd[:, r, :] @ w.conj().T).astype(np.complex64)
        return y

    def _presence(self, mean: np.ndarray, noise_dyn: float) -> dict[int, float | None]:
        """Changement statique par rapport a l'habitacle vide, par CLEAN : on prend le voxel ou le
        changement est le plus fort (toutes zones confondues), on l'attribue a sa zone, on retire sa
        reponse exacte (lobes secondaires compris) et on recommence. Resultat par zone : composante la
        plus forte, en dB au-dessus du bruit de la difference (moyenne de T images - moyenne de reference)."""
        if not self.reference or "mean_cube" not in self.reference:
            return {z: None for z in self.zmap.zones}
        diff = (mean - self.reference["mean_cube"]).astype(np.complex128)
        zm, n = self.zmap, self.cfg.num_virt_ant
        T = min(self.n, self.W)
        noise_diff = noise_dyn * (1.0 / T + 1.0 / max(self.reference.get("num_frames", T), 1))
        energy = {z: 0.0 for z in zm.zones}
        first = None
        for _ in range(self.p.clean_max_iter):
            pw = np.abs(np.sum(diff[zm.all_rbin] * zm.all_resp.conj(), axis=1) / n) ** 2
            k = int(np.argmax(pw))
            if first is None:
                first = pw[k]
            if pw[k] < max(noise_diff * 10 ** (self.p.clean_stop_db / 10), first * 10 ** (-self.p.clean_rel_stop_db / 10)):
                break
            r = zm.all_rbin[k]
            a = self._refine_direction(diff[r], zm.all_u[k], zm.all_w[k])
            c = np.vdot(a, diff[r]) / n
            diff[r] -= c * a
            zk = int(zm.all_zone[k])
            energy[zk] = max(energy[zk], float(abs(c) ** 2))   # composante la plus forte de la zone
        return {z: float(10 * np.log10(e / noise_diff + 1e-12)) for z, e in energy.items()}

    def _refine_direction(self, x_ant: np.ndarray, u0: float, w0: float) -> np.ndarray:
        """Reponse du reseau pour la direction (u, w) qui maximise la puissance autour de (u0, w0)."""
        st = self.p.angle_step
        g = np.linspace(-st, st, 17)
        uu, ww = np.meshgrid(u0 + g, w0 + g, indexing="ij")
        uu, ww = uu.ravel(), ww.ravel()
        resp = np.conj(self.cfg.steering(self.zmap.su * uu, self.zmap.sw * ww))
        k = int(np.argmax(np.abs(resp.conj() @ x_ant)))
        return resp[k]

    def _analyze_zone(self, Y: np.ndarray, noise: float) -> dict:
        """Y : (T, voxels) series dynamiques de la zone -> respiration / apnee / mouvement."""
        p, fs = self.p, self.fs
        nc = self.nc
        dyn = np.mean(np.abs(Y) ** 2, axis=0)
        Ys = Y[-self.Ws * nc:]
        dyn_short = np.mean(np.abs(Ys - Ys.mean(axis=0)) ** 2, axis=0)
        out = {"breath": False, "bpm": None, "q": 0.0, "amp_mm": 0.0, "moving": False,
               "apnea": True, "disp": None, "dyn_best": float(dyn.max()) if dyn.size else 0.0,
               "heart": None, "v": None, "f_peak": None}
        cand = np.flatnonzero(10 * np.log10(dyn / noise + 1e-30) > p.dyn_snr_min_db)
        if cand.size == 0:
            return out
        cand = cand[np.argsort(dyn[cand])[::-1][:p.num_candidates]]
        tt = np.arange(Y.shape[0] // nc)
        scored = []
        for v in cand:
            # deroulement sur tous les bursts (pas de 40 ms), puis une valeur par image
            phi = np.unwrap(np.angle(Y[:, v])).reshape(-1, nc).mean(axis=1)
            d = phi * self.lam / (4 * np.pi)
            d = d - np.polyval(np.polyfit(tt, d, 1), tt)
            f_bpm, q, frac, prom = _band_peak(d, fs, p.breath_band_bpm, with_prominence=True)
            scored.append((q * frac * np.sqrt(dyn[v]), v, d, f_bpm, q, frac, prom))
        scored.sort(key=lambda s: s[0], reverse=True)
        # vote pondere sur le rythme (comme l'algo TI), puis meilleur candidat proche du vote
        rates = np.array([sc[3] if sc[3] is not None else np.nan for sc in scored])
        weights = np.array([sc[0] for sc in scored])
        chosen = scored[0]
        ok = np.isfinite(rates)
        if ok.any():
            grid = np.arange(p.breath_band_bpm[0], p.breath_band_bpm[1] + 0.5, 0.5)
            votes = [weights[ok][np.abs(rates[ok] - g) <= 1.5].sum() for g in grid]
            f_vote = grid[int(np.argmax(votes))]
            close = [sc for sc in scored if sc[3] is not None and abs(sc[3] - f_vote) <= 1.5]
            if close:
                chosen = close[0]
        _, v, d, f_bpm, q, frac, prom = chosen
        d_b = _fft_filter(d, fs, p.breath_band_bpm[0] / 60, p.breath_band_bpm[1] / 60)
        out.update(disp=d_b, dyn_best=float(dyn[v]), q=q, v=int(v), f_peak=f_bpm,
                   amp_mm=float(np.std(d_b[-self.Ws:]) * 2 * np.sqrt(2) * 1e3))
        out["moving"] = bool(np.std(d[-self.Ws:]) * 1e3 > p.move_mm and frac < 0.6)
        amp_old = float(np.std(d_b[:-self.Ws]) * 2 * np.sqrt(2) * 1e3) if len(d_b) > self.Ws + 10 else 0.0
        no_motion = 10 * np.log10(dyn_short[v] / noise + 1e-30) < p.apnea_snr_db
        weak = out["amp_mm"] < p.apnea_amp_mm or (amp_old > 0 and out["amp_mm"] < p.apnea_rel * amp_old)
        out["apnea"] = bool(no_motion or weak)
        out["breath"] = bool(q >= p.breath_q_min and frac >= p.breath_frac_min and prom >= p.breath_prom_min
                             and not out["apnea"])
        out["bpm"] = f_bpm if out["breath"] else None
        half = len(d) // 2
        if not out["breath"] and not out["apnea"] and half >= self.Ws:
            # seconde chance sur la moitie recente (ex. apres un mouvement qui a perturbe la fenetre)
            d2 = d[-half:]
            t2 = np.arange(half)
            d2 = d2 - np.polyval(np.polyfit(t2, d2, 1), t2)
            f2, q2, frac2, prom2 = _band_peak(d2, fs, p.breath_band_bpm, with_prominence=True)
            if (q2 >= p.breath_q_min and frac2 >= p.breath_frac_min and prom2 >= p.breath_prom_min
                    and f2 is not None and f2 >= p.breath_band_bpm[0] + 2.0):
                out.update(breath=True, bpm=f2, f_peak=f2, q=q2)
        if out["breath"]:
            out["heart"] = self._heart(d, f_bpm)
        return out

    def _heart(self, d: np.ndarray, breath_bpm: float) -> float | None:
        """Experimental : pic cardiaque apres retrait de la respiration et de ses harmoniques."""
        p, fs = self.p, self.fs
        n = len(d)
        spec = np.fft.rfft((d - d.mean()) * np.hanning(n))
        f = np.fft.rfftfreq(n, 1 / fs)
        keep = (f >= p.heart_band_bpm[0] / 60) & (f <= p.heart_band_bpm[1] / 60)
        for k in range(1, 8):
            keep &= np.abs(f - k * breath_bpm / 60) > 0.07
        spec[~keep] = 0
        resid = np.fft.irfft(spec, n)
        h_bpm, h_q, _ = _band_peak(resid, fs, p.heart_band_bpm)
        return h_bpm if h_q >= 8.0 else None

    def _decide(self, z: int, a: dict):
        r, p = self.results[z], self.p
        r.presence_db = a["static"]
        r.breath_quality = a["q"]
        r.breath_amp_mm = a["amp_mm"]
        r.note = f"signal du siege {a['leak_from']}" if a.get("leak_from") is not None else ""
        if self.fw_occ is not None and self.fw_occ.ndim == 2 and z < self.fw_occ.shape[1]:
            r.firmware_occ = float(1 - self.fw_occ[0, z])
        # hysteresis cote securite : il faut confirm_s de respiration continue pour lever une alerte
        self._streak[z] = self._streak.get(z, 0) + 1 if (a["breath"] or a["moving"]) else 0
        confirmed = self._streak[z] * p.update_s >= p.confirm_s
        if a["moving"] and confirmed:
            r.state, r.breath_bpm, r.heart_bpm, r.no_breath_s = "MOUVEMENT", None, None, 0.0
            return
        if a["breath"] and (confirmed or r.state in ("RESPIRE", "MOUVEMENT", "INIT")):
            r.state, r.breath_bpm, r.heart_bpm, r.no_breath_s = "RESPIRE", a["bpm"], a["heart"], 0.0
            return
        r.breath_bpm, r.heart_bpm = (a["bpm"], a["heart"]) if a["breath"] else (None, None)
        occupied = None if r.presence_db is None else (
            r.presence_db >= p.presence_db and a["static_rel"] >= p.presence_rel_db)
        if a["breath"] or a["moving"]:
            # respiration pas encore confirmee : on continue de compter
            r.no_breath_s += p.update_s
            if r.state not in ("SURVEILLANCE", "ALERTE", "INCERTAIN"):
                r.state = "SURVEILLANCE"
            elif r.state == "SURVEILLANCE" and r.no_breath_s >= p.alert_after_s:
                r.state = "ALERTE"
            return
        if occupied is None:
            r.state = "INCERTAIN"
            r.no_breath_s += p.update_s
        elif occupied:
            r.no_breath_s += p.update_s
            r.state = "ALERTE" if r.no_breath_s >= p.alert_after_s else "SURVEILLANCE"
        else:
            r.state, r.no_breath_s = "VIDE", 0.0

def build_reference(cfg: RadarConfig, frames) -> dict:
    """Moyenne complexe du radar cube (compense) sur un enregistrement de l'habitacle vide."""
    acc, count = None, 0
    for fr in frames:
        if fr.radar_cube is None:
            continue
        x = fr.radar_cube.mean(axis=0).astype(np.complex128) * cfg.rx_comp[None, :]   # moyenne des bursts
        acc = x if acc is None else acc + x
        count += 1
    if count == 0:
        raise ValueError("Aucune trame avec radar cube dans l'enregistrement de reference")
    return {"mean_cube": (acc / count).astype(np.complex64), "num_frames": count}


def save_reference(ref: dict, path: str | Path):
    np.savez(path, mean_cube=ref["mean_cube"], num_frames=ref["num_frames"])


def load_reference(path: str | Path) -> dict:
    d = np.load(path)
    return {"mean_cube": d["mean_cube"], "num_frames": int(d["num_frames"])}
