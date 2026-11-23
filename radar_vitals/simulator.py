"""Simulateur de l'AWRL6844 en cabine : produit un enregistrement UART identique a la carte.

Modele (volontairement simple, a valider avec les vraies donnees) :
- chaque objet = points reflecteurs (position monde, amplitude) ; amplitude ~ sqrt(SER) / R^2
- occupant : thorax + abdomen (bougent avec respiration + coeur), tete, cuisses, jambes
- siege + habitacle : reflecteurs fixes (presents avec ou sans occupant)
- signal ADC reel par antenne virtuelle -> fenetre -> FFT distance -> 64 cases (comme le firmware)
- gain fixe (pas de normalisation par image) pour garder les niveaux statiques comparables
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .cfg_parser import RadarConfig, C
from .nuage import sensor_to_spherical
from . import tlv


@dataclass
class Occupant:
    zone: int
    kind: str = "adulte"              # "adulte" | "enfant"
    breath_bpm: float = 15.0
    breath_mm: float = 5.0            # amplitude crete-crete du thorax
    heart_bpm: float = 70.0
    heart_mm: float = 0.3
    rcs: float = 1.0
    # periodes sans respiration [(debut_s, fin_s)] ; fin=None -> jusqu'a la fin
    apnea: list[tuple[float, float | None]] = field(default_factory=list)
    # periodes de gros mouvements volontaires [(debut_s, fin_s)]
    moving: list[tuple[float, float]] = field(default_factory=list)
    heart_stops_with_breath: bool = False

    def breathing_at(self, t: float) -> bool:
        return not any(a <= t and (b is None or t < b) for a, b in self.apnea)

    def moving_at(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.moving)


@dataclass
class Scene:
    duration_s: float = 120.0
    occupants: list[Occupant] = field(default_factory=list)
    noise_counts: float = 6.0          # bruit apres FFT (ecart-type, en LSB)
    seed: int = 1


def _zone_center(cfg: RadarConfig, zone: int, cub: int) -> np.ndarray:
    boxes = cfg.cuboids[zone]
    x1, x2, y1, y2, z1, z2 = boxes[min(cub, len(boxes) - 1)]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2])


class CabinSimulator:
    def __init__(self, cfg: RadarConfig, scene: Scene):
        self.cfg, self.scene = cfg, scene
        self.rng = np.random.default_rng(scene.seed)
        self.rng_pts = np.random.default_rng(scene.seed + 1000)   # nuage de points (independant)
        self.n_ant = cfg.num_virt_ant
        # erreurs de phase materielles = inverse de la compensation du cfg
        self.hw_phase = 1.0 / cfg.rx_comp
        win = np.hanning(cfg.num_adc_samples + 2)[1:-1]
        self.window = win / win.sum()
        self.t_adc = np.arange(cfg.num_adc_samples) / cfg.fs_hz
        self.gain = self._calibrate_gain()
        self._build_static()
        self._build_occupants()

    # ------------------------------------------------------------ scene
    def _build_static(self):
        """Reflecteurs fixes : sieges (5 zones) + structure de l'habitacle (identiques d'une
        simulation a l'autre : meme voiture)."""
        rng = np.random.default_rng(2024)
        pts, amps = [], []
        for z in range(self.cfg.num_zones):
            back = _zone_center(self.cfg, z, 0) + np.array([0, 0.12, 0])   # dossier, derriere le thorax
            cushion = _zone_center(self.cfg, z, 1)
            for p, a in ((back, 0.25), (cushion, 0.30)):
                for _ in range(3):
                    pts.append(p + rng.normal(0, 0.04, 3))
                    amps.append(a * rng.uniform(0.6, 1.0))
        # habitacle : pavillon, montants, vitres, planche de bord, sol
        for p, a in (([0, 0.3, 1.15], 1.5), ([0.75, 0.9, 1.0], 1.0), ([-0.75, 0.9, 1.0], 1.0),
                     ([0.75, 1.9, 0.9], 0.8), ([-0.75, 1.9, 0.9], 0.8), ([0, -0.2, 0.7], 2.0),
                     ([0, 2.3, 0.9], 1.2), ([0.3, 1.2, 0.1], 0.6), ([-0.3, 1.2, 0.1], 0.6)):
            pts.append(np.array(p, float))
            amps.append(a)
        self.static_pts = np.array(pts)
        self.static_amp = np.array(amps)

    def _build_occupants(self):
        self.occ_parts = []
        for occ in self.scene.occupants:
            chest = _zone_center(self.cfg, occ.zone, 0) + np.array([0, -0.02, 0.05])
            parts = [  # (position, amplitude, couplage respiration, couplage coeur)
                (chest, 1.0, 1.0, 1.0),
                (chest + np.array([0, 0.0, -0.18]), 0.7, 0.6, 0.3),   # abdomen
                (chest + np.array([0, 0.03, 0.25]), 0.5, 0.1, 0.1),   # tete
                (_zone_center(self.cfg, occ.zone, 1), 0.8, 0.05, 0.0),  # cuisses
                (_zone_center(self.cfg, occ.zone, 2), 0.5, 0.0, 0.0),   # jambes
            ]
            scale = np.sqrt(occ.rcs) * (0.6 if occ.kind == "enfant" else 1.0)
            self.occ_parts.append((occ, [(np.asarray(p), a * scale, cb, ch) for p, a, cb, ch in parts]))
        # etats internes : phase de respiration, derive de rythme, balancement
        self._breath_phase = [self.rng.uniform(0, 2 * np.pi) for _ in self.scene.occupants]
        self._heart_phase = [self.rng.uniform(0, 2 * np.pi) for _ in self.scene.occupants]
        self._breath_gain = [1.0 for _ in self.scene.occupants]
        self._sway = [np.zeros(3) for _ in self.scene.occupants]

    # ------------------------------------------------------------ physique
    def _displacements(self, t: float, dt: float) -> list[tuple[float, float, np.ndarray]]:
        """Pour chaque occupant : (deplacement respiration m, deplacement coeur m, decalage corps m)."""
        out = []
        for i, occ in enumerate(self.scene.occupants):
            rate = occ.breath_bpm / 60 * (1 + 0.04 * np.sin(2 * np.pi * t / 37 + i))
            self._breath_phase[i] += 2 * np.pi * rate * dt
            target = 1.0 if occ.breathing_at(t) else 0.0
            self._breath_gain[i] += (target - self._breath_gain[i]) * min(1.0, dt / 1.5)
            th = self._breath_phase[i]
            shape = np.sin(th) + 0.25 * np.sin(2 * th + 0.6)   # inspiration/expiration asymetriques
            d_breath = 0.5 * occ.breath_mm * 1e-3 * shape / 1.1 * self._breath_gain[i]
            alive_heart = occ.breathing_at(t) or not occ.heart_stops_with_breath
            self._heart_phase[i] += 2 * np.pi * occ.heart_bpm / 60 * dt
            hp = self._heart_phase[i] % (2 * np.pi)
            pulse = np.exp(-((hp - 1.0) ** 2) / 0.15) - 0.4 * np.exp(-((hp - 2.2) ** 2) / 0.3)
            d_heart = occ.heart_mm * 1e-3 * pulse * (1.0 if alive_heart else 0.0)
            # balancement lent (vivant) ou gros mouvements
            if occ.moving_at(t):
                self._sway[i] = 0.03 * np.array([np.sin(1.3 * t), np.sin(0.9 * t + 1), 0.5 * np.sin(2.1 * t)])
            else:
                amp = 0.3e-3 if occ.breathing_at(t) else 0.0
                self._sway[i] = 0.98 * self._sway[i] + self.rng.normal(0, amp, 3) * np.sqrt(dt)
            out.append((d_breath, d_heart, self._sway[i].copy()))
        return out

    def _range_fft(self, pts: np.ndarray, amps: np.ndarray, radial_shift: np.ndarray) -> np.ndarray:
        """Spectre distance [range, ant] pour une liste de reflecteurs (positions monde)."""
        cfg = self.cfg
        t_s = cfg.world_to_sensor(pts)                     # (P, 3)
        r = np.linalg.norm(t_s, axis=1) + radial_shift      # deplacement radial (respiration)
        u, w = t_s[:, 0] / np.linalg.norm(t_s, axis=1), t_s[:, 2] / np.linalg.norm(t_s, axis=1)
        visible = t_s[:, 1] > 0.05 * np.linalg.norm(t_s, axis=1)   # devant le capteur
        amp = amps * visible / np.maximum(r, 0.15) ** 2
        tau = 2 * r / C
        f_if = cfg.slope_hz_s * tau                                  # (P,)
        f_start = cfg.start_freq_ghz * 1e9 + cfg.slope_hz_s * cfg.adc_skip_samples / cfg.fs_hz
        phase0 = 2 * np.pi * f_start * tau                           # (P,)
        spatial = np.pi * (np.outer(u, cfg.ant_geom0) + np.outer(w, cfg.ant_geom1))  # (P, A)
        # signal ADC reel : somme de cosinus -> (A, N)
        arg = 2 * np.pi * f_if[:, None, None] * self.t_adc[None, None, :] \
            + phase0[:, None, None] + spatial[:, :, None]
        adc = np.einsum("p,pan->an", amp, np.cos(arg))
        spec = np.fft.fft(adc * self.window[None, :], n=cfg.range_fft_size, axis=1)
        # erreurs de phase des voies (l'inverse de compRangeBiasAndRxChanPhase)
        return spec[:, :cfg.num_range_bins].T * 2 * self.hw_phase[None, :]   # (range, ant)

    def _scene_cube(self, t: float, dt: float) -> np.ndarray:
        pts, amps, shift = [self.static_pts], [self.static_amp], [np.zeros(len(self.static_amp))]
        disp = self._displacements(t, dt)
        for (occ, parts), (d_b, d_h, sway) in zip(self.occ_parts, disp):
            for p, a, cb, ch in parts:
                pts.append((p + sway)[None, :])
                amps.append(np.array([a]))
                shift.append(np.array([cb * d_b + ch * d_h]))
        return self._range_fft(np.vstack(pts), np.concatenate(amps), np.concatenate(shift))

    def _calibrate_gain(self) -> float:
        """Gain fixe (comme la carte) : depend uniquement de l'habitacle, pas des occupants."""
        self._build_static()
        spec = self._range_fft(self.static_pts, self.static_amp, np.zeros(len(self.static_amp)))
        return 2500.0 / max(np.abs(spec).max(), 1e-12)

    # ------------------------------------------------------------ sorties
    def _occupancy_tlv(self, t: float) -> bytes:
        """Imite le TLV 1041 du firmware (3 classes x 5 zones, Q7). Le firmware se fie au mouvement :
        un occupant qui ne respire plus finit classe 'vide'."""
        probs = np.zeros((3, 5))
        probs[0, :] = 0.95
        for occ in self.scene.occupants:
            if occ.zone >= 5:
                continue
            alive = occ.breathing_at(t) or occ.moving_at(t)
            p_occ = 0.9 if alive else 0.1
            p_occ = float(np.clip(p_occ + self.rng.normal(0, 0.03), 0, 1))
            cls = 2 if occ.kind == "enfant" else 1
            probs[:, occ.zone] = 0
            probs[cls, occ.zone] = p_occ
            probs[0, occ.zone] = 1 - p_occ
        return tlv.encode_occupancy(probs)

    def _point_cloud(self, t: float) -> np.ndarray:
        """Imite le TLV 3001 : quelques points sur les occupants vivants (plus s'ils bougent),
        presque rien sur un occupant immobile qui ne respire plus, un peu de fouillis.
        Modele grossier, seulement pour tester l'affichage sans carte."""
        rng = self.rng_pts
        pos, dopp = [], []
        for (occ, parts), sway in zip(self.occ_parts, self._sway):
            if occ.moving_at(t):
                n, sd, dsd = rng.poisson(14), 0.08, 0.25
            elif occ.breathing_at(t):
                n, sd, dsd = rng.poisson(5 if occ.kind == "adulte" else 3), 0.05, 0.02
            else:
                n, sd, dsd = rng.poisson(0.3), 0.05, 0.005
            if n == 0:
                continue
            w = np.array([a * (cb + 0.2) for _, a, cb, _ in parts])
            idx = rng.choice(len(parts), size=n, p=w / w.sum())
            for i in idx:
                pos.append(parts[i][0] + sway + rng.normal(0, sd, 3))
                dopp.append(rng.normal(0, dsd))
        for _ in range(rng.poisson(0.4)):          # fouillis
            pos.append(rng.uniform([-0.8, 0.0, 0.1], [0.8, 2.2, 1.2]))
            dopp.append(rng.normal(0, 0.05))
        if not pos:
            return np.zeros((0, 5))
        xyz_s = self.cfg.world_to_sensor(np.array(pos))
        keep = xyz_s[:, 1] > 0.05
        sph = sensor_to_spherical(xyz_s[keep])
        snr = rng.uniform(8, 22, keep.sum())
        return np.column_stack([sph, np.array(dopp)[keep], snr])

    def run(self, out_bin: str | Path, truth_json: str | Path | None = None, progress: bool = True) -> int:
        cfg = self.cfg
        if not cfg.radcube_export_enabled or cfg.radcube_num_chirps == 0:
            raise ValueError("Le cfg doit activer dbgGuiMonitor (radCubeFreshChunk) et exportRadCubeChunk")
        n_frames = int(self.scene.duration_s * cfg.frame_rate_hz)
        burst_dt = cfg.burst_period_us * 1e-6
        chirp_idx = cfg.exported_chirp_indices()
        truth = []
        t_prev = 0.0
        with open(out_bin, "wb") as fh:
            for k in range(n_frames):
                t0 = k / cfg.frame_rate_hz
                chirps = []
                for ci in chirp_idx:
                    t = t0 + ci * burst_dt
                    spec = self._scene_cube(t, max(t - t_prev, 1e-3))
                    t_prev = t
                    noise = self.rng.normal(0, self.scene.noise_counts, spec.shape) \
                        + 1j * self.rng.normal(0, self.scene.noise_counts, spec.shape)
                    chirps.append(spec * self.gain + noise)
                cube = np.stack(chirps)          # [chirp, range, ant]
                pts = self._point_cloud(t0)
                tlvs = [(tlv.TLV_POINT_CLOUD, tlv.encode_point_cloud(pts))] if len(pts) else []
                tlvs += [(tlv.TLV_RADAR_CUBE, tlv.encode_radar_cube(cube, cfg.num_rx, cfg.num_tx)),
                         (tlv.TLV_OCC_CLASS, self._occupancy_tlv(t0))]
                fh.write(tlv.encode_packet(k, tlvs, num_obj=len(pts)))
                truth.append({"t": round(t0, 3), "zones": {
                    str(o.zone): {"breathing": o.breathing_at(t0), "moving": o.moving_at(t0),
                                  "breath_bpm": o.breath_bpm, "heart_bpm": o.heart_bpm, "kind": o.kind}
                    for o in self.scene.occupants}})
                if progress and k % 100 == 0:
                    print(f"  simulation {k}/{n_frames} images")
        if truth_json:
            Path(truth_json).write_text(json.dumps({
                "frame_rate_hz": cfg.frame_rate_hz,
                "occupants": [asdict(o) for o in self.scene.occupants],
                "frames": truth}, indent=1))
        return n_frames
