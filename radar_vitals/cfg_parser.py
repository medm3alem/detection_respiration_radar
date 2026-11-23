"""Lecture d'un fichier .cfg de la demo in-cabin TI (AWRL6844) et parametres derives.

Conventions reprises des sources TI (toolbox 4.30) :
- Fs = 200 MHz / DigOutputSampRate (xWRL6844, cf. guide de migration x432 -> 6844)
- echantillonnage reel : numRangeBins = rangeFftSize / 2
- antGeometry0 = indice azimut (m), antGeometry1 = indice elevation (n), en lambda/2
- vecteur directeur TI : exp(-j*pi*(m*u + n*w)), u = x/R, w = z/R (repere capteur)
- repere capteur : x lateral, y axe de visee, z elevation
- monde = Rz(azim) * Ry(0) * Rx(elev) * capteur + sensorPosition  (occupancy_demo_gui.m)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

C = 299_792_458.0


def _rot_tw(elev_rad: float, yaw_rad: float, azim_rad: float) -> np.ndarray:
    ce, se = np.cos(elev_rad), np.sin(elev_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    ca, sa = np.cos(azim_rad), np.sin(azim_rad)
    rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass
class RadarConfig:
    lines: list[str]
    # front-end
    rx_mask: int = 15
    tx_mask: int = 15
    dig_out_samp_rate: float = 50
    num_adc_samples: int = 128
    ramp_end_us: float = 38.25
    idle_us: float = 4.0
    adc_skip_samples: int = 28
    slope_mhz_us: float = 102.0
    start_freq_ghz: float = 57.5
    chirps_per_burst: int = 4
    chirps_accum: int = 12
    burst_period_us: float = 4000.0
    bursts_per_frame: int = 32
    frame_period_ms: float = 200.0
    running_mode: int = 2
    # antennes
    ant_geom0: np.ndarray = field(default_factory=lambda: np.zeros(16))
    ant_geom1: np.ndarray = field(default_factory=lambda: np.zeros(16))
    range_bias_m: float = 0.0
    rx_comp: np.ndarray = field(default_factory=lambda: np.ones(16, complex))
    # montage
    sensor_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    azim_tilt_deg: float = 0.0
    elev_tilt_deg: float = 0.0
    # zones : {zone: [(x1,x2,y1,y2,z1,z2), ...]}
    cuboids: dict[int, list[tuple[float, ...]]] = field(default_factory=dict)
    # debug export
    radcube_export_enabled: bool = False
    radcube_chirp_start: int = 0
    radcube_chirp_step: int = 1
    radcube_num_chirps: int = 0

    # ---------- grandeurs derivees ----------
    @property
    def num_rx(self) -> int:
        return bin(self.rx_mask & 0xF).count("1") if self.rx_mask <= 0xF else 4

    @property
    def num_tx(self) -> int:
        return bin(self.tx_mask & 0xF).count("1") if self.tx_mask <= 0xF else 4

    @property
    def num_virt_ant(self) -> int:
        return self.num_rx * self.num_tx

    @property
    def fs_hz(self) -> float:
        return 200e6 / self.dig_out_samp_rate

    @property
    def slope_hz_s(self) -> float:
        return self.slope_mhz_us * 1e12

    @property
    def range_fft_size(self) -> int:
        return 1 << int(np.ceil(np.log2(self.num_adc_samples)))

    @property
    def num_range_bins(self) -> int:
        return self.range_fft_size // 2

    @property
    def range_step_m(self) -> float:
        return C * self.fs_hz / (2 * self.slope_hz_s * self.range_fft_size)

    @property
    def bandwidth_hz(self) -> float:
        return self.slope_hz_s * self.num_adc_samples / self.fs_hz

    @property
    def center_freq_hz(self) -> float:
        t_center = (self.adc_skip_samples + self.num_adc_samples / 2) / self.fs_hz
        return self.start_freq_ghz * 1e9 + self.slope_hz_s * t_center

    @property
    def wavelength_m(self) -> float:
        return C / self.center_freq_hz

    @property
    def frame_rate_hz(self) -> float:
        return 1000.0 / self.frame_period_ms

    @property
    def rot_tw(self) -> np.ndarray:
        return _rot_tw(np.radians(self.elev_tilt_deg), 0.0, np.radians(self.azim_tilt_deg))

    @property
    def num_zones(self) -> int:
        return (max(self.cuboids) + 1) if self.cuboids else 0

    def exported_chirp_indices(self) -> list[int]:
        return [self.radcube_chirp_start + i * self.radcube_chirp_step for i in range(self.radcube_num_chirps)]

    # ---------- conversions de repere ----------
    def sensor_to_world(self, pts_sensor: np.ndarray) -> np.ndarray:
        """pts (..., 3) repere capteur -> monde."""
        return pts_sensor @ self.rot_tw.T + self.sensor_pos

    def world_to_sensor(self, pts_world: np.ndarray) -> np.ndarray:
        return (np.asarray(pts_world) - self.sensor_pos) @ self.rot_tw

    def steering(self, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Vecteurs directeurs TI, shape (len(u), num_virt_ant)."""
        u = np.atleast_1d(u)[:, None]
        w = np.atleast_1d(w)[:, None]
        return np.exp(-1j * np.pi * (self.ant_geom0[None, :] * u + self.ant_geom1[None, :] * w))

    def summary(self) -> str:
        return (
            f"{self.num_tx}TX x {self.num_rx}RX = {self.num_virt_ant} antennes virtuelles | "
            f"Fs={self.fs_hz/1e6:.1f} MHz, {self.num_adc_samples} ech., B={self.bandwidth_hz/1e9:.2f} GHz | "
            f"{self.num_range_bins} cases x {self.range_step_m*100:.2f} cm = {self.num_range_bins*self.range_step_m:.2f} m | "
            f"lambda={self.wavelength_m*1e3:.2f} mm | {self.frame_rate_hz:.1f} images/s | "
            f"{self.num_zones} zones | export radar cube: "
            + (f"bursts {self.exported_chirp_indices()}" if self.radcube_export_enabled else "desactive")
        )


def parse_cfg(path: str | Path) -> RadarConfig:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines()]
    cfg = RadarConfig(lines=[ln for ln in lines if ln and not ln.startswith("%")])
    for ln in cfg.lines:
        tok = ln.split()
        cmd, a = tok[0], tok[1:]
        try:
            if cmd == "channelCfg":
                cfg.rx_mask, cfg.tx_mask = int(a[0], 0), int(a[1], 0)
            elif cmd == "chirpComnCfg":
                cfg.dig_out_samp_rate = float(a[0])
                cfg.num_adc_samples = int(a[3])
                cfg.ramp_end_us = float(a[5])
            elif cmd == "chirpTimingCfg":
                cfg.idle_us = float(a[0])
                cfg.adc_skip_samples = int(a[1])
                cfg.slope_mhz_us = float(a[3])
                cfg.start_freq_ghz = float(a[4])
            elif cmd == "frameCfg":
                cfg.chirps_per_burst = int(a[0])
                cfg.chirps_accum = int(a[1])
                cfg.burst_period_us = float(a[2])
                cfg.bursts_per_frame = int(a[3])
                cfg.frame_period_ms = float(a[4])
            elif cmd == "runningMode":
                cfg.running_mode = int(a[0])
            elif cmd == "antGeometry0":
                cfg.ant_geom0 = np.array([float(v) for v in a])
            elif cmd == "antGeometry1":
                cfg.ant_geom1 = np.array([float(v) for v in a])
            elif cmd == "compRangeBiasAndRxChanPhase":
                vals = [float(v) for v in a]
                cfg.range_bias_m = vals[0]
                pairs = np.array(vals[1:]).reshape(-1, 2)
                cfg.rx_comp = pairs[:, 0] + 1j * pairs[:, 1]
            elif cmd == "sensorPosition":
                cfg.sensor_pos = np.array([float(a[0]), float(a[1]), float(a[2])])
                cfg.azim_tilt_deg = float(a[3])
                cfg.elev_tilt_deg = float(a[4])
            elif cmd == "cuboidDef":
                zone = int(a[0])
                box = tuple(float(v) for v in a[2:8])
                cfg.cuboids.setdefault(zone, []).append(box)
            elif cmd == "dbgGuiMonitor":
                cfg.radcube_export_enabled = len(a) >= 9 and int(a[8]) == 1
            elif cmd == "exportRadCubeChunk":
                cfg.radcube_chirp_start = int(a[0])
                cfg.radcube_chirp_step = int(a[1])
                cfg.radcube_num_chirps = int(a[2])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Ligne de cfg invalide : '{ln}' ({exc})") from exc
    if len(cfg.ant_geom0) != cfg.num_virt_ant or len(cfg.ant_geom1) != cfg.num_virt_ant:
        raise ValueError("antGeometry0/1 doivent avoir une valeur par antenne virtuelle")
    if len(cfg.rx_comp) != cfg.num_virt_ant:
        raise ValueError("compRangeBiasAndRxChanPhase doit contenir 1 + 2 x nb antennes virtuelles valeurs")
    return cfg


def with_radcube_export(cfg_text: str, chirp_start: int = 0, chirp_step: int = 10, num_chirps: int = 3) -> str:
    """Ajoute dbgGuiMonitor / exportRadCubeChunk juste avant sensorStart (si absents)."""
    out, inserted = [], False
    has_dbg = any(ln.strip().startswith("dbgGuiMonitor") for ln in cfg_text.splitlines())
    has_exp = any(ln.strip().startswith("exportRadCubeChunk") for ln in cfg_text.splitlines())
    for ln in cfg_text.splitlines():
        if ln.strip().startswith("sensorStart") and not inserted:
            out.append("% --- ajout hackathon : export du radar cube pour les signes vitaux ---")
            if not has_dbg:
                out.append("dbgGuiMonitor 0 0 0 0 0 0 0 0 1")
            if not has_exp:
                out.append(f"exportRadCubeChunk {chirp_start} {chirp_step} {num_chirps}")
            inserted = True
        out.append(ln)
    if not inserted:
        raise ValueError("Pas de ligne sensorStart dans le cfg")
    return "\n".join(out) + "\n"
