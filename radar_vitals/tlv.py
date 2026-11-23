"""Format des trames UART de la demo in-cabin TI 6844 (modes SBR / CPD).

Repris de mmwave_demo_mss.c/.h et parseDataFromDevice.m (toolbox 4.30) :
- en-tete 40 octets : magic 0x0102 0x0304 0x0506 0x0708, version, totalPacketLen,
  platform, frameNumber, timeCpuCycles, numDetectedObj[2] (uint16), numTLVs, subFrameNumber
- chaque TLV : type (uint32), length (uint32, taille de la charge utile), charge utile
- trame completee a un multiple de 32 octets
- TLV 2007 (radar cube) : 3 x uint32 [numAnt, numRangeBins, numChirps] (ordre reel ecrit par
  mmwave_demo_mss.c, dim[0..2]), puis int16 (imag, reel) ordonnes [chirp][range][tx][rx]
- TLV 3001 (nuage de points, coordonnees spheriques, repere capteur) : 5 float32 d'unites
  [elevation, azimut, doppler, distance, snr], puis par point 8 octets
  int8 elevation, int8 azimut, int16 doppler, uint16 distance, uint16 snr
  (envoye seulement s'il y a au moins un point ; snr en dB au format Q8)
- TLV 1041 (classification) : uint8 Q7, ordre [zone][classe] (classe la plus rapide)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Iterator

import numpy as np

MAGIC = struct.pack("<4H", 0x0102, 0x0304, 0x0506, 0x0708)
HEADER_FMT = "<8s5I2H2I"
HEADER_LEN = struct.calcsize(HEADER_FMT)  # 40
TL_FMT = "<2I"
TL_LEN = 8
SEGMENT_LEN = 32

TLV_STATS = 6
TLV_RX_CHAN_COMP = 318
TLV_OCC_CLASS = 1041
TLV_OCC_HEIGHT = 1042
TLV_RADAR_CUBE = 2007
TLV_MACRO_DOPPLER = 2008
TLV_POINT_CLOUD = 3001
TLV_OCC_FEATURES = 3002

PLATFORM_6844 = 0xA6432  # valeur ecrite par le firmware in-cabin
NUM_ZONES = 5            # zones SBR/CPD de la demo in-cabin

# unites du nuage de points ecrites par le firmware (mmwave_demo_mss.c, MmwDemo_ClassifierTask)
POINT_UNITS = (np.pi / 2 / 127, np.pi / 2 / 127, 0.00028, 0.00025, 1 / 256)  # elev, azim, doppler, range, snr
POINT_DTYPE = np.dtype([("elev", "i1"), ("azim", "i1"), ("doppler", "<i2"), ("range", "<u2"), ("snr", "<u2")])


@dataclass
class Frame:
    frame_number: int
    num_detected_obj: int
    tlv_types: list[int] = field(default_factory=list)
    radar_cube: np.ndarray | None = None          # complex64 [chirp, range, ant]
    points: np.ndarray | None = None              # float [N, 5] : distance m, azimut rad, elevation rad, doppler m/s, snr dB
    occupancy: np.ndarray | None = None           # float [classe, zone] (0..1)
    heights: np.ndarray | None = None             # float [zone]
    rx_chan_comp: np.ndarray | None = None        # float [1 + 2*nAnt]
    macro_doppler: np.ndarray | None = None       # uint32 [fft, zone]
    raw_tlvs: dict[int, bytes] = field(default_factory=dict)


# ---------------------------------------------------------------- decodage
def _decode_radar_cube(payload: bytes, num_ant_hint: int = 16) -> np.ndarray:
    d0, d1, d2 = struct.unpack_from("<3I", payload, 0)
    data = np.frombuffer(payload, dtype="<i2", offset=12)
    cplx = data[1::2].astype(np.float32) + 1j * data[0::2].astype(np.float32)
    if d0 * d1 * d2 == cplx.size and d0 == num_ant_hint:
        n_ant, n_rng, n_chirps = d0, d1, d2              # firmware : [numAnt, numRangeBins, numChirps]
    else:                                                # anciens enregistrements du simulateur
        n_ant, n_rng = num_ant_hint, d0
        n_chirps = cplx.size // max(1, n_rng * n_ant)
    if n_chirps * n_rng * n_ant != cplx.size or n_chirps == 0:
        raise ValueError(f"Taille radar cube incoherente (dims {d0}, {d1}, {d2} ; {cplx.size} valeurs)")
    return cplx.reshape(n_chirps, n_rng, n_ant).astype(np.complex64)


def _decode_point_cloud(payload: bytes) -> np.ndarray:
    if len(payload) < 20:
        return np.zeros((0, 5))
    elev_u, azim_u, dopp_u, rng_u, snr_u = struct.unpack_from("<5f", payload, 0)
    n = (len(payload) - 20) // POINT_DTYPE.itemsize
    p = np.frombuffer(payload, dtype=POINT_DTYPE, count=n, offset=20)
    return np.column_stack([p["range"] * rng_u, p["azim"] * azim_u, p["elev"] * elev_u,
                            p["doppler"] * dopp_u, p["snr"] * snr_u]).astype(float)


def decode_occupancy(payload: bytes, num_zones: int = NUM_ZONES) -> np.ndarray:
    """TLV 1041 -> [classe, zone]. SBR : (vide, occupe) ; CPD : (vide, puis 2 probabilites adulte/enfant)."""
    v = np.frombuffer(payload, dtype=np.uint8).astype(float) / 128.0
    if v.size % num_zones:
        return v
    return v.reshape(num_zones, -1).T.copy()


def decode_packet(packet: bytes, num_ant: int = 16) -> Frame:
    (_, _version, total_len, _platform, frame_number, _cycles,
     n_obj, _n_obj2, num_tlvs, _subframe) = struct.unpack_from(HEADER_FMT, packet, 0)
    fr = Frame(frame_number=frame_number, num_detected_obj=n_obj)
    off = HEADER_LEN
    for _ in range(num_tlvs):
        if off + TL_LEN > len(packet):
            break
        t, length = struct.unpack_from(TL_FMT, packet, off)
        off += TL_LEN
        payload = packet[off:off + length]
        off += length
        fr.tlv_types.append(t)
        if t == TLV_RADAR_CUBE:
            fr.radar_cube = _decode_radar_cube(payload, num_ant)
        elif t == TLV_POINT_CLOUD:
            fr.points = _decode_point_cloud(payload)
        elif t == TLV_OCC_CLASS:
            fr.occupancy = decode_occupancy(payload)
        elif t == TLV_OCC_HEIGHT:
            fr.heights = np.frombuffer(payload, dtype="<f4").copy()
        elif t == TLV_RX_CHAN_COMP:
            fr.rx_chan_comp = np.frombuffer(payload, dtype="<f4").copy()
        elif t == TLV_MACRO_DOPPLER:
            fft_size, n_zones = struct.unpack_from("<2I", payload, 0)
            v = np.frombuffer(payload, dtype="<u4", offset=8)
            if v.size == fft_size * n_zones:
                fr.macro_doppler = v.reshape(n_zones, fft_size).T.copy()
        else:
            fr.raw_tlvs[t] = bytes(payload)
    return fr


class PacketBuffer:
    """Decoupage incremental d'un flux d'octets en trames (pour le direct sur port serie)."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        self.buf.extend(data)
        out = []
        while True:
            idx = self.buf.find(MAGIC)
            if idx < 0:
                del self.buf[:-7]
                break
            if idx > 0:
                del self.buf[:idx]
            if len(self.buf) < HEADER_LEN:
                break
            total_len = struct.unpack_from("<I", self.buf, 12)[0]
            if total_len < HEADER_LEN or total_len > 4_000_000:
                del self.buf[:8]  # faux magic, on continue la recherche
                continue
            if len(self.buf) < total_len:
                break
            out.append(bytes(self.buf[:total_len]))
            del self.buf[:total_len]
        return out


def iter_packets(stream: BinaryIO, chunk: int = 65536) -> Iterator[bytes]:
    """Decoupe un fichier enregistre en trames completes."""
    pb = PacketBuffer()
    while True:
        data = stream.read(chunk)
        if not data:
            break
        yield from pb.feed(data)


def iter_frames(stream: BinaryIO, num_ant: int = 16) -> Iterator[Frame]:
    for pkt in iter_packets(stream):
        try:
            yield decode_packet(pkt, num_ant)
        except (ValueError, struct.error) as exc:
            print(f"[tlv] trame ignoree : {exc}")


# ---------------------------------------------------------------- encodage (simulateur)
def encode_radar_cube(cube: np.ndarray, num_rx: int = 4, num_tx: int = 4) -> bytes:
    """cube complexe [chirp, range, ant] -> charge utile TLV 2007 (int16 imag, reel), en-tete comme le firmware."""
    n_chirps, n_rng, n_ant = cube.shape
    re = np.clip(np.round(cube.real), -32768, 32767).astype("<i2")
    im = np.clip(np.round(cube.imag), -32768, 32767).astype("<i2")
    inter = np.empty(cube.shape + (2,), dtype="<i2")
    inter[..., 0] = im
    inter[..., 1] = re
    return struct.pack("<3I", n_ant, n_rng, n_chirps) + inter.tobytes()


def encode_point_cloud(points: np.ndarray) -> bytes:
    """points [N, 5] (distance, azimut, elevation, doppler, snr dB) -> charge utile TLV 3001."""
    elev_u, azim_u, dopp_u, rng_u, snr_u = POINT_UNITS
    p = np.zeros(len(points), dtype=POINT_DTYPE)
    p["elev"] = np.clip(np.round(points[:, 2] / elev_u), -128, 127)
    p["azim"] = np.clip(np.round(points[:, 1] / azim_u), -128, 127)
    p["doppler"] = np.clip(np.round(points[:, 3] / dopp_u), -32768, 32767)
    p["range"] = np.clip(np.round(points[:, 0] / rng_u), 0, 65535)
    p["snr"] = np.clip(np.round(points[:, 4] / snr_u), 0, 65535)
    return struct.pack("<5f", elev_u, azim_u, dopp_u, rng_u, snr_u) + p.tobytes()


def encode_occupancy(probs: np.ndarray) -> bytes:
    """[classe, zone] (0..1) -> charge utile TLV 1041 (ordre firmware : [zone][classe])."""
    return np.clip(np.round(np.asarray(probs).T * 128), 0, 255).astype(np.uint8).tobytes()


def encode_packet(frame_number: int, tlvs: list[tuple[int, bytes]], num_obj: int = 0) -> bytes:
    body = b"".join(struct.pack(TL_FMT, t, len(p)) + p for t, p in tlvs)
    packet_len = HEADER_LEN + len(body)
    total_len = SEGMENT_LEN * ((packet_len + SEGMENT_LEN - 1) // SEGMENT_LEN)
    header = struct.pack(HEADER_FMT, MAGIC, 0x06010304, total_len, PLATFORM_6844,
                         frame_number, 0, num_obj, 0, len(tlvs), 0xFFFFFFFF)
    return header + body + b"\x00" * (total_len - packet_len)