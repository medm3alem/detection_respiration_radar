"""Lecture des trames dans un fil a part : depuis la carte (port serie) ou depuis un enregistrement."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from . import tlv


class Lecteur(threading.Thread):
    """Lit les trames dans un fil a part et les depose dans une file."""

    def __init__(self, num_ant: int):
        super().__init__(daemon=True)
        self.num_ant = num_ant
        self.q: queue.Queue = queue.Queue(maxsize=500)
        self.stop_evt = threading.Event()
        self.pret = threading.Event()
        self.fini = threading.Event()
        self.n_bytes = 0
        self.n_frames = 0
        self.n_bad = 0
        self.erreur: Exception | None = None
        self.bloquant = False

    def _push_packet(self, pkt: bytes):
        try:
            fr = tlv.decode_packet(pkt, self.num_ant)
        except Exception:      # trame tronquee ou TLV inattendu : on la saute
            self.n_bad += 1
            return
        fr.t_recv = time.time()   # heure de reception (pour le debit)
        self.n_frames += 1
        if self.bloquant:          # relecture de fichier : on attend l'affichage, on ne jette rien
            while not self.stop_evt.is_set():
                try:
                    self.q.put(fr, timeout=0.2)
                    return
                except queue.Full:
                    continue
            return
        try:
            self.q.put_nowait(fr)
        except queue.Full:     # affichage trop lent : on jette la plus ancienne
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put_nowait(fr)


class LecteurSerie(Lecteur):
    def __init__(self, port: str, baud: int, sortie: Path | None, num_ant: int):
        super().__init__(num_ant)
        self.port, self.baud, self.sortie = port, baud, sortie

    def run(self):
        import serial
        ser = fh = None
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.1)
            ser.reset_input_buffer()
            fh = open(self.sortie, "wb") if self.sortie else None
            self.pret.set()
            pb = tlv.PacketBuffer()
            while not self.stop_evt.is_set():
                chunk = ser.read(8192)
                if not chunk:
                    continue
                self.n_bytes += len(chunk)
                if fh:
                    fh.write(chunk)
                for pkt in pb.feed(chunk):
                    self._push_packet(pkt)
        except Exception as exc:  # port occupe, carte debranchee...
            self.erreur = exc
        finally:
            self.pret.set()
            if fh:
                fh.close()
            if ser:
                ser.close()
            self.fini.set()


class LecteurFichier(Lecteur):
    def __init__(self, path: Path, frame_rate: float, vitesse: float, num_ant: int):
        super().__init__(num_ant)
        self.path, self.dt = path, (1.0 / (frame_rate * vitesse) if vitesse > 0 else 0.0)
        self.bloquant = True

    def run(self):
        try:
            self.pret.set()
            with open(self.path, "rb") as fh:
                for pkt in tlv.iter_packets(fh):
                    if self.stop_evt.is_set():
                        break
                    self.n_bytes += len(pkt)
                    self._push_packet(pkt)
                    if self.dt:
                        time.sleep(self.dt)
        except Exception as exc:
            self.erreur = exc
        finally:
            self.fini.set()
