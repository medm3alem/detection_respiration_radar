"""Liaison serie avec l'EVM AWRL6844 (ports XDS110) : detection des ports et envoi du cfg.

Ports de l'EVM sous Windows (gestionnaire de peripheriques) :
    "XDS110 Class Application/User UART (COMx)"  -> commandes, 115 200 bauds
    "XDS110 Class Auxiliary Data Port (COMy)"    -> donnees, 1 250 000 bauds
"""
from __future__ import annotations

import time

BAUD_CLI = 115_200
BAUD_DATA = 1_250_000


def list_ports():
    from serial.tools import list_ports as lp
    ports = list(lp.comports())
    if not ports:
        print("Aucun port serie. Carte branchee sur l'USB XDS110 ? Drivers XDS110 installes (UniFlash ou CCS) ?")
    for p in ports:
        print(f"  {p.device:8s} {p.description}")
    return ports


def guess_ports() -> tuple[str | None, str | None]:
    """(port commandes, port donnees) d'apres la description des ports XDS110."""
    from serial.tools import list_ports as lp
    cli = data = None
    for p in lp.comports():
        d = (p.description or "").lower()
        if "application/user uart" in d or "user uart" in d:
            cli = p.device
        elif "auxiliary data port" in d or "data port" in d:
            data = p.device
    return cli, data


def cfg_lines(cfg_text: str) -> list[str]:
    """Lignes a envoyer : sans commentaires (%) ni lignes vides."""
    return [ln.strip() for ln in cfg_text.splitlines() if ln.strip() and not ln.strip().startswith("%")]


def send_cfg(port: str, lines: list[str], baud: int = BAUD_CLI, verbose: bool = True) -> bool:
    """Envoie le cfg ligne par ligne et attend la fin de la reponse (invite 'mmwDemo:/>') apres
    chaque commande. Une commande est en ERREUR si la reponse contient 'Error', meme suivie de
    'Done' (sensorStart repond toujours 'Done', apres avoir affiche ses erreurs)."""
    import serial
    ok = True
    silent = 0
    with serial.Serial(port, baud, timeout=0.1) as ser:
        ser.reset_input_buffer()
        for ln in lines:
            ser.write((ln + "\n").encode())
            resp = b""
            t0 = time.time()
            limit = 15.0 if ln.startswith("sensorStart") else 2.0   # calibration au demarrage
            while time.time() - t0 < limit:
                resp += ser.read(ser.in_waiting or 1)
                after_echo = resp.split(ln.encode()[:12], 1)[-1]
                if b"/>" in after_echo and (b"Done" in after_echo or b"rror" in after_echo):
                    break
            txt = resp.decode(errors="replace").replace("\r", "")
            status = "ERREUR" if "rror" in txt else ("OK" if "Done" in txt else "?")
            if verbose:
                print(f"  [{status:6s}] {ln}")
                if status != "OK":
                    for rl in txt.split("\n"):
                        rl = rl.strip()
                        if rl and not rl.endswith("/>") and rl != ln.strip() and rl != "Done":
                            print(f"           | {rl}")
            if status == "ERREUR" and not ln.startswith("sensorStop"):
                ok = False
            silent = silent + 1 if not resp else 0
            if silent == 3:
                print("  !! La carte ne repond pas : mode fonctionnel (S7/S8) + appui sur RESET ? "
                      "bon port de commandes ? firmware flashe ?")
                return False
            time.sleep(0.02)
    return ok
