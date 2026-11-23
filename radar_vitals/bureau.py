"""Essai sur table : carte de l'environnement et respiration des personnes devant le radar.

Pas de zones voiture ni de reference a vide : on travaille dans le repere capteur.
A partir du radar cube (TLV 2007, 3 bursts par image) sur une fenetre glissante :

- carte statique : puissance de la moyenne temporelle, par case distance x direction
  (ce qui renvoie l'onde : murs, meubles, la personne) ;
- carte des micro-mouvements : puissance des variations dans la bande respiratoire (6 a 45 /min) ;
- cibles : les maxima de la carte des micro-mouvements, separes d'au moins 50 cm (une par personne,
  jusqu'a 4) ; chaque cible est suivie separement (piste P1, P2...) ;
- respiration : phase du signal de la cible deroulee sur les bursts (pas de 40 ms), convertie en
  deplacement du thorax (lambda / 4 pi), puis pic du spectre entre 6 et 45 /min ;
- pause / apnee (par personne) : une fois la respiration confirmee, la position est verrouillee et on surveille
  l'amplitude crete a crete sur des fenetres courtes (6 s). Si elle tombe sous 30 % de l'amplitude
  habituelle, on compte la duree de la pause ; au-dela de 10 s, c'est une apnee (alerte).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .cfg_parser import RadarConfig
from .processing import _band_peak, _fft_filter
from .tlv import Frame


@dataclass
class ParamsBureau:
    fenetre_s: float = 20.0          # fenetre d'analyse
    min_s: float = 8.0               # duree minimale avant de donner un rythme
    r_min: float = 0.3               # zone de recherche de la personne (m)
    r_max: float = 2.5
    u_max: float = 0.85              # |sin(azimut)| maximal de la grille
    w_max: float = 0.6               # |sin(elevation)| maximal de la grille
    pas_u: float = 0.05
    pas_w: float = 0.1
    bande_bpm: tuple = (6.0, 45.0)
    q_min: float = 4.0               # pic / mediane du spectre dans la bande
    frac_min: float = 0.35           # part de l'energie dans la bande
    prom_min: float = 2.0            # pic / voisins a +-3 /min
    amplitude_min_mm: float = 0.3    # mouvement crete a crete minimal (chaise vide : ~0,0 mm ; personne : 4-7 mm)
    pause_fenetre_s: float = 6.0     # fenetre courte pour l'amplitude instantanee
    pause_ratio: float = 0.3         # pause si amplitude courte < 30 % de l'amplitude habituelle
    reference_s: float = 60.0        # amplitude habituelle = quartile bas des 60 dernieres secondes
    faible_min_mm: float = 0.5       # respiration faible : au moins 0,5 mm crete a crete sur 6 s ...
    faible_q_min: float = 8.0        # ... un pic net sur les 10 dernieres secondes ...
    faible_ecart_bpm: float = 3.0    # ... au meme rythme que le dernier rythme confirme (+-3 /min)
    apnee_s: float = 10.0            # pause plus longue = apnee (alerte)
    confirmation_s: float = 7.0      # pendant une pause, un mouvement doit durer 7 s pour compter comme une reprise
                                     # (un mouvement isole, deglutition, sursaut, ne remet pas le compteur a zero)
    suivi_max_m: float = 0.25        # la cible verrouillee peut glisser jusqu'a 25 cm (la personne bouge)
    changement_db: float = 3.0       # on change de cible si une autre case est plus forte de 3 dB
    max_cibles: int = 4              # nombre maximal de personnes suivies
    separation_m: float = 0.5        # deux personnes sont a plus de 50 cm l'une de l'autre
    seuil_cible_db: float = 10.0     # une nouvelle cible bouge au moins 10 fois plus que le fond ...
    relatif_db: float = 40.0         # ... et au plus 40 dB sous la plus forte (un passager arriere renvoie
                                     # bien moins ; les lobes secondaires sont ecartes ensuite comme echos)
    abandon_s: float = 30.0          # une cible jamais confirmee est oubliee au bout de 30 s ...
    perdue_s: float = 8.0            # ... ou des qu'elle n'est plus un maximum de micro-mouvement depuis 8 s
    confirmation_cible_s: float = 2.0  # une nouvelle personne : rythme net pendant 2 s d'affilee ...
    nouvelle_amplitude_mm: float = 0.8   # ... et au moins 0,8 mm de mouvement (sinon : bruit, cuisses...)
    fusion_corr: float = 0.8         # meme rythme et mouvement identique : echo d'une personne deja suivie
    trou_max: int = 5                # trames perdues comblees (au-dela : on repart de zero)
    flip_azimuth: bool = False
    flip_elevation: bool = False


class Piste:
    """Une personne suivie : position, respiration, pause et apnee."""

    def __init__(self, pos: tuple[int, int], t: float):
        self.num: int | None = None                   # numero donne a la confirmation (P1, P2...)
        self.pos = pos                                # (case distance, indice direction)
        self.confirmee = False                        # respiration confirmee : position verrouillee
        self.t_creation = t
        self.amp_ref: deque = deque()                 # (instant s, amplitude mm) des analyses "respire", 60 s
        self.debut_pause: float | None = None         # instant (s, temps du flux) du debut de la pause
        self.fin_pause: float | None = None           # instant de la derniere reprise apres une pause
        self.mouvement_depuis: float | None = None    # pendant une pause : debut d'un mouvement a confirmer
        self.dernier_bpm: float | None = None         # dernier rythme confirme (respiration faible)
        self.ok_depuis: float | None = None           # debut de la serie d'analyses "rythme net" (confirmation)
        self.vu: float = t                            # dernier instant ou c'etait un maximum de micro-mouvement

    @property
    def nom(self) -> str:
        return f"P{self.num}" if self.num is not None else "?"


class CarteRespiration:
    def __init__(self, cfg: RadarConfig, p: ParamsBureau | None = None):
        if not cfg.radcube_export_enabled:
            raise ValueError("Le cfg n'exporte pas le radar cube : utiliser cfg/bureau_vitals.cfg "
                             "(lignes dbgGuiMonitor ... 1 et exportRadCubeChunk)")
        self.cfg, self.p = cfg, p or ParamsBureau()
        p = self.p
        self.fs = cfg.frame_rate_hz
        self.nc = max(1, cfg.radcube_num_chirps)
        self.R, self.A = cfg.num_range_bins, cfg.num_virt_ant
        self.W = int(round(p.fenetre_s * self.fs))
        self.buf = np.zeros((self.W, self.nc, self.R, self.A), np.complex64)
        self.n = 0
        self.last_fn: int | None = None
        self.perdues = 0
        self.redemarrages = 0
        self.gigue = 0                               # numeros de trame decales sans perte
        self.lam = cfg.wavelength_m
        self.u = np.arange(-p.u_max, p.u_max + 1e-9, p.pas_u)
        self.w = np.arange(-p.w_max, p.w_max + 1e-9, p.pas_w)
        uu, ww = np.meshgrid(self.u, self.w, indexing="ij")
        self.uu, self.ww = uu.ravel(), ww.ravel()
        su = -1.0 if p.flip_azimuth else 1.0
        sw = -1.0 if p.flip_elevation else 1.0
        # y = x . steering / N  (meme convention que processing.ZoneMap)
        self.S = (cfg.steering(su * self.uu, sw * self.ww) / self.A).astype(np.complex64)   # (D, A)
        self.ranges = np.arange(self.R) * cfg.range_step_m - cfg.range_bias_m
        valid_dir = self.uu ** 2 + self.ww ** 2 < 0.95
        self.masque = ((self.ranges >= p.r_min) & (self.ranges <= p.r_max))[:, None] & valid_dir[None, :]
        rr = np.maximum(self.ranges, 0.0)[:, None]
        self.XYZ = np.stack([rr * self.uu[None, :],
                             np.sqrt(np.maximum(rr ** 2 * (1 - self.uu ** 2 - self.ww ** 2)[None, :], 0.0)),
                             rr * self.ww[None, :]], axis=-1)            # (R, D, 3) position de chaque case
        self.pistes: list[Piste] = []
        self.n_confirmees = 0
        self.fantomes: list = []                      # positions de doublons recents (lobes, echos), a ignorer
        self.t_flux = 0                               # trames depuis le debut (ne repart pas a zero)

    # ------------------------------------------------------------------ entree
    def push(self, fr: Frame) -> bool:
        if fr.radar_cube is None:
            return False
        cube = fr.radar_cube
        if cube.shape != (self.nc, self.R, self.A):
            if cube.shape[1:] != (self.R, self.A):
                return False
            cube = np.repeat(cube.mean(axis=0, keepdims=True), self.nc, axis=0)
        cube = cube * self.cfg.rx_comp[None, None, :].astype(np.complex64)
        if self.last_fn is not None:
            # Le numero de trame de l'en-tete est lu par le firmware au moment de l'envoi UART :
            # il peut sauter de 2 puis se repeter alors qu'aucune trame n'est perdue.
            # On tolere donc -3..+2 (gigue), on comble 3..trou_max, et on repart de zero au-dela.
            gap = fr.frame_number - self.last_fn - 1
            if gap > self.p.trou_max or gap < -3:     # gros trou ou redemarrage du capteur
                self.n = 0
                self.redemarrages += 1
            elif gap >= 2:                             # vrai trou : on repete la derniere image
                self.perdues += gap
                last = self.buf[(self.n - 1) % self.W].copy()
                for _ in range(gap):
                    self.buf[self.n % self.W] = last
                    self.n += 1
                    self.t_flux += 1
            elif gap != 0:
                self.gigue += 1
        self.last_fn = fr.frame_number
        self.buf[self.n % self.W] = cube
        self.n += 1
        self.t_flux += 1
        return True

    @property
    def duree_s(self) -> float:
        return min(self.n, self.W) / self.fs

    def _fenetre(self) -> np.ndarray:
        T = min(self.n, self.W)
        idx = np.arange(self.n - T, self.n) % self.W
        return self.buf[idx]

    # ------------------------------------------------------------------ analyse
    def analyser(self) -> dict | None:
        p = self.p
        X = self._fenetre()                         # (T, nc, R, A)
        T = X.shape[0]
        if T < 4:
            return None
        Xf = X.mean(axis=1)                         # une valeur par image (T, R, A)
        m = Xf.mean(axis=0)                         # (R, A)
        stat = np.abs(m @ self.S.T) ** 2            # (R, D)

        # micro-mouvements dans la bande respiratoire
        Xd = (Xf - m) * np.hanning(T)[:, None, None].astype(np.float32)
        nfft = max(64, 1 << int(np.ceil(np.log2(T))))
        F = np.fft.fft(Xd, nfft, axis=0)            # signal complexe : frequences + et -
        f_bpm = np.abs(np.fft.fftfreq(nfft, 1 / self.fs) * 60)
        band = (f_bpm >= p.bande_bpm[0]) & (f_bpm <= p.bande_bpm[1])
        Yb = np.einsum("fra,da->frd", F[band], self.S, optimize=True)
        dyn = np.sum(np.abs(Yb) ** 2, axis=0)       # (R, D)

        nu, nw = len(self.u), len(self.w)
        res = {
            "duree_s": T / self.fs, "fenetre_s": p.fenetre_s, "perdues": self.perdues,
            "ranges": self.ranges, "u": self.u, "w": self.w,
            "carte_stat": stat.reshape(self.R, nu, nw).max(axis=2),
            "carte_dyn": dyn.reshape(self.R, nu, nw).max(axis=2),
            "cible": None, "bpm": None, "respire": False,
        }

        dm = np.where(self.masque, dyn, 0.0)
        if dm.max() <= 0:
            return res
        bruit = float(np.median(dm[self.masque])) + 1e-30
        maintenant = self.t_flux / self.fs

        # 1) personnes deja confirmees : analysees en premier (leur mouvement sert a reconnaitre les echos)
        confirmes = [pi for pi in self.pistes if pi.confirmee]
        res_conf = {id(pi): self._analyser_piste(pi, X, dm, bruit, maintenant) for pi in confirmes}
        # deux pistes confirmees qui sont la meme personne (elles se rejoignent, ou l'une est l'echo exact
        # de l'autre : correlation >= 0,9 au meme rythme) : on garde la plus ancienne
        for a in list(confirmes):
            for b in list(confirmes):
                if a is b or a not in confirmes or b not in confirmes or a.num > b.num:
                    continue
                d = self._dist(a.pos, b.pos)
                if d < p.suivi_max_m or self._echo_de(res_conf[id(b)], res_conf[id(a)], d, strict=True):
                    confirmes.remove(b)
        # 2) candidats : maxima de micro-mouvements (pas avant 8 s de donnees : la carte n'a pas de sens)
        candidats = self._candidats(dm, bruit) if T / self.fs >= p.min_s else []
        tentatives = [pi for pi in self.pistes if not pi.confirmee]
        for pi in tentatives:                  # elles suivent le candidat le plus fort a moins de 50 cm
            proches = [c for c in candidats if self._dist(c, pi.pos) <= p.separation_m]
            if proches:
                pi.vu = maintenant
                if dm[proches[0]] > dm[pi.pos] * 10 ** (p.changement_db / 10):
                    pi.pos = proches[0]
        # 3) un candidat loin de toutes les pistes = une nouvelle personne possible, sauf si son mouvement
        #    est celui d'une personne deja suivie (lobe secondaire du reseau, echo sur une paroi, cuisses...)
        for c in candidats:
            if len(confirmes) >= p.max_cibles or len(tentatives) >= p.max_cibles:
                break
            if any(self._dist(c, pi.pos) <= p.separation_m for pi in confirmes + tentatives):
                continue
            mc = self._mesure(X, *c)
            if any(self._echo_de(mc, res_conf[id(pi)], self._dist(c, pi.pos)) for pi in confirmes):
                self.fantomes.append((c, maintenant + p.abandon_s))
                continue
            tentatives.append(Piste(c, maintenant))
        # 4) nouvelles personnes possibles : analyse, puis echos et doublons ecartes
        garder_t = []
        res_t = {}
        r_tent = {id(pi): self._analyser_piste(pi, X, dm, bruit, maintenant) for pi in tentatives}
        for pi in sorted(tentatives, key=lambda q: -r_tent[id(q)]["contraste_db"]):   # la plus forte d'abord
            r = r_tent[id(pi)]
            echo = any(self._echo_de(r, res_conf[id(q)], self._dist(pi.pos, q.pos)) for q in confirmes) \
                or any(self._echo_de(r, res_t[id(q)], self._dist(pi.pos, q.pos)) for q in garder_t)
            double = any(self._dist(pi.pos, q.pos) <= p.separation_m for q in garder_t)
            vieille = (maintenant - pi.t_creation >= p.abandon_s and not pi.confirmee) \
                or maintenant - pi.vu >= p.perdue_s
            if echo or double:
                self.fantomes.append((pi.pos, maintenant + p.abandon_s))
            elif not vieille or pi.confirmee:
                garder_t.append(pi)
                res_t[id(pi)] = r
        self.fantomes = [(pos, t) for pos, t in self.fantomes if t > maintenant]
        self.pistes = confirmes + garder_t
        tous = {**res_conf, **res_t}
        # numeros P1, P2... dans l'ordre de confirmation
        for pi in sorted(self.pistes, key=lambda q: -tous[id(q)]["contraste_db"]):
            if pi.confirmee and pi.num is None and self.n_confirmees_actives() >= p.max_cibles:
                pi.confirmee = False                   # deja assez de personnes suivies
                tous[id(pi)].update(respire=False, bpm=None, etat="incertain")
            if pi.confirmee and pi.num is None:
                self.n_confirmees += 1
                pi.num = self.n_confirmees
            tous[id(pi)]["num"], tous[id(pi)]["nom"] = pi.num, pi.nom
            tous[id(pi)]["confirmee"] = pi.confirmee
        cibles = [tous[id(pi)] for pi in self.pistes]

        confirmees = sorted([c for c in cibles if c["confirmee"]], key=lambda c: c["num"])
        res["cibles"] = confirmees
        # cible principale (affichage detaille) : une alerte d'abord, sinon P1, sinon le meilleur candidat
        ordre = {"apnee": 0, "pause": 1}
        if confirmees:
            principale = min(confirmees, key=lambda c: (ordre.get(c["etat"], 2), c["num"]))
        elif cibles:
            principale = max(cibles, key=lambda c: c["contraste_db"])
        else:
            return res
        res["principale"] = principale
        res.update({k: v for k, v in principale.items() if k not in ("num", "confirmee")})
        return res

    # ------------------------------------------------------------------ pistes
    def n_confirmees_actives(self) -> int:
        return sum(1 for pi in self.pistes if pi.confirmee and pi.num is not None)

    def _dist(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return float(np.linalg.norm(self.XYZ[a] - self.XYZ[b]))

    def _candidats(self, dm: np.ndarray, bruit: float) -> list[tuple[int, int]]:
        """Maxima de micro-mouvements : au-dessus du fond, pas trop loin du plus fort, separes de 50 cm,
        hors des positions reconnues recemment comme echos."""
        p = self.p
        seuil = max(bruit * 10 ** (p.seuil_cible_db / 10), dm.max() * 10 ** (-p.relatif_db / 10))
        flat = dm.ravel()
        n = min(600, flat.size)
        top = np.argpartition(flat, -n)[-n:]
        top = top[np.argsort(flat[top])[::-1]]
        out: list[tuple[int, int]] = []
        for k in top:
            if flat[k] < seuil:
                break
            c = (int(k // dm.shape[1]), int(k % dm.shape[1]))
            if any(self._dist(c, f) <= p.suivi_max_m for f, _ in self.fantomes):
                continue
            if all(self._dist(c, o) > p.separation_m for o in out):
                out.append(c)
                if len(out) >= 3 * p.max_cibles:
                    break
        return out

    def _echo_de(self, m: dict, ref: dict, dist: float, strict: bool = False) -> bool:
        """Le mouvement m est-il celui de la personne ref (meme rythme, mouvement correle) ?
        Lobe secondaire ou echo : mouvement identique ; autre partie du corps (cuisses, tete) : en partie
        identique et a moins de 80 cm. Deux personnes distinctes ne respirent pas a l'identique 20 s durant."""
        p = self.p
        fa = m["filt"] * 1e3 if "filt" in m else m["deplacement_mm"]
        fb = ref["deplacement_mm"]
        ba, bb = m.get("f_pk", m.get("bpm_brut")), ref["bpm_brut"]
        if (ba is None or bb is None or abs(ba - bb) > 1.5 or fa.size != fb.size
                or fa.size < int(p.min_s * self.fs) or fa.std() == 0 or fb.std() == 0):
            return False
        corr = abs(float(np.corrcoef(fa, fb)[0, 1]))
        if strict:                                   # deux personnes deja confirmees : il faut l'identite
            return corr >= 0.9
        return corr >= p.fusion_corr or (dist <= p.separation_m + 0.3 and corr >= 0.4)

    def _analyser_piste(self, pi: Piste, X: np.ndarray, dm: np.ndarray, bruit: float, maintenant: float) -> dict:
        p = self.p
        T = X.shape[0]
        plus_fort = lambda a, b: dm[a] > dm[b] * 10 ** (p.changement_db / 10)   # noqa: E731
        m = self._mesure(X, *pi.pos)
        seuil_m = None
        calme = 0.0
        if pi.amp_ref:
            seuil_m = max(p.amplitude_min_mm, p.pause_ratio * self._amp_habituelle(pi)) * 1e-3
            calme = self._duree_calme(m["disp"], seuil_m)          # 0 : respire ; None : fenetre calme
        # La personne a bouge ? On ne deplace la position verrouillee que vers une position proche
        # (<= 25 cm : la meme personne), plus forte, ou l'on mesure vraiment une respiration.
        # Une position eloignee (tete, mains, voisin, objet) ne prend jamais la place.
        # Jamais pendant une pause : la fenetre de 20 s contient encore la respiration d'avant, une
        # position voisine aurait l'air de "respirer" et remettrait le compteur d'apnee a zero.
        if pi.confirmee and pi.debut_pause is None and pi.mouvement_depuis is None:
            pres = (np.linalg.norm(self.XYZ - self.XYZ[pi.pos], axis=-1) <= p.suivi_max_m) & self.masque
            dl = np.where(pres, dm, 0.0)
            best = np.unravel_index(int(np.argmax(dl)), dl.shape)
            best = (int(best[0]), int(best[1]))
            if best != pi.pos and plus_fort(best, pi.pos):
                autre = self._mesure(X, *best)
                if autre["ok"]:
                    # on y mesure une vraie respiration : la personne respire, et l'amplitude habituelle
                    # doit etre celle de la nouvelle position (le thorax ne renvoie pas comme le ventre)
                    pi.pos = best
                    m = autre
                    pi.amp_ref.clear()
                    pi.amp_ref.append((maintenant, m["amplitude_mm"]))
                    seuil_m = max(p.amplitude_min_mm, p.pause_ratio * m["amplitude_mm"]) * 1e-3
                    calme = 0.0
        r0, d0 = pi.pos
        contraste_db = float(10 * np.log10(dm[r0, d0] / bruit + 1e-30))
        ok = m["ok"]

        # pause / apnee : seulement une fois la respiration confirmee a cet endroit
        etat = "respire" if ok else "incertain"
        pause_s = 0.0
        if not pi.confirmee:
            # nouvelle personne : il faut un rythme net pendant 2 s d'affilee (un pic isole ne suffit pas)
            ok = ok and m["amplitude_mm"] >= p.nouvelle_amplitude_mm
            pi.ok_depuis = (pi.ok_depuis if pi.ok_depuis is not None else maintenant) if ok else None
            if ok and maintenant - pi.ok_depuis >= p.confirmation_cible_s:
                pi.confirmee = True                    # numero donne apres le menage des doublons
            elif ok:
                ok, etat = False, "incertain"
        faible = None
        if seuil_m is not None and calme != 0.0:
            faible = self._respiration_faible(pi, m["disp"])
            if faible is not None:
                calme = 0.0                            # elle respire, moins fort : pas de pause
        mouvement = False
        if seuil_m is not None:
            if calme == 0.0:
                if pi.debut_pause is not None:
                    # pendant une pause : ca rebouge. Reprise seulement si cela dure (sinon : sursaut isole)
                    if pi.mouvement_depuis is None:
                        pi.mouvement_depuis = maintenant
                    if maintenant - pi.mouvement_depuis >= p.confirmation_s:
                        pi.fin_pause = pi.mouvement_depuis
                        pi.debut_pause = None
                        pi.mouvement_depuis = None
                    else:
                        mouvement = True
            else:
                pi.mouvement_depuis = None
                debut = maintenant - (calme if calme is not None else T / self.fs)
                if pi.debut_pause is None or debut < pi.debut_pause:
                    pi.debut_pause = debut
            if pi.debut_pause is not None:
                pause_s = maintenant - pi.debut_pause
                etat = "apnee" if pause_s >= p.apnee_s else "pause"
                ok = False                                         # le rythme de la fenetre est perime
                faible = None
            elif faible is not None:
                etat = "faible"
                ok = True
                m = dict(m, f_pk=faible["bpm"])
                pi.amp_ref.append((maintenant, faible["p2p_mm"]))     # l'amplitude habituelle s'adapte
            elif pi.fin_pause is not None and maintenant - pi.fin_pause < p.fenetre_s:
                # Juste apres une pause, la fenetre de 20 s melange apnee et respiration : son rythme est
                # faux. On ne mesure le rythme que sur la partie posterieure a la reprise (au moins 8 s).
                n_apres = int((maintenant - pi.fin_pause) * self.fs)
                ok, etat = False, "reprise"
                if n_apres >= int(p.min_s * self.fs):
                    m2 = self._mesure(X[-n_apres:], r0, d0)
                    if m2["ok"]:
                        ok, etat = True, "respire"
                        m = dict(m, f_pk=m2["f_pk"], q=m2["q"], frac=m2["frac"], prom=m2["prom"])
        if ok and etat == "respire":
            pi.dernier_bpm = m["f_pk"]
            pi.amp_ref.append((maintenant, m["amplitude_mm"]))
        if ok:
            while pi.amp_ref and pi.amp_ref[0][0] < maintenant - p.reference_s:
                pi.amp_ref.popleft()

        spec = np.abs(np.fft.rfft((m["disp"] - m["disp"].mean()) * np.hanning(T), 2048)) ** 2
        u0, w0 = float(self.uu[d0]), float(self.ww[d0])
        rr = float(self.ranges[m["r"]])
        x, z = rr * u0, rr * w0
        y_ = float(np.sqrt(max(rr ** 2 - x ** 2 - z ** 2, 0.0)))
        return {
            "num": pi.num, "nom": pi.nom, "confirmee": pi.confirmee,
            "cible": {"case": int(m["r"]), "distance_m": rr, "u": u0, "w": w0, "xyz": (x, y_, z),
                      "azimut_deg": float(np.degrees(np.arcsin(np.clip(u0, -1, 1)))),
                      "elevation_deg": float(np.degrees(np.arcsin(np.clip(w0, -1, 1)))),
                      "verrouillee": pi.confirmee},
            "contraste_db": contraste_db,
            "seuil_pause_mm": seuil_m * 1e3 if seuil_m is not None else None,
            "respire": bool(ok), "bpm": float(m["f_pk"]) if (ok and m["f_pk"] is not None) else None,
            "bpm_brut": m["f_pk"], "q": float(m["q"]), "frac": float(m["frac"]), "prom": float(m["prom"]),
            "amplitude_mm": m["amplitude_mm"], "etat": etat, "pause_s": float(pause_s), "t_flux_s": maintenant,
            "mouvement": mouvement,
            "fin_pause_estimee_s": pi.mouvement_depuis if mouvement else maintenant,
            "t": np.arange(T) / self.fs - T / self.fs, "deplacement_mm": m["filt"] * 1e3,
            "deplacement_brut_mm": (m["disp"] - m["disp"].mean()) * 1e3,
            "spectre_bpm": np.fft.rfftfreq(2048, 1 / self.fs) * 60, "spectre": spec / (spec.max() + 1e-30),
        }

    # ------------------------------------------------------------------ outils
    def _respiration_faible(self, pi: Piste, disp: np.ndarray) -> dict | None:
        """Pause apparente, mais respiration plus faible au meme rythme ? (fin de fenetre : 10 s)"""
        p = self.p
        if pi.dernier_bpm is None:
            return None
        n10, n6 = int(round(10 * self.fs)), int(round(p.pause_fenetre_s * self.fs))
        if len(disp) < n10:
            return None
        seg = disp[-n10:]
        p2p = float(np.ptp(disp[-n6:]))
        f, q, frac, prom = _band_peak(seg - np.polyval(np.polyfit(np.arange(n10), seg, 1), np.arange(n10)),
                                      self.fs, p.bande_bpm, with_prominence=True)
        if (f is not None and p2p >= p.faible_min_mm * 1e-3 and q >= p.faible_q_min and frac >= 0.5
                and abs(f - pi.dernier_bpm) <= p.faible_ecart_bpm):
            return {"bpm": float(f), "p2p_mm": p2p * 1e3}
        return None

    def _amp_habituelle(self, pi: Piste) -> float:
        """Quartile bas des amplitudes recentes : quelques grandes inspirations (reprise apres une
        apnee, soupir) ne doivent pas faire prendre la respiration normale pour une pause."""
        return float(np.percentile([a for _, a in pi.amp_ref], 25))

    def _mesure(self, X: np.ndarray, r0: int, d0: int) -> dict:
        """Deplacement et rythme a une position (case distance r0 +-1, direction d0)."""
        p = self.p
        T = X.shape[0]
        best_sc = None
        cases = [r for r in (r0 - 1, r0, r0 + 1) if 0 <= r < self.R]
        ys = {r: (X[:, :, r, :] @ self.S[d0]).reshape(-1) for r in cases}   # series fines (bursts)
        puiss = {r: float(np.mean(np.abs(y) ** 2)) for r, y in ys.items()}
        p_max = max(puiss.values())
        for r in cases:
            # une case voisine trop faible est dominee par le bruit : sa phase saute au hasard et
            # donnerait de faux grands deplacements (fausse reprise de respiration pendant une apnee)
            if puiss[r] < p_max / 4:
                continue
            y = ys[r]
            phi = np.unwrap(np.angle(y)).reshape(T, self.nc).mean(axis=1)
            disp = phi * self.lam / (4 * np.pi)
            tt = np.arange(T)
            disp = disp - np.polyval(np.polyfit(tt, disp, 2), tt)   # derive lente (posture, chauffe)
            f_pk, q, frac, prom = _band_peak(disp, self.fs, p.bande_bpm, with_prominence=True)
            score = q * frac
            if best_sc is None or score > best_sc[0]:
                best_sc = (score, r, disp, f_pk, q, frac, prom)
        _, r, disp, f_pk, q, frac, prom = best_sc
        filt = _fft_filter(disp, self.fs, p.bande_bpm[0] / 60, p.bande_bpm[1] / 60)
        amplitude_mm = float(np.std(filt) * 2 * np.sqrt(2) * 1e3)
        # q, frac et prom sont relatifs (insensibles a l'echelle) : un objet immobile peut avoir un "beau" pic.
        # On exige donc aussi un vrai mouvement, et un pic qui ne colle pas au bas de la bande (derive).
        au_bord = f_pk is None or f_pk < p.bande_bpm[0] + 1.0
        ok = (T / self.fs >= p.min_s and q >= p.q_min and frac >= p.frac_min and prom >= p.prom_min
              and not au_bord and amplitude_mm >= p.amplitude_min_mm)
        return {"r": r, "disp": disp, "filt": filt, "f_pk": f_pk, "q": q, "frac": frac, "prom": prom,
                "amplitude_mm": amplitude_mm, "ok": bool(ok)}

    def _duree_calme(self, disp: np.ndarray, seuil_m: float) -> float | None:
        """Depuis combien de secondes l'amplitude crete a crete (fenetres de 6 s) est sous le seuil.

        Calculee sur le deplacement brut (tendance lente deja retiree) : le filtre passe-bande FFT
        "sonne" apres un arret brutal et masquerait le debut de l'apnee.
        0 : on respire encore ; None : toute la fenetre est calme. La fin de la derniere fenetre de 6 s
        qui contenait une respiration arrive environ 6 s apres le dernier mouvement : on rajoute ces 6 s.
        """
        L = int(round(self.p.pause_fenetre_s * self.fs))
        if len(disp) < L + 2:
            return 0.0
        win = np.lib.stride_tricks.sliding_window_view(disp, L)
        p2p = win.max(axis=1) - win.min(axis=1)
        au_dessus = np.flatnonzero(p2p >= seuil_m)
        if au_dessus.size and au_dessus[-1] == p2p.size - 1:
            return 0.0
        if not au_dessus.size:
            return None
        return (p2p.size - 1 - au_dessus[-1]) / self.fs + self.p.pause_fenetre_s