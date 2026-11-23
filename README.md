# Détection de la respiration par radar après un accident (TI AWRL6844)

Projet du hackathon sécurité routière Renault (ENSTA Paris, 2026–2027).
Un radar millimétrique 60 GHz placé dans l'habitacle dit aux secours, **avant même d'ouvrir la porte**,
combien de personnes sont à bord et si chacune **respire** (et à quel rythme), ou si elle est en **apnée**.

- **Carte :** Texas Instruments AWRL6844 (carte d'évaluation), avec le firmware « in-cabin sensing »
  précompilé de la Radar Toolbox TI. Aucune recompilation : on exporte les données brutes (radar cube)
  sur la liaison série et tout le traitement est fait en Python.
- **Sans caméra, à travers les vêtements, de jour comme de nuit.**

## État (23 nov. 2026)

| Essai sur table (carte réelle) | Attendu | Mesuré |
|---|---|---|
| Métronome, 10 respirations/min, à 0,5 m | 10 /min | 10,0 /min (9,7–10,3) |
| Métronome, 15 respirations/min, à 0,5 m | 15 /min | 15,1 /min (14,4–15,4) |
| Chaise vide | aucun rythme | aucun rythme |
| Apnée de 25 s | alerte | alerte 10 à 14 s après le début de l'apnée |
| Personne à 1,5 m | trouvée | trouvée, apnée détectée |

- **Plusieurs personnes** (jusqu'à 4, chacune suivie séparément : P1, P2...) : vérifié en simulation,
  pas encore sur la carte.
- **Pas encore fait :**
  - détection d'une personne déjà sans respiration au démarrage du radar ;
  - essais en voiture (attribution des cibles aux sièges) ;
  - écran pour les secours.

## Contenu

| Fichier | Rôle |
|---|---|
| `respiration_bureau.py` | **outil principal** : carte de la pièce, rythme respiratoire et apnée de chaque personne, en direct ou en relecture |
| `nuage_points.py` | affichage du nuage de points de la carte (remplace le visualiseur MATLAB de TI) |
| `capture.py` | envoi de la configuration à la carte et enregistrement du flux |
| `simuler.py`, `traiter.py` | simulateur de la carte (habitacle et accident) et traitement siège par siège |
| `radar_vitals/` | bibliothèque : lecture série et TLV (`serie.py`, `tlv.py`), configuration (`cfg_parser.py`), respiration et apnée (`bureau.py`), simulateur |
| `cfg/` | configurations envoyées à la carte. `cpd_ti.cfg` et `sbr_ti.cfg` viennent de la Radar Toolbox TI ; `bureau_vitals.cfg` et `cpd_vitals.cfg` y ajoutent l'export des données brutes. |
| `GUIDE_CARTE.md` | pas à pas pour la carte : switches, flashage avec UniFlash, ports COM, dépannage |
| `tests/` | tests automatiques (`python -m pytest tests`) ; `python tests/visualiser_tests.py` dessine une figure par test dans `tests/figures/` |

## Démarrage rapide

```
pip install -r requirements.txt
python respiration_bureau.py --port-cli COM6 --port-data COM8 --duree 90 --distance-min 0.45 --distance-max 1.8
```

La carte doit être flashée et en mode fonctionnel (voir `GUIDE_CARTE.md`). Les essais sont enregistrés
dans `enregistrements/` (non versionné) et se rejouent avec `--fichier enregistrements/bureau_xxx.bin`.

## Installation

Python 3.10 ou plus récent (3.14 OK) :

```
pip install -r requirements.txt
```

## 1. Sans carte : simulation

```
python simuler.py
```

- Simule 20 s d'habitacle vide (la référence), puis 120 s d'accident :
  - **conducteur :** respire, puis arrêt respiratoire à 60 s ;
  - **passager avant :** respire et bouge entre 30 et 36 s ;
  - **arrière gauche :** vide ;
  - **arrière milieu :** un enfant qui respire vite ;
  - **arrière droit :** un adulte inconscient qui ne respire pas.
- Affiche l'état des 5 sièges au fil du temps, puis une évaluation par rapport à la vérité.
- Enregistre `resultats_sim/synthese.png`.

Options : `--duree 60`, `--graine 3` (autre tirage aléatoire), `--bruit 12`.
Pour d'autres scénarios, modifiez `default_scene()` dans `simuler.py` (classe `Occupant`).

Pour rejouer un enregistrement :

```
python traiter.py resultats_sim/scenario.bin --reference resultats_sim/reference_vide.npz --verite resultats_sim/verite.json
```

## 2. Avec la carte

Le pas à pas complet (switches S7/S8, UniFlash, dépannage) est dans **`GUIDE_CARTE.md`**.

1. **Flasher** `demo_in_cabin_sensing_6844_system.release.appimage` avec UniFlash : mode flash (S8 = OFF, S7 = OFF), RESET, Load Image.
2. **Mode fonctionnel :** S8 = OFF, S7 = ON, puis RESET.
3. **Premier essai, le nuage de points** (remplace le visualiseur MATLAB de TI) :
   ```
   python nuage_points.py
   ```
   Le script envoie `cfg/cpd_ti.cfg`, affiche les points en 3D, vue de dessus et vue de côté, et enregistre le flux dans `enregistrements/`.
4. **Sur table, une ou plusieurs personnes immobiles** (carte de la pièce, rythme respiratoire et apnée de chacune) :
   ```
   python respiration_bureau.py --port-cli COM6 --port-data COM8 --duree 90 --distance-min 0.45 --distance-max 1.8
   ```
   - Jusqu'à 4 personnes, à plus de 50 cm les unes des autres, suivies séparément (P1, P2...) : rythme, pause, alerte d'apnée au-delà de 10 s.
   - Options : `--cibles 1` (une seule personne), `--separation 0.5` (m), `--apnee 10` (s), `--seuil-pause 0.3`.
   - Relecture d'un essai : `--fichier enregistrements\bureau_xxx.bin`.
   - Limite : une personne qui ne respire déjà plus au lancement n'est pas vue (aucun mouvement). Il faut la référence « habitacle vide » de `traiter.py`.
5. **Enregistrer l'habitacle vide** (référence, portes fermées, personne dedans) :
   ```
   python capture.py --duree 60 --sortie vide.bin
   ```
6. **Faire un essai en direct :**
   ```
   python capture.py --duree 120 --sortie essai1.bin --direct --reference vide.bin
   ```
7. **Rejouer et produire la figure :**
   ```
   python traiter.py essai1.bin --reference vide.bin
   ```
8. **Vérifier le sens des angles** (une seule fois) : enregistrer une personne seule sur le siège conducteur, puis lancer `traiter.py` avec et sans `--inverser-azimut`. Le bon réglage est celui où le conducteur « RESPIRE » et le passager est « VIDE ».

`cfg/cpd_vitals.cfg` est le `cpd.cfg` de TI avec **deux lignes en plus**, qui demandent au firmware
précompilé d'envoyer une partie du radar cube (TLV 2007) :

```
dbgGuiMonitor 0 0 0 0 0 0 0 0 1
exportRadCubeChunk 0 10 3        % bursts 0, 10 et 20 de chaque image (0, 40 et 80 ms)
```

Le débit est d'environ 61 ko/s pour une capacité UART d'environ 125 ko/s. Si des trames se perdent, passez à `exportRadCubeChunk 0 16 2`.
`capture.py` sauvegarde le cfg réellement envoyé à côté du `.bin` (même nom, `.cfg`), et `traiter.py` le relit.

## Comment ça marche

| Étape | Où | Principe |
|---|---|---|
| Format UART | `radar_vitals/tlv.py` | En-tête 40 octets + TLV, repris de `mmwave_demo_mss.c` (TLV 2007 = radar cube, 1041 = classes CPD TI) |
| Géométrie | `radar_vitals/cfg_parser.py` | Antennes (`antGeometry0/1`), montage (`sensorPosition`), zones (`cuboidDef`), mêmes conventions que la GUI TI |
| Zones | `ZoneMap` | Voxels (case distance × direction) dont la position tombe dans les cuboïdes de chaque siège |
| Présence | `_presence` | Moyenne courante − moyenne de l'habitacle vide, puis CLEAN : on attribue chaque changement à sa zone en retirant ses lobes secondaires |
| Respiration | `_analyze_zone` | Phase du voxel → déplacement (λ/4π), déroulée sur les 3 bursts, puis spectre 6–45/min et vote entre voxels (comme l'algo TI `vitalsign.c`) |
| Sièges voisins | `_analyze`, `_nulled_series` | Sièges traités du plus « vivant » au moins vivant ; chacun avec un zéro vers l'empreinte des sources déjà reconnues (supprime les fuites) |
| Décision | `_decide` | VIDE / RESPIRE / MOUVEMENT / SURVEILLANCE / ALERTE (sans respiration ≥ 10 s) / INCERTAIN ; 3 s de respiration continue pour lever une alerte |

Tous les seuils sont dans `Params` (`radar_vitals/processing.py`).

## Résultats en simulation (3 tirages, scénario par défaut)

- **Rythme respiratoire :** erreur médiane de 0,3 à 0,6/min.
- **Victimes sans respiration :** toujours en **ALERTE**, 19 à 37 s après l'arrêt (10 s de seuil + latence de la fenêtre).
- **Points faibles :**
  - **Siège arrière gauche vide :** 16 à 39 s de fausse alerte sur 120 s.
  - **Respirations « vues à tort » :** quelques secondes sur des victimes, de 3 à 14 s.

## Limites connues (à vérifier sur la vraie carte)

- **Sièges arrière.** Vus depuis le plafonnier, les trois sièges arrière ne sont qu'à environ 0,4 d'écart en cosinus directeur. Le réseau 4×4 résout environ 0,5 : il y a donc de la diaphonie entre eux. C'est la principale source d'erreur restante. Pistes : autre position de montage, 2e radar, croiser avec la classification TI (TLV 1041).
- **Sièges avant.** Ils sont à environ ±57° : le conducteur « apparaît » aussi côté passager à −2 dB. C'est corrigé par l'annulation, mais ça reste fragile.
- **Cœur.** À 5 images/s, les harmoniques du battement se replient : le rythme cardiaque est **expérimental**, à ne pas afficher aux secours. Il faudrait une cadence plus élevée.
- **Déroulement de phase.** Une respiration profonde fait bouger le thorax de plus de λ/4 en 200 ms : d'où les 3 bursts par image. Avec 1 seul burst, le rythme devient faux.
- **Seuils de présence** (`presence_db`, `presence_rel_db`). Réglés sur la simulation : **à recaler** sur des enregistrements réels (vide, adulte, enfant ou poupée, sac d'eau).
- **Référence vide.** Elle suppose un habitacle identique : un accident (déformation, airbags) la fausse en partie.
- **Sens des angles.** Il est déduit des sources TI, mais pas encore vérifié. Si une personne sur le siège conducteur apparaît côté passager, ajouter `--inverser-azimut` à `traiter.py` / `capture.py` (`--inverser-elevation` pour avant / arrière). Le cfg envoyé à la carte ne change pas.
- **Calibration de phase.** Sans `compRangeBiasAndRxChanPhase` mesuré (coin réflecteur + `measureRangeBiasAndRxChanPhase 1 2 0.2`), la séparation des sièges sera moins bonne.

## Fichiers

```
GUIDE_CARTE.md    flasher la carte, switches, premier essai, dépannage
nuage_points.py   nuage de points en direct ou depuis un enregistrement (remplace la GUI MATLAB)
respiration_bureau.py  essai sur table : carte de la pièce + rythme et apnée de chaque personne (jusqu'à 4)
simuler.py        simulation complète + évaluation + figure
traiter.py        traite un enregistrement (.bin) simulé ou réel
capture.py        pilote la carte : envoi du cfg, enregistrement, direct
cfg/cpd_ti.cfg    cfg CPD d'origine TI (nuage de points)
cfg/sbr_ti.cfg    cfg SBR d'origine TI
cfg/cpd_vitals.cfg  cpd_ti.cfg + export du radar cube (signes vitaux)
cfg/bureau_vitals.cfg  cpd_ti.cfg + export du radar cube, nuage de points seul (essai sur table)
radar_vitals/
  cfg_parser.py   lecture du cfg, géométrie
  tlv.py          format des trames UART (lecture + écriture) : nuage 3001, radar cube 2007, classes 1041
  nuage.py        nuage de points : repère capteur → voiture, points par siège
  serie.py        ports XDS110, envoi du cfg
  lecture.py      lecture des trames dans un fil à part (carte ou fichier)
  bureau.py       carte statique / micro-mouvements, choix de la cible, rythme respiratoire
  simulator.py    simulateur d'habitacle (occupants, respiration, cœur, mouvements)
  processing.py   traitement : zones, présence, respiration, fuites, états
  report.py       affichage, évaluation, figure
tests/test_base.py          les 11 tests
tests/visualiser_tests.py   une figure par test -> tests/figures/
```
