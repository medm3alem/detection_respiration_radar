# Carte AWRL6844EVM : flasher le firmware in-cabin et voir le nuage de points

Ce guide remplace le visualiseur MATLAB de TI (`occupancy_demo_gui`, qui demande le MATLAB Runtime R2026a) par `nuage_points.py`.

## Ce qu'il faut

- **La carte** AWRL6844EVM et un câble micro-USB *de données*, branché sur le connecteur **XDS110** (J5). L'alimentation par jack n'est pas nécessaire.
- **UniFlash**, à télécharger sur ti.com/tool/UNIFLASH. Il installe aussi les drivers XDS110.
- **Le firmware :** `demo_in_cabin_sensing_6844_system.release.appimage` (prendre `.release`, pas `.debug`). Il se trouve dans :
  `radar_toolbox_4_30_00_06\source\ti\examples\Automotive_InCabin_Security_and_Safety\AWRL68xx_InCabin_Demos\prebuilt_binaries\`
- **Le kit `radar_vitals`**, installé dans son venv.

## Les switches de mode (S7, S8)

D'après le guide de l'EVM (SWRU630, tableau 3-2) :

| Mode | S8 (SOP1) | S7 (SOP0) |
|---|---|---|
| Flash | 0 (OFF) | 0 (OFF) |
| **Fonctionnel** (utilisation normale) | 0 (OFF) | **1 (ON)** |
| Debug (CCS) | 1 (ON) | 1 (ON) |

- **Sens des switches :** « 1 » correspond à ON, comme sur les autres cartes TI basse consommation. Vérifie quand même la sérigraphie à côté de S7 et S8.
- **Prise en compte :** après chaque changement de switch, appuie sur **RESET (S2)**. Sinon, la carte reste dans l'ancien mode.
- **Confirmé sur notre carte :** le firmware répond avec S7 sur ON et S8 sur OFF.
- **Routage des liaisons série (S1, S4).** Le firmware in-cabin utilise deux UART :
  - **UART-B à 115 200 bauds pour les commandes.** Elle arrive par défaut sur le port « Application/User UART » (S1.2 = OFF, S4.5 = OFF).
  - **UART-A à 1,25 Mbaud pour les données.** Pour qu'elle arrive sur l'USB XDS110, il faut **S1.4 = ON** (UART-A au lieu de CAN-FD A) et **S1.5 = ON** (UART-A vers le XDS110), d'après les tableaux 3-3 et 3-4 du guide de l'EVM. Après un changement, appuie sur RESET.

## Étape 1 : drivers et ports COM

1. Installe UniFlash, puis branche la carte sur l'USB XDS110.
2. Ouvre le Gestionnaire de périphériques, rubrique *Ports (COM et LPT)*. Deux ports doivent apparaître :
   - « XDS110 Class Application/User UART (COMx) » : les **commandes** ;
   - « XDS110 Class Auxiliary Data Port (COMy) » : les **données**.

   Tu peux aussi lancer `python nuage_points.py --lister-ports`.
3. S'ils n'apparaissent pas, installe le pack « XDS Emulation Software » de TI, puis redémarre le PC.

## Étape 2 : flasher le firmware (UniFlash)

1. Ouvre UniFlash. Dans *New Configuration*, tape `6844`, choisis **AWRL6844**, puis clique sur **Start**.
2. Dans l'onglet *Settings & Utilities*, renseigne *COM Port* avec le port **commandes** (Application/User UART), par exemple `COM5`.
3. Dans l'onglet *Program*, à la ligne *Meta Image 1*, clique sur *Browse* et choisis le `.appimage`. Laisse les autres lignes vides.
4. Mets la carte en **mode flash** (S8 = OFF, S7 = OFF), appuie sur **RESET**, puis clique **tout de suite** sur **Load Image**. En mode flash, la carte n'attend qu'environ 15 s après le reset. Si le flash échoue, refais RESET puis Load Image aussitôt.
5. Attends le message `[SUCCESS] Program Load completed successfully`.

Le flash n'est à faire qu'une fois : le firmware reste en mémoire.

**En cas d'erreur :**
- **Mauvais port :** il faut le port *User UART*, pas le port *Data*.
- **Mode non pris en compte :** les switches ne sont pas en mode flash, ou tu n'as pas appuyé sur RESET.
- **Port occupé :** un autre programme l'utilise (`nuage_points.py`, un terminal…). Ferme-le.
- **UniFlash ouvert avant de brancher la carte :** ferme-le, puis rouvre-le.

## Étape 3 : passer en mode fonctionnel

Règle S8 sur OFF et S7 sur ON, puis appuie sur **RESET**. Le firmware démarre et attend sa configuration : il n'émet rien avant d'avoir reçu `sensorStart`.

## Étape 4 : voir le nuage de points

1. Pose la carte debout sur une table, antennes face à la pièce, sans objet métallique juste devant.
2. Lance le visualiseur :
   ```
   cd "C:\Users\USER\Desktop\hackaton renault\radar_vitals"
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ..\.venv\Scripts\Activate.ps1
   python nuage_points.py
   ```
3. Le script :
   - trouve les deux ports XDS110 ;
   - envoie `cfg/cpd_ti.cfg` (le cfg CPD d'origine de TI) : chaque ligne doit afficher `[OK]` ;
   - ouvre la fenêtre (vue 3D, vue de dessus, vue de côté, panneau d'infos) ;
   - enregistre tout dans `enregistrements\nuage_<date>.bin`, avec le `.cfg` envoyé.
4. Place-toi entre 0,5 et 1,5 m devant les antennes et bouge un bras : des points doivent apparaître. Le panneau doit indiquer environ 5 trames/s.
5. Ferme la fenêtre pour arrêter : le script envoie alors `sensorStop 0`.

### Vérifier le sens des axes (une fois)

Dans la vue de dessus, le radar est en bas et regarde vers le haut de l'écran.

| Test | Ce qu'on doit voir | Sinon |
|---|---|---|
| Debout derrière la carte (dans le sens des antennes), quelqu'un passe à ta droite | Les points vont à droite | `--inverser-azimut` |
| Lever le bras | Les points montent dans la vue de côté | `--inverser-elevation` |
| Haut/bas et gauche/droite sont mélangés | – | La carte est tournée d'un quart de tour : tourne-la |
| Se placer à 1 m mesuré | Y ≈ 1 m | Noter l'écart (calibration) |

### Options utiles

- **Mode SBR** (rappel de ceinture) au lieu de CPD : `python nuage_points.py --cfg cfg/sbr_ti.cfg`.
- **Couleur selon la vitesse** (ce qui bouge vers le radar ou s'en éloigne) : `--couleur doppler`.
- **Plus fluide**, vues 2D seulement : `--sans-3d`.
- **Relire un enregistrement :** `python nuage_points.py --fichier enregistrements\nuage_xxx.bin`.
- **Une fois le radar monté dans la voiture :** `--repere monde`, avec `sensorPosition` et les `cuboidDef` du cfg mis à jour.
- **Sans carte :** après `python simuler.py`, lance `python nuage_points.py --fichier resultats_sim/scenario.bin --repere monde`.

## Étape 5 : carte de la pièce et respiration d'une personne

Ce script lit le radar cube (le signal brut de chaque antenne) au lieu du nuage de points. Il dessine la carte de ce qui renvoie l'onde, repère ce qui bouge un peu (la respiration), puis mesure le rythme.

1. Garde le même réglage : S7 sur ON, S8 sur OFF, puis RESET.
2. La personne s'assoit face au radar, à 0,5–1,5 m (environ 1 m est idéal), le torse face aux antennes. Elle reste immobile et respire normalement. Rien d'autre ne doit bouger devant le radar (ventilateur, rideau, écran qui pivote…).
3. Lance le script :
   ```
   python respiration_bureau.py
   ```
   Il envoie `cfg/bureau_vitals.cfg`. C'est le cfg CPD de TI, qui exporte en plus le radar cube (3 bursts par image) et n'envoie que le nuage de points en plus.
4. Lis la fenêtre :
   - **Carte statique :** ce qui renvoie l'onde (murs, meubles, la personne), vue de dessus, avec le nuage de points TI en bleu.
   - **Micro-mouvements :** ce qui bouge dans la bande respiratoire. Le cercle vert est la cible suivie.
   - **Courbe :** le mouvement du thorax en mm, sur les 20 dernières secondes.
   - **Spectre :** le pic donne le rythme.
   - **Rythme :** affiché en grand, en vert quand il est net. Le premier rythme apparaît au bout d'environ 10 s, et il se stabilise après 20 s.
5. Vérifie le résultat avec une respiration de rythme connu. Compte tes respirations pendant 60 s, ou cale-toi sur un minuteur : 4 s d'inspiration + 4 s d'expiration donnent 7,5 /min, 2 s + 2 s donnent 15 /min.
6. Ferme la fenêtre pour arrêter. Dans `enregistrements\`, tu trouves `bureau_<date>.bin`, le `.cfg` envoyé et la figure `.png`. Pour relire un enregistrement :
   ```
   python respiration_bureau.py --fichier enregistrements\bureau_xxx.bin
   ```

| Problème | Que faire |
|---|---|
| « Les trames n'ont pas de radar cube » | Le cfg n'exporte pas le radar cube : utilise `cfg/bureau_vitals.cfg` |
| Beaucoup de « trames comblées » ou de « redémarrages » | La liaison série sature. Dans `cfg/bureau_vitals.cfg`, mets `guiMonitor 0 0 0 0 0 0 0 0` (supprime le nuage de points), ou `exportRadCubeChunk 0 16 2` (2 bursts au lieu de 3) |
| La cible saute sur autre chose (ventilateur, écran) | Encadre la personne avec `--distance-min 0.6 --distance-max 1.4` |
| La courbe est régulière mais le rythme reste à « -- » | Respiration trop faible ou trop lente (moins de 6 /min) : rapproche-toi du radar |

## Dépannage

| Symptôme | Cause probable |
|---|---|
| « La carte ne répond pas » pendant l'envoi du cfg | Pas en mode fonctionnel (S7 = ON, S8 = OFF), RESET oublié, firmware pas flashé, ou ports inversés |
| Des lignes du cfg en `[ERREUR]` | Le firmware ne correspond pas au cfg (par exemple la démo du SDK au lieu de la démo in-cabin) |
| Cfg OK (y compris `sensorStart`) mais « 0 octet reçu » | L'UART-A n'est pas routée vers l'USB : mettre S1.4 = ON et S1.5 = ON, puis RESET. Sinon, vérifier le port de données |
| Trames reçues mais aucun point | Rien ne bouge dans le champ : se placer devant et bouger. Essayer aussi `--cfg cfg/sbr_ti.cfg` |
| Trames perdues | Fermer les autres programmes, utiliser `--sans-3d` |

## Ce que contient la trame (pour le code)

- **TLV 3001, nuage de points.** La charge utile commence par 5 `float32` donnant les unités, dans l'ordre élévation, azimut, Doppler, distance, SNR. Ensuite, chaque point occupe 8 octets : élévation `int8`, azimut `int8`, Doppler `int16`, distance `uint16`, SNR `uint16`. Le SNR est en dB au format Q8. Le TLV n'est envoyé que s'il y a au moins un point.
- **Conversion en coordonnées capteur :**
  - x = r·cos(él)·sin(az)
  - y = r·cos(él)·cos(az)
  - z = r·sin(él)

  Puis on passe au repère voiture avec `sensorPosition`, comme le fait `parseDataFromDevice.m`.
- **TLV 1041, classification TI.** Valeurs `uint8` au format Q7, rangées [zone][classe].
  - En SBR : (vide, occupé).
  - En CPD : (vide, puis deux probabilités adulte/enfant). La source TI et la GUI MATLAB ne les nomment pas dans le même ordre.
