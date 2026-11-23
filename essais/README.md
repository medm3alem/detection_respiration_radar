# Essais de validation sur table (carte AWRL6844, 22 nov. 2026)

Chaque essai comprend trois fichiers :
- `.bin` : flux brut de la carte, rejouable sans carte ;
- `.cfg` : configuration envoyée à la carte ;
- `.png` : figure de fin d'essai.

Montage : carte debout sur une table, antennes vers la personne assise, immobile, mains au repos.
Métronome : on change d'état (inspirer / expirer) à chaque battement, donc 20 battements/min = 10 respirations/min.
Les temps sont comptés depuis la première trame reçue.

| Fichier | Distance | Protocole | Attendu | Résultat |
|---|---|---|---|---|
| `bureau_20261122_172618` | 0,5 m | respiration libre, 90 s | pas de référence | ≈ 21 /min, mouvement de 5–6 mm |
| `bureau_20261122_181108` | 0,5 m | métronome à 20, 60 s | 10 /min | 10,0 /min (9,7–10,3), sur 103 analyses |
| `bureau_20261122_181225` | 0,5 m | métronome à 30, 60 s | 15 /min | 15,1 /min (14,4–15,4), sur 107 analyses |
| `bureau_20261122_180932` | – | chaise vide, 60 s | aucun rythme | aucun rythme, 0,0 mm de mouvement |
| `bureau_20261122_184243` | 0,5 m | respiration, apnée de 25 s, reprise ; 90 s | une alerte | une pause de 32 à 58 s (≈ 26 s), alerte à 44 s |
| `bureau_20261122_202758` | 1,5 m | respiration, apnée, reprise ; 90 s | une alerte | personne trouvée à 1,4–1,6 m, mouvement de 1,6–4,4 mm, alerte à 52 s |

Pour rejouer un essai (depuis la racine du dépôt) :

```
python respiration_bureau.py --fichier essais/bureau_20261122_184243.bin --distance-min 0.45 --distance-max 1.8
```

Ces enregistrements servent de tests de non-régression : après chaque modification du traitement, on
les rejoue et on vérifie qu'on retrouve les mêmes résultats.
