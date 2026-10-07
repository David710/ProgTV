# ProgTV

Guide TV Flask, classement local PyTorch et explications Ollama.
Python 3.11 ou supérieur.

## Installation et lancement

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cd app_progTV
../.venv/bin/flask --app app run
```

La page fonctionne sans Ollama ; seules les explications en dépendent.
Pour celles-ci, installer Ollama et récupérer `gemma3:12b-it-qat`.
Ne pas exposer le serveur de développement Flask en production.

## Préparation des données

Depuis `app_progTV`, lancer `python progtv.py` dans l'environnement installé.
La source distante et les modèles doivent être accessibles. Les fichiers sont
stockés relativement au module, indépendamment du répertoire de lancement.
Exemple de tâche cron quotidienne à 6 h (adapter les chemins et le fuseau du serveur) :

```cron
0 6 * * * /chemin/ProgTV/.venv/bin/python /chemin/ProgTV/app_progTV/progtv.py >> /chemin/ProgTV/preparation.log 2>&1
```

Le cache noté est remplacé atomiquement après une préparation réussie.
Les téléchargements et les embeddings sont calculés hors des requêtes web.
En cas d'absence de données du jour, le site utilise le dernier cache valide
et indique sa date ; les programmes terminés ne sont pas recommandés.

## Parcourir les programmes

- **Maintenant** : programmes commencés et pas encore terminés.
- **Ce soir** : programme de chaque chaîne en cours à 21 h.
- **Demain** : toutes les diffusions qui commencent le lendemain, entre minuit
  inclus et minuit suivant exclu, selon le calendrier Europe/Paris.
- **Suggestions** : les cinq prochaines diffusions les mieux classées parmi
  celles qui correspondent aux filtres.

Recherche dans le titre, résumé, chaîne et catégorie, sans distinction de casse
ou d'accents. Les caractères sont recherchés littéralement (pas de regex).
Filtres combinables par chaîne, catégorie et durée maximale en minutes.
Les filtres restent sélectionnés lors d'un changement de vue ; Réinitialiser
les efface en conservant la vue. L'URL contient la sélection pour pouvoir la
recharger ou la partager. Les données disponibles peuvent ne pas couvrir demain.

### API de consultation

`GET /api/programs?view=now|tonight|tomorrow|suggestions` (défaut : `tonight`).
`GET /api/suggestions` reste disponible.
Les deux routes acceptent `q`, `channel`, `category` et `max_duration` (entier
entre 1 et 1440). Les filtres sont appliqués avant la limite des suggestions.
Réponses : liste JSON compatible avec la version précédente ; paramètres invalides
400, données absentes 503. Les en-têtes `X-Programs-Date` et `X-Programs-Stale`
indiquent la fraîcheur, `X-Programs-View-Date` la date consultée et
`X-Programs-Filters` les chaînes/catégories de la période, au format JSON.

## Modèle et compatibilité

Les anciens poids seuls ne permettent pas une prédiction fiable. Il faut
réentraîner avec `python progtv.py --train df_programs_tf1_note.pkl` depuis
`app_progTV`, après avoir
placé le jeu annoté dans `app_progTV/train`. Colonnes attendues : `cat`, `rating`,
`embeddings` et `note`. Les embeddings doivent utiliser exactement le même modèle
et le même pooling que `generate_embeddings` (CamemBERT, moyenne masquée).
Régénérer les embeddings d'entraînement si nécessaire.

L'entraînement sauvegarde les poids et un fichier `.preprocessing.pkl` contenant
les mappings et le normaliseur. Les catégories inconnues utilisent la valeur -1.
Les artefacts pickle doivent provenir d'une source de confiance.
La note affichée est un score d'affinité brut, pas une probabilité.
Les dates des anciens caches sans fuseau sont interprétées comme UTC, conformément
à leur construction à partir des timestamps Unix.

## Tests

```sh
python -m unittest discover -s tests -v
node --check app_progTV/static/app.js
node tests/test_frontend.cjs
```

Les tests JavaScript utilisent un DOM simulé, sans dépendance npm ; ils ne
remplacent pas une vérification visuelle dans un navigateur.
