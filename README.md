# ProgTV

Guide TV Flask, classement local PyTorch et explications Ollama.
Python 3.11 ou supérieur.

## Installation et lancement

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
./run.sh
```

Après installation, une seule commande met à jour les
programmes du jour puis lance l'application sur **http://127.0.0.1:5000** :

```sh
./run.sh
```

Le script utilise `.venv/bin/python`, fonctionne aussi depuis un autre dossier
via son chemin absolu et arrête le lancement si la mise à jour échoue. Le modèle
compatible est entraîné automatiquement au premier lancement.
La préparation peut prendre du temps, surtout au premier téléchargement de
CamemBERT. Arrêter le serveur avec Ctrl+C.

```sh
./run.sh --update-only  # actualiser sans démarrer le serveur
./run.sh --port 8000   # actualiser puis lancer sur un autre port
./run.sh --help
```

Pour un environnement Python déjà installé ailleurs :
`PROGTV_PYTHON=/chemin/venv/bin/python ./run.sh`.

La page fonctionne sans Ollama ; seules les explications en dépendent.
Pour celles-ci, installer Ollama et récupérer `gemma3:12b-it-qat`.
Ne pas exposer le serveur de développement Flask en production.

## Préparation des données

Depuis `app_progTV`, lancer `python progtv.py` dans l'environnement installé.
La source distante et les modèles doivent être accessibles. Les fichiers sont
stockés relativement au module, indépendamment du répertoire de lancement.
Exemple de tâche cron quotidienne à 6 h (adapter les chemins et le fuseau du serveur) :

```cron
0 6 * * * /chemin/ProgTV/run.sh --update-only >> /chemin/ProgTV/preparation.log 2>&1
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

Les poids historiques ne contiennent pas leur prétraitement. Au premier lancement,
`./run.sh` entraîne automatiquement `app_progTV/train/trained_model_v2.pth`
et son fichier `.preprocessing.pkl`, à partir de `df_programs_tf1_note.pkl`
à la racine du dépôt (ou dans `app_progTV/train`). Les poids historiques sont
conservés ; les lancements suivants réutilisent le modèle compatible.

Pour réentraîner explicitement, depuis la racine de ProgTV :

```sh
.venv/bin/python app_progTV/progtv.py --train df_programs_tf1_note.pkl
```

Un chemin absolu vers un autre jeu annoté est également accepté. Colonnes
attendues : `cat`, `rating`, `embeddings`, `note`. Les embeddings doivent utiliser
le même modèle et le même pooling que `generate_embeddings` (CamemBERT, moyenne
masquée). Les embeddings fournis ont été produits avec CamemBERT ; les régénérer
si vous changez de méthode. Le jeu initial est limité à 164 programmes d'une
chaîne : la qualité du classement sur les autres chaînes reste à évaluer.

Les artefacts générés sont ignorés par Git. Garder les poids et le fichier de
prétraitement ensemble. Les catégories inconnues utilisent la valeur -1.
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
