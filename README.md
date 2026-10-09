# ProgTV

Guide TV Flask, classement local PyTorch et explications Ollama.
Python 3.11 ou supérieur.

## Installation et lancement

Avec l'environnement Conda déjà installé :

```sh
conda activate progTV_pytorch
python -m pip install -r requirements.txt
./run.sh
```

Le lanceur choisit `PROGTV_PYTHON` si défini, sinon le Python de l'environnement
Conda actif (hors `base`), puis `.venv/bin/python` comme solution de repli.
La procédure `.venv` existante reste possible.

Après installation, une seule commande met à jour les
programmes du jour puis lance l'application sur **http://127.0.0.1:5000** :

```sh
./run.sh
```

Le script utilise le Python sélectionné ci-dessus, fonctionne depuis un autre dossier
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
- **Ce soir** : première émission de chaque chaîne commençant entre 21 h et
  21 h 30 (bornes incluses), d’au moins 40 minutes. À défaut, première émission
  entre 20 h 30 et 21 h (21 h exclue), d’au moins 60 minutes et se terminant
  à 21 h 30 ou après. Horaires Europe/Paris. Cette estimation par horaires et
  durée écarte l’access court et les bulletins ; une chaîne sans candidat est
  omise. Les filtres s’appliquent après la sélection du prime time.
- **Demain** : toutes les diffusions qui commencent le lendemain, entre minuit
  inclus et minuit suivant exclu, selon le calendrier Europe/Paris.
- **Suggestions** : les cinq prochaines diffusions les mieux classées parmi
  celles qui correspondent aux filtres.

Les vues Maintenant, Ce soir et Demain affichent d’abord TF1, France 2,
France 3, Canal+, France 5, M6 et Arte, puis les chaînes de la TNT.
Les programmes sont triés par horaire à l’intérieur de chaque chaîne ;
les chaînes supplémentaires suivent par ordre alphabétique. Le filtre des chaînes
et les préférences suivent le même ordre. Suggestions conserve le classement
par affinité.

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

## Préférences et avis personnels

Ouvrir **Mes préférences et mes avis** pour choisir les catégories et chaînes
préférées, les catégories à exclure, les mots/expressions appréciés ou à éviter
et la durée maximale habituelle. Enregistrer pour recalculer les suggestions.
Les filtres temporaires du guide restent indépendants ; les contraintes se cumulent.

Sur chaque carte, choisir **J'aime**, **Pas pour moi** ou **Déjà vu**. Un seul
avis est actif par contenu ; cliquer à nouveau l'annule. L'historique permet
également d'annuler un avis même si sa diffusion a disparu du cache.
Le vote conserve la carte et le focus sur le bouton utilisé. Les suggestions
sont recalculées à la prochaine consultation de la vue.
Les rediffusions au même titre, résumé et catégorie partagent le même avis,
indépendamment de la chaîne. Les épisodes aux résumés différents restent distincts.

Les avis et les préférences sont stockés dans
`app_progTV/instance/profiles.sqlite3` (ignoré par Git), avec un profil anonyme
par navigateur identifié par un cookie HttpOnly de durée un an. Il n'y a pas de
compte ni de synchronisation entre appareils ; supprimer le cookie crée un
nouveau profil. Sauvegarder la base pour conserver les données côté serveur.
Chaque vote conserve également un instantané JSON du programme (résumé, chaîne,
horaires et autres données affichées). Les bases existantes sont migrées
automatiquement ; le contenu des anciens votes ne peut pas être reconstitué
après disparition du cache.
`PROGTV_DATABASE` permet de choisir un autre chemin de base.

Les programmes vus/écartés et les contenus exclus sont retirés des **suggestions**,
mais restent visibles dans les autres vues. Le classement normalise le rang du
score du modèle et ajoute des bonus explicites pour les préférences et les avis,
ainsi qu'un ajustement limité par genre à partir des avis positifs/négatifs.
Les raisons sont affichées sur les cartes ; `note_pred` reste inchangée. Le modèle
PyTorch n'est pas réentraîné à chaque avis. Les explications Ollama utilisent le
profil et les avis, sans goûts prédéfinis, et leur cache tient compte de ce contexte.

Les compteurs d'avis et `like_ratio` sont descriptifs : ce taux n'est pas une
mesure de précision ni une preuve d'amélioration. Les pondérations initiales
restent à évaluer sur des retours indépendants des données de classement.

### Favoris et rappels

Chaque carte permet d’**ajouter une diffusion aux favoris**, indépendamment de
J’aime, Pas pour moi ou Déjà vu. Ouvrir **Mes favoris et rappels** pour retrouver
le titre, le résumé, la chaîne et la date complète. Les favoris sont conservés
par navigateur dans la même base SQLite et restent consultables après disparition
du cache ; les rediffusions sont des favoris distincts. Ajouter ou retirer un
favori conserve les cartes et la position dans la page.

Le lien **Télécharger le calendrier** exporte cette diffusion au format `.ics`.
Choisir un rappel de 5, 15, 30 ou 60 minutes avant le début (15 par défaut),
ou Sans rappel. Importer le fichier dans son application calendrier pour activer
l’événement et son alerte. Les heures sont exportées en UTC pour conserver
l’instant exact, y compris lors des changements d’heure. L’export d’un favori
reste disponible hors cache. Une diffusion terminée est signalée dans la liste.

Le fichier contient les horaires enregistrés lors de l’ajout du favori ;
il ne constitue pas un abonnement actualisé automatiquement. Retirer un favori
ne supprime pas un événement déjà importé. Les notifications dépendent de
l’application calendrier, pas d’un service d’alertes dans ProgTV.

### API du profil

- `GET /api/profile` : préférences, choix disponibles, historique, compteurs et favoris.
- `PUT /api/profile` : remplacer les préférences (objet JSON ; champs omis remis
  à leur valeur par défaut). Liste de 30 textes maximum, 100 caractères chacun.
- `PUT /api/programs/<id>/feedback` : `{"value":"like"}`, `dislike`, `seen` ou
  `null` pour annuler. Programme absent du cache : 404.
- `DELETE /api/feedback/<content_id>` : annuler un avis enregistré.
- `PUT /api/programs/<id>/favorite` : ajouter une diffusion au cache aux favoris.
- `DELETE /api/programs/<id>/favorite` : retirer un favori, même hors cache.
- `GET /api/programs/<id>/calendar?reminder=15` : fichier `.ics` ; rappel
  0 (aucun), 5, 15, 30 ou 60 minutes. Export hors cache réservé aux favoris
  du profil courant. Horaires invalides : 422.

Les listes de programmes incluent désormais `content_id`, `feedback` et `favorite` ; les
suggestions ajoutent `recommendation_score` et `recommendation_reasons`.
Les réponses personnalisées sont marquées `private, no-store`.

### Styles

Les nouveaux contrôles utilisent Tailwind préfixé `tw-`, sans réinitialisation
CSS globale ; Bootstrap reste utilisé par le guide existant. Le CSS compilé est
committé et servi localement : Node n'est pas requis pour lancer l'application.
Pour modifier les styles : `npm ci --cache .cache/npm`, puis `npm run build:css`.

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
Les dates des anciens caches sans fuseau sont interprétées comme UTC.
Une correction de **−2 heures** s’applique aux heures de début et de fin de
la source TV avant l’affichage en Europe/Paris. Les caches existants sont corrigés
à la lecture, sans recalcul des embeddings ni modification des fichiers.
Les nouveaux caches portent un marqueur pour éviter une double correction.
Les durées restent identiques ; la correction tient compte des changements de date.
Cet ajustement fixe correspond au décalage constaté sur la source actuelle ;
il devra être réévalué si les horaires fournis par l’API changent.

## Tests

```sh
mkdir -p .test-tmp
TMPDIR="$PWD/.test-tmp" PROGTV_DATABASE="$PWD/.test-tmp/tests.sqlite3" python -m unittest discover -s tests -v
node --check app_progTV/static/app.js
node tests/test_frontend.cjs
```

Les tests JavaScript utilisent un DOM simulé, sans dépendance npm ; ils ne
remplacent pas une vérification visuelle dans un navigateur.
