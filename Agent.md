# Suivi de ProgTV

Mis à jour : 2026-10-08.

## Règle de maintenance
Mettre ce fichier à jour à chaque évolution : comportement livré, validations,
limitations et prochaine étape. Ne pas annoncer comme terminée une tâche non vérifiée.

## Application
Guide TV Flask et JavaScript, classement PyTorch sur embeddings CamemBERT,
explications locales Ollama. Le dépôt de l'application est ce dossier.

## Priorités
1. Stabiliser : prétraitement identique à l'entraînement et à la prédiction,
   horaires Europe/Paris, sélection déterministe, données absentes et périmées,
   rendu sûr, explications sans course entre requêtes.
2. Usage quotidien : maintenant/ce soir/demain, recherche et filtres,
   interface mobile, explications à la demande et cache.
3. Personnalisation : préférences, retours, favoris/rappels, évaluation du classement.
4. Profils du foyer ; comparaison mesurée des modèles d'embeddings.

## Journal
- 2026-10-07 : analyse initiale et création de ce suivi. Aucun changement applicatif
  livré. Dépendances et données quotidiennes absentes de l'environnement.

## Validation prévue
Tests ciblés : encodage et normalisation, changements d'heure, absence de données,
suggestions sans doublons et nombre de résultats. Vérification syntaxique Python/JS.

- 2026-10-07 : premier lot de stabilisation implémenté.
  - Mappings appris sur le train et normaliseur sauvegardés/réutilisés ;
    catégories inconnues à -1 ; évaluation MSE sur le jeu de test.
  - Refus explicite des poids historiques sans prétraitement. CLI `--train`.
  - Horaires avec fuseau Europe/Paris, cache historique naïf interprété UTC.
  - Prime time : émission en cours à 21 h ; suggestions : au plus n diffusions,
    ordre déterministe et filtrage facultatif des chaînes.
  - Identifiants par diffusion ; cache noté remplacé atomiquement ; chemins
    indépendants du dossier de lancement et timeout du téléchargement.
  - Page accessible sans données ; API 503 explicite ; repli sur dernier cache
    valide et indication de sa fraîcheur.
  - Rendu DOM textuel sûr, images HTTP(S), affichage mobile, score d'affinité
    brut, états chargement/erreur/vide et annulation des requêtes obsolètes.
  - Explications au clic par identifiant avec timeout Ollama et cache mémoire
    borné ; suppression du chargement global concurrent des commentaires.
  - README, dépendances et exemple cron quotidien ajoutés.
  - Validation : 11 tests unittest réussis dans un venv temporaire avec PyTorch
    CPU, syntaxe JavaScript et Python valide, `git diff --check` réussi.

- 2026-10-07 : deuxième lot, navigation temporelle et filtres.
  - Vues Maintenant (début inclus, fin exclue), Ce soir (21 h), Demain
    (journée complète selon le calendrier Europe/Paris), Suggestions conservée.
  - Recherche textuelle littérale sur titre/résumé/chaîne/catégorie, insensible
    à la casse et aux accents ; filtres chaîne, catégorie et durée maximale.
  - Filtrage avant le top 5 ; choix de filtres issus de la période non filtrée
    pour pouvoir sortir d'une sélection vide ; tri chronologique hors suggestions.
  - Sélection conservée entre vues et dans l'URL ; réinitialisation des filtres,
    recherche temporisée et protection contre les réponses obsolètes.
  - Navigation et formulaire accessibles au clavier, état sélectionné annoncé,
    nombre de résultats et message explicite si demain n'est pas disponible.
  - API compatible avec les listes JSON historiques, validation 400 des vues
    et durées, métadonnées des filtres et de la date consultée dans les en-têtes.
  - Validation : 20 tests Python et 4 tests JavaScript sur DOM simulé réussis ;
    syntaxe Python/JS et `git diff --check` réussis. Cas couverts : bornes horaires,
    journée du changement d'heure, filtres combinés avant classement, recherche
    littérale/accents, restauration URL, réinitialisation, réponses obsolètes,
    rendu textuel sûr et commentaires au clic.
  - Limites : pas de vérification visuelle navigateur ni de données source réelles.
    Les nouvelles vues ne créent pas les programmes absents du cache.

- 2026-10-07 : lanceur synthétique `./run.sh` ajouté.
  - Actualise les programmes puis lance Flask sur 127.0.0.1:5000 ; refuse de
    démarrer en cas d'échec de préparation. Ne modifie pas les anciens caches.
  - Mode `--update-only`, options Flask transmises (ex. `--port 8000`), aide,
    exécutable Python configurable via PROGTV_PYTHON ; chemins absolus internes.
  - Installation initiale, commande quotidienne et exemple cron documentés.
  - Validation : syntaxe Bash, aide et essais isolés avec Python simulé : ordre
    préparation/lancement, mode actualisation seule, arguments, arrêt sur échec,
    environnement absent, lancement depuis un autre dossier et chemins avec espaces.
  - Limite : préparation réelle non exécutée, faute de modèle réentraîné et de
    jeu annoté. Le script n'installe pas automatiquement les dépendances.

- 2026-10-07 : correction du premier démarrage sans prétraitement historique.
  - Le jeu annoté est désormais présent sur disque à la racine : 164 lignes,
    embeddings CamemBERT (768 dimensions), notes de 0 à 16.
  - Entraînement automatique si les artefacts compatibles sont absents ;
    sauvegarde dans trained_model_v2.pth et son prétraitement, sans écraser
    trained_model.pth/trained_model_old.pth ; réutilisation aux lancements suivants.
  - Recherche du jeu dans train puis à la racine, chemin absolu accepté,
    validation des entrées numériques et graine d'entraînement fixée.
  - Sortie du lanceur non tamponnée pour suivre la préparation en direct.
  - Entraînement réel effectué : 164 programmes, MSE de test 2,4592 ; modèle
    compatible sauvegardé et relu. Artefacts générés ignorés par Git.
  - Descriptions absentes/numériques converties en texte avant tokenisation ;
    cache des résumés identiques et limitation des threads CPU dans la CLI.
  - Validation : 23 tests Python et 4 tests JavaScript réussis ; Bash/Python et
    `git diff --check` valides. Bootstrap testé : entraînement une seule fois,
    poids historiques inchangés, chemins et descriptions atypiques.
  - Mise à jour réelle du 2026-10-07 réussie via `./run.sh --update-only` après
    une première expiration du délai de réponse de la source. Page et API
    vérifiées avec le client Flask : HTTP 200, données fraîches, 19 programmes
    maintenant, 19 ce soir, 572 demain et 5 suggestions.
  - Le jeu initial est limité à une chaîne ; cette MSE ne valide pas encore
    la qualité du classement multi-chaînes.

- 2026-10-08 : préférences et avis personnels implémentés.
  - Profil anonyme par navigateur (cookie HttpOnly SameSite=Lax), SQLite local
    dans app_progTV/instance ; aucun compte ou synchronisation entre appareils.
  - Catégories/chaînes préférées, exclusions, mots appréciés/à éviter et durée
    habituelle ; validation, enregistrement, remise à zéro et état d'erreur.
  - Avis J'aime/Pas pour moi/Déjà vu exclusifs, persistants et annulables sur la
    carte ou dans l'historique, même après disparition du programme du cache.
  - Identité de contenu par titre/résumé/catégorie normalisés pour partager les
    avis entre rediffusions ; séparation des épisodes aux résumés différents.
  - Suggestions ajustées avant le top 5, exclusions des vus/écartés, bonus
    explicites et signal de genre limité ; raisons affichées, note_pred intacte.
  - Vues temporelles non masquées par les préférences ; profil sans effet sur le
    classement initial s'il est vide. Les contraintes des filtres se cumulent.
  - Explications Ollama fondées sur préférences/avis ; cache contextualisé,
    sans anciens goûts en dur. Compteurs descriptifs, pas de précision revendiquée.
  - Contrôles Tailwind compilés localement et versionnés, préfixe tw- sans
    remplacer Bootstrap ; aucun besoin de Node pour utiliser le site.
  - Lanceur : environnement Conda actif prioritaire après PROGTV_PYTHON,
    .venv conservé en repli. Vérifications dans Conda progTV_pytorch.
  - Validation : 39 tests Python et 7 tests JavaScript réussis ; compilation
    Tailwind, syntaxe Python/JS/Bash et git diff --check ; sélection du Python
    explicite/Conda/venv et modes du lanceur vérifiés. Vérification sur cache
    réel avec une base isolée : 19 chaînes, 5 suggestions, enregistrement de
    préférences, exclusion Déjà vu et annulation : HTTP 200.
  - Test ancien des descriptions mis à jour pour tenir compte de la déduplication
    des embeddings. Données temporaires et caches placés dans ProgTV.
  - Limites : pas de validation visuelle navigateur ni d'appel Ollama réel ;
    pondérations à mesurer sur des retours indépendants, pas de réentraînement
    automatique sur les avis. Effacer le cookie ouvre un nouveau profil.

## Prochaine étape
Ajouter les favoris et les rappels/export calendrier, puis évaluer le classement
personnalisé sur des retours indépendants. Les profils nommés du foyer restent
une évolution ultérieure.

## Limites du premier lot
- Limitation initiale résolue : jeu annoté désormais présent, modèle v2 entraîné
  sur les données réelles ; poids historiques conservés.
- Source distante et génération CamemBERT désormais vérifiées sur les données
  réelles. Génération Ollama réelle toujours non testée.
- Vérification navigateur visuelle non effectuée ; JavaScript vérifié syntaxiquement.
- Cache des commentaires local au processus, non persistant. Les goûts en dur
  ont été remplacés par le profil et les avis ; modèle initial non réentraîné
  automatiquement sur ces retours.
- Planification quotidienne documentée, pas installée sur le système.
- Fichiers modèle/prétraitement séparés : garder les deux artefacts ensemble.

# ne pas faire
- modifier ou supprimer des dossiers à l'exterieur du dossier de travail ProgTV
- ne pas désactiver la carte wifi

# bonnes pratiques
- utiliser conda comme gestionnaire d'environnement
- utiliser PEP8
- utiliser tailwind css
- git commit et push sur github après chaque nouvelles feature