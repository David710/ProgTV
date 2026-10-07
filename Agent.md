# Suivi de ProgTV

Mis à jour : 2026-10-07.

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

## Limitations connues
Les poids historiques seuls ne contiennent pas les encodeurs et le normaliseur.
Le modèle devra être réentraîné pour produire un artefact cohérent.
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

## Prochaine étape
Ajouter les vues maintenant/demain puis la recherche et les filtres. Ensuite
préférences et retours utilisateur, avec mesure de qualité des recommandations.

## Limites du premier lot
- Aucun réentraînement réel effectué : jeu annoté absent du disque. Les poids
  historiques sont conservés. Test d'entraînement effectué sur données synthétiques.
- Source distante, téléchargement CamemBERT et génération Ollama réels non testés.
- Vérification navigateur visuelle non effectuée ; JavaScript vérifié syntaxiquement.
- Cache des commentaires local au processus, non persistant ; profil de goûts
  encore écrit en dur. La source du classement historique n'est pas migrée.
- Planification quotidienne documentée, pas installée sur le système.
- Fichiers modèle/prétraitement séparés : garder les deux artefacts ensemble.
