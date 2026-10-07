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
