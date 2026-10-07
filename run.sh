#!/usr/bin/env bash
# Met à jour les programmes avant de démarrer le serveur local.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PROGTV_PYTHON:-$project_dir/.venv/bin/python}"

if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage : ./run.sh [options de flask run]
        ./run.sh --update-only

Sans option : actualiser les programmes, puis servir http://127.0.0.1:5000.
--update-only : actualiser les programmes sans lancer le serveur.
Exemple : ./run.sh --port 8000

Prérequis : .venv avec requirements.txt installé, modèle réentraîné avec
son fichier de prétraitement, accès à la source TV et à CamemBERT.
PROGTV_PYTHON permet de choisir un autre exécutable Python.
EOF
    exit 0
fi

if [[ ! -x "$python_bin" ]]; then
    printf '%s\n' 'Environnement Python absent. Depuis le dossier ProgTV, exécutez :' >&2
    printf '%s\n' 'python3 -m venv .venv' '.venv/bin/pip install -r requirements.txt' >&2
    exit 1
fi

update_only=false
if [[ "${1:-}" == "--update-only" ]]; then
    update_only=true
    shift
    if [[ $# -ne 0 ]]; then
        printf '%s\n' '--update-only ne prend pas d’autres options.' >&2
        exit 2
    fi
fi

printf '%s\n' 'Mise à jour des programmes du jour…'
if ! "$python_bin" "$project_dir/app_progTV/progtv.py"; then
    printf '%s\n' 'Mise à jour échouée ; serveur non lancé. Les anciens programmes sont conservés.' >&2
    printf '%s\n' 'Si le modèle est historique, suivez la rubrique « Modèle et compatibilité » du README.' >&2
    exit 1
fi

if [[ "$update_only" == true ]]; then
    exit 0
fi

exec "$python_bin" -m flask --app "$project_dir/app_progTV/app.py" run "$@"
