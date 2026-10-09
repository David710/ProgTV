from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
from zoneinfo import ZoneInfo

from flask import Flask, Response, g, jsonify, render_template, request
import pandas as pd

import progtv
from calendar_export import calendar_event
from personalization import (
    ProfileStore, annotate, content_id, feedback_metrics, personalize,
    validate_preferences,
)

app = Flask(__name__)
app.config['PROFILE_DATABASE'] = os.environ.get(
    'PROGTV_DATABASE', str(Path(app.instance_path) / 'profiles.sqlite3')
)
logger = logging.getLogger(__name__)
COMMENT_CACHE = {}
FIELDS = [
    'id', 'name', 'start', 'end', 'icon', 'rating', 'cat', 'desc',
    'note_pred', 'duration', 'channel_name', 'channel_icon', 'content_id',
    'feedback', 'favorite', 'recommendation_score', 'recommendation_reasons',
]


def profile_id():
    if 'profile_id' not in g:
        value = request.cookies.get('progtv_profile', '')
        if re.fullmatch(r'[a-f0-9]{32}', value):
            g.profile_id = value
        else:
            g.profile_id = secrets.token_hex(16)
        g.new_profile = g.profile_id != value
    return g.profile_id


def profile_store():
    return ProfileStore(app.config['PROFILE_DATABASE'])


@app.after_request
def remember_profile(response):
    if getattr(g, 'new_profile', False):
        response.set_cookie(
            'progtv_profile', g.profile_id, max_age=31536000,
            httponly=True, samesite='Lax',
        )
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'private, no-store'
    return response


@app.errorhandler(sqlite3.Error)
def profile_storage_error(error):
    logger.exception('Stockage du profil indisponible')
    return jsonify(error='Profil temporairement indisponible. Réessayez.'), 503


def load_programs():
    tv = progtv.TVProgram()
    today = datetime.now(ZoneInfo('Europe/Paris')).date().isoformat()
    paths = sorted(
        Path(tv.download_folder).glob('progtv_rated_*.pkl'), reverse=True,
    )
    paths = [path for path in paths
             if path.stem.removeprefix('progtv_rated_') <= today]
    # Un cache du jour corrompu ne doit pas masquer le dernier cache valide.
    for path in paths:
        try:
            data = tv.read_programs(path)
            if data is not None:
                tv.flatten_programs(data)
                return tv, data, path.stem.removeprefix('progtv_rated_')
        except Exception:
            logger.exception('Cache TV invalide : %s', path.name)
    return tv, None, None


def serialize(frame):
    records = []
    for row in frame.reindex(columns=FIELDS).to_dict(orient='records'):
        record = {}
        for key, value in row.items():
            if isinstance(value, (datetime, pd.Timestamp)):
                value = value.isoformat()
            elif value is None or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                value = None
            record[key] = value
        records.append(record)
    return records


def response_for(suggestions=False):
    view = ('suggestions' if suggestions
            else request.args.get('view', 'tonight'))
    if view not in {'now', 'tonight', 'tomorrow', 'suggestions'}:
        return jsonify(error='Vue inconnue.'), 400
    duration = request.args.get('max_duration', '').strip()
    try:
        max_duration = int(duration) if duration else None
        if max_duration is not None and not 1 <= max_duration <= 1440:
            raise ValueError()
    except ValueError:
        return jsonify(error=(
            'La durée maximale doit être comprise entre 1 et 1440 minutes.'
        )), 400
    tv, data, data_date = load_programs()
    if data is None:
        return jsonify(error=(
            'Aucun programme disponible. Lancez la préparation des données.'
        )), 503
    frame, choices, selected_date = tv.select_programs(
        data, view=view, query=request.args.get('q', ''),
        channel=request.args.get('channel', ''),
        category=request.args.get('category', ''),
        max_duration=max_duration, limit=10**9,
    )
    store = profile_store()
    identity = profile_id()
    feedback = store.feedback(identity)
    if view == 'suggestions':
        frame = personalize(
            frame, store.preferences(identity), feedback,
        ).head(5)
    else:
        frame = annotate(frame, feedback)
    favorites = store.favorites(identity)
    frame['favorite'] = frame['id'].isin(favorites)
    response = jsonify(serialize(frame))
    response.headers['X-Programs-Date'] = data_date
    response.headers['X-Programs-View-Date'] = selected_date.isoformat()
    response.headers['X-Programs-Filters'] = json.dumps(
        choices, ensure_ascii=True,
    )
    today = datetime.now(ZoneInfo('Europe/Paris')).date().isoformat()
    response.headers['X-Programs-Stale'] = str(data_date != today).lower()
    return response


@app.route('/')
def index():
    profile_id()
    return render_template('index.html')


@app.route('/api/programs')
def get_programs():
    return response_for()


@app.route('/api/suggestions')
def get_suggestions():
    return response_for(suggestions=True)


def find_program(program_id):
    tv, data, _ = load_programs()
    if data is None:
        return tv, None, (jsonify(error='Programmes indisponibles.'), 503)
    programs = tv.flatten_programs(data)
    matches = programs[programs['id'] == program_id]
    if matches.empty:
        return tv, None, (jsonify(error='Programme introuvable.'), 404)
    # Sérialiser remplace également les valeurs manquantes par null.
    return tv, serialize(matches.iloc[:1])[0], None


@app.route('/api/profile', methods=['GET', 'PUT'])
def profile():
    store = profile_store()
    identity = profile_id()
    if request.method == 'PUT':
        try:
            preferences = validate_preferences(request.get_json(silent=True))
        except ValueError as error:
            return jsonify(error=str(error)), 400
        saved = store.save_preferences(identity, preferences)
        return jsonify(preferences=saved)
    feedback = store.feedback(identity)
    choices = {'categories': [], 'channels': []}
    tv, data, _ = load_programs()
    if data is not None:
        programs = tv.flatten_programs(data)
        columns = [('categories', 'cat'), ('channels', 'channel_name')]
        for key, column in columns:
            if column in programs:
                choices[key] = sorted(
                    programs[column].dropna().astype(str)
                    .loc[lambda values: values != ''].unique(),
                    key=progtv.channel_sort_key if key == 'channels' else None,
                )
    return jsonify(
        preferences=store.preferences(identity),
        feedback=list(feedback.values()),
        favorites=list(store.favorites(identity).values()),
        metrics=feedback_metrics(feedback), choices=choices,
    )


@app.route('/api/programs/<program_id>/feedback', methods=['PUT'])
def save_feedback(program_id):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or set(payload) != {'value'}:
        return jsonify(error='Un champ value est requis.'), 400
    value = payload['value']
    if value is not None and value not in ('like', 'dislike', 'seen'):
        return jsonify(error=(
            'Retour attendu : like, dislike, seen ou null.'
        )), 400
    _, program, error = find_program(program_id)
    if error is not None:
        return error
    key = profile_store().save_feedback(profile_id(), program, value)
    return jsonify(content_id=key, feedback=value)


@app.route('/api/feedback/<key>', methods=['DELETE'])
def remove_feedback(key):
    profile_store().remove_feedback(profile_id(), key)
    return jsonify(deleted=True)


@app.route('/api/programs/<program_id>/favorite', methods=['PUT', 'DELETE'])
def favorite(program_id):
    store = profile_store()
    identity = profile_id()
    if request.method == 'DELETE':
        store.remove_favorite(identity, program_id)
        return jsonify(favorite=False)
    _, program, error = find_program(program_id)
    if error is not None:
        return error
    store.save_favorite(identity, program)
    return jsonify(favorite=True)


@app.route('/api/programs/<program_id>/calendar')
def program_calendar(program_id):
    try:
        reminder = int(request.args.get('reminder', '15'))
        if reminder not in (0, 5, 15, 30, 60):
            raise ValueError()
    except ValueError:
        return jsonify(error='Rappel attendu : 0, 5, 15, 30 ou 60 minutes.'), 400
    # L’instantané du favori reste accessible lorsque le cache a disparu.
    program = profile_store().favorites(profile_id()).get(program_id)
    if program is None:
        _, program, error = find_program(program_id)
        if error is not None:
            return error
    try:
        body = calendar_event(program, reminder)
    except (ValueError, TypeError):
        return jsonify(error='Horaires indisponibles pour cet export.'), 422
    return Response(body, content_type='text/calendar; charset=utf-8', headers={
        'Content-Disposition': 'attachment; filename="progtv.ics"',
    })


@app.route('/api/programs/<program_id>/comment')
def get_comment(program_id):
    tv, program, error = find_program(program_id)
    if error is not None:
        return error
    description = str(program.get('desc') or '')
    store = profile_store()
    identity = profile_id()
    preferences = store.preferences(identity)
    feedback = store.feedback(identity)
    preferences['feedback_context'] = {
        'programme': feedback.get(content_id(program), {}).get('value'),
        'categories_avec_avis_positifs': sorted({
            row['category'] for row in feedback.values()
            if row['value'] == 'like' and row['category']
        }),
    }
    cache_key = (
        program_id, description, json.dumps(preferences, sort_keys=True),
    )
    if cache_key not in COMMENT_CACHE:
        try:
            comment = tv.get_ollama_comment(
                description, preferences=preferences,
            )
        except Exception:
            logger.exception('Génération du commentaire impossible')
            return jsonify(error=(
                'Explication temporairement indisponible. Réessayez.'
            )), 503
        if len(COMMENT_CACHE) >= 256:
            COMMENT_CACHE.pop(next(iter(COMMENT_CACHE)))
        COMMENT_CACHE[cache_key] = comment
    return jsonify(id=program_id, comment=COMMENT_CACHE[cache_key])


if __name__ == '__main__':
    app.run()
