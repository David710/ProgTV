from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import math
import logging
import json

from flask import Flask, jsonify, render_template, request
import pandas as pd
import progtv

app = Flask(__name__)
logger = logging.getLogger(__name__)
COMMENT_CACHE = {}
FIELDS = ['id', 'name', 'start', 'end', 'icon', 'rating', 'cat', 'desc',
          'note_pred', 'duration', 'channel_name', 'channel_icon']


def load_programs():
    tv = progtv.TVProgram()
    today = datetime.now(ZoneInfo('Europe/Paris')).date().isoformat()
    paths = sorted(Path(tv.download_folder).glob('progtv_rated_*.pkl'), reverse=True)
    paths = [path for path in paths if path.stem.removeprefix('progtv_rated_') <= today]
    if not paths:
        return tv, None, None
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
            elif value is None or (isinstance(value, float) and not math.isfinite(value)):
                value = None
            record[key] = value
        records.append(record)
    return records


def response_for(suggestions=False):
    view = 'suggestions' if suggestions else request.args.get('view', 'tonight')
    if view not in {'now', 'tonight', 'tomorrow', 'suggestions'}:
        return jsonify(error='Vue inconnue.'), 400
    duration = request.args.get('max_duration', '').strip()
    try:
        max_duration = int(duration) if duration else None
        if max_duration is not None and not 1 <= max_duration <= 1440:
            raise ValueError()
    except ValueError:
        return jsonify(error='La durée maximale doit être comprise entre 1 et 1440 minutes.'), 400
    tv, data, data_date = load_programs()
    if data is None:
        return jsonify(error='Aucun programme disponible. Lancez la préparation des données.'), 503
    frame, choices, selected_date = tv.select_programs(
        data, view=view, query=request.args.get('q', ''),
        channel=request.args.get('channel', ''), category=request.args.get('category', ''),
        max_duration=max_duration,
    )
    response = jsonify(serialize(frame))
    response.headers['X-Programs-Date'] = data_date
    response.headers['X-Programs-View-Date'] = selected_date.isoformat()
    response.headers['X-Programs-Filters'] = json.dumps(choices, ensure_ascii=True)
    response.headers['X-Programs-Stale'] = str(data_date != datetime.now(ZoneInfo('Europe/Paris')).date().isoformat()).lower()
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/programs')
def get_programs():
    return response_for()


@app.route('/api/suggestions')
def get_suggestions():
    return response_for(suggestions=True)


@app.route('/api/programs/<program_id>/comment')
def get_comment(program_id):
    tv, data, _ = load_programs()
    if data is None:
        return jsonify(error='Programmes indisponibles.'), 503
    programs = tv.flatten_programs(data)
    matches = programs[programs['id'] == program_id]
    if matches.empty:
        return jsonify(error='Programme introuvable.'), 404
    description = str(matches.iloc[0].get('desc', '') or '')
    cache_key = (program_id, description)
    if cache_key not in COMMENT_CACHE:
        try:
            comment = tv.get_ollama_comment(description)
        except Exception:
            logger.exception('Génération du commentaire impossible')
            return jsonify(error='Explication temporairement indisponible. Réessayez.'), 503
        if len(COMMENT_CACHE) >= 256:
            COMMENT_CACHE.pop(next(iter(COMMENT_CACHE)))
        COMMENT_CACHE[cache_key] = comment
    return jsonify(id=program_id, comment=COMMENT_CACHE[cache_key])


if __name__ == '__main__':
    app.run()
