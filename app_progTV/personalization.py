"""Profil local, retours persistants et ajustement explicite du classement."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import unicodedata

import pandas as pd

DEFAULT_PREFERENCES = {
    'liked_categories': [],
    'preferred_channels': [],
    'disliked_categories': [],
    'keywords': [],
    'avoid_keywords': [],
    'max_duration': None,
}
FEEDBACK_VALUES = {'like', 'dislike', 'seen'}


def normalized(value):
    if pd.api.types.is_scalar(value) and pd.isna(value):
        value = ''
    text = unicodedata.normalize('NFKD', str(value).casefold())
    return ''.join(char for char in text if not unicodedata.combining(char))


def content_id(program):
    """Une rediffusion au même titre/résumé partage le même retour."""
    parts = [normalized(program.get(key, '')).strip()
             for key in ('name', 'desc', 'cat')]
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()


def validate_preferences(payload):
    if not isinstance(payload, dict):
        raise ValueError('Les préférences doivent être un objet JSON.')
    if set(payload) - set(DEFAULT_PREFERENCES):
        raise ValueError('Champ de préférence inconnu.')
    result = {}
    for key, default in DEFAULT_PREFERENCES.items():
        value = payload.get(key, default)
        if key == 'max_duration':
            if value is not None and (
                type(value) is not int or not 1 <= value <= 1440
            ):
                raise ValueError(
                    'Durée maximale : entier de 1 à 1440 minutes.'
                )
        else:
            if not isinstance(value, list) or len(value) > 30:
                raise ValueError('Chaque liste accepte au plus 30 valeurs.')
            if any(not isinstance(item, str) or not item.strip()
                   or len(item) > 100 for item in value):
                raise ValueError(
                    'Valeurs attendues : textes de 1 à 100 caractères.'
                )
            value = list(dict.fromkeys(item.strip() for item in value))
        result[key] = value
    if set(result['liked_categories']) & set(result['disliked_categories']):
        raise ValueError('Une catégorie ne peut être préférée et exclue.')
    return result


class ProfileStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS preferences (
                    profile_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    profile_id TEXT NOT NULL,
                    content_id TEXT NOT NULL,
                    program_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    value TEXT NOT NULL,
                    program_json TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (profile_id, content_id)
                );
            ''')

            columns = {row[1] for row in db.execute('PRAGMA table_info(feedback)')}
            if 'program_json' not in columns:
                db.execute('ALTER TABLE feedback ADD COLUMN program_json TEXT')

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def preferences(self, profile_id):
        with closing(self.connect()) as db:
            row = db.execute(
                'SELECT payload FROM preferences WHERE profile_id = ?',
                (profile_id,),
            ).fetchone()
        return json.loads(row[0]) if row else validate_preferences({})

    def save_preferences(self, profile_id, preferences):
        preferences = validate_preferences(preferences)
        with closing(self.connect()) as db, db:
            db.execute(
                'INSERT INTO preferences VALUES (?, ?) '
                'ON CONFLICT(profile_id) DO UPDATE SET '
                'payload=excluded.payload',
                (profile_id, json.dumps(preferences, ensure_ascii=False)),
            )
        return preferences

    def feedback(self, profile_id):
        with closing(self.connect()) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                'SELECT content_id, program_id, name, category, value, '
                'updated_at '
                'FROM feedback WHERE profile_id = ? '
                'ORDER BY updated_at DESC, content_id', (profile_id,),
            ).fetchall()
        return {row['content_id']: dict(row) for row in rows}

    def save_feedback(self, profile_id, program, value):
        if value not in FEEDBACK_VALUES and value is not None:
            raise ValueError('Retour attendu : like, dislike, seen ou null.')
        key = content_id(program)
        with closing(self.connect()) as db, db:
            if value is None:
                db.execute(
                    'DELETE FROM feedback WHERE profile_id=? AND content_id=?',
                    (profile_id, key),
                )
            else:
                db.execute(
                    'INSERT INTO feedback '
                    '(profile_id, content_id, program_id, name, category, '
                    'value, program_json) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?) '
                    'ON CONFLICT(profile_id, content_id) DO UPDATE SET '
                    'program_id=excluded.program_id, name=excluded.name, '
                    'category=excluded.category, value=excluded.value, '
                    'program_json=excluded.program_json, '
                    'updated_at=CURRENT_TIMESTAMP',
                    (profile_id, key, str(program['id']),
                     str(program.get('name') or ''),
                     str(program.get('cat') or ''), value,
                     json.dumps(program, ensure_ascii=False, allow_nan=False)),
                )
        return key

    def remove_feedback(self, profile_id, key):
        with closing(self.connect()) as db, db:
            db.execute(
                'DELETE FROM feedback WHERE profile_id=? AND content_id=?',
                (profile_id, key),
            )


def annotate(programs, feedback):
    programs = programs.copy()
    programs['content_id'] = [content_id(row)
                              for row in programs.to_dict('records')]
    programs['feedback'] = programs['content_id'].map(
        lambda key: feedback.get(key, {}).get('value')
    )
    return programs


def personalize(programs, preferences, feedback):
    """Reranker sans modifier note_pred ; les exclusions précèdent le top 5."""
    programs = annotate(programs, feedback)
    text = pd.Series('', index=programs.index)
    for column in ('name', 'desc', 'cat'):
        if column in programs:
            text += ' ' + programs[column].fillna('').astype(str)
    text = text.map(normalized)
    categories = programs.get('cat', pd.Series('', index=programs.index))
    excluded = categories.isin(preferences['disliked_categories'])
    for keyword in preferences['avoid_keywords']:
        excluded |= text.str.contains(normalized(keyword), regex=False)
    excluded |= programs['feedback'].isin(['dislike', 'seen'])
    if preferences['max_duration'] is not None:
        excluded |= programs['duration'] > preferences['max_duration']
    programs = programs[~excluded].copy()
    # Le rang normalisé ne dépend pas de l'échelle des notes d'entraînement.
    scores = pd.to_numeric(programs['note_pred'], errors='coerce')
    programs['recommendation_score'] = scores.rank(pct=True).fillna(0.0)
    learned = {}
    for entry in feedback.values():
        if entry['value'] in {'like', 'dislike'}:
            votes = learned.setdefault(entry['category'], [])
            votes.append(1 if entry['value'] == 'like' else -1)
    reasons = []
    for index, row in programs.iterrows():
        explanation = []
        bonus = 0.0
        if row.get('cat') in preferences['liked_categories']:
            bonus += 0.25
            explanation.append('Catégorie préférée')
        if row['channel_name'] in preferences['preferred_channels']:
            bonus += 0.15
            explanation.append('Chaîne préférée')
        matched = [word for word in preferences['keywords']
                   if normalized(word) in text.loc[index]]
        if matched:
            bonus += 0.2
            explanation.append(
                'Correspond à vos goûts : ' + ', '.join(matched)
            )
        if row['feedback'] == 'like':
            bonus += 0.35
            explanation.append('Vous avez aimé ce programme')
        votes = learned.get(row.get('cat'), [])
        adjustment = 0.15 * sum(votes) / (len(votes) + 2)
        bonus += adjustment
        if adjustment > 0:
            explanation.append('Genre apprécié dans vos retours')
        programs.at[index, 'recommendation_score'] += bonus
        reasons.append(explanation or ['Classement du modèle'])
    programs['recommendation_reasons'] = reasons
    return programs.sort_values(
        ['recommendation_score', 'note_pred', 'start', 'id'],
        ascending=[False, False, True, True],
    )


def feedback_metrics(feedback):
    counts = {value: sum(row['value'] == value for row in feedback.values())
              for value in FEEDBACK_VALUES}
    rated = counts['like'] + counts['dislike']
    return {
        **counts, 'rated': rated,
        'like_ratio': counts['like'] / rated if rated else None,
    }
