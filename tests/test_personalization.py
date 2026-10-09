import json
from pathlib import Path
import tempfile
import sqlite3
import unittest
from unittest.mock import patch

from test_core import TVProgram, web, pd
from personalization import (
    ProfileStore, content_id, feedback_metrics, personalize,
    validate_preferences,
)


def candidate_data():
    tomorrow = pd.Timestamp.now(tz='Europe/Paris').normalize() + pd.DateOffset(days=1)
    programs = pd.DataFrame([
        dict(name=f'Programme {i}', desc=f'Résumé {i}', cat='Film' if i < 5 else 'Sport',
             rating='', icon='', start=tomorrow + pd.Timedelta(hours=i),
             end=tomorrow + pd.Timedelta(hours=i + 1), note_pred=10.0 - i)
        for i in range(6)
    ])
    return pd.DataFrame([dict(name='TF1', icon='', programs=programs)])


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.programs = TVProgram().flatten_programs(candidate_data())
        self.preferences = validate_preferences({})

    def test_empty_profile_preserves_baseline_and_original_scores(self):
        result = personalize(self.programs, self.preferences, {})
        self.assertEqual(result['id'].tolist(), self.programs['id'].tolist())
        self.assertEqual(result['note_pred'].tolist(), self.programs['note_pred'].tolist())

    def test_seen_and_dislike_exclude_reruns_before_limit(self):
        item = self.programs.iloc[0].to_dict()
        for value in ['seen', 'dislike']:
            feedback = {content_id(item): dict(value=value, category='Film')}
            result = personalize(self.programs, self.preferences, feedback).head(5)
            self.assertNotIn(item['id'], result['id'].tolist())
            self.assertEqual(len(result), 5)

    def test_like_promotes_program_previously_outside_top_five(self):
        item = self.programs.iloc[-1].to_dict()
        feedback = {content_id(item): dict(value='like', category='Sport')}
        result = personalize(self.programs, self.preferences, feedback).head(5)
        self.assertIn(item['id'], result['id'].tolist())
        self.assertIn('Vous avez aimé ce programme', result.loc[5, 'recommendation_reasons'])

    def test_preferences_and_keywords_affect_order_and_exclusions(self):
        preferences = validate_preferences({'liked_categories': ['Sport'], 'keywords': ['resume 5']})
        result = personalize(self.programs, preferences, {})
        self.assertLess(result['id'].tolist().index(self.programs.iloc[5]['id']), 5)
        preferences = validate_preferences({'disliked_categories': ['Film']})
        self.assertEqual(personalize(self.programs, preferences, {})['cat'].tolist(), ['Sport'])
        preferences = validate_preferences({'avoid_keywords': ['RÉSUMÉ 5']})
        self.assertEqual(len(personalize(self.programs, preferences, {})), 5)
        preferences = validate_preferences({'max_duration': 59})
        self.assertTrue(personalize(self.programs, preferences, {}).empty)

    def test_feedback_learning_is_bounded_and_ignores_seen(self):
        preferences = validate_preferences({})
        feedback = {'other': {'category': 'Sport', 'value': 'like'}}
        result = personalize(self.programs, preferences, feedback)
        self.assertIn('Genre apprécié dans vos retours', result.loc[5, 'recommendation_reasons'])
        feedback['other']['value'] = 'seen'
        result = personalize(self.programs, preferences, feedback)
        self.assertEqual(result.loc[5, 'recommendation_reasons'], ['Classement du modèle'])

    def test_suggestions_deduplicate_titles_across_episodes_and_channels(self):
        programs = self.programs.copy()
        repeated = programs.iloc[[0]].copy()
        repeated['id'] = 'rerun'
        repeated['name'] = '  PROGRAMME   0 '
        repeated['desc'] = 'Un autre épisode'
        repeated['channel_name'] = 'France 2'
        repeated['start'] += pd.Timedelta(days=1)
        programs = pd.concat([programs, repeated], ignore_index=True)
        result = personalize(programs, self.preferences, {}).head(5)
        self.assertEqual(len(result), 5)
        self.assertNotIn('rerun', result['id'].tolist())
        self.assertEqual(result['name'].tolist().count('Programme 0'), 1)
        # Le bonus d’une chaîne préférée choisit sa diffusion pour ce titre.
        preferences = validate_preferences({'preferred_channels': ['France 2']})
        result = personalize(programs, preferences, {}).head(5)
        self.assertIn('rerun', result['id'].tolist())
        self.assertNotIn(programs.iloc[0]['id'], result['id'].tolist())
        self.assertNotIn('_suggestion_title', result.columns)

    def test_content_key_matches_missing_values_and_ignores_channel(self):
        base = dict(name='Film', cat='Film', desc=None, channel_name='TF1')
        rerun = dict(base, desc=float('nan'), channel_name='France 2')
        self.assertEqual(content_id(base), content_id(rerun))
        self.assertNotEqual(content_id(base), content_id(dict(base, desc='Autre épisode')))

    def test_metrics_are_descriptive_and_empty_ratio_is_not_zero(self):
        self.assertIsNone(feedback_metrics({})['like_ratio'])
        metrics = feedback_metrics({str(i): dict(value=value) for i, value in enumerate(['like', 'like', 'dislike', 'seen'])})
        self.assertEqual(metrics['rated'], 3)
        self.assertAlmostEqual(metrics['like_ratio'], 2 / 3)


class ProfileAPITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous_database = web.app.config['PROFILE_DATABASE']
        web.app.config['PROFILE_DATABASE'] = str(Path(self.directory.name) / 'profiles.sqlite3')
        self.data = candidate_data()
        self.tv = TVProgram()
        self.loader = patch.object(web, 'load_programs', return_value=(self.tv, self.data, '2026-10-08'))
        self.loader.start()
        self.client = web.app.test_client()
        web.COMMENT_CACHE.clear()

    def tearDown(self):
        self.loader.stop()
        web.app.config['PROFILE_DATABASE'] = self.previous_database
        self.directory.cleanup()

    def test_profiles_persist_and_are_isolated(self):
        first = self.client.get('/api/profile')
        self.assertEqual(first.status_code, 200)
        self.assertIn('HttpOnly', first.headers['Set-Cookie'])
        self.assertIn('SameSite=Lax', first.headers['Set-Cookie'])
        response = self.client.put('/api/profile', json={'keywords': ['polar']})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get('/api/profile').json['preferences']['keywords'], ['polar'])
        other = web.app.test_client()
        self.assertEqual(other.get('/api/profile').json['preferences']['keywords'], [])
        cookie = self.client.get_cookie('progtv_profile').value
        reopened = web.app.test_client()
        reopened.set_cookie('progtv_profile', cookie)
        self.assertEqual(reopened.get('/api/profile').json['preferences']['keywords'], ['polar'])
        self.assertEqual(response.headers['Cache-Control'], 'private, no-store')

    def test_bad_preferences_are_rejected_without_erasing_saved_values(self):
        self.client.put('/api/profile', json={'keywords': ['action']})
        for body in [[], {'max_duration': True}, {'max_duration': 0},
                     {'max_duration': 1441}, {'keywords': 'polar'},
                     {'keywords': ['']}, {'keywords': ['x' * 101]},
                     {'keywords': ['a'] * 31}, {'unknown': 1},
                     {'liked_categories': ['Film'], 'disliked_categories': ['Film']}]:
            self.assertEqual(self.client.put('/api/profile', json=body).status_code, 400, body)
        self.assertEqual(self.client.get('/api/profile').json['preferences']['keywords'], ['action'])

    def test_feedback_update_undo_and_isolation(self):
        program = self.tv.flatten_programs(self.data).iloc[0]
        url = f"/api/programs/{program['id']}/feedback"
        for value in ['like', 'seen', 'dislike']:
            response = self.client.put(url, json={'value': value})
            self.assertEqual(response.status_code, 200)
            rows = self.client.get('/api/profile').json['feedback']
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['value'], value)
            self.assertNotIn('profile_id', rows[0])
        other = web.app.test_client()
        other.delete('/api/feedback/' + response.json['content_id'])
        self.assertEqual(len(self.client.get('/api/profile').json['feedback']), 1)
        self.client.delete('/api/feedback/' + response.json['content_id'])
        self.assertEqual(self.client.get('/api/profile').json['feedback'], [])
        self.client.put(url, json={'value': 'like'})
        self.client.put(url, json={'value': None})
        self.assertEqual(self.client.get('/api/profile').json['feedback'], [])

    def test_full_content_persists_after_cache_disappears(self):
        program = self.client.get('/api/programs?view=tomorrow').json[0]
        url = f"/api/programs/{program['id']}/feedback"
        self.assertEqual(self.client.put(url, json={'value': 'like'}).status_code, 200)
        with sqlite3.connect(web.app.config['PROFILE_DATABASE']) as db:
            saved = json.loads(db.execute(
                'SELECT program_json FROM feedback'
            ).fetchone()[0])
        for field in ['name', 'desc', 'cat', 'start', 'end', 'channel_name']:
            self.assertEqual(saved[field], program[field])
        with patch.object(web, 'load_programs', return_value=(self.tv, None, None)):
            cookie = self.client.get_cookie('progtv_profile').value
            reopened = web.app.test_client()
            reopened.set_cookie('progtv_profile', cookie)
            self.assertEqual(reopened.get('/api/profile').json['feedback'][0]['value'], 'like')

    def test_existing_database_migrates_without_losing_votes(self):
        path = web.app.config['PROFILE_DATABASE']
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE feedback (profile_id TEXT, content_id TEXT, '
                       'program_id TEXT, name TEXT, category TEXT, value TEXT, '
                       'updated_at TEXT DEFAULT CURRENT_TIMESTAMP, '
                       'PRIMARY KEY (profile_id, content_id))')
            db.execute("INSERT INTO feedback (profile_id, content_id, program_id, "
                       "name, category, value) VALUES ('p', 'c', 'i', 'Film', 'Film', 'like')")
        store = ProfileStore(path)
        self.assertEqual(store.feedback('p')['c']['value'], 'like')
        program = self.client.get('/api/programs?view=tomorrow').json[0]
        store.save_feedback('p', program, 'like')
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute(
                'SELECT COUNT(program_json) FROM feedback'
            ).fetchone()[0], 1)

    def test_feedback_validation_and_unknown_program(self):
        program = self.tv.flatten_programs(self.data).iloc[0]
        url = f"/api/programs/{program['id']}/feedback"
        for body in [{}, [], {'value': 'bad'}, {'value': []}, {'value': 'like', 'x': 2}]:
            self.assertEqual(self.client.put(url, json=body).status_code, 400)
        self.assertEqual(self.client.put('/api/programs/missing/feedback', json={'value': 'like'}).status_code, 404)

    def test_suggestions_personalize_before_limit_without_hiding_tv_views(self):
        programs = self.tv.flatten_programs(self.data)
        first, last = programs.iloc[0], programs.iloc[-1]
        self.client.put(f"/api/programs/{last['id']}/feedback", json={'value': 'like'})
        suggestions = self.client.get('/api/suggestions').json
        self.assertIn(last['id'], [row['id'] for row in suggestions])
        self.client.put(f"/api/programs/{first['id']}/feedback", json={'value': 'seen'})
        suggestions = self.client.get('/api/suggestions').json
        self.assertEqual(len(suggestions), 5)
        self.assertNotIn(first['id'], [row['id'] for row in suggestions])
        tomorrow = self.client.get('/api/programs?view=tomorrow').json
        self.assertEqual(len(tomorrow), 6)
        self.assertEqual(next(row['feedback'] for row in tomorrow if row['id'] == first['id']), 'seen')
        self.client.put('/api/profile', json={'disliked_categories': ['Film']})
        self.assertEqual([row['cat'] for row in self.client.get('/api/suggestions').json], ['Sport'])

    def test_api_suggestions_unique_titles_without_hiding_airings_in_tv_views(self):
        programs = self.data.iloc[0]['programs'].copy()
        repeated = programs.iloc[[0]].copy()
        repeated['start'] += pd.Timedelta(minutes=30)
        repeated['end'] += pd.Timedelta(minutes=30)
        repeated['desc'] = 'Autre épisode du même programme'
        self.data.at[0, 'programs'] = pd.concat([programs, repeated], ignore_index=True)
        suggestions = self.client.get('/api/suggestions').json
        self.assertEqual(len(suggestions), 5)
        self.assertEqual(len({row['name'] for row in suggestions}), 5)
        tomorrow = self.client.get('/api/programs?view=tomorrow').json
        self.assertEqual(len(tomorrow), 7)

    def test_comment_cache_includes_preferences_and_prompt_receives_them(self):
        program = self.tv.flatten_programs(self.data).iloc[0]
        url = f"/api/programs/{program['id']}/comment"
        with patch.object(self.tv, 'get_ollama_comment', return_value='Explication') as generate:
            self.client.get(url)
            self.client.get(url)
            self.assertEqual(generate.call_count, 1)
            self.client.put('/api/profile', json={'keywords': ['polar']})
            self.client.get(url)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(generate.call_args.kwargs['preferences']['keywords'], ['polar'])

    def test_feedback_invalidates_comment_cache(self):
        program = self.tv.flatten_programs(self.data).iloc[0]
        url = f"/api/programs/{program['id']}/comment"
        with patch.object(self.tv, 'get_ollama_comment', return_value='Explication') as generate:
            self.client.get(url)
            self.client.put(f"/api/programs/{program['id']}/feedback", json={'value': 'like'})
            self.client.get(url)
            self.assertEqual(generate.call_count, 2)
            context = generate.call_args.kwargs['preferences']['feedback_context']
            self.assertEqual(context['programme'], 'like')
            self.assertEqual(context['categories_avec_avis_positifs'], ['Film'])

    def test_sqlite_failure_returns_recoverable_json_error(self):
        with patch.object(web, 'profile_store', side_effect=sqlite3.OperationalError('locked')):
            response = self.client.get('/api/profile')
            self.assertEqual(response.status_code, 503)
            self.assertIn('error', response.json)

    def test_profile_available_without_tv_data(self):
        with patch.object(web, 'load_programs', return_value=(self.tv, None, None)):
            response = self.client.get('/api/profile')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json['choices'], {'categories': [], 'channels': []})
            self.assertEqual(self.client.put('/api/profile', json={}).status_code, 200)
