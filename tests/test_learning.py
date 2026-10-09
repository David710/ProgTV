from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from test_core import TVProgram, pd, web
from test_personalization import candidate_data
from personalization import ProfileStore, content_id, personalize, validate_preferences
import explanations


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'profile.sqlite3'
        self.store = ProfileStore(self.path)
        self.programs = web.serialize(TVProgram().flatten_programs(candidate_data()))

    def tearDown(self):
        self.directory.cleanup()

    def test_learning_reverses_votes_without_overwriting_preferences(self):
        manual = self.store.save_preferences('p', {'keywords': ['polar']})
        first = dict(self.programs[0], desc='Voyage histoire cuisine locale')
        second = dict(self.programs[1], desc='Cuisine histoire découverte')
        self.store.save_feedback('p', first, 'like')
        tastes = self.store.tastes('p')
        self.assertEqual(tastes['liked_count'], 1)
        self.assertEqual(tastes['categories'], [{'value': 'Film', 'count': 1}])
        self.assertEqual(tastes['channels'], [{'value': 'TF1', 'count': 1}])
        self.assertEqual(tastes['keywords'], [])
        self.store.save_feedback('p', second, 'like')
        tastes = self.store.tastes('p')
        self.assertEqual({entry['value'] for entry in tastes['keywords']}, {'cuisine', 'histoire'})
        reopened = ProfileStore(self.path)
        self.assertEqual(reopened.tastes('p'), tastes)
        self.assertEqual(reopened.tastes('other')['liked_count'], 0)
        self.assertEqual(reopened.preferences('p'), manual)
        self.store.save_feedback('p', second, 'seen')
        self.assertEqual(self.store.tastes('p')['keywords'], [])
        self.store.remove_feedback('p', content_id(first))
        self.assertEqual(self.store.tastes('p')['liked_count'], 0)
        self.assertEqual(self.store.preferences('p'), manual)

    def test_identical_summaries_do_not_artificially_strengthen_topics(self):
        for i in range(2):
            self.store.save_feedback('p', dict(self.programs[i], desc='Histoire cuisine locale'), 'like')
        tastes = self.store.tastes('p')
        self.assertEqual(tastes['liked_count'], 2)
        self.assertEqual(tastes['keywords'], [])

    def test_legacy_votes_have_categories_but_no_invented_description(self):
        with self.store.connect() as db:
            db.execute("INSERT INTO feedback (profile_id, content_id, program_id, name, category, value) "
                       "VALUES ('p', 'k', 'i', 'Film', 'Film', 'like')")
        tastes = self.store.tastes('p')
        self.assertEqual(tastes['liked_count'], 1)
        self.assertEqual(tastes['keywords'], [])
        self.assertEqual(tastes['sources'][0]['description'], '')

    def test_learned_topics_affect_ranking_and_manual_exclusions_win(self):
        for i in range(2):
            self.store.save_feedback('p', dict(self.programs[i], desc=f'Histoire cuisine locale {i}'), 'like')
        feedback = self.store.feedback('p')
        candidates = TVProgram().flatten_programs(candidate_data())
        candidates.at[5, 'desc'] = 'Histoire et cuisine du monde'
        preferences = validate_preferences({})
        result = personalize(candidates, preferences, feedback, self.store.tastes('p'))
        self.assertTrue(any('Thèmes de vos J’aime' in reason for reason in result.loc[5, 'recommendation_reasons']))
        disabled = personalize(candidates, validate_preferences({'learn_from_likes': False}), feedback)
        self.assertFalse(any('Thèmes de vos J’aime' in reason for reason in disabled.loc[5, 'recommendation_reasons']))
        excluded = personalize(candidates, validate_preferences({'avoid_keywords': ['histoire']}), feedback)
        self.assertNotIn(candidates.iloc[5]['id'], excluded['id'].tolist())
        with self.assertRaises(ValueError):
            validate_preferences({'learn_from_likes': 'oui'})


class ExplanationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = ProfileStore(Path(self.directory.name) / 'profile.sqlite3')
        self.program = web.serialize(TVProgram().flatten_programs(candidate_data()))[0]
        self.program['desc'] = 'Un voyage historique autour des recettes locales.'
        self.preferences = validate_preferences({'keywords': ['recettes']})
        self.context = explanations.context_for(self.program, self.preferences, {}, self.store.tastes('p'))

    def tearDown(self):
        self.directory.cleanup()

    def test_base_is_yielded_before_llm_and_cache_survives_restart_and_rerun(self):
        generate = Mock(return_value='Le résumé indique : « recettes locales ».')
        events = explanations.explanation_events(self.store, 'p', self.program, self.context, generate)
        first = next(events)
        self.assertEqual(first['type'], 'base')
        generate.assert_not_called()
        final = list(events)[-1]
        self.assertEqual(final['type'], 'done')
        self.assertFalse(final['cached'])
        reopened = ProfileStore(self.store.path)
        rerun = dict(self.program, id='other-airing', start='2026-10-20T20:00:00+02:00')
        cached = list(explanations.explanation_events(reopened, 'p', rerun, self.context, generate))[-1]
        self.assertTrue(cached['cached'])
        self.assertEqual(generate.call_count, 1)
        other = list(explanations.explanation_events(reopened, 'other', self.program, self.context, generate))[-1]
        self.assertFalse(other['cached'])
        self.assertEqual(generate.call_count, 2)

    def test_cache_changes_with_context_model_prompt_and_duration(self):
        old = explanations.cache_key(self.program, self.context)
        for field, value in [('duration', 40), ('desc', 'Autre résumé'), ('cat', 'Sport')]:
            self.assertNotEqual(old, explanations.cache_key(dict(self.program, **{field: value}), self.context))
        changed = explanations.context_for(self.program, validate_preferences({'keywords': ['histoire']}), {}, self.store.tastes('p'))
        self.assertNotEqual(old, explanations.cache_key(self.program, changed))
        with patch.object(explanations, 'MODEL', 'different'):
            self.assertNotEqual(old, explanations.cache_key(self.program, self.context))
        with patch.object(explanations, 'PROMPT_VERSION', explanations.PROMPT_VERSION + 1):
            self.assertNotEqual(old, explanations.cache_key(self.program, self.context))

    def test_missing_summary_skips_llm_and_failures_do_not_enter_cache(self):
        missing = dict(self.program, desc=None)
        context = explanations.context_for(missing, self.preferences, {}, self.store.tastes('p'))
        generate = Mock(side_effect=RuntimeError('offline'))
        result = list(explanations.explanation_events(self.store, 'p', missing, context, generate))[-1]
        generate.assert_not_called()
        self.assertIn('résumé est absent', result['comment'])
        result = list(explanations.explanation_events(self.store, 'p', self.program, self.context, generate))[-1]
        self.assertEqual(result['type'], 'fallback')
        self.assertIn('recettes', result['comment'])
        self.assertIsNone(self.store.cached_explanation('p', explanations.cache_key(self.program, self.context)))

    def test_only_exact_quotes_are_accepted_and_llm_settings_are_bounded(self):
        for excerpt, accepted in [('recettes locales', True), ('football', False), ('', True), ('x' * 241, False)]:
            response = {'message': {'content': json.dumps({'explanation': 'Ce programme rejoint votre intérêt pour les recettes.', 'excerpt': excerpt})}, 'done_reason': 'stop'}
            with patch.object(explanations.ollama, 'Client') as client:
                client.return_value.chat.return_value = response
                if accepted:
                    result = explanations.generate_explanation(self.program, self.preferences, ['Mot apprécié : recettes'])
                    self.assertIn('rejoint votre intérêt', result)
                    self.assertEqual('Le résumé indique' in result, bool(excerpt))
                else:
                    with self.assertRaises(ValueError):
                        explanations.generate_explanation(self.program)
                options = client.return_value.chat.call_args.kwargs
                self.assertFalse(options['think'])
                self.assertEqual(options['keep_alive'], '30m')
                self.assertEqual(options['options']['num_predict'], 300)
                payload = json.loads(options['messages'][1]['content'])
                self.assertNotIn('preferences', payload)
                self.assertEqual(payload['correspondance_etablie'], accepted)
                self.assertEqual(payload['titre'], self.program['name'])
                self.assertEqual(payload['duree_minutes'], self.program['duration'])

    def test_missing_empty_or_overlong_explanation_is_rejected(self):
        for explanation in (None, '', '   ', 'x' * 501):
            response = {'message': {'content': json.dumps({
                'explanation': explanation, 'excerpt': 'recettes locales'})}}
            with patch.object(explanations.ollama, 'Client') as client:
                client.return_value.chat.return_value = response
                with self.assertRaises(ValueError):
                    explanations.generate_explanation(self.program)

    def test_same_request_concurrently_generates_only_once(self):
        entered = threading.Event()
        release = threading.Event()
        def generate(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(3))
            return 'Citation'
        generator = Mock(side_effect=generate)
        def request():
            return list(explanations.explanation_events(self.store, 'p', self.program, self.context, generator))[-1]
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(request)
            self.assertTrue(entered.wait(3))
            second = pool.submit(request)
            release.set()
            replies = [first.result(), second.result()]
        self.assertEqual(generator.call_count, 1)
        self.assertEqual(sum(reply['cached'] for reply in replies), 1)


class LearningAPITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous_database = web.app.config['PROFILE_DATABASE']
        web.app.config['PROFILE_DATABASE'] = str(Path(self.directory.name) / 'profile.sqlite3')
        self.tv = TVProgram()
        self.data = candidate_data()
        self.loader = patch.object(web, 'load_programs', return_value=(self.tv, self.data, '2026-10-10'))
        self.loader.start()
        self.client = web.app.test_client()
        self.program = self.client.get('/api/programs?view=tomorrow').json[0]

    def tearDown(self):
        self.loader.stop()
        web.app.config['PROFILE_DATABASE'] = self.previous_database
        self.directory.cleanup()

    def test_votes_update_export_and_removal_reverses_learning(self):
        url = f"/api/programs/{self.program['id']}/feedback"
        self.client.put(url, json={'value': 'like'})
        export = self.client.get('/api/profile/tastes')
        self.assertEqual(export.status_code, 200)
        self.assertIn('attachment', export.headers['Content-Disposition'])
        self.assertEqual(export.json['learned_tastes']['liked_count'], 1)
        self.assertEqual(self.client.get('/api/profile').json['learned_tastes']['liked_count'], 1)
        self.assertEqual(web.app.test_client().get('/api/profile/tastes').json['learned_tastes']['liked_count'], 0)
        self.client.delete('/api/feedback/' + content_id(self.program))
        self.assertEqual(self.client.get('/api/profile/tastes').json['learned_tastes']['liked_count'], 0)

    def test_stream_starts_with_grounded_reasons_and_falls_back_offline(self):
        url = f"/api/programs/{self.program['id']}/comment?stream=1"
        with patch.object(self.tv, 'get_ollama_comment', side_effect=RuntimeError('offline')) as generate:
            response = self.client.get(url, buffered=False)
            self.assertEqual(response.mimetype, 'application/x-ndjson')
            self.assertEqual(response.headers['Cache-Control'], 'private, no-store')
            first = next(response.response)
            self.assertEqual(json.loads(first)['type'], 'base')
            generate.assert_not_called()
            rest = b''.join(response.response)
            self.assertEqual(json.loads(rest)['type'], 'fallback')
            response.close()


class ProgramCacheTests(unittest.TestCase):
    def test_cache_reuses_data_and_refreshes_after_file_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'programs.pkl'
            data = candidate_data()
            data.to_pickle(path)
            stat = path.stat()
            first = web.cached_programs(str(path), stat.st_mtime_ns, stat.st_size)
            second = web.cached_programs(str(path), stat.st_mtime_ns, stat.st_size)
            self.assertIs(first, second)
            changed = candidate_data()
            changed.iloc[0]['programs'].at[0, 'name'] = 'Programme actualisé'
            changed.to_pickle(path)
            stat = path.stat()
            refreshed = web.cached_programs(str(path), stat.st_mtime_ns, stat.st_size)
            self.assertEqual(refreshed.iloc[0]['programs'].iloc[0]['name'], 'Programme actualisé')
            self.assertIsNot(first, refreshed)
