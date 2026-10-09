import sys
import tempfile
import unittest
from datetime import datetime, date
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app_progTV'))
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from progtv import TVProgram, NeuralNetwork
import app as web
web.app.config['LLM_WARMUP'] = False


def dataset():
    return pd.DataFrame([{'name': 'TF1', 'icon': '', 'programs': pd.DataFrame([
        {'name': 'Film', 'start': pd.Timestamp('2026-10-07T20:30:00+02:00'),
         'end': pd.Timestamp('2026-10-07T22:30:00+02:00'), 'note_pred': .8,
         'desc': '<script>unsafe</script>', 'cat': 'Film', 'rating': 'Tout public', 'icon': ''},
        {'name': 'Film', 'start': pd.Timestamp('2026-10-08T20:30:00+02:00'),
         'end': pd.Timestamp('2026-10-08T22:30:00+02:00'), 'note_pred': .9,
         'desc': '', 'cat': 'Film', 'rating': 'Tout public', 'icon': ''}
    ])}])


class ProgramsTests(unittest.TestCase):
    def setUp(self):
        self.tv = TVProgram()

    def test_prime_includes_program_already_started(self):
        result = self.tv.get_prime_programs(dataset(), date(2026, 10, 7))
        self.assertEqual(result['name'].tolist(), ['Film'])
        self.assertEqual(result.iloc[0]['start'].hour, 20)

    def test_suggestions_limit_filter_and_distinct_airings(self):
        now = datetime(2026, 10, 7, 12, tzinfo=ZoneInfo('Europe/Paris'))
        results = self.tv.get_best_programs(dataset(), n=2, now=now)
        self.assertEqual(len(results), 2)
        self.assertEqual(results['id'].nunique(), 2)
        self.assertEqual(results.iloc[0]['note_pred'], .9)
        self.assertEqual(len(self.tv.get_best_programs(dataset(), n=1, now=now)), 1)
        self.assertTrue(self.tv.get_best_programs(dataset(), whitelist=['M6'], now=now).empty)
        self.assertTrue(self.tv.get_best_programs(dataset(), n=0, now=now).empty)

    def test_daylight_saving_conversion(self):
        unix = [pd.Timestamp('2026-03-29T00:30Z').timestamp(), pd.Timestamp('2026-03-29T01:30Z').timestamp()]
        results = self.tv.format_programs([{'start': t, 'end': t + 3600} for t in unix])
        self.assertEqual(results['start'].dt.hour.tolist(), [23, 0])
        self.assertEqual((results['end'] - results['start']).dt.total_seconds().tolist(), [3600, 3600])

    def test_api_times_are_corrected_without_changing_duration(self):
        timestamp = pd.Timestamp('2026-10-08T22:00+02:00').timestamp()
        result = self.tv.format_programs([
            dict(start=timestamp, end=timestamp + 2700)])
        self.assertEqual(result.iloc[0]['start'], pd.Timestamp('2026-10-08T20:00+02:00'))
        self.assertEqual(result.iloc[0]['end'], pd.Timestamp('2026-10-08T20:45+02:00'))

    def test_legacy_cache_correction_is_idempotent_and_preserves_scores(self):
        original = dataset()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.pkl'
            original.to_pickle(path)
            corrected = self.tv.read_programs(path)
            programs = corrected.iloc[0]['programs']
            self.assertEqual(programs.iloc[0]['start'],
                             pd.Timestamp('2026-10-07T18:30+02:00'))
            self.assertEqual(programs['note_pred'].tolist(), [.8, .9])
            corrected.to_pickle(path)
            reopened = self.tv.read_programs(path)
            pd.testing.assert_frame_equal(programs, reopened.iloc[0]['programs'])
            self.assertEqual(original.iloc[0]['programs'].iloc[0]['start'],
                             pd.Timestamp('2026-10-07T20:30+02:00'))

    def test_new_api_cache_does_not_receive_correction_twice(self):
        start = pd.Timestamp('2026-10-08T22:00+02:00').timestamp()
        raw = pd.DataFrame([dict(name='TF1', programs=[
            dict(name='JT 20h', start=start, end=start + 2700, note_pred=.5)])])
        corrected = self.tv.filter_programs(raw, ['TF1'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'new.pkl'
            corrected.to_pickle(path)
            reopened = self.tv.read_programs(path)
        self.assertEqual(reopened.iloc[0]['programs'].iloc[0]['start'],
                         pd.Timestamp('2026-10-08T20:00+02:00'))

    def test_prediction_reuses_saved_preprocessing(self):
        with tempfile.TemporaryDirectory() as folder:
            self.tv.download_folder = Path(folder)
            self.tv.category_maps = {'rating': {'Tout public': 3}, 'cat': {'Film': 7}}
            self.tv.scaler = StandardScaler().fit([[0, 0, 0], [6, 14, 2]])
            model = NeuralNetwork(3)
            model.eval()
            path = Path(folder) / 'model.pth'
            self.tv.save_model(model, path)
            other = TVProgram()
            other.download_folder = Path(folder)
            loaded = other.load_model(path, 770)
            data = dataset()
            data.iloc[0]['programs']['embeddings_camembert'] = [np.array([1.]), np.array([1.])]
            results = other.rate_programs(loaded, data, 'camembert')
            with torch.no_grad():
                expected = loaded(torch.zeros((2, 3))).numpy().ravel()
            np.testing.assert_allclose(results.iloc[0]['programs']['note_pred'], expected)
            encoded = other.encode_categories(pd.DataFrame({'rating': ['unknown'], 'cat': ['unknown']}))
            np.testing.assert_equal(encoded, [[-1, -1]])

    def test_training_saves_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            self.tv.train_folder = Path(folder)
            pd.DataFrame({
                'cat': ['Film', 'Sport'] * 10, 'rating': ['Tout public'] * 20,
                'embeddings': [np.array([i / 20]) for i in range(20)],
                'note': [i / 20 for i in range(20)]
            }).to_pickle(Path(folder) / 'training.pkl')
            with patch('builtins.print'):
                model = self.tv.train_model('training.pkl')
                restored = TVProgram().load_model(Path(folder) / 'trained_model_v2.pth', 770)
            self.assertEqual(restored.fc1.in_features, 3)
            model.eval()
            with torch.no_grad():
                np.testing.assert_allclose(model(torch.zeros(1, 3)).numpy(), restored(torch.zeros(1, 3)).numpy())

    def test_first_launch_trains_once_and_preserves_legacy_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            self.tv.train_folder = Path(folder)
            legacy = Path(folder) / 'trained_model.pth'
            legacy.write_bytes(b'historical weights')
            pd.DataFrame({
                'cat': ['Film'] * 10, 'rating': ['Tout public'] * 10,
                'embeddings': [np.array([i / 10]) for i in range(10)],
                'note': [i / 10 for i in range(10)]
            }).to_pickle(Path(folder) / 'df_programs_tf1_note.pkl')
            with patch('builtins.print'), patch.object(self.tv, 'train_model', wraps=self.tv.train_model) as train:
                model = self.tv.ensure_model()
                self.tv.ensure_model()
                train.assert_called_once()
            self.assertEqual(model.fc1.in_features, 3)
            self.assertEqual(legacy.read_bytes(), b'historical weights')

    def test_embeddings_accept_missing_or_numeric_descriptions(self):
        tokenizer = Mock(return_value={"attention_mask": torch.ones(1, 1, dtype=torch.long)})
        model = Mock(return_value=SimpleNamespace(last_hidden_state=torch.zeros(1, 1, 768)))
        transformers = SimpleNamespace(
            CamembertTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
            CamembertModel=SimpleNamespace(from_pretrained=Mock(return_value=model)))
        data = pd.DataFrame([dict(programs=pd.DataFrame({'desc': [None, np.nan, 12]}))])
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {'transformers': transformers}):
            results = self.tv.generate_embeddings(data, 'camembert', Path(folder) / 'programs.pkl')
        self.assertEqual([call.args[0] for call in tokenizer.call_args_list], ['', '12.0'])
        self.assertEqual(len(results.iloc[0]['programs']['embeddings_camembert']), 3)

    def test_training_file_can_be_absolute_and_errors_are_clear(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'training.pkl'
            source.write_bytes(b'placeholder')
            self.assertEqual(self.tv.training_file(source), source)
            with self.assertRaisesRegex(FileNotFoundError, 'Jeu annoté introuvable'):
                self.tv.training_file(Path(folder) / 'missing.pkl')

    def test_empty_programs_are_supported(self):
        data = pd.DataFrame([{'name': 'TF1', 'programs': pd.DataFrame()}])
        self.assertTrue(self.tv.get_prime_programs(data).empty)
        self.assertTrue(self.tv.get_best_programs(data).empty)

    def test_legacy_weights_fail_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'réentraînez'):
                self.tv.load_model(Path(folder) / 'legacy.pth', 770)


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = web.app.test_client()


    def test_page_and_missing_data(self):
        with patch.object(web, 'load_programs', return_value=(TVProgram(), None, None)):
            self.assertEqual(self.client.get('/').status_code, 200)
            response = self.client.get('/api/programs')
            self.assertEqual(response.status_code, 503)
            self.assertIn('error', response.json)

    def test_dates_include_offset(self):
        records = web.serialize(TVProgram().flatten_programs(dataset()))
        self.assertTrue(records[0]['start'].endswith('+02:00'))

    def test_comments_cached_and_errors_recoverable(self):
        tv = TVProgram()
        program_id = tv.flatten_programs(dataset()).iloc[0]['id']
        with patch.object(web, 'load_programs', return_value=(tv, dataset(), '2026-10-07')):
            with patch.object(tv, 'get_ollama_comment', return_value='<b>Explication</b>') as generate:
                for _ in range(2):
                    response = self.client.get(f'/api/programs/{program_id}/comment')
                    self.assertIn('<b>Explication</b>', response.json['comment'])
                generate.assert_called_once()
            self.assertEqual(self.client.get('/api/programs/unknown/comment').status_code, 404)

            with patch.object(tv, 'get_ollama_comment', side_effect=RuntimeError('offline')):
                self.client.put('/api/profile', json={'keywords': ['changed']})
                response = self.client.get(f'/api/programs/{program_id}/comment')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json['type'], 'fallback')

    def test_last_valid_cache_and_freshness(self):
        with tempfile.TemporaryDirectory() as folder:
            tv = TVProgram()
            tv.download_folder = Path(folder)
            dataset().to_pickle(Path(folder) / 'progtv_rated_2026-10-06.pkl')
            (Path(folder) / 'progtv_rated_2026-10-07.pkl').write_bytes(b'invalid')
            with patch.object(web.progtv, 'TVProgram', return_value=tv):
                _, data, data_date = web.load_programs()
                self.assertIsNotNone(data)
                self.assertEqual(data_date, '2026-10-06')
                response = self.client.get('/api/programs')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers['X-Programs-Stale'], 'true')


if __name__ == '__main__':
    unittest.main()
