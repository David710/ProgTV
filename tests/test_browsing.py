import json
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from test_core import TVProgram, web, pd


def guide():
    rows = [
        ('France 2', 'Cuisine été', '2026-10-07T20:00+02:00', '2026-10-07T21:00+02:00', .2, 'Cuisine', 'Crème et café'),
        ('France 2', 'Polar', '2026-10-07T21:00+02:00', '2026-10-07T22:30+02:00', .9, 'Film', 'Enquête'),
        ('TF1', 'Action', '2026-10-07T20:30+02:00', '2026-10-07T22:30+02:00', .8, 'Film', 'Action'),
        ('TF1', 'Sport', '2026-10-08T00:00+02:00', '2026-10-08T01:00+02:00', .7, 'Sport', ''),
        ('TF1', 'Demain soir', '2026-10-08T21:00+02:00', '2026-10-08T22:00+02:00', .6, 'Film', ''),
        ('TF1', 'Après-demain', '2026-10-09T00:00+02:00', '2026-10-09T01:00+02:00', .5, 'Film', ''),
    ]
    channels = {}
    for channel, name, start, end, score, category, desc in rows:
        channels.setdefault(channel, []).append(dict(
            name=name, start=pd.Timestamp(start), end=pd.Timestamp(end),
            note_pred=score, cat=category, desc=desc, icon='', rating='Tout public'))
    return pd.DataFrame([dict(name=channel, icon='', programs=pd.DataFrame(programs))
                         for channel, programs in channels.items()])


class BrowsingTests(unittest.TestCase):
    def setUp(self):
        self.tv = TVProgram()
        self.now = datetime(2026, 10, 7, 21, tzinfo=ZoneInfo('Europe/Paris'))

    def select(self, **kwargs):
        return self.tv.select_programs(guide(), now=self.now, **kwargs)

    def test_now_includes_start_and_excludes_end(self):
        programs, _, _ = self.select(view='now')
        self.assertEqual(set(programs['name']), {'Polar', 'Action'})
        programs, _, _ = self.select(view='tonight')
        self.assertEqual(set(programs['name']), {'Polar', 'Action'})

    def test_tomorrow_full_day_with_exclusive_end(self):
        programs, _, selected_date = self.select(view='tomorrow')
        self.assertEqual(programs['name'].tolist(), ['Sport', 'Demain soir'])
        self.assertEqual(str(selected_date), '2026-10-08')

    def test_tomorrow_uses_local_day_at_utc_midnight(self):
        _, _, selected_date = self.tv.select_programs(
            guide(), view='tomorrow', now=pd.Timestamp('2026-10-07T22:30Z'))
        self.assertEqual(str(selected_date), '2026-10-09')

    def test_tomorrow_daylight_saving_is_calendar_day(self):
        programs = pd.DataFrame([
            dict(name=name, start=pd.Timestamp(start), end=pd.Timestamp(start).to_pydatetime() + timedelta(minutes=30), note_pred=.5)
            for name, start in [('early', '2026-10-25T00:00+02:00'),
                                ('late', '2026-10-25T23:30+01:00'),
                                ('next', '2026-10-26T00:00+01:00')]])
        data = pd.DataFrame([dict(name='TF1', programs=programs)])
        result, _, _ = self.tv.select_programs(data, view='tomorrow', now=pd.Timestamp('2026-10-24T12:00+02:00'))
        self.assertEqual(result['name'].tolist(), ['early', 'late'])

    def test_search_accents_case_and_literal_regex(self):
        self.now = datetime(2026, 10, 7, 12, tzinfo=ZoneInfo('Europe/Paris'))
        for query in ['  CUISINE ETE ', 'creme', 'CAFÉ', 'france 2']:
            programs, _, _ = self.select(view='suggestions', query=query)
            self.assertIn('Cuisine été', programs['name'].tolist())
        programs, _, _ = self.select(view='suggestions', query='.*')
        self.assertTrue(programs.empty)

    def test_filters_before_top_five_and_choices_do_not_disappear(self):
        programs, choices, _ = self.select(view='suggestions', category='Sport', channel='TF1', max_duration=60, limit=1)
        self.assertEqual(programs['name'].tolist(), ['Sport'])
        lower_ranked, _, _ = self.select(view='suggestions', category='Film', max_duration=60, limit=1)
        self.assertEqual(lower_ranked['name'].tolist(), ['Demain soir'])
        self.assertEqual(choices['categories'], ['Film', 'Sport'])
        programs, choices, _ = self.select(view='suggestions', channel='absent')
        self.assertTrue(programs.empty)
        self.assertEqual(choices['channels'], ['TF1'])
        programs, _, _ = self.select(view='tonight', max_duration=90)
        self.assertEqual(programs['name'].tolist(), ['Polar'])

    def test_prime_time_skips_access_and_short_weather_bulletins(self):
        rows = []
        schedules = {
            'TF1': [('Ici tout commence', '20:30', 35),
                    ('Météo', '21:05', 5), ('Film', '21:10', 110)],
            'France 2': [('Access long', '20:30', 70), ('Soirée', '21:15', 90)],
            'Arte': [('Film tôt', '20:55', 100)],
            'M6': [('Access seul', '20:30', 35)],
            'W9': [('Épisode 1', '21:00', 40), ('Épisode 2', '21:40', 40)],
        }
        for channel, schedule in schedules.items():
            programs = []
            for name, time, duration in schedule:
                start = pd.Timestamp(f'2026-10-07T{time}+02:00')
                programs.append(dict(name=name, start=start,
                                     end=start + pd.Timedelta(minutes=duration),
                                     note_pred=.5))
            rows.append(dict(name=channel, programs=pd.DataFrame(programs)))
        data = pd.DataFrame(rows)
        result, _, _ = self.tv.select_programs(data, view='tonight', now=self.now)
        self.assertEqual(result['name'].tolist(), ['Film', 'Soirée', 'Film tôt', 'Épisode 1'])
        self.assertEqual(self.tv.get_prime_programs(data, self.now.date())['id'].tolist(),
                         result['id'].tolist())
        # Le filtrage ne doit pas faire remonter l’access à la place du prime.
        result, _, _ = self.tv.select_programs(
            data, view='tonight', now=self.now, channel='TF1', max_duration=60)
        self.assertTrue(result.empty)

    def test_prime_time_hour_and_duration_boundaries(self):
        programs = []
        cases = [('trop tôt', '20:29', 180, False),
                 ('repli limite', '20:30', 60, True),
                 ('repli court', '20:59', 59, False),
                 ('épisode court', '21:00', 39, False),
                 ('épisode limite', '21:00', 40, True),
                 ('début limite', '21:30', 40, True),
                 ('trop tard', '21:31', 120, False)]
        for name, time, duration, _ in cases:
            start = pd.Timestamp(f'2026-10-07T{time}+02:00')
            programs.append(dict(name=name, programs=pd.DataFrame([
                dict(name=name, start=start, end=start + pd.Timedelta(minutes=duration),
                     note_pred=.5)])))
        result, _, _ = self.tv.select_programs(
            pd.DataFrame(programs), view='tonight', now=self.now)
        self.assertEqual(set(result['name']), {name for name, _, _, valid in cases if valid})

    def test_channel_order_precedes_time_and_sorts_filter_choices(self):
        rows = []
        for channel in ['Chérie 25', 'Z locale', 'France 3', 'Arte',
                        'W9', 'France 2', 'TF1', 'M6', 'A locale',
                        'Canal+', 'France 5', 'TF1 Séries-Films']:
            programs = pd.DataFrame([
                dict(name=f'{channel} tard', start=pd.Timestamp('2026-10-08T21:00+02:00'),
                     end=pd.Timestamp('2026-10-08T23:00+02:00'), note_pred=.5),
                dict(name=f'{channel} tôt', start=pd.Timestamp('2026-10-08T20:00+02:00'),
                     end=pd.Timestamp('2026-10-08T22:00+02:00'), note_pred=.8),
            ])
            rows.append(dict(name=channel, programs=programs))
        data = pd.DataFrame(rows)
        expected = ['TF1', 'France 2', 'France 3', 'Canal+', 'France 5',
                    'M6', 'Arte', 'W9', 'TF1 Séries-Films', 'Chérie 25',
                    'A locale', 'Z locale']
        result, choices, _ = self.tv.select_programs(
            data, view='tomorrow', now=self.now)
        self.assertEqual(choices['channels'], expected)
        self.assertEqual(result['channel_name'].tolist(),
                         [channel for channel in expected for _ in range(2)])
        self.assertEqual(result['name'].tolist(),
                         [f'{channel} {time}' for channel in expected
                          for time in ['tôt', 'tard']])
        for view in ['now', 'tonight']:
            result, _, _ = self.tv.select_programs(
                data, view=view, now=pd.Timestamp('2026-10-08T21:00+02:00'))
            self.assertEqual(result['channel_name'].drop_duplicates().tolist(), expected)

    def test_channel_name_variants_share_the_same_rank(self):
        from progtv import channel_sort_key
        for first, second in [('Arte', 'ARTE'), ('TF1 Séries-Films', 'TF1 SERIES FILMS'),
                              ('La chaine l’Équipe', "L'Equipe"),
                              ('Canal+', 'CANAL +')]:
            self.assertEqual(channel_sort_key(first), channel_sort_key(second))

    def test_empty_views_and_invalid_view(self):
        data = pd.DataFrame([dict(name='TF1', programs=pd.DataFrame())])
        for view in ['now', 'tonight', 'tomorrow', 'suggestions']:
            programs, choices, _ = self.tv.select_programs(data, view=view, now=self.now)
            self.assertTrue(programs.empty)
            self.assertEqual(choices, {'channels': [], 'categories': []})
        with self.assertRaises(ValueError):
            self.select(view='unknown')


class BrowsingAPITests(unittest.TestCase):
    def setUp(self):
        self.client = web.app.test_client()
        self.tv = TVProgram()

    def test_invalid_parameters_are_400_without_data_access(self):
        with patch.object(web, 'load_programs') as load:
            for query in ['view=unknown', 'max_duration=0', 'max_duration=-1',
                          'max_duration=1441', 'max_duration=abc', 'max_duration=1.5']:
                response = self.client.get('/api/programs?' + query)
                self.assertEqual(response.status_code, 400, query)
                self.assertIn('error', response.json)
            load.assert_not_called()

    def test_api_filter_arguments_and_metadata(self):
        real_select = self.tv.select_programs
        now = pd.Timestamp('2026-10-07T21:00+02:00')
        with patch.object(web, 'load_programs', return_value=(self.tv, guide(), '2026-10-07')):
            with patch.object(self.tv, 'select_programs', side_effect=lambda *args, **kwargs: real_select(*args, **kwargs, now=now)):
                response = self.client.get('/api/programs?view=tomorrow&channel=TF1&category=Sport&max_duration=60&q=sport')
                self.assertEqual(response.status_code, 200)
                self.assertEqual([p['name'] for p in response.json], ['Sport'])
                self.assertEqual(response.headers['X-Programs-View-Date'], '2026-10-08')
                choices = json.loads(response.headers['X-Programs-Filters'])
                self.assertEqual(choices['categories'], ['Film', 'Sport'])
                response = self.client.get('/api/suggestions?category=Sport')
                self.assertEqual([p['name'] for p in response.json], ['Sport'])
                response = self.client.get('/api/programs')
                self.assertEqual(set(p['name'] for p in response.json), {'Polar', 'Action'})


if __name__ == '__main__':
    unittest.main()
