from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_core import TVProgram, web, pd
from test_personalization import candidate_data
from calendar_export import calendar_event


class FavoritesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous_database = web.app.config['PROFILE_DATABASE']
        web.app.config['PROFILE_DATABASE'] = str(Path(self.directory.name) / 'profile.sqlite3')
        self.tv = TVProgram()
        self.loader = patch.object(web, 'load_programs', return_value=(self.tv, candidate_data(), '2026-10-09'))
        self.loader.start()
        self.client = web.app.test_client()
        self.program = self.client.get('/api/programs?view=tomorrow').json[0]
        self.url = f"/api/programs/{self.program['id']}/favorite"
        self.calendar = f"/api/programs/{self.program['id']}/calendar"

    def tearDown(self):
        self.loader.stop()
        web.app.config['PROFILE_DATABASE'] = self.previous_database
        self.directory.cleanup()

    def test_persistence_isolation_and_independence_from_votes(self):
        feedback = f"/api/programs/{self.program['id']}/feedback"
        self.client.put(feedback, json={'value': 'like'})
        for _ in range(2):
            self.assertEqual(self.client.put(self.url).json, {'favorite': True})
        profile = self.client.get('/api/profile').json
        self.assertEqual(len(profile['favorites']), 1)
        self.assertEqual(profile['favorites'][0]['desc'], self.program['desc'])
        rows = self.client.get('/api/programs?view=tomorrow').json
        self.assertTrue(next(row for row in rows if row['id'] == self.program['id'])['favorite'])
        other = web.app.test_client()
        self.assertEqual(other.get('/api/profile').json['favorites'], [])
        other.delete(self.url)
        self.assertEqual(len(self.client.get('/api/profile').json['favorites']), 1)
        reopened = web.app.test_client()
        reopened.set_cookie('progtv_profile', self.client.get_cookie('progtv_profile').value)
        with patch.object(web, 'load_programs', return_value=(self.tv, None, None)):
            self.assertEqual(len(reopened.get('/api/profile').json['favorites']), 1)
            self.assertEqual(reopened.get(self.calendar).status_code, 200)
            self.assertEqual(other.get(self.calendar).status_code, 503)
            self.assertEqual(reopened.delete(self.url).status_code, 200)
            self.assertEqual(reopened.get('/api/profile').json['favorites'], [])
        self.assertEqual(self.client.get('/api/profile').json['feedback'][0]['value'], 'like')

    def test_download_validation_and_timezone(self):
        response = self.client.get(self.calendar)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'text/calendar')
        self.assertIn('attachment', response.headers['Content-Disposition'])
        self.assertEqual(response.headers['Cache-Control'], 'private, no-store')
        start = pd.Timestamp(self.program['start']).tz_convert('UTC').strftime('%Y%m%dT%H%M%SZ')
        self.assertIn(f'DTSTART:{start}\r\n', response.text)
        self.assertIn('TRIGGER:-PT15M', response.text)
        self.assertNotIn('VALARM', self.client.get(self.calendar + '?reminder=0').text)
        for minutes in ['5', '30', '60']:
            self.assertIn(f'TRIGGER:-PT{minutes}M', self.client.get(self.calendar + '?reminder=' + minutes).text)
        for value in ['-1', 'abc', 'True', '1440', '1.5']:
            self.assertEqual(self.client.get(self.calendar + '?reminder=' + value).status_code, 400)
        self.assertEqual(self.client.put('/api/programs/missing/favorite').status_code, 404)
        self.assertEqual(self.client.get('/api/programs/missing/calendar').status_code, 404)
        with patch.object(web, 'find_program', return_value=(self.tv, dict(self.program, start=None), None)):
            self.assertEqual(self.client.get(self.calendar).status_code, 422)


class CalendarTests(unittest.TestCase):
    def test_unicode_folding_injection_and_dst(self):
        program = dict(id='abc', name='Été, polar; \\ suite',
                       desc=('é' * 120) + '\r\nBEGIN:VEVENT', channel_name='TF1',
                       start='2026-10-25T02:30:00+02:00',
                       end='2026-10-25T02:30:00+01:00')
        exported = calendar_event(program)
        self.assertTrue(exported.endswith('END:VCALENDAR\r\n'))
        self.assertTrue(all(len(line.encode('utf-8')) <= 75 for line in exported.split('\r\n')))
        unfolded = exported.replace('\r\n ', '')
        self.assertEqual(unfolded.count('\r\nBEGIN:VEVENT\r\n'), 1)
        self.assertIn('DTSTART:20261025T003000Z', unfolded)
        self.assertIn('DTEND:20261025T013000Z', unfolded)
        self.assertIn('Été\\, polar\\; \\\\ suite', unfolded)
        self.assertIn('\\nBEGIN:VEVENT', unfolded)
        uid = next(line for line in unfolded.split('\r\n') if line.startswith('UID:'))
        self.assertIn(uid, calendar_event(program, 0).replace('\r\n ', ''))

    def test_invalid_dates(self):
        for start, end in [('2026-10-09T20:00:00', '2026-10-09T21:00:00'),
                           ('2026-10-09T20:00:00+02:00', '2026-10-09T19:00:00+02:00'),
                           ('bad', None)]:
            with self.assertRaises(ValueError):
                calendar_event(dict(id='abc', start=start, end=end))
