"""Export iCalendar d’une diffusion, avec un rappel facultatif."""

from datetime import datetime, timezone
import hashlib


def escape_text(value):
    text = str(value or '').replace('\\', '\\\\')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text.replace('\n', '\\n').replace(';', '\\;').replace(',', '\\,')


def fold_line(line):
    # RFC 5545 : au plus 75 octets, sans couper un caractère UTF-8.
    parts = []
    current = ''
    for char in line:
        if len((current + char).encode('utf-8')) > 75:
            parts.append(current)
            current = ' '
        current += char
    parts.append(current)
    return '\r\n'.join(parts)


def calendar_event(program, reminder=15):
    times = []
    for field in ('start', 'end'):
        value = datetime.fromisoformat(str(program.get(field) or ''))
        if value.tzinfo is None:
            raise ValueError('Horaire sans fuseau.')
        times.append(value.astimezone(timezone.utc))
    if times[1] <= times[0]:
        raise ValueError('Durée invalide.')
    uid = hashlib.sha256(str(program['id']).encode()).hexdigest()
    lines = ['BEGIN:VCALENDAR', 'VERSION:2.0',
             'PRODID:-//ProgTV//Guide TV//FR', 'CALSCALE:GREGORIAN',
             'BEGIN:VEVENT', f'UID:{uid}@progtv.local',
             'DTSTAMP:' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
             'DTSTART:' + times[0].strftime('%Y%m%dT%H%M%SZ'),
             'DTEND:' + times[1].strftime('%Y%m%dT%H%M%SZ'),
             'SUMMARY:' + escape_text(program.get('name')),
             'LOCATION:' + escape_text(program.get('channel_name')),
             'DESCRIPTION:' + escape_text(program.get('desc'))]
    if reminder:
        lines += ['BEGIN:VALARM', f'TRIGGER:-PT{reminder}M',
                  'ACTION:DISPLAY', 'DESCRIPTION:' + escape_text(program.get('name')),
                  'END:VALARM']
    lines += ['END:VEVENT', 'END:VCALENDAR']
    return '\r\n'.join(fold_line(line) for line in lines) + '\r\n'
