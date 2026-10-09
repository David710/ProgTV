"""Goûts calculés à partir des avis, sans inférence de préférences par un LLM."""

from collections import Counter
import hashlib
import json
import re
import unicodedata

STOP_WORDS = set('''avec alors apres avant aussi autre autres avoir cette ces cet
chez comme dans depuis deux des donc dont elle elles entre est etre fait font
fois ils leur leurs lui mais meme moins monde nous nouveau nouvelle nouvelles
par pas pendant plus pour programme programmes retrouve sans ses sont sous sur
tout tous toute toutes tres une vous chaque jour jours episode episodes serie
saison emission emissions presente presentee premier premiere donne donnees
cette ceux cela ainsi afin aux bien face film encore lorsque peut qui que quoi
quel quelle quels quelles son sa se on ne du de la le les un et ou au en il ce
'''.split())


def normalize(text):
    value = unicodedata.normalize('NFKD', str(text or '').casefold())
    return ''.join(char for char in value if not unicodedata.combining(char))


def words(text):
    return {word for word in re.findall(r'[a-z]{4,}', normalize(text))
            if word not in STOP_WORDS}


def learn_tastes(feedback):
    liked = [dict(row, content_id=key) for key, row in feedback.items()
             if row['value'] == 'like']
    categories = Counter(row['category'] for row in liked if row['category'])
    channels = Counter()
    documents = Counter()
    sources = []
    descriptions = set()
    for row in liked:
        program = row.get('program') or {}
        channel = program.get('channel_name')
        if channel:
            channels[channel] += 1
        # Chaque mot compte une seule fois par contenu apprécié.
        description = normalize(program.get('desc', '')).strip()
        if description and description not in descriptions:
            documents.update(words(description))
            descriptions.add(description)
        sources.append({'content_id': row['content_id'], 'name': row.get('name', ''),
                        'category': row['category'],
                        'description': program.get('desc') or '',
                        'channel': channel or ''})
    keywords = [{'value': word, 'count': count}
                for word, count in sorted(documents.items(),
                                          key=lambda item: (-item[1], item[0]))
                if count >= 2][:20]

    def counted(counter):
        return [{'value': value, 'count': count}
                for value, count in sorted(counter.items(),
                                           key=lambda item: (-item[1], item[0]))]

    if not liked:
        summary = 'Aucun goût appris : ajoutez des avis J’aime.'
    else:
        summary = f'{len(liked)} contenu(s) aimé(s).'
        if categories:
            summary += ' Genres appréciés : ' + ', '.join(
                entry['value'] for entry in counted(categories)[:5]
            ) + '.'
        if channels:
            summary += ' Chaînes de vos J’aime : ' + ', '.join(
                entry['value'] for entry in counted(channels)[:3]
            ) + '.'
        if keywords:
            summary += ' Thèmes récurrents dans les résumés : ' + ', '.join(
                entry['value'] for entry in keywords[:8]
            ) + '.'
    digest = hashlib.sha256(json.dumps(
        sorted(sources, key=lambda row: row['content_id']), sort_keys=True,
        ensure_ascii=False,
    ).encode()).hexdigest()
    return {'version': 1, 'fingerprint': digest, 'liked_count': len(liked),
            'categories': counted(categories), 'channels': counted(channels),
            'keywords': keywords, 'summary': summary, 'sources': sources}
