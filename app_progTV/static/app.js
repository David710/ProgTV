const programsDiv = document.getElementById('programs');
const pageTitle = document.getElementById('page-title');
const statusDiv = document.getElementById('status');
let activeRequest;

function element(tag, className, text) {
    const node = document.createElement(tag);
    node.className = className;
    if (text != null) node.textContent = text;
    return node;
}

function safeImage(url, alt, className) {
    const img = element('img', className);
    img.alt = alt;
    img.loading = 'lazy';
    try {
        const parsed = new URL(url);
        if (['https:', 'http:'].includes(parsed.protocol)) img.src = parsed.href;
    } catch (_) { /* Une URL absente ne doit pas casser la carte. */ }
    img.addEventListener('error', () => { img.hidden = true; });
    return img;
}

function formatDuration(minutes) {
    if (!Number.isFinite(minutes) || minutes < 0) return 'Durée inconnue';
    const total = Math.round(minutes);
    return total >= 60 ? `${Math.floor(total / 60)}h ${total % 60}min` : `${total}min`;
}

function renderProgram(program, suggestions) {
    const card = element('article', 'card p-2');
    card.append(safeImage(program.icon, program.name || 'Programme', 'card-img-top'));
    const body = element('div', 'card-body');
    const header = element('div', 'd-flex align-items-center gap-2 mb-3');
    header.append(safeImage(program.channel_icon, program.channel_name || 'Chaîne', 'channel-icon'));
    const date = new Date(program.start);
    const time = Number.isNaN(date.getTime()) ? 'Horaire inconnu' : new Intl.DateTimeFormat('fr-FR', {
        timeZone: 'Europe/Paris', hour: '2-digit', minute: '2-digit',
        ...(suggestions ? { weekday: 'long' } : {})
    }).format(date);
    header.append(element('span', 'badge text-bg-light', time));
    body.append(header, element('h2', 'h5 card-title', program.name),
        element('p', 'text-muted', program.channel_name),
        element('p', 'card-text', program.desc || 'Résumé indisponible.'));
    const details = element('div', 'd-flex flex-wrap gap-2 mb-3');
    const score = Number.isFinite(program.note_pred) ? `Affinité : ${program.note_pred.toFixed(2)}` : 'Affinité indisponible';
    for (const label of [program.rating, program.cat, score, formatDuration(program.duration)]) {
        if (label) details.append(element('span', 'badge text-bg-secondary', label));
    }
    body.append(details);
    if (suggestions) {
        const button = element('button', 'btn btn-outline-primary', 'Pourquoi je vais aimer ?');
        button.type = 'button';
        const comment = element('p', 'ai-comments mt-3');
        comment.hidden = true;
        let loaded = false;
        button.addEventListener('click', async () => {
            if (loaded) {
                comment.hidden = !comment.hidden;
                button.textContent = comment.hidden ? 'Pourquoi je vais aimer ?' : 'Cacher le commentaire';
                return;
            }
            button.disabled = true;
            comment.hidden = false;
            comment.textContent = 'Préparation de l’explication…';
            try {
                const response = await fetch(`/api/programs/${encodeURIComponent(program.id)}/comment`);
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Explication indisponible.');
                comment.textContent = data.comment;
                loaded = true;
                button.textContent = 'Cacher le commentaire';
            } catch (error) {
                comment.textContent = error.message;
            } finally {
                button.disabled = false;
            }
        });
        body.append(button, comment);
    }
    card.append(body);
    return card;
}

async function loadPrograms(suggestions = false) {
    if (activeRequest) activeRequest.abort();
    const controller = new AbortController();
    activeRequest = controller;
    const date = new Intl.DateTimeFormat('fr-FR', {
        timeZone: 'Europe/Paris', weekday: 'long', day: 'numeric', month: 'long'
    }).format(new Date());
    pageTitle.textContent = `${suggestions ? 'Suggestions' : 'Ce soir'} — ${date}`;
    programsDiv.replaceChildren();
    programsDiv.setAttribute('aria-busy', 'true');
    statusDiv.textContent = 'Chargement des programmes…';
    document.getElementById('prog-day-link').classList.toggle('active', !suggestions);
    document.getElementById('suggestions-link').classList.toggle('active', suggestions);
    try {
        const response = await fetch(suggestions ? '/api/suggestions' : '/api/programs', { signal: controller.signal });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Impossible de charger les programmes.');
        statusDiv.textContent = response.headers.get('X-Programs-Stale') === 'true'
            ? `Dernières données disponibles : ${response.headers.get('X-Programs-Date')}.`
            : '';
        if (!data.length) statusDiv.textContent += ' Aucun programme disponible pour cette sélection.';
        programsDiv.replaceChildren(...data.map(program => renderProgram(program, suggestions)));
    } catch (error) {
        if (error.name !== 'AbortError') statusDiv.textContent = error.message;
    } finally {
        if (activeRequest === controller) programsDiv.setAttribute('aria-busy', 'false');
    }
}

for (const [id, suggestions] of [['prog-day-link', false], ['suggestions-link', true]]) {
    document.getElementById(id).addEventListener('click', event => {
        event.preventDefault();
        loadPrograms(suggestions);
    });
}
loadPrograms();
