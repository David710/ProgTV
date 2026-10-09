const programsDiv = document.getElementById('programs');
const pageTitle = document.getElementById('page-title');
const statusDiv = document.getElementById('status');
let activeRequest;
let explanationVersion = 0;
const commentRequests = new Set();

function invalidateExplanations() {
    explanationVersion += 1;
    for (const controller of commentRequests) controller.abort();
    commentRequests.clear();
}

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
    if (suggestions && program.recommendation_reasons?.length) {
        body.append(element('p', 'tw-text-sm tw-text-blue-800', program.recommendation_reasons.join(' · ')));
    }
    if (suggestions) {
        const button = element('button', 'btn btn-outline-primary', 'Pourquoi ce programme ?');
        button.type = 'button';
        const comment = element('p', 'ai-comments mt-3 tw-whitespace-pre-line');
        comment.setAttribute('role', 'status');
        comment.hidden = true;
        let loadedVersion = -1;
        const renderedVersion = explanationVersion;
        button.addEventListener('click', async () => {
            if (loadedVersion === explanationVersion) {
                comment.hidden = !comment.hidden;
                button.textContent = comment.hidden ? 'Pourquoi ce programme ?' : 'Cacher l’explication';
                return;
            }
            const version = explanationVersion;
            const controller = new AbortController();
            commentRequests.add(controller);
            button.disabled = true;
            comment.hidden = false;
            comment.textContent = renderedVersion === version && program.recommendation_reasons?.length
                ? program.recommendation_reasons.join(' · ') + '. Complément en préparation…'
                : 'Analyse de vos préférences et avis…';
            let complete = false;
            try {
                const response = await fetch(`/api/programs/${encodeURIComponent(program.id)}/comment?stream=1`, {
                    signal: controller.signal
                });
                if (!response.ok) {
                    const data = await response.json();
                    throw new Error(data.error || 'Explication indisponible.');
                }
                const apply = data => {
                    if (controller.signal.aborted || version !== explanationVersion) return;
                    if (data.comment) comment.textContent = data.comment;
                    if (data.type === 'fallback') {
                        comment.textContent += '\n' + data.message;
                        button.textContent = 'Réessayer le complément';
                        complete = true;
                    } else if (data.type === 'done' || !data.type) {
                        loadedVersion = version;
                        button.textContent = 'Cacher l’explication';
                        complete = true;
                    }
                };
                if (response.headers.get('Content-Type')?.includes('application/x-ndjson')) {
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';
                    try {
                        while (true) {
                            const { done, value } = await reader.read();
                            buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
                            let newline;
                            while ((newline = buffer.indexOf('\n')) >= 0) {
                                const line = buffer.slice(0, newline);
                                buffer = buffer.slice(newline + 1);
                                if (line.trim()) apply(JSON.parse(line));
                            }
                            if (done) {
                                if (buffer.trim()) apply(JSON.parse(buffer));
                                break;
                            }
                        }
                    } finally {
                        reader.releaseLock();
                    }
                } else {
                    apply(await response.json());
                }
                if (!complete && !controller.signal.aborted) throw new Error('Complément interrompu. Réessayez.');
            } catch (error) {
                if (controller.signal.aborted) {
                    comment.textContent = 'Vos goûts ont changé. Cliquez pour actualiser l’explication.';
                } else {
                    comment.textContent += '\n' + error.message;
                }
                button.textContent = 'Réessayer l’explication';
            } finally {
                if (controller.signal.aborted) {
                    comment.textContent = 'Vos goûts ont changé. Cliquez pour actualiser l’explication.';
                    button.textContent = 'Réessayer l’explication';
                }
                button.disabled = false;
                commentRequests.delete(controller);
            }
        });
        body.append(button, comment);
    }
    if (window.Personalization) body.append(window.Personalization.feedbackControls(program));
    if (window.Personalization) body.append(window.Personalization.favoriteControls(program));
    card.append(body);
    return card;
}

const views = { now: 'Maintenant', tonight: 'Ce soir', tomorrow: 'Demain', suggestions: 'Suggestions' };
const initialParams = new URLSearchParams(window.location.search);
let currentView = Object.hasOwn(views, initialParams.get('view')) ? initialParams.get('view') : 'tonight';
const searchInput = document.getElementById('search');
const channelSelect = document.getElementById('channel');
const categorySelect = document.getElementById('category');
const durationInput = document.getElementById('max-duration');
const filtersForm = document.getElementById('filters');
let searchTimer;

function updateChoices(select, values, selected, allLabel) {
    const options = [element('option', '', allLabel)];
    options[0].value = '';
    const choices = [...values];
    if (selected && !choices.includes(selected)) choices.push(selected);
    for (const value of choices) {
        const option = element('option', '', value);
        option.value = value;
        options.push(option);
    }
    select.replaceChildren(...options);
    select.value = selected;
}

searchInput.value = initialParams.get('q') || '';
durationInput.value = initialParams.get('max_duration') || '';
updateChoices(channelSelect, [], initialParams.get('channel') || '', 'Toutes les chaînes');
updateChoices(categorySelect, [], initialParams.get('category') || '', 'Toutes les catégories');

function filterParams() {
    const params = new URLSearchParams({ view: currentView });
    for (const [key, input] of [['q', searchInput], ['channel', channelSelect],
        ['category', categorySelect], ['max_duration', durationInput]]) {
        if (input.value.trim()) params.set(key, input.value.trim());
    }
    return params;
}

function updateNavigation() {
    for (const button of document.querySelectorAll('[data-view]')) {
        const selected = button.dataset.view === currentView;
        button.classList.toggle('active', selected);
        button.setAttribute('aria-pressed', String(selected));
    }
}

async function loadPrograms() {
    invalidateExplanations();
    if (activeRequest) activeRequest.abort();
    if (!filtersForm.reportValidity()) {
        programsDiv.setAttribute('aria-busy', 'false');
        statusDiv.textContent = 'Vérifiez les filtres saisis.';
        return;
    }
    const controller = new AbortController();
    activeRequest = controller;
    const params = filterParams();
    const requestedView = currentView;
    window.history.replaceState(null, '', `${window.location.pathname}?${params}`);
    updateNavigation();
    pageTitle.textContent = views[requestedView];
    programsDiv.replaceChildren();
    programsDiv.setAttribute('aria-busy', 'true');
    statusDiv.textContent = 'Chargement des programmes…';
    try {
        const endpoint = requestedView === 'suggestions' ? '/api/suggestions' : '/api/programs';
        const response = await fetch(`${endpoint}?${params}`, { signal: controller.signal });
        const data = await response.json();
        if (controller.signal.aborted || activeRequest !== controller) return;
        if (!response.ok) throw new Error(data.error || 'Impossible de charger les programmes.');
        const selectedDate = response.headers.get('X-Programs-View-Date');
        if (selectedDate) {
            const date = new Intl.DateTimeFormat('fr-FR', {
                timeZone: 'Europe/Paris', weekday: 'long', day: 'numeric', month: 'long'
            }).format(new Date(`${selectedDate}T12:00:00+00:00`));
            pageTitle.textContent = `${views[requestedView]} — ${date}`;
        }
        const choices = JSON.parse(response.headers.get('X-Programs-Filters') || '{"channels":[],"categories":[]}');
        updateChoices(channelSelect, choices.channels, params.get('channel') || '', 'Toutes les chaînes');
        updateChoices(categorySelect, choices.categories, params.get('category') || '', 'Toutes les catégories');
        const messages = [];
        if (response.headers.get('X-Programs-Stale') === 'true') {
            messages.push(`Dernières données disponibles : ${response.headers.get('X-Programs-Date')}.`);
        }
        if (data.length) {
            messages.push(`${data.length} programme${data.length > 1 ? 's' : ''}.`);
        } else if (requestedView === 'suggestions') {
            messages.push('Aucune suggestion ne correspond à vos filtres, préférences ou avis. Vous pouvez les modifier dans « Mes préférences et mes avis ».');
        } else if (['q', 'channel', 'category', 'max_duration'].some(key => params.has(key))) {
            messages.push('Aucun programme ne correspond à ces filtres. Vous pouvez les réinitialiser.');
        } else {
            messages.push(requestedView === 'tomorrow'
                ? 'Aucun programme disponible pour demain dans les données actuelles.'
                : 'Aucun programme disponible pour cette période.');
        }
        statusDiv.textContent = messages.join(' ');
        programsDiv.replaceChildren(...data.map(program => renderProgram(program, requestedView === 'suggestions')));
    } catch (error) {
        if (!controller.signal.aborted && activeRequest === controller) statusDiv.textContent = error.message;
    } finally {
        if (activeRequest === controller) programsDiv.setAttribute('aria-busy', 'false');
    }
}

function reloadFilters(delay = 0) {
    clearTimeout(searchTimer);
    if (activeRequest) activeRequest.abort();
    if (delay) searchTimer = setTimeout(loadPrograms, delay);
    else loadPrograms();
}

for (const button of document.querySelectorAll('[data-view]')) {
    button.addEventListener('click', () => {
        currentView = button.dataset.view;
        reloadFilters();
    });
}
searchInput.addEventListener('input', () => reloadFilters(250));
for (const input of [channelSelect, categorySelect, durationInput]) {
    input.addEventListener('change', () => reloadFilters());
}
filtersForm.addEventListener('submit', event => {
    event.preventDefault();
    reloadFilters();
});
document.getElementById('reset-filters').addEventListener('click', () => {
    for (const input of [searchInput, channelSelect, categorySelect, durationInput]) input.value = '';
    reloadFilters();
});
loadPrograms();

if (window.Personalization) {
    window.Personalization.onProfileChange = invalidateExplanations;
    window.Personalization.onChange = () => {
        reloadFilters();
    };
}
