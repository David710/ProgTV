const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Petit DOM de test pour exercer les requêtes et le rendu sans dépendance navigateur.
class Element {
    constructor(tag = 'div') {
        this.tag = tag;
        this.value = '';
        this.textContent = '';
        this.children = [];
        this.listeners = {};
        this.attributes = {};
        this.dataset = {};
        this.classList = { toggle() {} };
    }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    addEventListener(event, callback) { this.listeners[event] = callback; }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key]; }
    focus(options) { this.focusOptions = options; }
    reportValidity() { return true; }
    get selectedOptions() { return this.children.filter(option => option.selected); }
}

function harness(search = '', withProfile = false) {
    const ids = ['programs', 'page-title', 'status', 'search', 'channel', 'category',
        'max-duration', 'filters', 'reset-filters', 'preferences-form', 'preferences-fields',
        'preferences-status', 'feedback-history', 'feedback-metrics', 'liked-categories',
        'disliked-categories', 'preferred-channels', 'keywords', 'avoid-keywords',
        'preferred-duration', 'reset-preferences', 'favorites-history', 'favorites-status'];
    const nodes = Object.fromEntries(ids.map(id => [id, new Element()]));
    const buttons = ['now', 'tonight', 'tomorrow', 'suggestions'].map(view => {
        const button = new Element('button');
        button.dataset.view = view;
        return button;
    });
    const requests = [];
    const profile = {
        preferences: { liked_categories: [], disliked_categories: [], preferred_channels: [],
            keywords: [], avoid_keywords: [], max_duration: null },
        choices: { channels: ['TF1'], categories: ['Film'] }, feedback: [], favorites: [],
        metrics: { like: 0, dislike: 0, seen: 0 }
    };
    const window = {
        location: { search, pathname: '/' },
        history: { replaceState(_, __, url) { window.lastUrl = url; } }
    };
    const document = {
        getElementById: id => nodes[id],
        querySelectorAll: selector => selector === '[data-favorite-id]'
            ? descendants(nodes.programs).filter(node => node.dataset.favoriteId) : buttons,
        createElement: tag => new Element(tag)
    };
    const context = { document, window, URL, URLSearchParams, AbortController,
        Intl, Date, setTimeout, clearTimeout,
        fetch: (url, options) => {
            if (withProfile && url === '/api/profile' && options?.method === 'GET') {
                return Promise.resolve({ ok: true, json: async () => profile });
            }
            return new Promise(resolve => requests.push({ url, options, resolve }));
        }
    };
    if (withProfile) vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../app_progTV/static/profile.js'), 'utf8'), context);
    vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../app_progTV/static/app.js'), 'utf8'), context);
    return { nodes, buttons, requests, window, profile };
}

function respond(request, data = [], headers = {}, status = 200) {
    request.resolve({
        ok: status === 200,
        json: async () => data,
        headers: { get: key => ({
            'X-Programs-View-Date': '2026-10-08',
            'X-Programs-Filters': JSON.stringify({ channels: ['TF1'], categories: ['Film'] }),
            ...headers
        })[key] || null }
    });
}
const flush = () => new Promise(resolve => setImmediate(resolve));

const program = {
    id: 'stable-id', name: '<script>alert(1)</script>', desc: '<b>Résumé</b>',
    channel_name: 'TF1', icon: 'javascript:alert(1)', channel_icon: '',
    start: '2026-10-08T20:30:00+02:00', cat: 'Film', duration: 90, note_pred: .7
};

function descendants(node) { return [node, ...node.children.flatMap(descendants)]; }

test('restore filters from URL, display tomorrow date and reset without changing view', async () => {
    const h = harness('?view=tomorrow&q=caf%C3%A9&channel=TF1&category=Film&max_duration=90');
    const params = new URL(h.requests[0].url, 'http://localhost').searchParams;
    assert.equal(params.get('view'), 'tomorrow');
    assert.equal(params.get('q'), 'café');
    assert.equal(params.get('max_duration'), '90');
    respond(h.requests[0], [program]);
    await flush();
    assert.match(h.nodes['page-title'].textContent, /Demain.*8 octobre/);
    assert.equal(h.nodes.channel.value, 'TF1');
    assert.equal(h.nodes.category.value, 'Film');
    assert.equal(h.nodes.programs.children.length, 1);
    assert.equal(h.nodes.programs.attributes['aria-busy'], 'false');
    h.nodes['reset-filters'].listeners.click();
    assert.equal(h.requests[1].url, '/api/programs?view=tomorrow');
    assert.equal(h.window.lastUrl, '/?view=tomorrow');
    respond(h.requests[1]);
    await flush();
    assert.match(h.nodes.status.textContent, /Aucun programme disponible pour demain/);
});

test('obsolete response cannot overwrite newer view, including errors', async () => {
    const h = harness();
    h.buttons.find(button => button.dataset.view === 'now').listeners.click();
    assert.equal(h.requests[0].options.signal.aborted, true);
    respond(h.requests[1], [program]);
    await flush();
    respond(h.requests[0], { error: 'ancienne erreur' }, {}, 503);
    await flush();
    assert.match(h.nodes['page-title'].textContent, /^Maintenant/);
    assert.equal(h.nodes.programs.children.length, 1);
    assert.equal(h.nodes.status.textContent, '1 programme.');
});

test('literal text rendering and unsafe image URL rejected', async () => {
    const h = harness('?view=suggestions');
    assert.match(h.requests[0].url, /^\/api\/suggestions\?/);
    respond(h.requests[0], [program]);
    await flush();
    const elements = descendants(h.nodes.programs);
    assert.ok(elements.some(node => node.textContent === program.name));
    assert.ok(elements.some(node => node.textContent === program.desc));
    assert.ok(elements.filter(node => node.tag === 'img').every(node => !node.src));
    const button = elements.find(node => node.tag === 'button');
    const pending = button.listeners.click();
    assert.equal(h.requests[1].url, '/api/programs/stable-id/comment');
    respond(h.requests[1], { comment: '<script>comment</script>' });
    await pending;
    assert.ok(elements.some(node => node.textContent === '<script>comment</script>'));
});

test('debounced search preserves choices even with no matching programs', async () => {
    const h = harness('?view=now&channel=TF1');
    respond(h.requests[0], [program]);
    await flush();
    h.nodes.search.value = 'aucun';
    h.nodes.search.listeners.input();
    h.nodes.search.value = 'introuvable';
    h.nodes.search.listeners.input();
    await new Promise(resolve => setTimeout(resolve, 280));
    assert.equal(h.requests.length, 2);
    assert.match(h.requests[1].url, /q=introuvable/);
    respond(h.requests[1], []);
    await flush();
    assert.equal(h.nodes.channel.value, 'TF1');
    assert.ok(h.nodes.channel.children.some(option => option.value === 'TF1'));
    assert.match(h.nodes.status.textContent, /réinitialiser/);
});


test('preferences save comma-separated tastes and reload suggestions', async () => {
    const h = harness('?view=suggestions', true);
    respond(h.requests[0], [program]);
    await flush();
    h.nodes.keywords.value = 'action, polar';
    h.nodes['preferred-duration'].value = '90';
    h.nodes['liked-categories'].children[0].selected = true;
    h.nodes['preferences-form'].listeners.submit({ preventDefault() {} });
    assert.equal(h.requests[1].url, '/api/profile');
    assert.equal(h.requests[1].options.method, 'PUT');
    const payload = JSON.parse(h.requests[1].options.body);
    assert.deepEqual(payload.keywords, ['action', 'polar']);
    assert.deepEqual(payload.liked_categories, ['Film']);
    assert.equal(payload.max_duration, 90);
    h.profile.preferences = payload;
    respond(h.requests[1], { preferences: payload });
    await flush();
    assert.equal(h.nodes['preferences-status'].textContent, 'Préférences enregistrées.');
    assert.equal(h.nodes['preferences-fields'].disabled, false);
    assert.match(h.requests[2].url, /^\/api\/suggestions/);
    respond(h.requests[2], [program]);
    await flush();
});

test('feedback can be toggled and button states update only on success', async () => {
    const h = harness('?view=tomorrow', true);
    const item = { ...program, feedback: null };
    respond(h.requests[0], [item]);
    await flush();
    const card = h.nodes.programs.children[0];
    const like = descendants(h.nodes.programs).find(node => node.textContent === 'J’aime');
    const pending = like.listeners.click();
    assert.equal(h.requests[1].url, '/api/programs/stable-id/feedback');
    assert.deepEqual(JSON.parse(h.requests[1].options.body), { value: 'like' });
    respond(h.requests[1], { feedback: 'like' });
    await pending;
    assert.equal(like.attributes['aria-pressed'], 'true');
    assert.equal(h.requests.length, 2);
    assert.equal(h.nodes.programs.children[0], card);
    assert.equal(like.focusOptions.preventScroll, true);
    const pressed = descendants(h.nodes.programs).find(node => node.textContent === 'J’aime');
    const undo = pressed.listeners.click();
    assert.deepEqual(JSON.parse(h.requests[2].options.body), { value: null });
    respond(h.requests[2], { error: 'stockage indisponible' }, {}, 503);
    await undo;
    assert.equal(pressed.attributes['aria-pressed'], 'true');
    assert.equal(pressed.disabled, false);
    assert.ok(descendants(h.nodes.programs).some(node => node.textContent === 'stockage indisponible'));
});

test('history undo removes an excluded program even after its schedule expires', async () => {
    const h = harness('?view=suggestions', true);
    h.profile.feedback = [{ name: '<b>Ancien film</b>', value: 'seen', content_id: 'old-content' }];
    respond(h.requests[0], []);
    await flush();
    const button = descendants(h.nodes['feedback-history']).find(node => node.tag === 'button');
    const pending = button.listeners.click();
    assert.equal(h.requests[1].url, '/api/feedback/old-content');
    assert.equal(h.requests[1].options.method, 'DELETE');
    h.profile.feedback = [];
    respond(h.requests[1], { deleted: true });
    await pending;
    assert.match(h.nodes['feedback-history'].children[0].textContent, /Aucun avis/);
    respond(h.requests[2], [program]);
    await flush();
});

for (const label of ['J’aime', 'Pas pour moi', 'Déjà vu']) {
    test(`${label} preserves the suggestion card and focus`, async () => {
        const h = harness('?view=suggestions', true);
        respond(h.requests[0], [{ ...program, feedback: null }]);
        await flush();
        const card = h.nodes.programs.children[0];
        const button = descendants(card).find(node => node.textContent === label);
        const pending = button.listeners.click();
        respond(h.requests[1], { feedback: 'saved' });
        await pending;
        assert.equal(h.requests.length, 2);
        assert.equal(h.nodes.programs.children[0], card);
        assert.equal(button.focusOptions.preventScroll, true);
        assert.equal(button.disabled, false);
        assert.equal(button.attributes['aria-pressed'], 'true');
    });
}


test('favorites toggle without replacing cards and calendar reminder updates the link', async () => {
    const h = harness('?view=tomorrow', true);
    respond(h.requests[0], [{ ...program, favorite: false }]);
    await flush();
    const card = h.nodes.programs.children[0];
    const button = descendants(card).find(node => node.textContent === 'Ajouter aux favoris');
    const pending = button.listeners.click();
    assert.equal(h.requests[1].options.method, 'PUT');
    assert.equal(h.requests[1].url, '/api/programs/stable-id/favorite');
    h.profile.favorites = [{ ...program, end: '2026-10-08T22:00:00+02:00' }];
    respond(h.requests[1], { favorite: true });
    await pending;
    assert.equal(h.requests.length, 2);
    assert.equal(h.nodes.programs.children[0], card);
    assert.equal(button.attributes['aria-pressed'], 'true');
    assert.equal(button.focusOptions.preventScroll, true);
    assert.ok(descendants(h.nodes['favorites-history']).some(node => node.textContent === program.name));
    const select = descendants(card).find(node => node.tag === 'select');
    const link = descendants(card).find(node => node.tag === 'a');
    assert.equal(link.href, '/api/programs/stable-id/calendar?reminder=15');
    select.value = '60';
    select.listeners.change();
    assert.equal(link.href, '/api/programs/stable-id/calendar?reminder=60');
    const undo = button.listeners.click();
    assert.equal(h.requests[2].options.method, 'DELETE');
    h.profile.favorites = [];
    respond(h.requests[2], { favorite: false });
    await undo;
    assert.equal(button.attributes['aria-pressed'], 'false');
    assert.equal(h.nodes.programs.children[0], card);
});

test('favorite storage failure keeps the card state and history removal updates it', async () => {
    const h = harness('?view=tomorrow', true);
    h.profile.favorites = [{ ...program, end: '2026-10-08T22:00:00+02:00' }];
    respond(h.requests[0], [{ ...program, favorite: true }]);
    await flush();
    const button = descendants(h.nodes.programs).find(node => node.dataset.favoriteId);
    const pending = button.listeners.click();
    respond(h.requests[1], { error: 'stockage indisponible' }, {}, 503);
    await pending;
    assert.equal(button.attributes['aria-pressed'], 'true');
    assert.equal(button.disabled, false);
    const remove = descendants(h.nodes['favorites-history']).find(node => node.tag === 'button');
    const undo = remove.listeners.click();
    h.profile.favorites = [];
    respond(h.requests[2], { favorite: false });
    await undo;
    assert.equal(button.attributes['aria-pressed'], 'false');
    assert.equal(h.requests.length, 3);
});
