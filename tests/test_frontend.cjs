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
    reportValidity() { return true; }
}

function harness(search = '') {
    const ids = ['programs', 'page-title', 'status', 'search', 'channel', 'category',
        'max-duration', 'filters', 'reset-filters'];
    const nodes = Object.fromEntries(ids.map(id => [id, new Element()]));
    const buttons = ['now', 'tonight', 'tomorrow', 'suggestions'].map(view => {
        const button = new Element('button');
        button.dataset.view = view;
        return button;
    });
    const requests = [];
    const window = {
        location: { search, pathname: '/' },
        history: { replaceState(_, __, url) { window.lastUrl = url; } }
    };
    const document = {
        getElementById: id => nodes[id],
        querySelectorAll: () => buttons,
        createElement: tag => new Element(tag)
    };
    const context = { document, window, URL, URLSearchParams, AbortController,
        Intl, Date, setTimeout, clearTimeout,
        fetch: (url, options) => new Promise(resolve => requests.push({ url, options, resolve }))
    };
    vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../app_progTV/static/app.js'), 'utf8'), context);
    return { nodes, buttons, requests, window };
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
