/* Profil anonyme identifié par le cookie HttpOnly posé par la page Flask. */
(() => {
    const form = document.getElementById('preferences-form');
    const fields = document.getElementById('preferences-fields');
    const status = document.getElementById('preferences-status');
    const history = document.getElementById('feedback-history');
    const metrics = document.getElementById('feedback-metrics');
    const controls = {
        liked_categories: document.getElementById('liked-categories'),
        disliked_categories: document.getElementById('disliked-categories'),
        preferred_channels: document.getElementById('preferred-channels'),
        keywords: document.getElementById('keywords'),
        avoid_keywords: document.getElementById('avoid-keywords'),
        max_duration: document.getElementById('preferred-duration')
    };
    const labels = { like: 'J’aime', dislike: 'Pas pour moi', seen: 'Déjà vu' };
    const defaults = { liked_categories: [], disliked_categories: [], preferred_channels: [],
        keywords: [], avoid_keywords: [], max_duration: null };
    let ready = false;
    let generation = 0;

    function node(tag, text, className = '') {
        const result = document.createElement(tag);
        result.textContent = text;
        result.className = className;
        return result;
    }

    async function request(url, method = 'GET', payload) {
        const options = { method };
        if (payload !== undefined) {
            options.headers = { 'Content-Type': 'application/json' };
            options.body = JSON.stringify(payload);
        }
        const response = await fetch(url, options);
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Enregistrement impossible.');
        return data;
    }

    function fillPreferences(preferences, choices) {
        for (const [key, select] of Object.entries(controls)) {
            if (['keywords', 'avoid_keywords'].includes(key)) {
                select.value = preferences[key].join(', ');
            } else if (key === 'max_duration') {
                select.value = preferences[key] ?? '';
            } else {
                const available = key === 'preferred_channels' ? choices.channels : choices.categories;
                const values = [...new Set([...available, ...preferences[key]])];
                select.replaceChildren(...values.map(value => {
                    const option = node('option', value);
                    option.value = value;
                    option.selected = preferences[key].includes(value);
                    return option;
                }));
            }
        }
    }

    async function refresh(fill = false) {
        const version = ++generation;
        try {
            const data = await request('/api/profile');
            if (version !== generation) return;
            if (fill || !ready) fillPreferences(data.preferences, data.choices);
            ready = true;
            fields.disabled = false;
            metrics.textContent = `${data.metrics.like} aimé(s), ${data.metrics.dislike} écarté(s), ${data.metrics.seen} déjà vu(s).`;
            history.replaceChildren(...data.feedback.map(entry => {
                const row = node('div', '', 'tw-flex tw-items-center tw-justify-between tw-gap-2 tw-text-sm');
                const text = node('span', `${entry.name} — ${labels[entry.value]}`);
                const undo = node('button', 'Annuler', 'tw-rounded tw-border tw-border-solid tw-border-slate-300 tw-px-2 tw-py-1');
                undo.type = 'button';
                undo.setAttribute('aria-label', `Annuler l’avis sur ${entry.name}`);
                undo.addEventListener('click', async () => {
                    undo.disabled = true;
                    try {
                        await request(`/api/feedback/${encodeURIComponent(entry.content_id)}`, 'DELETE');
                        await refresh();
                        window.Personalization.onChange?.();
                    } catch (error) {
                        status.textContent = error.message;
                    } finally {
                        undo.disabled = false;
                    }
                });
                row.append(text, undo);
                return row;
            }));
            if (!data.feedback.length) history.append(node('p', 'Aucun avis enregistré.'));
            if (fill) status.textContent = '';
        } catch (error) {
            status.textContent = error.message;
        }
    }

    function readPreferences() {
        const result = {};
        for (const [key, input] of Object.entries(controls)) {
            if (['keywords', 'avoid_keywords'].includes(key)) {
                result[key] = input.value.split(',').map(word => word.trim()).filter(Boolean);
            } else if (key === 'max_duration') {
                result[key] = input.value ? Number(input.value) : null;
            } else {
                result[key] = Array.from(input.selectedOptions, option => option.value);
            }
        }
        return result;
    }

    async function save(preferences) {
        fields.disabled = true;
        status.textContent = 'Enregistrement…';
        try {
            await request('/api/profile', 'PUT', preferences);
            await refresh(true);
            status.textContent = 'Préférences enregistrées.';
            window.Personalization.onChange?.();
        } catch (error) {
            status.textContent = error.message;
        } finally {
            fields.disabled = !ready;
        }
    }

    form.addEventListener('submit', event => {
        event.preventDefault();
        if (ready && form.reportValidity()) save(readPreferences());
    });
    document.getElementById('reset-preferences').addEventListener('click', () => {
        if (ready) save(defaults);
    });

    function feedbackControls(program) {
        const container = node('div', '', 'tw-mt-4 tw-flex tw-flex-wrap tw-gap-2');
        const message = node('span', '', 'tw-w-full tw-text-sm');
        message.setAttribute('role', 'status');
        let selected = program.feedback;
        const buttons = [];
        const paint = () => {
            for (const [value, button] of buttons) {
                button.setAttribute('aria-pressed', String(selected === value));
                button.className = selected === value
                    ? 'tw-rounded tw-bg-blue-700 tw-text-white tw-px-3 tw-py-2'
                    : 'tw-rounded tw-border tw-border-solid tw-border-slate-300 tw-px-3 tw-py-2';
            }
        };
        for (const [value, label] of Object.entries(labels)) {
            const button = node('button', label);
            button.type = 'button';
            button.setAttribute('aria-label', `${label} : ${program.name}`);
            button.addEventListener('click', async () => {
                for (const [, control] of buttons) control.disabled = true;
                const next = selected === value ? null : value;
                try {
                    await request(`/api/programs/${encodeURIComponent(program.id)}/feedback`, 'PUT', { value: next });
                    selected = next;
                    program.feedback = next;
                    paint();
                    message.textContent = next ? 'Avis enregistré.' : 'Avis annulé.';
                    await refresh();
                    // Garder les cartes en place pendant l’évaluation.
                } catch (error) {
                    message.textContent = error.message;
                } finally {
                    for (const [, control] of buttons) control.disabled = false;
                    if (document.activeElement === document.body || document.activeElement === button) {
                        button.focus({ preventScroll: true });
                    }
                }
            });
            buttons.push([value, button]);
            container.append(button);
        }
        container.append(message);
        paint();
        return container;
    }

    window.Personalization = { refresh, feedbackControls, onChange: null };
    refresh(true);
})();
