/* Profil anonyme identifié par le cookie HttpOnly posé par la page Flask. */
(() => {
    const form = document.getElementById('preferences-form');
    const fields = document.getElementById('preferences-fields');
    const status = document.getElementById('preferences-status');
    const history = document.getElementById('feedback-history');
    const favoritesHistory = document.getElementById('favorites-history');
    const favoritesStatus = document.getElementById('favorites-status');
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
            renderFavorites(data.favorites || []);
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

    const controlStyle = 'tw-rounded tw-border tw-border-solid tw-border-slate-300 tw-px-3 tw-py-2';

    function calendarControls(program) {
        const group = node('div', '', 'tw-flex tw-flex-wrap tw-items-center tw-gap-2');
        const label = node('label', 'Rappel : ', 'tw-text-sm');
        const select = node('select', '', controlStyle);
        select.setAttribute('aria-label', `Rappel pour ${program.name}`);
        for (const [value, text] of [[0, 'Sans rappel'], [5, '5 min avant'],
            [15, '15 min avant'], [30, '30 min avant'], [60, '1 h avant']]) {
            const option = node('option', text);
            option.value = String(value);
            option.selected = value === 15;
            select.append(option);
        }
        select.value = '15';
        label.append(select);
        const link = node('a', 'Télécharger le calendrier', controlStyle);
        link.setAttribute('aria-label', `Télécharger le calendrier : ${program.name}`);
        const update = () => {
            link.href = `/api/programs/${encodeURIComponent(program.id)}/calendar?reminder=${select.value}`;
        };
        update();
        select.addEventListener('change', update);
        group.append(label, link);
        return group;
    }

    function paintFavorite(button, selected) {
        button.setAttribute('aria-pressed', String(selected));
        button.textContent = selected ? 'Retirer des favoris' : 'Ajouter aux favoris';
    }

    function renderFavorites(programs) {
        const ids = new Set(programs.map(program => program.id));
        for (const button of document.querySelectorAll('[data-favorite-id]')) {
            paintFavorite(button, ids.has(button.dataset.favoriteId));
        }
        favoritesHistory.replaceChildren(...programs.map(program => {
            const row = node('div', '', 'tw-rounded tw-border tw-border-solid tw-border-slate-200 tw-p-4');
            const date = new Date(program.start);
            const formatted = Number.isNaN(date.getTime()) ? 'Horaire inconnu' : new Intl.DateTimeFormat('fr-FR', {
                timeZone: 'Europe/Paris', weekday: 'long', day: 'numeric', month: 'long',
                year: 'numeric', hour: '2-digit', minute: '2-digit'
            }).format(date);
            row.append(node('h3', program.name, 'tw-font-semibold'),
                node('p', `${program.channel_name} — ${formatted}`, 'tw-text-sm'),
                node('p', program.desc || 'Résumé indisponible.', 'tw-text-sm'));
            if (new Date(program.end).getTime() <= Date.now()) {
                row.append(node('p', 'Diffusion terminée.', 'tw-text-sm tw-text-slate-600'));
            }
            const remove = node('button', 'Retirer des favoris', controlStyle);
            remove.type = 'button';
            remove.setAttribute('aria-label', `Retirer des favoris : ${program.name}`);
            remove.addEventListener('click', async () => {
                remove.disabled = true;
                try {
                    await request(`/api/programs/${encodeURIComponent(program.id)}/favorite`, 'DELETE');
                    await refresh();
                    favoritesStatus.textContent = 'Favori retiré.';
                } catch (error) {
                    favoritesStatus.textContent = error.message;
                } finally {
                    remove.disabled = false;
                }
            });
            row.append(remove, calendarControls(program));
            return row;
        }));
        if (!programs.length) favoritesHistory.append(node('p', 'Aucun favori enregistré.'));
    }

    function favoriteControls(program) {
        const container = node('div', '', 'tw-mt-4 tw-space-y-3');
        const button = node('button', '', controlStyle);
        button.type = 'button';
        button.dataset.favoriteId = program.id;
        button.setAttribute('aria-label', `Favori : ${program.name}`);
        paintFavorite(button, Boolean(program.favorite));
        const message = node('p', '', 'tw-text-sm');
        message.setAttribute('role', 'status');
        button.addEventListener('click', async () => {
            const selected = button.getAttribute('aria-pressed') === 'true';
            button.disabled = true;
            try {
                await request(`/api/programs/${encodeURIComponent(program.id)}/favorite`, selected ? 'DELETE' : 'PUT');
                paintFavorite(button, !selected);
                message.textContent = selected ? 'Favori retiré.' : 'Favori enregistré pour cette diffusion.';
                await refresh();
            } catch (error) {
                message.textContent = error.message;
            } finally {
                button.disabled = false;
                if (document.activeElement === document.body || document.activeElement === button) {
                    button.focus({ preventScroll: true });
                }
            }
        });
        container.append(button, calendarControls(program), message);
        return container;
    }

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

    window.Personalization = { refresh, feedbackControls, favoriteControls, onChange: null };
    refresh(true);
})();
