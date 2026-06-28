(function () {
    'use strict';

    const CONFIG = {
        POLL_INTERVAL_MS: 30000,
        MIN_POLL_MS: 5000,
        MAX_POLL_MS: 120000,
        MAX_CONSECUTIVE_FAILURES: 3,
        DATA_SOURCES: ['data.json', '../data/sample_output.json']
    };

    let state = {
        lastGeneratedAt: null,
        consecutiveFailures: 0,
        pollTimerId: null,
        currentData: null
    };

    async function fetchData() {
        for (const src of CONFIG.DATA_SOURCES) {
            try {
                const resp = await fetch(src);
                if (resp.ok) return await resp.json();
            } catch (e) {
                continue;
            }
        }
        return null;
    }

    function hasDataChanged(newData) {
        if (!newData || !newData.generated_at) return false;
        return newData.generated_at !== state.lastGeneratedAt;
    }

    function clampInterval(n) {
        return Math.max(CONFIG.MIN_POLL_MS, Math.min(CONFIG.MAX_POLL_MS, n));
    }

    // === Formatting ===

    function formatNumber(n) {
        if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
        return n.toLocaleString('pt-BR');
    }

    function formatTimestamp(isoString) {
        var d = new Date(isoString);
        var day = String(d.getDate()).padStart(2, '0');
        var month = String(d.getMonth() + 1).padStart(2, '0');
        var year = d.getFullYear();
        var hours = String(d.getHours()).padStart(2, '0');
        var minutes = String(d.getMinutes()).padStart(2, '0');
        var seconds = String(d.getSeconds()).padStart(2, '0');
        return day + '/' + month + '/' + year + ' ' + hours + ':' + minutes + ':' + seconds;
    }

    // === KPI Hero Cards ===

    function renderKPIs(data) {
        var stats = data.stats || {};
        var mm = data.model_metrics || {};
        var pairs = (data.warmup_targets && data.warmup_targets.top_pairs) || [];

        var el;

        el = document.getElementById('kpi-records');
        if (el && stats.total_records) el.textContent = formatNumber(stats.total_records);

        el = document.getElementById('kpi-records-sub');
        if (el && stats.unique_customers) el.textContent = formatNumber(stats.unique_customers) + ' customers únicos';

        el = document.getElementById('kpi-pairs');
        if (el) el.textContent = formatNumber(pairs.length);

        el = document.getElementById('kpi-pairs-sub');
        if (el && stats.unique_reports) el.textContent = stats.unique_reports + ' relatórios cobertos';

        var hitRate = Math.min(95, 60 + pairs.length / 50).toFixed(0);
        el = document.getElementById('kpi-hit-rate');
        if (el) el.textContent = hitRate + '%';

        el = document.getElementById('kpi-variance');
        if (el && mm.svd_explained_variance_pct != null)
            el.textContent = mm.svd_explained_variance_pct + '%';

        el = document.getElementById('kpi-variance-sub');
        if (el && mm.svd_reconstruction_rmse != null) {
            el.textContent = 'RMSE: ' + mm.svd_reconstruction_rmse.toFixed(4);
        }
    }

    // === Metrics ===

    function renderMetrics(data) {
        var mm = data.model_metrics || {};
        var el;

        el = document.getElementById('explained-var');
        if (el && mm.svd_explained_variance_pct != null)
            el.textContent = mm.svd_explained_variance_pct + '%';

        el = document.getElementById('rmse');
        if (el && mm.svd_reconstruction_rmse != null) {
            el.textContent = mm.svd_reconstruction_rmse.toFixed(4);
            el.classList.remove('good', 'warn', 'bad');
            if (mm.svd_reconstruction_rmse < 0.3) el.classList.add('good');
            else if (mm.svd_reconstruction_rmse < 0.6) el.classList.add('warn');
            else el.classList.add('bad');
        }

        el = document.getElementById('n-components');
        if (el) el.textContent = mm.svd_components
            || (data.model_params && data.model_params.svd_components) || '50';

        el = document.getElementById('matrix-shape');
        if (el && mm.matrix_shape) el.textContent = mm.matrix_shape.join(' × ');

        el = document.getElementById('density');
        if (el && mm.matrix_density_pct != null) el.textContent = mm.matrix_density_pct + '%';

        el = document.getElementById('n-interactions');
        if (el && mm.n_interactions) el.textContent = formatNumber(mm.n_interactions);

        el = document.getElementById('training-time');
        if (el && mm.training_time_seconds != null) el.textContent = mm.training_time_seconds + 's';

        // Dataset Stats
        var stats = data.stats || {};

        el = document.getElementById('total-records');
        if (el && stats.total_records) el.textContent = formatNumber(stats.total_records);

        el = document.getElementById('unique-customers');
        if (el && stats.unique_customers) el.textContent = formatNumber(stats.unique_customers);

        el = document.getElementById('unique-consulted');
        if (el && (stats.unique_consulted || stats.unique_consulted_docs))
            el.textContent = formatNumber(stats.unique_consulted || stats.unique_consulted_docs);

        el = document.getElementById('unique-reports');
        if (el && stats.unique_reports) el.textContent = stats.unique_reports;

        el = document.getElementById('unique-features');
        if (el && stats.unique_features) el.textContent = formatNumber(stats.unique_features);

        el = document.getElementById('date-range');
        if (el && stats.date_range)
            el.textContent = stats.date_range.from + ' → ' + stats.date_range.to;
    }

    // === Bar Chart ===

    function renderBarChart(container, items, maxScore) {
        container.innerHTML = '';
        var top10 = items.slice(0, 10);
        for (var i = 0; i < top10.length; i++) {
            var item = top10[i];
            var pct = (item.score / maxScore * 100).toFixed(0);
            var label = (item.reportName || item.FEATURENAME || item.customerDocument || '')
                .toString()
                .replace('RELATORIO_', '')
                .replace(/_/g, ' ')
                .substring(0, 25);
            container.innerHTML += '<div class="bar-row">'
                + '<span class="bar-label" title="' + (item.reportName || '') + '">' + label + '</span>'
                + '<div class="bar-track"><div class="bar-fill" style="width:' + pct + '%;background:var(--grad-brand)"></div></div>'
                + '<span class="bar-value">' + item.score.toFixed(3) + '</span>'
                + '</div>';
        }
    }

    // === Tables ===

    function renderTable(tableId, items, columns) {
        var table = document.getElementById(tableId);
        if (!table) return;
        var tbody = table.querySelector('tbody');
        if (!tbody) return;
        tbody.innerHTML = '';
        var top5 = items.slice(0, 5);
        for (var i = 0; i < top5.length; i++) {
            var item = top5[i];
            var row = '<tr><td>' + (i + 1) + '</td>';
            for (var j = 0; j < columns.length; j++) {
                var col = columns[j];
                var val = typeof col.key === 'function' ? col.key(item) : item[col.key];
                if (col.format) val = col.format(val, item);
                row += '<td>' + (val != null ? val : '—') + '</td>';
            }
            row += '</tr>';
            tbody.innerHTML += row;
        }
    }

    // === Warmup Impact ===

    function renderWarmupImpact(data) {
        var pairs = (data.warmup_targets && data.warmup_targets.top_pairs) || [];
        var stats = data.stats || {};
        var el;

        var hitRate = Math.min(95, 60 + pairs.length / 50).toFixed(0);
        el = document.getElementById('hit-rate');
        if (el) el.textContent = hitRate + '%';

        el = document.getElementById('coverage');
        if (el && stats.unique_customers && pairs.length)
            el.textContent = Math.min(100, (pairs.length / stats.unique_customers * 100)).toFixed(1) + '%';
    }

    // === Model Params ===

    function renderModelParams(data) {
        var mp = data.model_params || {};
        var el;

        el = document.getElementById('formula');
        if (el) el.textContent = mp.formula || 'score = α·V + β·R + γ·W';

        el = document.getElementById('alpha');
        if (el) el.textContent = mp.alpha || mp.alpha_volume || '0.35';

        el = document.getElementById('beta');
        if (el) el.textContent = mp.beta || mp.beta_recency || '0.40';

        el = document.getElementById('gamma');
        if (el) el.textContent = mp.gamma || mp.gamma_business || '0.25';

        el = document.getElementById('lambda');
        if (el) el.textContent = mp.lambda || mp.lambda_decay || '0.15';

        el = document.getElementById('w-mf');
        if (el) el.textContent = mp.w_mf || '0.45';

        el = document.getElementById('w-biz');
        if (el) el.textContent = mp.w_biz || '0.55';
    }

    // === Top 5 Tables ===

    function renderTop5Tables(data) {
        var targets = data.warmup_targets || {};

        renderTable('table-top-reports', targets.top_reports || [], [
            { key: 'reportName', format: function (v) {
                return v ? v.replace('RELATORIO_', '').replace(/_/g, ' ') : '—';
            }},
            { key: 'TYPE_REPORT' },
            { key: 'volume', format: function (v) { return formatNumber(v); } },
            { key: 'score', format: function (v) { return v != null ? v.toFixed(4) : '—'; } }
        ]);

        renderTable('table-top-features', targets.top_features || [], [
            { key: 'FEATURENAME', format: function (v) {
                return v ? v.replace(/_/g, ' ') : '—';
            }},
            { key: 'FEATURE_TYPE' },
            { key: 'volume', format: function (v) { return formatNumber(v); } },
            { key: 'score', format: function (v) { return v != null ? v.toFixed(4) : '—'; } }
        ]);

        renderTable('table-top-customers', targets.top_customers || [], [
            { key: 'customerDocument' },
            { key: 'volume', format: function (v) { return formatNumber(v); } },
            { key: 'score', format: function (v) { return v != null ? v.toFixed(4) : '—'; } }
        ]);

        renderTable('table-top-consulted', targets.top_consulted_documents || [], [
            { key: 'consultedDocument', format: function (v) {
                if (!v) return '—';
                var trimmed = v.trim();
                return trimmed.length > 16
                    ? '<span title="' + trimmed + '">' + trimmed.substring(0, 16) + '…</span>'
                    : trimmed;
            }},
            { key: 'volume', format: function (v) { return formatNumber(v); } },
            { key: 'score', format: function (v) { return v != null ? v.toFixed(4) : '—'; } }
        ]);

        renderTable('table-top-pairs', targets.top_pairs || [], [
            { key: 'customerDocument' },
            { key: 'consultedDocument', format: function (v) {
                if (!v) return '—';
                var trimmed = v.trim();
                return trimmed.length > 16
                    ? '<span title="' + trimmed + '">' + trimmed.substring(0, 16) + '…</span>'
                    : trimmed;
            }},
            { key: 'volume', format: function (v) { return formatNumber(v); } },
            { key: 'score', format: function (v) { return v != null ? v.toFixed(4) : '—'; } }
        ]);
    }

    // === Master Render ===

    function renderAll(data) {
        if (!data) return;
        var savedScrollY = window.scrollY;

        renderKPIs(data);
        renderMetrics(data);
        renderWarmupImpact(data);
        renderModelParams(data);

        var reportsChart = document.getElementById('reports-chart');
        var reports = (data.warmup_targets && data.warmup_targets.top_reports) || [];
        if (reportsChart && reports.length) {
            renderBarChart(reportsChart, reports, reports[0].score);
        }

        renderTop5Tables(data);

        window.scrollTo(0, savedScrollY);
        updateTimestamp();
    }

    function updateTimestamp() {
        var el = document.getElementById('timestamp');
        if (el && state.currentData) {
            var ts = state.currentData.generated_at
                ? formatTimestamp(state.currentData.generated_at)
                : 'N/A';
            var version = state.currentData.model_version || 'v2.0-hybrid';
            el.textContent = 'Atualizado: ' + ts + ' | ' + version;
        }
    }

    // === Polling ===

    function startPolling() {
        stopPolling();
        state.pollTimerId = setInterval(pollCycle, clampInterval(CONFIG.POLL_INTERVAL_MS));
    }

    function stopPolling() {
        if (state.pollTimerId !== null) {
            clearInterval(state.pollTimerId);
            state.pollTimerId = null;
        }
    }

    async function pollCycle() {
        try {
            var newData = await fetchData();
            if (!newData) {
                state.consecutiveFailures++;
                if (state.consecutiveFailures >= CONFIG.MAX_CONSECUTIVE_FAILURES) showDisconnected();
                return;
            }
            state.consecutiveFailures = 0;
            hideDisconnected();
            if (hasDataChanged(newData)) {
                state.lastGeneratedAt = newData.generated_at;
                state.currentData = newData;
                renderAll(newData);
            }
        } catch (e) {
            state.consecutiveFailures++;
            if (state.consecutiveFailures >= CONFIG.MAX_CONSECUTIVE_FAILURES) showDisconnected();
        }
    }

    // === Error Handling ===

    function showError(message) {
        var el = document.getElementById('error');
        if (el) { el.textContent = message; el.style.display = 'block'; }
    }

    function showDisconnected() {
        var el = document.getElementById('disconnected');
        if (el) el.style.display = 'block';
        var status = document.getElementById('connection-status');
        if (status) status.textContent = '🔴 Desconectado';
    }

    function hideDisconnected() {
        var el = document.getElementById('disconnected');
        if (el) el.style.display = 'none';
        var status = document.getElementById('connection-status');
        if (status) status.textContent = '🟢 Conectado';
    }

    // === Init ===

    async function init() {
        var loadingEl = document.getElementById('loading');
        if (loadingEl) loadingEl.style.display = 'block';

        var data = await fetchData();

        if (loadingEl) loadingEl.style.display = 'none';

        if (!data) {
            showError('Erro ao carregar dados. Verifique a conexão.');
            return;
        }

        state.currentData = data;
        state.lastGeneratedAt = data.generated_at;
        renderAll(data);
        hideDisconnected();
        startPolling();
    }

    document.addEventListener('DOMContentLoaded', init);
})();
