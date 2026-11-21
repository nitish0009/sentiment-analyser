const API_BASE = "/api";
const MAX_CHARS = 5000;

const SAMPLES = {
    "positive-film": (
        "I absolutely loved this film. The acting was excellent, the story kept me engaged " +
        "from start to finish, and the ending was deeply satisfying. I would happily watch it again."
    ),
    "negative-film": (
        "A complete waste of time. The plot made no sense, the dialogue was painful, and the " +
        "acting felt wooden throughout. I kept checking how much time was left."
    ),
    "positive-product": (
        "An outstanding product. It arrived on time, was easy to set up, and works exactly as " +
        "described. The build quality feels solid and I would recommend it without hesitation."
    ),
    "low-confidence": "I liked parts of it and disliked other parts."
};

const EMPTY_STATE = `
    <div class="empty-state">
        <div class="empty-rule" aria-hidden="true"></div>
        <h3>Your prediction will appear here</h3>
        <p>Type a review and press <strong>Analyze sentiment</strong> to see the predicted class, confidence and probability split.</p>
    </div>
`;

const elements = {
    input: document.getElementById("review-input"),
    charCount: document.getElementById("char-count"),
    analyze: document.getElementById("analyze-button"),
    clear: document.getElementById("clear-button"),
    result: document.getElementById("result-panel"),
    error: document.getElementById("api-error"),
    statusPill: document.getElementById("model-status"),
    statusText: document.getElementById("model-status-text"),
    sampleButtons: Array.from(document.querySelectorAll("[data-sample]"))
};

let busy = false;

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function formatPercent(value, digits = 2) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "\u2014";
}

function formatInteger(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toLocaleString("en-US") : "\u2014";
}

function setText(id, text) {
    const node = document.getElementById(id);
    if (node) node.textContent = text;
}

function setError(message) {
    elements.error.textContent = message;
    elements.error.hidden = !message;
}

function setStatus(ready, label) {
    elements.statusPill.classList.toggle("not-ready", !ready);
    elements.statusText.textContent = label;
}

function updateCounter() {
    const length = elements.input.value.length;
    elements.charCount.textContent = `${formatInteger(length)} / ${formatInteger(MAX_CHARS)}`;
}

function setBusy(value) {
    busy = value;
    elements.analyze.disabled = value;
    elements.analyze.textContent = value ? "Analyzing\u2026" : "Analyze sentiment";
}

function showEmpty() {
    elements.result.className = "result-panel empty";
    elements.result.innerHTML = EMPTY_STATE;
}

function showLoading() {
    elements.result.className = "result-panel";
    elements.result.innerHTML = `
        <div class="result-loading">
            <div>
                <div class="spinner" aria-hidden="true"></div>
                <p>Running preprocessing and the saved classifier\u2026</p>
            </div>
        </div>
    `;
}

function renderPrediction(prediction) {
    const positive = prediction.sentiment === "positive";
    const label = positive ? "Positive" : "Negative";
    const mark = positive ? "P" : "N";
    const css = positive ? "positive" : "negative";
    const confidence = Number(prediction.confidence_percent);
    const negativeProbability = Number(prediction.probabilities?.negative || 0);
    const positiveProbability = Number(prediction.probabilities?.positive || 0);
    const negativePercent = (negativeProbability * 100).toFixed(1);
    const positivePercent = (positiveProbability * 100).toFixed(1);
    const warning = prediction.low_confidence
        ? `<div class="low-confidence"><strong>Low confidence.</strong> The model is still choosing between positive and negative; this is not a neutral prediction.</div>`
        : "";
    const modelName = escapeHtml(prediction.model_name || "Saved model");
    const cleaned = prediction.cleaned_text
        ? escapeHtml(prediction.cleaned_text)
        : "No usable tokens remained after preprocessing.";

    elements.result.className = "result-panel";
    elements.result.innerHTML = `
        <div class="result-card result-${css}">
            <div class="result-head">
                <div class="result-mark" aria-hidden="true">${mark}</div>
                <div>
                    <div class="result-label">Predicted sentiment</div>
                    <div class="result-value">${label}</div>
                </div>
                <div class="confidence-chip">${confidence.toFixed(1)}%</div>
            </div>
            <div class="confidence-caption">
                Confidence level: <strong>${escapeHtml(prediction.confidence_level)}</strong> &middot;
                This is the model's probability for the predicted class.
            </div>
            <div class="prob-row">
                <div class="prob-label"><span>Negative probability</span><span>${negativePercent}%</span></div>
                <div class="prob-track"><div class="prob-fill neg" style="width:${negativePercent}%"></div></div>
            </div>
            <div class="prob-row">
                <div class="prob-label"><span>Positive probability</span><span>${positivePercent}%</span></div>
                <div class="prob-track"><div class="prob-fill pos" style="width:${positivePercent}%"></div></div>
            </div>
            ${warning}
            <div class="result-meta">
                <span>${modelName}</span>
                <span>${formatInteger(prediction.input_word_count)} input words</span>
                <span>${formatInteger(prediction.cleaned_token_count)} tokens used</span>
            </div>
            <details class="result-details">
                <summary>See preprocessing details</summary>
                <p>${cleaned}</p>
            </details>
        </div>
    `;
}

function showPredictionError(message) {
    elements.result.className = "result-panel";
    elements.result.innerHTML = `<div class="result-error" role="alert">${escapeHtml(message)}</div>`;
}

async function fetchJson(path, options = {}) {
    const response = await fetch(path, options);
    let payload = {};
    try {
        payload = await response.json();
    } catch {
        payload = {};
    }
    if (!response.ok) {
        throw new Error(payload.error || `Request failed with status ${response.status}.`);
    }
    return payload;
}

async function analyze() {
    if (busy) return;

    const text = elements.input.value.trim();
    if (!text) {
        showPredictionError("Please enter a review before analyzing.");
        elements.input.focus();
        return;
    }
    if (text.length > MAX_CHARS) {
        showPredictionError(`Please keep the review under ${formatInteger(MAX_CHARS)} characters.`);
        return;
    }

    setError("");
    setBusy(true);
    showLoading();

    try {
        const result = await fetchJson(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        renderPrediction(result);
    } catch (error) {
        showPredictionError(error.message || "The prediction request failed.");
    } finally {
        setBusy(false);
    }
}

function clearInput() {
    elements.input.value = "";
    updateCounter();
    setError("");
    showEmpty();
    elements.input.focus();
}

function loadSample(key) {
    const text = SAMPLES[key];
    if (!text) return;
    elements.input.value = text;
    updateCounter();
    setError("");
    analyze();
}

function renderMetadata(data) {
    const models = data.models || {};
    const selectedKey = data.model_key || Object.keys(models)[0];
    const selected = models[selectedKey] || {};
    const metrics = selected.metrics || {};

    setText("selected-model-name", data.model_name || selected.display_name || "Selected model");
    setText("selected-accuracy", formatPercent(metrics.accuracy, 2));
    setText("selected-model-note", (
        "Selected by the highest weighted F1-score on the held-out test set. " +
        "The comparison is based on measured performance, not a manual choice."
    ));
    setText("metric-accuracy", formatPercent(metrics.accuracy, 2));
    setText("metric-precision", formatPercent(metrics.precision, 2));
    setText("metric-recall", formatPercent(metrics.recall, 2));
    setText("metric-f1", formatPercent(metrics.f1_score, 2));

    const split = data.split || {};
    setText("metric-caption", (
        `Measured on ${formatInteger(split.test_rows)} test reviews. ` +
        "Precision, recall and F1 are weighted by class support."
    ));

    const dataset = data.dataset || {};
    const summary = dataset.summary || {};
    setText("dataset-title", dataset.name || "IMDb Movie Reviews");
    setText("dataset-rows", formatInteger(summary.rows_loaded));
    setText("dataset-duplicates", formatInteger(summary.duplicates_removed));
    setText("dataset-clean", formatInteger(summary.final_rows));
    setText(
        "dataset-balance",
        `${formatInteger(summary.positive_reviews)} / ${formatInteger(summary.negative_reviews)}`
    );
    setText("dataset-length", `${Number(summary.average_words || 0).toFixed(1)} words`);

    setText("split-rows", `${formatInteger(split.train_rows)} / ${formatInteger(split.test_rows)}`);
    setText(
        "split-strategy",
        split.stratified ? "Stratified 80 / 20" : "80 / 20 hold-out"
    );
    setText("split-random", split.random_state ?? "\u2014");

    const tfidf = data.tfidf || {};
    const ngrams = tfidf.ngram_range || [];
    setText("tfidf-features", formatInteger(tfidf.n_features_learned));
    setText(
        "tfidf-ngrams",
        ngrams.length === 2 && ngrams[0] === 1 && ngrams[1] === 2
            ? "Unigram + bigram"
            : ngrams.join(" to ")
    );
    setText("tfidf-fitted", tfidf.fitted_on || "training split only");

    const tableBody = document.getElementById("comparison-body");
    tableBody.innerHTML = "";
    for (const [key, payload] of Object.entries(models)) {
        const rowMetrics = payload.metrics || {};
        const isSelected = key === selectedKey;
        const row = document.createElement("tr");
        if (isSelected) row.className = "selected-row";
        row.innerHTML = `
            <td>${escapeHtml(payload.display_name || key)}</td>
            <td>${Number(rowMetrics.accuracy || 0).toFixed(4)}</td>
            <td>${Number(rowMetrics.precision || 0).toFixed(4)}</td>
            <td>${Number(rowMetrics.recall || 0).toFixed(4)}</td>
            <td>${Number(rowMetrics.f1_score || 0).toFixed(4)}</td>
            <td>${isSelected ? "Yes" : ""}</td>
        `;
        tableBody.appendChild(row);
    }

    const images = Array.from(document.querySelectorAll(".chart-card img"));
    (data.charts || []).forEach((chart, index) => {
        if (!images[index]) return;
        images[index].src = chart.src;
        images[index].alt = chart.alt || chart.title || "";
    });
}

async function loadMetadata() {
    try {
        const data = await fetchJson(`${API_BASE}/metadata`);
        renderMetadata(data);
    } catch (error) {
        setError("The measured results could not be loaded. " + error.message);
    }
}

async function loadHealth() {
    try {
        const health = await fetchJson(`${API_BASE}/health`);
        setStatus(Boolean(health.model_ready), health.model_ready ? "Model ready" : "Model not ready");
    } catch {
        setStatus(false, "Model unavailable");
    }
}

elements.input.addEventListener("input", updateCounter);
elements.input.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        analyze();
    }
});
elements.analyze.addEventListener("click", analyze);
elements.clear.addEventListener("click", clearInput);
elements.sampleButtons.forEach((button) => {
    button.addEventListener("click", () => loadSample(button.dataset.sample));
});

document.querySelectorAll(".chart-card img").forEach((image) => {
    image.addEventListener("error", () => {
        const figure = image.closest("figure");
        if (figure) figure.hidden = true;
    });
});

updateCounter();
showEmpty();
loadHealth();
loadMetadata();
