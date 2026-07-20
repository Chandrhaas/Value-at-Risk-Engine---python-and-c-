// ==========================================================================
// VaR Engine — frontend logic
// Calls POST /portfolio/var (same-origin, served by FastAPI StaticFiles
// mount, so no CORS setup needed) and renders the combined Monte Carlo +
// Neural Network response.
// ==========================================================================

const form = document.getElementById("varForm");
const submitBtn = document.getElementById("submitBtn");
const submitLabel = document.getElementById("submitLabel");
const spinner = document.getElementById("spinner");
const errorBox = document.getElementById("errorBox");
const resultsSection = document.getElementById("results");

const weightsToggle = document.getElementById("weightsToggle");
const weightsField = document.getElementById("weightsField");

// -------------------- helpers --------------------

function formatCurrency(value) {
  const sign = value < 0 ? "-" : "";
  return `${sign}$${Math.abs(value).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function formatPercent(value) {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}%`;
}

function parseTickers(raw) {
  return raw
    .split(",")
    .map((t) => t.trim().toUpperCase())
    .filter(Boolean);
}

function parseWeights(raw) {
  if (!raw || !raw.trim()) return null;
  return raw
    .split(",")
    .map((w) => parseFloat(w.trim()))
    .filter((w) => !Number.isNaN(w));
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitLabel.hidden = isLoading;
  spinner.hidden = !isLoading;
}

// -------------------- weights toggle --------------------

weightsToggle.addEventListener("click", () => {
  const isHidden = weightsField.hidden;
  weightsField.hidden = !isHidden;
  weightsToggle.textContent = isHidden ? "− Custom weights" : "+ Custom weights";
});

// -------------------- tabs --------------------

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// -------------------- delta annotation --------------------
// The one thing the page is built to say clearly: the gap between the two
// methods isn't "which one is right" -- it's roughly how much diversification
// credit the Monte Carlo side is giving that the NN's conservative summed
// estimate doesn't.

function renderDelta(elementId, mcValue, nnValue) {
  const el = document.getElementById(elementId);
  const diff = nnValue - mcValue;
  const pctBase = Math.abs(mcValue);
  const pct = pctBase > 0 ? (Math.abs(diff) / pctBase) * 100 : 0;
  const direction = diff >= 0 ? "higher" : "lower";
  el.textContent = `NN ${pct.toFixed(0)}% ${direction}`;
}

// -------------------- rendering --------------------

function renderTickerTable(perTicker) {
  const tbody = document.getElementById("tickerTableBody");
  tbody.innerHTML = "";

  Object.values(perTicker).forEach((row) => {
    const tr = document.createElement("tr");
    const var95Class = row.var_95 < 0 ? "value-flag" : "";
    const var99Class = row.var_99 < 0 ? "value-flag" : "";

    tr.innerHTML = `
      <td>${row.ticker}</td>
      <td>${row.as_of_date}</td>
      <td>${formatPercent(row.q05_return)}</td>
      <td>${formatPercent(row.q01_return)}</td>
      <td>${(row.weight * 100).toFixed(1)}%</td>
      <td>${formatCurrency(row.dollar_value)}</td>
      <td class="${var95Class}">${formatCurrency(row.var_95)}</td>
      <td class="${var99Class}">${formatCurrency(row.var_99)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderResults(data) {
  const mc = data.monte_carlo;
  const nn = data.neural_network;

  document.getElementById("mcVar95").textContent = formatCurrency(mc.var_95);
  document.getElementById("mcVar99").textContent = formatCurrency(mc.var_99);
  document.getElementById("nnVar95").textContent = formatCurrency(nn.var_95);
  document.getElementById("nnVar99").textContent = formatCurrency(nn.var_99);

  document.getElementById("mcVar95Full").textContent = formatCurrency(mc.var_95);
  document.getElementById("mcVar99Full").textContent = formatCurrency(mc.var_99);
  document.getElementById("mcCvar").textContent = formatCurrency(mc.cvar);
  document.getElementById("mcMessage").textContent = mc.message;

  document.getElementById("nnVar95Full").textContent = formatCurrency(nn.var_95);
  document.getElementById("nnVar99Full").textContent = formatCurrency(nn.var_99);
  document.getElementById("nnMessage").textContent = nn.message;

  document.getElementById("comparisonNote").textContent = data.comparison_note;

  renderDelta("deltaVar95", mc.var_95, nn.var_95);
  renderDelta("deltaVar99", mc.var_99, nn.var_99);

  renderTickerTable(nn.per_ticker);

  resultsSection.hidden = false;
}

// -------------------- submit --------------------

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideError();

  const tickers = parseTickers(document.getElementById("tickers").value);
  const portfolioSize = parseFloat(document.getElementById("portfolioSize").value);
  const years = parseInt(document.getElementById("years").value, 10);
  const numSimulations = parseInt(document.getElementById("numSimulations").value, 10);
  const weights = parseWeights(document.getElementById("weights").value);

  if (tickers.length === 0) {
    showError("Enter at least one ticker.");
    return;
  }
  if (weights && weights.length !== tickers.length) {
    showError(`Weights count (${weights.length}) must match ticker count (${tickers.length}).`);
    return;
  }
  if (weights) {
    const sum = weights.reduce((a, b) => a + b, 0);
    if (Math.abs(sum - 1) > 0.01) {
      showError(`Weights must sum to 1.0 (got ${sum.toFixed(4)}).`);
      return;
    }
  }

  setLoading(true);
  try {
    const response = await fetch("/portfolio/var", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        portfolio_size: portfolioSize,
        tickers,
        num_simulations: numSimulations,
        years,
        weights,
      }),
    });

    if (!response.ok) {
      let detail = response.statusText;
      try {
        const errBody = await response.json();
        detail = errBody.detail || detail;
      } catch (_) {
        // response wasn't JSON, fall back to statusText
      }
      throw new Error(detail);
    }

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    showError(err.message || "Something went wrong. Check the console and try again.");
  } finally {
    setLoading(false);
  }
});