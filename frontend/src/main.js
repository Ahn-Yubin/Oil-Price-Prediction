let chartRef = null;
let refreshTimer = null;
let candleSeriesRef = null;
let predSeriesRef = null;
let predUpperSeriesRef = null;
let predLowerSeriesRef = null;
let predBandFillRef = null;
let predBandMaskRef = null;
let predTailFillRef = null;
let predTailMaskRef = null;
let forecastModelSeriesRefs = new Map();
let createLineSeriesRef = null;
let isResizeBound = false;
let isCrosshairBound = false;
let activeDataKey = null;
let latestPayload = null;
let predictedByTime = new Map();
let forecastByTime = new Map();
let requestVersion = 0;
let modelCatalog = new Map();

const MODEL_LABELS = {
  motif: "Motif",
  pattern_mlp: "Pattern MLP",
  deep_lstm_tcn_fusion: "Deep LSTM+TCN",
  llm_context_seq_moe: "LLM Context MoE",
  random_walk: "Random Walk",
  drift: "Drift",
  seasonal_naive: "Seasonal Naive",
  volatility_scaled_naive: "Vol-Scaled Naive",
};

function normalizeSymbolText(value) {
  return String(value || "").trim().toUpperCase();
}

function updateSymbolSuggestionState() {
  const input = document.getElementById("symbol-input");
  const current = normalizeSymbolText(input ? input.value : "");
  document.querySelectorAll(".symbol-option").forEach((item) => {
    const sym = normalizeSymbolText(item.dataset.symbol || item.textContent || "");
    item.classList.toggle("active", !!current && sym === current);
  });
}

function openSymbolDropdown() {
  const dd = document.getElementById("symbol-dropdown");
  dd?.classList.remove("hidden");
  document.getElementById("symbol-input")?.setAttribute("aria-expanded", "true");
}

function closeSymbolDropdown() {
  const dd = document.getElementById("symbol-dropdown");
  dd?.classList.add("hidden");
  document.getElementById("symbol-input")?.setAttribute("aria-expanded", "false");
}

async function loadData(symbol, interval, models = "") {
  const ts = Date.now();
  const modelQuery = models ? `&models=${encodeURIComponent(models)}` : "";
  const forecastUrl = `/api/forecast?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}${modelQuery}&_ts=${ts}`;
  let response = await fetch(forecastUrl, { cache: "no-store" });
  if (response.ok) {
    const forecast = await response.json();
    return convertForecastToChartPayload(forecast);
  }
  response = await fetch(
    `/api/chart?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}${modelQuery}&_ts=${ts}`,
    { cache: "no-store" },
  );
  if (!response.ok) {
    let message = "Failed to load chart data";
    try {
      const body = await response.json();
      if (body && body.detail) {
        message = typeof body.detail === "string" ? body.detail : body.detail.message || JSON.stringify(body.detail);
      }
    } catch (_err) {
      // ignore parse failure
    }
    throw new Error(message);
  }
  return response.json();
}

function convertForecastToChartPayload(forecast) {
  const candles = forecast.candles || [];
  const last = candles.length ? candles[candles.length - 1] : null;
  const anchor = last ? { time: last.time, value: forecast.current_price } : null;
  const points = forecast.forecast || [];
  const lineFrom = (key) => (anchor ? [anchor] : []).concat(points.map((p) => ({ time: p.time, value: p[key] })));
  const scenarioModels = [];
  if (forecast.scenarios) {
    [
      ["bull", "Bull", "#7ee787"],
      ["base", "Base", "#58a6ff"],
      ["bear", "Bear", "#ff7b72"],
    ].forEach(([id, label, color]) => {
      const data = forecast.scenarios[id] || [];
      scenarioModels.push({
        id: `scenario_${id}`,
        label,
        description: `${label} scenario`,
        color,
        points: (anchor ? [anchor] : []).concat(data.map((p) => ({ time: p.time, value: p.value }))),
      });
    });
  }
  const warningObjects = Array.isArray(forecast.warning_objects) ? forecast.warning_objects : [];
  const warningMessages = warningObjects.length
    ? warningObjects
        .filter((item) => ["warning", "error"].includes(String(item.severity || "warning")))
        .map((item) => item.message)
    : forecast.warnings || [];
  const infoMessages = warningObjects
    .filter((item) => String(item.severity || "warning") === "info")
    .map((item) => item.message);
  return {
    candles,
    predicted: lineFrom("p50"),
    predicted_lower: lineFrom("p10"),
    predicted_upper: lineFrom("p90"),
    predicted_tail_lower: lineFrom("p05"),
    predicted_tail_upper: lineFrom("p95"),
    forecast_models: forecast.model_paths?.length ? forecast.model_paths : scenarioModels,
    metrics: {
      mae: null,
      rmse: null,
      mape: null,
      model: forecast.model_version || "Forecast API",
    },
    symbol_input: forecast.data_status?.symbol_requested || forecast.symbol,
    symbol_resolved: forecast.symbol,
    interval_resolved: forecast.interval,
    interval_requested: forecast.data_status?.interval_requested || forecast.interval,
    updated_at: forecast.generated_at,
    data_source: forecast.data_status?.source,
    data_status: forecast.data_status,
    warning: warningMessages.join(" "),
    warnings: forecast.warnings || [],
    warning_objects: warningObjects,
    info_messages: infoMessages,
    forecast_horizon: points.length,
    confidence_level: null,
    asset_metadata: forecast.asset_metadata,
    regime: forecast.regime,
    cross_asset_context: forecast.cross_asset_context,
    selected_models: forecast.selected_models || [],
    primary_model: forecast.primary_model || null,
    artifact_status: forecast.artifact_status || {},
    calibration_status: forecast.calibration_status || {},
    llm_context_summary: forecast.llm_context_summary || {},
    explanation_hint: "Use /api/explanation for structured context.",
  };
}

function formatMetric(value, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Number(value).toFixed(3)}${suffix}`;
}

function setMetrics(metrics, updatedAt, forecastHorizon, confidenceLevel) {
  document.getElementById("mae-value").textContent = formatMetric(metrics.mae);
  document.getElementById("rmse-value").textContent = formatMetric(metrics.rmse);
  document.getElementById("mape-value").textContent = formatMetric(metrics.mape, "%");
  document.getElementById("chart-updated-value").textContent =
    `Updated ${new Date(updatedAt).toLocaleString()}`;
  const ci = confidenceLevel ? `${Math.round(confidenceLevel * 100)}% band` : "uncalibrated band";
  const horizonText = forecastHorizon ? `, ${forecastHorizon} steps` : "";
  document.getElementById("model-value").textContent = `${metrics.model || "-"} (${ci}${horizonText})`;
}

function setDataStatusBadge(dataStatus) {
  const badge = document.getElementById("data-status-badge");
  if (!badge) return;
  const status = String(dataStatus?.status || "unknown").toUpperCase();
  badge.textContent = `DATA ${status}`;
  badge.dataset.status = String(dataStatus?.status || "unknown").toLowerCase();
}

function setForecastBadges(payload) {
  const confidence = document.getElementById("confidence-badge");
  const regime = document.getElementById("regime-badge");
  const calibration = document.getElementById("calibration-badge");
  if (confidence) {
    const values = (payload.predicted || []).slice(1);
    confidence.textContent = values.length ? `CONF ${Math.round((payload.regime?.confidence || 0.5) * 100)}%` : "CONF -";
  }
  if (regime) {
    const probs = payload.regime || {};
    const entries = Object.entries(probs).filter(([key]) => key !== "confidence");
    const label = entries.length ? entries.sort((a, b) => Number(b[1]) - Number(a[1]))[0][0] : "unknown";
    regime.textContent = `REGIME ${String(label).replaceAll("_", " ").toUpperCase()}`;
  }
  if (calibration) {
    const status = String(payload.calibration_status?.calibration_status || "uncalibrated").toUpperCase();
    calibration.textContent = `BAND ${status}`;
    calibration.dataset.status = status.toLowerCase();
  }
  const modelValue = document.getElementById("model-value");
  if (modelValue && payload.primary_model) {
    const llm = payload.llm_context_summary?.enabled ? "LLM context on" : "LLM context off";
    modelValue.textContent = `${payload.primary_model} (${llm})`;
  }
}

async function loadExplanation(symbol, interval) {
  const panel = document.getElementById("explanation-panel");
  if (!panel) return;
  try {
    const res = await fetch(`/api/explanation?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}`, {
      cache: "no-store",
    });
    if (!res.ok) throw new Error("explanation unavailable");
    const body = await res.json();
    panel.querySelector("#explanation-summary").textContent = body.summary || "-";
    panel.querySelector("#explanation-drivers").replaceChildren(
      ...(body.main_drivers || []).map((driver) => {
        const li = document.createElement("li");
        li.textContent = driver;
        return li;
      }),
    );
    panel.querySelector("#explanation-warning").textContent = body.confidence_warning || "";
  } catch (_err) {
    panel.querySelector("#explanation-summary").textContent = "Explanation unavailable for this request.";
    panel.querySelector("#explanation-drivers").replaceChildren();
    panel.querySelector("#explanation-warning").textContent = "";
  }
}

async function loadBacktestDiagnostics(symbol, interval) {
  const table = document.getElementById("leaderboard-table");
  const status = document.getElementById("diagnostics-status");
  if (!table || !status) return;
  try {
    const res = await fetch(`/api/backtests?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}`, {
      cache: "no-store",
    });
    if (!res.ok) throw new Error("backtest unavailable");
    const body = await res.json();
    status.textContent = body.status || "missing";
    const rows = (body.leaderboard || []).slice(0, 8);
    if (!rows.length) {
      table.textContent = "No rolling leaderboard available.";
      return;
    }
    const cols = ["model", "rmse", "mae", "pinball_loss", "coverage_80", "directional_accuracy"];
    const header = document.createElement("div");
    header.className = "leaderboard-row leaderboard-head";
    cols.forEach((col) => {
      const cell = document.createElement("span");
      cell.textContent = col.replaceAll("_", " ").toUpperCase();
      header.appendChild(cell);
    });
    const bodyRows = rows.map((row) => {
      const line = document.createElement("div");
      line.className = "leaderboard-row";
      cols.forEach((col) => {
        const cell = document.createElement("span");
        const value = row[col];
        cell.textContent = typeof value === "number" ? value.toFixed(3) : value ?? "-";
        line.appendChild(cell);
      });
      return line;
    });
    table.replaceChildren(header, ...bodyRows);
  } catch (_err) {
    status.textContent = "missing";
    table.textContent = "Backtest diagnostics unavailable.";
  }
}

function setStatus(message, severity = "warning") {
  const banner = document.getElementById("status-banner");
  if (!message) {
    banner.classList.add("hidden");
    banner.textContent = "";
    banner.removeAttribute("data-severity");
    return;
  }
  banner.classList.remove("hidden");
  banner.dataset.severity = severity;
  banner.textContent = message;
}

function setInfoMessages(messages) {
  const banner = document.getElementById("info-banner");
  if (!banner) return;
  const values = (messages || []).filter(Boolean);
  if (!values.length) {
    banner.classList.add("hidden");
    banner.textContent = "";
    return;
  }
  banner.classList.remove("hidden");
  banner.textContent = values.join(" ");
}

function setForecastNotices(payload) {
  setStatus(payload.warning || null, "warning");
  setInfoMessages(payload.info_messages || []);
}

async function loadModelCatalog() {
  const select = document.getElementById("model-input");
  if (!select) return;
  try {
    const response = await fetch(`/api/models?_ts=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error("model catalog unavailable");
    const body = await response.json();
    modelCatalog = new Map((body.user_facing_models || []).map((item) => [item.id, item]));
    const current = select.value || "";
    select.replaceChildren();
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "Default";
    select.appendChild(defaultOption);
    (body.user_facing_models || []).forEach((model) => {
      const option = document.createElement("option");
      option.value = model.id;
      const label = MODEL_LABELS[model.id] || model.id;
      option.textContent = model.status === "artifact_missing" ? `${label} (Train required)` : label;
      option.dataset.status = model.status || "available";
      if (model.training_command) {
        option.dataset.trainingCommand = model.training_command;
        option.title = model.training_command;
      }
      select.appendChild(option);
    });
    select.value = [...select.options].some((option) => option.value === current) ? current : "";
  } catch (_err) {
    // Keep static options when the model catalog is unavailable.
  }
}

function selectedModelCatalogEntry() {
  const modelInput = document.getElementById("model-input");
  const modelId = modelInput ? modelInput.value || "" : "";
  return modelCatalog.get(modelId) || null;
}

function toUnixTime(time) {
  if (typeof time === "number") return time;
  if (time && typeof time === "object" && "year" in time && "month" in time && "day" in time) {
    return Math.floor(Date.UTC(time.year, time.month - 1, time.day) / 1000);
  }
  return null;
}

function toDisplayTime(time) {
  const ts = toUnixTime(time);
  if (ts === null) return "-";
  return new Date(ts * 1000).toLocaleString();
}

function formatPrice(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(4);
}

function primaryForecastModel(payload) {
  const models = payload?.forecast_models || [];
  return models.find((model) => model.id === payload?.primary_model) || models.find((model) => model.id === "motif") || models[0] || null;
}

function setLegendRows(rows) {
  const legend = document.getElementById("tv-legend");
  if (!legend) return;
  legend.replaceChildren();
  rows.forEach((row) => {
    const line = document.createElement("div");
    line.className = row.className ? `line ${row.className}` : "line";
    line.textContent = row.text;
    legend.appendChild(line);
  });
}

function renderLegend(symbol, interval, timeLabel, ohlc, pred, forecastValues = null) {
  const rows = [
    { text: `${symbol} · ${interval} · ${timeLabel}`, className: "head" },
    { text: `O ${formatPrice(ohlc?.open)}  H ${formatPrice(ohlc?.high)}` },
    { text: `L ${formatPrice(ohlc?.low)}  C ${formatPrice(ohlc?.close)}` },
    { text: `PRED ${formatPrice(pred)}`, className: "pred" },
  ];
  if (forecastValues) {
    Object.entries(forecastValues).forEach(([label, value]) => {
      rows.push({ text: `${label} ${formatPrice(value)}`, className: "model-pred" });
    });
  }
  setLegendRows(rows);
}

function refreshLegendDefault() {
  if (!latestPayload) return;
  const candles = latestPayload.candles || [];
  const lastCandle = candles.length ? candles[candles.length - 1] : null;
  const symbol = latestPayload.symbol_input || latestPayload.symbol_resolved || "-";
  const interval = (latestPayload.interval_resolved || "").toUpperCase() || "-";
  const pred = lastCandle ? predictedByTime.get(String(lastCandle.time)) ?? null : null;
  const modelValues = lastCandle ? forecastByTime.get(String(lastCandle.time)) ?? null : null;
  renderLegend(
    symbol,
    interval,
    lastCandle ? toDisplayTime(lastCandle.time) : "-",
    lastCandle
      ? {
          open: lastCandle.open,
          high: lastCandle.high,
          low: lastCandle.low,
          close: lastCandle.close,
        }
      : null,
    pred,
    modelValues,
  );
}

function updateLegendOnCrosshair(param) {
  if (!latestPayload) return;
  if (!param || !param.time) {
    refreshLegendDefault();
    return;
  }

  const seriesMap = param.seriesData || param.seriesPrices;
  const candle = seriesMap && typeof seriesMap.get === "function" ? seriesMap.get(candleSeriesRef) : null;
  const predRaw = seriesMap && typeof seriesMap.get === "function" ? seriesMap.get(predSeriesRef) : null;
  const predFromSeries = typeof predRaw === "number" ? predRaw : null;
  const crossTime = toUnixTime(param.time);
  const predFromLookup =
    crossTime !== null && predictedByTime.has(String(crossTime))
      ? predictedByTime.get(String(crossTime))
      : null;
  const forecastValues = crossTime !== null ? forecastByTime.get(String(crossTime)) ?? null : null;

  const symbol = latestPayload.symbol_input || latestPayload.symbol_resolved || "-";
  const interval = (latestPayload.interval_resolved || "").toUpperCase() || "-";

  if (candle && typeof candle === "object" && "open" in candle) {
    renderLegend(symbol, interval, toDisplayTime(param.time), candle, predFromSeries ?? predFromLookup, forecastValues);
    return;
  }

  // Future forecast region has no candle; keep OHLC empty and show prediction.
  renderLegend(symbol, interval, toDisplayTime(param.time), null, predFromSeries ?? predFromLookup, forecastValues);
}

function ensureChart() {
  if (chartRef) return;
  if (typeof LightweightCharts === "undefined") {
    throw new Error("LightweightCharts is not loaded");
  }
  const container = document.getElementById("oilChart");
  chartRef = LightweightCharts.createChart(container, {
    layout: {
      background: { type: "solid", color: "#151b23" },
      textColor: "#8b949e",
    },
    rightPriceScale: {
      borderColor: "#30363d",
    },
    timeScale: {
      borderColor: "#30363d",
      timeVisible: true,
    },
    grid: {
      vertLines: { color: "#21262d" },
      horzLines: { color: "#21262d" },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
    },
    width: container.clientWidth,
    height: 520,
  });

  const createLineSeries = (options) => {
    // v4 API: chart.addLineSeries(options)
    if (typeof chartRef.addLineSeries === "function") {
      return chartRef.addLineSeries(options);
    }
    // v5 API: chart.addSeries(LightweightCharts.LineSeries, options)
    if (typeof chartRef.addSeries === "function" && LightweightCharts.LineSeries) {
      return chartRef.addSeries(LightweightCharts.LineSeries, options);
    }
    throw new Error("Unsupported lightweight-charts API version");
  };
  createLineSeriesRef = createLineSeries;

  const createCandlestickSeries = (options) => {
    // v4 API: chart.addCandlestickSeries(options)
    if (typeof chartRef.addCandlestickSeries === "function") {
      return chartRef.addCandlestickSeries(options);
    }
    // v5 API: chart.addSeries(LightweightCharts.CandlestickSeries, options)
    if (typeof chartRef.addSeries === "function" && LightweightCharts.CandlestickSeries) {
      return chartRef.addSeries(LightweightCharts.CandlestickSeries, options);
    }
    throw new Error("Unsupported lightweight-charts API version");
  };

  const createAreaSeries = (options) => {
    // v4 API: chart.addAreaSeries(options)
    if (typeof chartRef.addAreaSeries === "function") {
      return chartRef.addAreaSeries(options);
    }
    // v5 API: chart.addSeries(LightweightCharts.AreaSeries, options)
    if (typeof chartRef.addSeries === "function" && LightweightCharts.AreaSeries) {
      return chartRef.addSeries(LightweightCharts.AreaSeries, options);
    }
    throw new Error("Unsupported lightweight-charts API version");
  };

  predBandFillRef = createAreaSeries({
    title: "P10-P90 Forecast Band",
    lineColor: "rgba(88, 166, 255, 0.0)",
    topColor: "rgba(88, 166, 255, 0.24)",
    bottomColor: "rgba(88, 166, 255, 0.04)",
    lineWidth: 1,
  });
  // Mask lower half so only [lower, upper] band stays visible.
  predBandMaskRef = createAreaSeries({
    lineColor: "rgba(21,27,35,0.0)",
    topColor: "rgba(21,27,35,1.0)",
    bottomColor: "rgba(21,27,35,1.0)",
    lineWidth: 1,
  });
  predTailFillRef = createAreaSeries({
    title: "P05-P95 Tail Band",
    lineColor: "rgba(188, 140, 255, 0.0)",
    topColor: "rgba(188, 140, 255, 0.12)",
    bottomColor: "rgba(188, 140, 255, 0.02)",
    lineWidth: 1,
  });
  predTailMaskRef = createAreaSeries({
    lineColor: "rgba(21,27,35,0.0)",
    topColor: "rgba(21,27,35,0.88)",
    bottomColor: "rgba(21,27,35,0.88)",
    lineWidth: 1,
  });

  candleSeriesRef = createCandlestickSeries({
    title: "Actual OHLC",
    upColor: "#2dd4bf",
    downColor: "#ff7b72",
    borderUpColor: "#2dd4bf",
    borderDownColor: "#ff7b72",
    wickUpColor: "#2dd4bf",
    wickDownColor: "#ff7b72",
  });
  predSeriesRef = createLineSeries({
    title: "Predicted",
    color: "#58a6ff",
    lineWidth: 2.2,
    lineStyle:
      LightweightCharts.LineStyle && LightweightCharts.LineStyle.Dashed !== undefined
        ? LightweightCharts.LineStyle.Dashed
        : 2,
  });
  predUpperSeriesRef = createLineSeries({
    title: "Pred Upper",
    color: "rgba(88, 166, 255, 0.56)",
    lineWidth: 1.2,
  });
  predLowerSeriesRef = createLineSeries({
    title: "Pred Lower",
    color: "rgba(88, 166, 255, 0.56)",
    lineWidth: 1.2,
  });

  if (!isResizeBound) {
    window.addEventListener("resize", () => {
      if (chartRef) {
        chartRef.applyOptions({ width: container.clientWidth });
      }
    });
    isResizeBound = true;
  }

  if (!isCrosshairBound && typeof chartRef.subscribeCrosshairMove === "function") {
    chartRef.subscribeCrosshairMove(updateLegendOnCrosshair);
    isCrosshairBound = true;
  }
}

function setInitialForecastView(payload) {
  const timeScale = chartRef.timeScale();
  if (typeof timeScale.setVisibleLogicalRange !== "function") {
    timeScale.fitContent();
    return;
  }

  const candles = payload.candles || [];
  const predicted = payload.predicted || [];
  if (!candles.length || predicted.length < 2) {
    timeScale.fitContent();
    return;
  }

  const lastCandle = candles[candles.length - 1];
  const forecastStartsOnLastCandle = String(predicted[0]?.time) === String(lastCandle?.time);
  const forecastStartIndex = forecastStartsOnLastCandle ? candles.length - 1 : candles.length;
  const interval = payload.interval_resolved || "1d";
  const futureBars = Math.max(1, predicted.length - 1);
  const rightPadding = Math.max(8, Math.round(futureBars * 0.18));
  const forecastPosition = 0.62;
  const minPastBarsByInterval = {
    "1d": 90,
    "1h": 160,
    "30m": 180,
    "15m": 220,
  };
  const pastBars = Math.max(
    minPastBarsByInterval[interval] || 90,
    24,
    Math.round(((futureBars + rightPadding) * forecastPosition) / (1 - forecastPosition)),
  );
  const from = Math.max(0, forecastStartIndex - pastBars);
  const to = forecastStartIndex + futureBars + rightPadding;

  if (to > from) {
    timeScale.setVisibleLogicalRange({ from, to });
    return;
  }
  timeScale.fitContent();
}

function setModelOverlayLegend(models) {
  const root = document.getElementById("model-overlay-legend");
  if (!root) return;
  root.replaceChildren();
  (models || []).forEach((model) => {
    const item = document.createElement("span");
    item.className = "model-chip";
    item.style.setProperty("--model-color", model.color || "#8b949e");
    item.textContent = model.label || model.id || "Model";
    root.appendChild(item);
  });
}

function rebuildForecastLookup(models) {
  const next = new Map();
  (models || []).forEach((model) => {
    const label = model.label || model.id || "Model";
    (model.points || []).forEach((point) => {
      const key = String(point.time);
      const entry = next.get(key) || {};
      entry[label] = point.value;
      next.set(key, entry);
    });
  });
  forecastByTime = next;
}

function renderForecastModelSeries(models) {
  if (!createLineSeriesRef) return;
  const overlays = (models || []).filter((model) => model.id !== "motif");
  const nextIds = new Set(overlays.map((model) => model.id));
  forecastModelSeriesRefs.forEach((series, id) => {
    if (!nextIds.has(id)) {
      if (typeof chartRef.removeSeries === "function") {
        chartRef.removeSeries(series);
      }
      forecastModelSeriesRefs.delete(id);
    }
  });

  overlays.forEach((model) => {
    let series = forecastModelSeriesRefs.get(model.id);
    if (!series) {
      series = createLineSeriesRef({
        title: model.label || model.id,
        color: model.color || "#8b949e",
        lineWidth: model.id === "ensemble" ? 2 : 1.4,
        lineStyle:
          LightweightCharts.LineStyle && LightweightCharts.LineStyle.Solid !== undefined
            ? LightweightCharts.LineStyle.Solid
            : 0,
      });
      forecastModelSeriesRefs.set(model.id, series);
    } else if (typeof series.applyOptions === "function") {
      series.applyOptions({
        title: model.label || model.id,
        color: model.color || "#8b949e",
        lineWidth: model.id === "ensemble" ? 2 : 1.4,
      });
    }
    series.setData(model.points || []);
  });
}

function renderChart(payload, resetView = false) {
  ensureChart();
  latestPayload = payload;
  predictedByTime = new Map((payload.predicted || []).map((p) => [String(p.time), p.value]));
  rebuildForecastLookup(payload.forecast_models || []);
  setModelOverlayLegend(payload.forecast_models || []);
  const timeScale = chartRef.timeScale();
  const savedRange =
    !resetView && typeof timeScale.getVisibleLogicalRange === "function"
      ? timeScale.getVisibleLogicalRange()
      : null;

  const upper = payload.predicted_upper || payload.predicted || [];
  const lower = payload.predicted_lower || payload.predicted || [];
  const tailUpper = payload.predicted_tail_upper || upper;
  const tailLower = payload.predicted_tail_lower || lower;
  const primaryModel = primaryForecastModel(payload);
  if (primaryModel && typeof predSeriesRef.applyOptions === "function") {
    predSeriesRef.applyOptions({
      title: `${primaryModel.label || "Primary"} Forecast`,
      color: primaryModel.color || "#d29922",
      lineWidth: 2.4,
    });
  }
  predTailFillRef.setData(tailUpper);
  predTailMaskRef.setData(tailLower);
  predBandFillRef.setData(upper);
  predBandMaskRef.setData(lower);
  candleSeriesRef.setData(payload.candles || []);
  predSeriesRef.setData(payload.predicted || []);
  predUpperSeriesRef.setData(upper);
  predLowerSeriesRef.setData(lower);
  renderForecastModelSeries(payload.forecast_models || []);
  refreshLegendDefault();

  if (resetView) {
    setInitialForecastView(payload);
    return;
  }

  if (savedRange && typeof timeScale.setVisibleLogicalRange === "function") {
    timeScale.setVisibleLogicalRange(savedRange);
    return;
  }
  timeScale.fitContent();
}

async function initDashboard(symbol, interval) {
  const reqId = ++requestVersion;
  const loadBtn = document.getElementById("load-button");
  const modelInput = document.getElementById("model-input");
  const selectedModels = modelInput ? modelInput.value || "" : "";
  if (loadBtn) loadBtn.disabled = true;
  try {
    const payload = await loadData(symbol, interval, selectedModels);
    if (reqId !== requestVersion) return;
    setMetrics(
      payload.metrics,
      payload.updated_at,
      payload.forecast_horizon,
      payload.confidence_level,
    );
    setDataStatusBadge(payload.data_status);
    setForecastBadges(payload);
    setForecastNotices(payload);
    loadExplanation(symbol, interval);
    loadBacktestDiagnostics(symbol, interval);
    try {
      const nextDataKey = `${payload.symbol_input}|${payload.interval_resolved}`;
      const shouldResetView = activeDataKey !== nextDataKey;
      renderChart(payload, shouldResetView);
      activeDataKey = nextDataKey;
    } catch (chartError) {
      console.error(chartError);
      document.getElementById("updated-value").textContent = `${new Date(
        payload.updated_at,
      ).toLocaleString()} (chart render failed)`;
    }
  } catch (error) {
    if (reqId !== requestVersion) return;
    console.error(error);
    document.getElementById("chart-updated-value").textContent = "Updated -";
    setStatus(`API 요청 실패: ${error?.message || "서버 로그를 확인하세요."}`, "warning");
    setInfoMessages([]);
  } finally {
    if (reqId === requestVersion && loadBtn) {
      loadBtn.disabled = false;
    }
  }
}

function bindControls() {
  const input = document.getElementById("symbol-input");
  const combobox = document.getElementById("symbol-combobox");
  const dropdown = document.getElementById("symbol-dropdown");
  const intervalInput = document.getElementById("interval-input");
  const modelInput = document.getElementById("model-input");
  const button = document.getElementById("load-button");
  const triggerSearch = () => {
    const symbol = (input.value || "NYMEX:CL1!").trim();
    const interval = intervalInput.value || "1d";
    const catalogEntry = selectedModelCatalogEntry();
    if (catalogEntry && catalogEntry.status === "artifact_missing") {
      const label = MODEL_LABELS[catalogEntry.id] || catalogEntry.id;
      setStatus(
        `${label} requires training before dashboard use. Command: ${catalogEntry.training_command}`,
        "warning",
      );
      setInfoMessages([]);
      modelInput.value = "";
      return;
    }
    // Force new chart request even for same symbol/interval.
    activeDataKey = null;
    initDashboard(symbol, interval);
  };

  dropdown?.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target.closest(".symbol-option") : null;
    if (!target) return;
    input.value = target.dataset.symbol || target.textContent || "";
    updateSymbolSuggestionState();
    closeSymbolDropdown();
    triggerSearch();
  });

  button.addEventListener("click", triggerSearch);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      closeSymbolDropdown();
      triggerSearch();
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      openSymbolDropdown();
    }
  });
  input.addEventListener("focus", () => {
    updateSymbolSuggestionState();
    openSymbolDropdown();
  });
  input.addEventListener("click", () => {
    updateSymbolSuggestionState();
    openSymbolDropdown();
  });
  input.addEventListener("input", () => {
    updateSymbolSuggestionState();
    // 입력값이 있어도 추천 드롭다운 목록은 계속 유지
    openSymbolDropdown();
  });
  input.addEventListener("change", updateSymbolSuggestionState);
  document.addEventListener("click", (event) => {
    if (!combobox?.contains(event.target)) {
      closeSymbolDropdown();
    }
  });
  intervalInput.addEventListener("change", triggerSearch);
  modelInput?.addEventListener("change", triggerSearch);
  updateSymbolSuggestionState();
}

function startAutoRefresh() {
  if (refreshTimer) {
    clearInterval(refreshTimer);
  }
  refreshTimer = setInterval(() => {
    const symbol = document.getElementById("symbol-input").value || "NYMEX:CL1!";
    const interval = document.getElementById("interval-input").value || "1d";
    initDashboard(symbol, interval);
  }, 15_000);
}

async function bootDashboard() {
  bindControls();
  await loadModelCatalog();
  initDashboard(
    document.getElementById("symbol-input").value || "NYMEX:CL1!",
    document.getElementById("interval-input").value || "1d",
  );
  startAutoRefresh();
}

bootDashboard();
