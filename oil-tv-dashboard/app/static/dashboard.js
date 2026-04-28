let chartRef = null;
let refreshTimer = null;
let candleSeriesRef = null;
let predSeriesRef = null;
let predUpperSeriesRef = null;
let predLowerSeriesRef = null;
let predBandFillRef = null;
let predBandMaskRef = null;
let isResizeBound = false;
let isCrosshairBound = false;
let activeDataKey = null;
let latestPayload = null;
let predictedByTime = new Map();
let requestVersion = 0;

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

async function loadData(symbol, interval) {
  const ts = Date.now();
  const response = await fetch(
    `/api/chart?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}&_ts=${ts}`,
    { cache: "no-store" },
  );
  if (!response.ok) {
    let message = "Failed to load chart data";
    try {
      const body = await response.json();
      if (body && body.detail) {
        message = String(body.detail);
      }
    } catch (_err) {
      // ignore parse failure
    }
    throw new Error(message);
  }
  return response.json();
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
  const ci = confidenceLevel ? `${Math.round(confidenceLevel * 100)}% CI` : "CI";
  const horizonText = forecastHorizon ? `, ${forecastHorizon} steps` : "";
  document.getElementById("model-value").textContent = `${metrics.model || "-"} (${ci}${horizonText})`;
}

function setStatus(message) {
  const banner = document.getElementById("status-banner");
  if (!message) {
    banner.classList.add("hidden");
    banner.textContent = "";
    return;
  }
  banner.classList.remove("hidden");
  banner.textContent = message;
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

function renderLegend(symbol, interval, timeLabel, ohlc, pred) {
  setLegendRows([
    { text: `${symbol} · ${interval} · ${timeLabel}`, className: "head" },
    { text: `O ${formatPrice(ohlc?.open)}  H ${formatPrice(ohlc?.high)}` },
    { text: `L ${formatPrice(ohlc?.low)}  C ${formatPrice(ohlc?.close)}` },
    { text: `PRED ${formatPrice(pred)}`, className: "pred" },
  ]);
}

function refreshLegendDefault() {
  if (!latestPayload) return;
  const candles = latestPayload.candles || [];
  const lastCandle = candles.length ? candles[candles.length - 1] : null;
  const symbol = latestPayload.symbol_input || latestPayload.symbol_resolved || "-";
  const interval = (latestPayload.interval_resolved || "").toUpperCase() || "-";
  const pred = lastCandle ? predictedByTime.get(String(lastCandle.time)) ?? null : null;
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

  const symbol = latestPayload.symbol_input || latestPayload.symbol_resolved || "-";
  const interval = (latestPayload.interval_resolved || "").toUpperCase() || "-";

  if (candle && typeof candle === "object" && "open" in candle) {
    renderLegend(symbol, interval, toDisplayTime(param.time), candle, predFromSeries ?? predFromLookup);
    return;
  }

  // Future forecast region has no candle; keep OHLC empty and show prediction.
  renderLegend(symbol, interval, toDisplayTime(param.time), null, predFromSeries ?? predFromLookup);
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
    title: "Forecast CI",
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

function renderChart(payload, resetView = false) {
  ensureChart();
  latestPayload = payload;
  predictedByTime = new Map((payload.predicted || []).map((p) => [String(p.time), p.value]));
  const timeScale = chartRef.timeScale();
  const savedRange =
    !resetView && typeof timeScale.getVisibleLogicalRange === "function"
      ? timeScale.getVisibleLogicalRange()
      : null;

  const upper = payload.predicted_upper || payload.predicted || [];
  const lower = payload.predicted_lower || payload.predicted || [];
  predBandFillRef.setData(upper);
  predBandMaskRef.setData(lower);
  candleSeriesRef.setData(payload.candles || []);
  predSeriesRef.setData(payload.predicted || []);
  predUpperSeriesRef.setData(upper);
  predLowerSeriesRef.setData(lower);
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
  if (loadBtn) loadBtn.disabled = true;
  try {
    const payload = await loadData(symbol, interval);
    if (reqId !== requestVersion) return;
    setMetrics(
      payload.metrics,
      payload.updated_at,
      payload.forecast_horizon,
      payload.confidence_level,
    );
    setStatus(payload.warning || null);
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
    setStatus(`API 요청 실패: ${error?.message || "서버 로그를 확인하세요."}`);
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
  const button = document.getElementById("load-button");
  const triggerSearch = () => {
    const symbol = (input.value || "NYMEX:CL1!").trim();
    const interval = intervalInput.value || "1d";
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

bindControls();
initDashboard(
  document.getElementById("symbol-input").value || "NYMEX:CL1!",
  document.getElementById("interval-input").value || "1d",
);
startAutoRefresh();
