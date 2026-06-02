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
let backtestActualSeriesRef = null;
let forecastModelSeriesRefs = new Map();
let createLineSeriesRef = null;
let isResizeBound = false;
let isCrosshairBound = false;
let isClickBound = false;
let activeDataKey = null;
let latestPayload = null;
let predictedByTime = new Map();
let forecastByTime = new Map();
let requestVersion = 0;
let modelCatalog = new Map();
let latestContextPayload = null;
let latestCommentaryPayload = null;
let selectedBacktestTime = null;
let activeBacktestOriginMarker = null;
let activeBacktestPayload = null;
let currentLanguage = "ko";
let lastContextKey = "";
let lastContextLoadMs = 0;
let lastCommentaryKey = "";
let lastCommentaryLoadMs = 0;
let chartRequestInFlight = false;
let contextRequestInFlight = false;
let commentaryRequestInFlight = false;
let commentaryRequestVersion = 0;
let loadingState = { chart: false, context: false, commentary: false, backtest: false };

const CONTEXT_REFRESH_MS = 300_000;
const COMMENTARY_REFRESH_MS = 600_000;
const PRICE_REFRESH_MS = 90_000;
const DEFAULT_SYMBOL = "CL=F";

const MODEL_LABELS = {
  oil_context_fusion: "Oil Context Fusion",
  motif: "Motif",
  pattern_mlp: "Pattern MLP",
  random_walk: "Random Walk",
  drift: "Drift",
  seasonal_naive: "Seasonal Naive",
  volatility_scaled_naive: "Vol-Scaled Naive",
};

const DEFAULT_MODEL_ORDER = [
  "oil_context_fusion",
  "motif",
  "pattern_mlp",
  "random_walk",
  "drift",
  "seasonal_naive",
  "volatility_scaled_naive",
];

const DEFAULT_VISIBLE_MODELS = new Set(["oil_context_fusion"]);

const MODEL_COLORS = {
  oil_context_fusion: "#2dd4bf",
  motif: "#d29922",
  pattern_mlp: "#58a6ff",
  random_walk: "#8b949e",
  drift: "#ff7b72",
  seasonal_naive: "#f2cc60",
  volatility_scaled_naive: "#db6d28",
};

const I18N = {
  ko: {
    appTitle: "유가 예측 대시보드",
    oilInstrument: "WTI 원유",
    symbol: "심볼",
    symbolPlaceholder: "CL=F 고정",
    interval: "주기",
    horizon: "예측 기간",
    backtest: "백테스트",
    live: "라이브",
    llmCommentary: "AI 시황 해설",
    newsContext: "뉴스와 시나리오",
    bull: "상방",
    base: "기준",
    bear: "하방",
    updated: "업데이트",
    data: "데이터",
    band: "밴드",
    conf: "신뢰도",
    regime: "국면",
    waiting: "검증 대기",
    liveMetric: "라이브 지표",
    backtestMetric: "백테스트 지표",
    noContext: "현재 보이는 기간에 표시할 실시간 뉴스/이벤트 컨텍스트가 없습니다.",
    origin: "기준 시점",
    loading: "로딩 중",
    actual: "실제값",
    llm: "AI 해설",
    fallback: "대체 경로",
    cached: "캐시",
    newsTimeline: "뉴스 이벤트",
    newsCountUnit: "건",
    bullishShort: "상방",
    bearishShort: "하방",
    mixedShort: "혼조",
    neutralShort: "중립",
    loadingChart: "차트 데이터 불러오는 중",
    refreshingChart: "시장 데이터 갱신 중",
    loadingNews: "실시간 뉴스 갱신 중",
    loadingCommentary: "AI 시황 갱신 중",
    commentaryGenerating: "해설 생성 중입니다.",
    selectOriginFirst: "차트에서 기준 시점을 선택한 뒤 백테스트로 전환하세요.",
    latestUpdates: "최신 업데이트",
    moreEvents: "이벤트 더 보기",
    sourceUnknown: "출처 미상",
    minutesAgo: "분 전",
    hoursAgo: "시간 전",
    justNow: "방금 전",
    chatTitle: "AI 시황 질문",
    chatScope: "현재 유가/뉴스/모델 기준",
    chatEmpty: "예측 결과나 시황에 대해 짧게 질문해보세요.",
    chatPlaceholder: "예: 지금 상승 예측의 핵심 근거가 뭐야?",
    chatAsk: "질문",
    chatLoading: "답변 생성 중",
    chatError: "답변을 불러오지 못했습니다.",
    trainRequired: "학습 필요",
    bandVolBased: "변동성 기반",
  },
  en: {
    appTitle: "Oil Price Forecast Dashboard",
    oilInstrument: "WTI Crude Oil",
    symbol: "Symbol",
    symbolPlaceholder: "CL=F fixed",
    interval: "Interval",
    horizon: "Forecast Length",
    backtest: "Backtest",
    live: "Live",
    llmCommentary: "AI Market Commentary",
    newsContext: "News & Scenario Context",
    bull: "Bull",
    base: "Base",
    bear: "Bear",
    updated: "Updated",
    data: "DATA",
    band: "BAND",
    conf: "CONF",
    regime: "REGIME",
    waiting: "Pending",
    liveMetric: "Live metric",
    backtestMetric: "Backtest metric",
    noContext: "No live news/event context records to show.",
    origin: "Origin",
    loading: "Loading",
    actual: "Actual",
    llm: "AI commentary",
    fallback: "fallback",
    cached: "cached",
    newsTimeline: "News events",
    newsCountUnit: "",
    bullishShort: "Bull",
    bearishShort: "Bear",
    mixedShort: "Mixed",
    neutralShort: "Neutral",
    loadingChart: "Loading chart data",
    refreshingChart: "Refreshing market data",
    loadingNews: "Refreshing live news",
    loadingCommentary: "Refreshing AI market commentary",
    commentaryGenerating: "Generating commentary.",
    selectOriginFirst: "Select an origin on the chart before switching to backtest.",
    latestUpdates: "Latest Updates",
    moreEvents: "Show more events",
    sourceUnknown: "Unknown source",
    minutesAgo: "min ago",
    hoursAgo: "h ago",
    justNow: "Just now",
    chatTitle: "Ask AI",
    chatScope: "Current oil/news/model",
    chatEmpty: "Ask a short question about the forecast or market context.",
    chatPlaceholder: "e.g. What supports the current upside view?",
    chatAsk: "Ask",
    chatLoading: "Generating answer",
    chatError: "Unable to load an answer.",
    trainRequired: "Train required",
    bandVolBased: "Vol based",
  },
};

function t(key) {
  return I18N[currentLanguage]?.[key] || I18N.ko[key] || key;
}

function currentOilSymbol() {
  const input = document.getElementById("symbol-input");
  if (input && input.value !== DEFAULT_SYMBOL) input.value = DEFAULT_SYMBOL;
  return DEFAULT_SYMBOL;
}

function setLanguage(language, refreshPanels = true) {
  const nextLanguage = language === "en" ? "en" : "ko";
  if (currentLanguage === nextLanguage && !refreshPanels) return;
  currentLanguage = nextLanguage;
  try {
    window.localStorage?.setItem("dashboard.language", currentLanguage);
  } catch {
    // Browser privacy modes can block localStorage; the in-memory language state is enough.
  }
  applyLanguage();
  if (!refreshPanels) return;
  lastContextKey = "";
  lastCommentaryKey = "";
  commentaryRequestVersion += 1;
  commentaryRequestInFlight = false;
  renderModelCommentaryLoading();
  const intervalInput = document.getElementById("interval-input");
  const horizonInput = document.getElementById("horizon-input");
  refreshDashboardPanels(
    currentOilSymbol(),
    intervalInput?.value || "1d",
    forecastModelsQuery(),
    horizonInput?.value || "",
    requestVersion,
    { forceContext: false, forceCommentary: true },
  );
}

function activeLoadingMessages() {
  const messages = [];
  if (loadingState.chart) messages.push(latestPayload ? t("refreshingChart") : t("loadingChart"));
  if (loadingState.context) messages.push(t("loadingNews"));
  if (loadingState.commentary) messages.push(t("loadingCommentary"));
  if (loadingState.backtest) messages.push(t("loading"));
  return messages;
}

function setLoadingState(kind, active) {
  loadingState = { ...loadingState, [kind]: Boolean(active) };
  const banner = document.getElementById("loading-banner");
  const message = document.getElementById("loading-message");
  const chartOverlay = document.getElementById("chart-loading-overlay");
  const chartMessage = document.getElementById("chart-loading-message");
  const messages = activeLoadingMessages();
  if (banner && message) {
    if (messages.length) {
      message.textContent = messages.join(" · ");
      banner.classList.remove("hidden");
    } else {
      banner.classList.add("hidden");
      message.textContent = "";
    }
  }
  if (chartOverlay && chartMessage) {
    const showChartOverlay = loadingState.chart && !latestPayload;
    chartMessage.textContent = t("loadingChart");
    chartOverlay.classList.toggle("hidden", !showChartOverlay);
  }
}

function applyLanguage() {
  document.documentElement.lang = currentLanguage;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.dataset.i18n;
    if (key) node.textContent = t(key);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    const key = node.dataset.i18nPlaceholder;
    if (key) node.setAttribute("placeholder", t(key));
  });
  const toggle = document.getElementById("language-toggle");
  if (toggle) {
    toggle.checked = currentLanguage === "en";
    toggle.setAttribute("aria-checked", currentLanguage === "en" ? "true" : "false");
  }
  if (latestPayload) {
    setMetrics(latestPayload.metrics || {}, latestPayload.updated_at, latestPayload.forecast_horizon, latestPayload.confidence_level);
    setDataStatusBadge(latestPayload.data_status);
    setChartDataWarning(latestPayload.data_status);
    setForecastBadges(latestPayload);
    setForecastNotices(latestPayload);
    renderMarketContextPanel(latestContextPayload);
    renderModelCommentary(latestCommentaryPayload);
  }
  setLoadingState("chart", loadingState.chart);
}

async function loadData(symbol, interval, models = "", horizon = "") {
  symbol = DEFAULT_SYMBOL;
  const ts = Date.now();
  const modelQuery = models ? `&models=${encodeURIComponent(models)}` : "";
  const horizonQuery = horizon ? `&horizon=${encodeURIComponent(horizon)}` : "";
  const forecastUrl = `/api/forecast?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}${horizonQuery}${modelQuery}&_ts=${ts}`;
  let response = await fetch(forecastUrl, { cache: "no-store" });
  if (response.ok) {
    const forecast = await response.json();
    return convertForecastToChartPayload(forecast);
  }
  response = await fetch(
    `/api/chart?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}${horizonQuery}${modelQuery}&_ts=${ts}`,
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

async function loadBacktestVisualization(symbol, interval, originTime, models = "", horizon = "") {
  symbol = DEFAULT_SYMBOL;
  const modelQuery = models ? `&models=${encodeURIComponent(models)}` : "";
  const horizonQuery = horizon ? `&horizon=${encodeURIComponent(horizon)}` : "";
  const response = await fetch(
    `/api/backtests/visualization?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}&origin_time=${encodeURIComponent(originTime)}${horizonQuery}${modelQuery}&_ts=${Date.now()}`,
    { cache: "no-store" },
  );
  if (!response.ok) {
    let message = "Failed to load backtest visualization";
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
    symbol_input: forecast.symbol || forecast.data_status?.symbol_resolved || forecast.data_status?.symbol_requested,
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
    band_explanation: forecast.band_explanation || {},
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

function localizeNotice(message) {
  const text = String(message || "");
  const enToKo = {
    "Quantile bands are residual-volatility adapters and are not validated coverage intervals yet.": "예측 밴드는 잔차 변동성 기반 보조 범위이며, 아직 검증된 신뢰구간이 아닙니다.",
    "Run rolling coverage calibration before labeling bands as validated confidence intervals.": "검증된 신뢰구간으로 표기하려면 rolling coverage calibration을 먼저 수행해야 합니다.",
    "Forecast bands are built from the selected model quantile/residual scale and recent realized volatility for this model/symbol/interval; measured coverage calibration is not available yet.": "예측 밴드는 선택 모델의 분위수/잔차 폭과 최근 실현 변동성을 함께 사용해 산출됩니다. 이 모델·종목·주기 조합의 실측 coverage 보정값은 아직 부족합니다.",
  };
  const koToEn = Object.fromEntries(Object.entries(enToKo).map(([en, ko]) => [ko, en]));
  return currentLanguage === "ko" ? enToKo[text] || text : koToEn[text] || text;
}

function bandExplanationText(payload) {
  const explanation = payload?.band_explanation || {};
  const status = String(explanation.status || payload?.calibration_status?.calibration_status || "volatility_estimated").toLowerCase();
  const primary = MODEL_LABELS[explanation.primary_model] || explanation.primary_model || payload?.primary_model || "model";
  const horizon = explanation.horizon || payload?.forecast_horizon || "-";
  const bandScale = Number(explanation.band_scale);
  const scaleText = Number.isFinite(bandScale) ? (currentLanguage === "ko" ? ` 밴드 스케일은 ${bandScale.toFixed(2)}입니다.` : ` Band scale is ${bandScale.toFixed(2)}.`) : "";
  if (currentLanguage === "ko") {
    if (status === "calibrated") {
      const n = explanation.n_origins ? `${explanation.n_origins}개 rolling 기준점` : "rolling backtest";
      return `예측 밴드는 ${primary} 경로에 ${n}의 conformal 보정을 적용해 산출됩니다.${scaleText}`;
    }
    return `예측 밴드는 ${primary}의 ${horizon}스텝 분위수/잔차 폭과 최근 실현 변동성을 결합해 산출됩니다. 실측 coverage 보정값이 부족한 조합도 P10-P90, P05-P95 범위를 일관되게 표시합니다.${scaleText}`;
  }
  if (status === "calibrated") {
    const n = explanation.n_origins ? `${explanation.n_origins} rolling origins` : "rolling backtest";
    return `Forecast bands apply conformal adjustment from ${n} to the ${primary} path.${scaleText}`;
  }
  return `Forecast bands combine the ${primary} ${horizon}-step quantile/residual scale with recent realized volatility, so P10-P90 and P05-P95 ranges are available for every supported symbol.${scaleText}`;
}

function localizeCalibrationStatus(calibrationStatus) {
  const status = String(calibrationStatus?.calibration_status || "volatility_estimated").toLowerCase();
  const nOrigins = Number(calibrationStatus?.n_origins || 0);
  const coverage80 = calibrationStatus?.coverage_80;
  if (currentLanguage === "ko") {
    if (status === "calibrated") {
      const coverage = coverage80 !== null && coverage80 !== undefined ? ` · P10-P90 적중률 ${(Number(coverage80) * 100).toFixed(1)}%` : "";
      return {
        label: "보정됨",
        title: `Rolling backtest ${nOrigins}개 기준점으로 conformal calibration을 적용했습니다.${coverage}`,
      };
    }
    return {
      label: t("bandVolBased"),
      title: "선택 모델의 분위수/잔차 폭과 최근 실현 변동성으로 예측 밴드를 산출합니다.",
    };
  }
  if (status === "calibrated") {
    const coverage = coverage80 !== null && coverage80 !== undefined ? ` · P10-P90 coverage ${(Number(coverage80) * 100).toFixed(1)}%` : "";
    return {
      label: "CALIBRATED",
      title: `Conformal calibration is applied from ${nOrigins} rolling backtest origins.${coverage}`,
    };
  }
  return {
    label: t("bandVolBased").toUpperCase(),
    title: "Bands are estimated from model residual/quantile scale and recent realized volatility.",
  };
}

function localizeRole(value) {
  const text = String(value || "-");
  if (currentLanguage !== "ko") return text;
  if (text === "context/event encoder only") return "컨텍스트/이벤트 인코더 전용";
  if (text === "deterministic_context_narrative") return "규칙 기반 시나리오";
  return text;
}

function localizeNewsSource(value) {
  const source = String(value || "unknown");
  if (currentLanguage !== "ko") return source.replaceAll("_", " ");
  if (source === "live_public_news") return "실시간 공개 뉴스";
  if (source === "live_public_news_cached") return "실시간 공개 뉴스 캐시";
  if (source === "live_public_news_unavailable") return "실시간 뉴스 실패";
  if (source === "offline_cache") return "오프라인 캐시";
  return source.replaceAll("_", " ");
}

function localizeScenarioText(value, kind = "") {
  const text = String(value || "-");
  if (currentLanguage !== "ko") return text;
  if (text.includes("median path currently leans")) {
    const direction = text.includes("upside") ? "상방" : text.includes("downside") ? "하방" : "중립";
    const regime = (text.match(/dominant regime is ([^.]+)/)?.[1] || "unknown").replaceAll("_", " ");
    return `선택 모델의 중앙 경로는 현재 ${direction}으로 기울어져 있으며, 우세 국면은 ${regime}입니다. 뉴스와 이벤트는 가격을 직접 찍는 용도가 아니라 예측 근거와 리스크를 설명하는 데 사용됩니다.`;
  }
  const known = {
    bull: "상방 시나리오는 추세와 뉴스 흐름이 우호적으로 이어질 때의 낙관 경로입니다.",
    base: "기준 시나리오는 현재 데이터와 뉴스 흐름을 종합했을 때 가장 중심에 놓이는 경로입니다.",
    bear: "하방 시나리오는 변동성이 커지거나 최근 뉴스/이벤트 압력이 불리하게 바뀔 때의 방어적 경로입니다.",
  };
  return known[kind] || text;
}

function setMetrics(metrics, updatedAt, forecastHorizon, confidenceLevel) {
  const mode = metrics.metric_mode || "live";
  const fallbackText = mode === "backtest" ? "-" : t("waiting");
  document.getElementById("mae-value").textContent = metrics.mae === null || metrics.mae === undefined ? fallbackText : formatMetric(metrics.mae);
  document.getElementById("rmse-value").textContent = metrics.rmse === null || metrics.rmse === undefined ? fallbackText : formatMetric(metrics.rmse);
  document.getElementById("mape-value").textContent = metrics.mape === null || metrics.mape === undefined ? fallbackText : formatMetric(metrics.mape, "%");
  document.getElementById("chart-updated-value").textContent =
    `${t("updated")} ${new Date(updatedAt).toLocaleString()}`;
  document.querySelectorAll(".metric-card h3").forEach((heading) => {
    heading.dataset.mode = mode;
    heading.title = mode === "backtest" ? t("backtestMetric") : t("liveMetric");
  });
}

function setDataStatusBadge(dataStatus) {
  const badge = document.getElementById("data-status-badge");
  if (!badge) return;
  const status = String(dataStatus?.status || "unknown").toUpperCase();
  badge.textContent = `${t("data")} ${status}`;
  badge.dataset.status = String(dataStatus?.status || "unknown").toLowerCase();
}

function setChartDataWarning(dataStatus) {
  const warning = document.getElementById("chart-data-warning");
  if (!warning) return;
  const status = String(dataStatus?.status || "").toLowerCase();
  const warnings = Array.isArray(dataStatus?.warnings) ? dataStatus.warnings : [];
  if (!["mock", "fallback", "stale", "error"].includes(status)) {
    warning.classList.add("hidden");
    warning.textContent = "";
    return;
  }
  const statusText =
    currentLanguage === "ko"
      ? {
          mock: "개발용 mock 데이터",
          fallback: "대체 데이터",
          stale: "오래된 데이터",
          error: "데이터 오류",
        }[status] || status
      : status.toUpperCase();
  const detail = warnings.length ? ` ${warnings.slice(0, 2).join(" ")}` : "";
  warning.textContent =
    currentLanguage === "ko"
      ? `현재 차트는 ${statusText}를 표시 중입니다. 실제 시장 데이터가 아닐 수 있습니다.${detail}`
      : `Chart is showing ${statusText}. It may not be live market data.${detail}`;
  warning.classList.remove("hidden");
}

function setForecastBadges(payload) {
  const confidence = document.getElementById("confidence-badge");
  const regime = document.getElementById("regime-badge");
  const calibration = document.getElementById("calibration-badge");
  if (confidence) {
    const values = (payload.predicted || []).slice(1);
    confidence.textContent = values.length ? `${t("conf")} ${Math.round((payload.regime?.confidence || 0.5) * 100)}%` : `${t("conf")} -`;
  }
  if (regime) {
    const probs = payload.regime || {};
    const entries = Object.entries(probs).filter(([key]) => key !== "confidence");
    const label = entries.length ? entries.sort((a, b) => Number(b[1]) - Number(a[1]))[0][0] : "unknown";
    regime.textContent = `${t("regime")} ${String(label).replaceAll("_", " ").toUpperCase()}`;
  }
  if (calibration) {
    const status = String(payload.calibration_status?.calibration_status || "volatility_estimated").toUpperCase();
    const calibrationText = localizeCalibrationStatus(payload.calibration_status);
    calibration.textContent = `${t("band")} ${calibrationText.label}`;
    calibration.dataset.status = status.toLowerCase() === "calibrated" ? "calibrated" : "estimated";
    calibration.title = bandExplanationText(payload) || calibrationText.title;
  }
}

function markerForContextPoint(point) {
  const bias = String(point.overall_bias || "neutral").toLowerCase();
  const impact = Number(point.impact_score || 0);
  const eventCount = Number(point.event_count || 0);
  const color =
    bias === "bullish"
      ? "#7ee787"
      : bias === "bearish"
        ? "#ff7b72"
        : bias === "mixed"
          ? "#f2cc60"
          : "#79c0ff";
  return {
    time: point.time,
    position: bias === "bearish" ? "aboveBar" : "belowBar",
    color,
    shape: bias === "bearish" ? "arrowDown" : bias === "bullish" ? "arrowUp" : "circle",
    text: `${bias.slice(0, 1).toUpperCase()} ${eventCount || Math.round(impact * 10)}`,
  };
}

function renderContextMarkers(contextPayload) {
  if (!candleSeriesRef || typeof candleSeriesRef.setMarkers !== "function") return;
  const markers = [];
  if (activeBacktestOriginMarker) {
    markers.push(activeBacktestOriginMarker);
  }
  candleSeriesRef.setMarkers(markers);
}

function directionLabelForBias(bias) {
  const normalized = String(bias || "neutral").toLowerCase();
  if (normalized === "bullish") return t("bullishShort");
  if (normalized === "bearish") return t("bearishShort");
  if (normalized === "mixed") return t("mixedShort");
  return t("neutralShort");
}

function formatRelativeTime(epochSeconds) {
  const deltaMs = Math.max(0, Date.now() - Number(epochSeconds || 0) * 1000);
  const minutes = Math.floor(deltaMs / 60_000);
  if (minutes < 1) return t("justNow");
  if (minutes < 60) return currentLanguage === "ko" ? `${minutes}${t("minutesAgo")}` : `${minutes} ${t("minutesAgo")}`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return currentLanguage === "ko" ? `${hours}${t("hoursAgo")}` : `${hours} ${t("hoursAgo")}`;
  return new Date(Number(epochSeconds) * 1000).toLocaleDateString();
}

function formatNewsSource(source) {
  const text = String(source || "").trim();
  if (!text) return t("sourceUnknown");
  const parts = text.split(":").map((part) => part.trim()).filter(Boolean);
  const last = parts.length ? parts[parts.length - 1] : text;
  const normalized = last.replaceAll("_", " ");
  if (normalized.toLowerCase() === "yahoo finance rss") return "Yahoo Finance";
  if (normalized.toLowerCase() === "google news rss") return "Google News";
  return normalized;
}

function eventFactorText(point) {
  const bias = directionLabelForBias(point?.overall_bias || "neutral");
  const impact = Number(point?.impact_score || 0);
  const uncertainty = Number(point?.uncertainty ?? 0);
  const count = Number(point?.event_count || 0);
  return currentLanguage === "ko"
    ? `입력 팩터: ${bias} · 중요도 ${impact.toFixed(2)} · 불확실성 ${uncertainty.toFixed(2)} · 이벤트 ${count}건`
    : `Input factors: ${bias} · impact ${impact.toFixed(2)} · uncertainty ${uncertainty.toFixed(2)} · events ${count}`;
}

function newsForContextPoint(contextPayload, point) {
  const pointDay = new Date(Number(point.time) * 1000).toISOString().slice(0, 10);
  const sameDay = (contextPayload?.news || []).filter((item) => {
    if (!item.time) return false;
    return new Date(Number(item.time) * 1000).toISOString().slice(0, 10) === pointDay;
  });
  return sameDay.length
    ? sameDay.slice(-5).reverse()
    : [
        {
          time: point.time,
          source: t("sourceUnknown"),
          headline: point.explanation || t("noContext"),
        },
      ];
}

function closeNewsPopover() {
  document.getElementById("news-detail-popover")?.classList.add("hidden");
  document.querySelectorAll(".chart-news-marker").forEach((button) => button.setAttribute("aria-expanded", "false"));
}

function showNewsPopover(point, newsItems, anchor) {
  const popover = document.getElementById("news-detail-popover");
  if (!popover) return;
  document.querySelectorAll(".chart-news-marker").forEach((button) => button.setAttribute("aria-expanded", "false"));
  anchor?.setAttribute("aria-expanded", "true");
  const head = document.createElement("div");
  head.className = "news-popover-head";
  const title = document.createElement("h4");
  title.className = "news-popover-title";
  const icon = document.createElement("span");
  icon.className = "news-popover-icon";
  icon.textContent = "N";
  const titleText = document.createElement("span");
  titleText.textContent = t("latestUpdates");
  title.append(icon, titleText);
  const close = document.createElement("button");
  close.type = "button";
  close.className = "news-popover-close";
  close.textContent = "×";
  close.addEventListener("click", closeNewsPopover);
  head.append(title, close);

  const list = document.createElement("div");
  list.className = "news-popover-list";
  newsItems.slice(0, 5).forEach((news) => {
    const row = document.createElement("article");
    row.className = "news-popover-item";
    const meta = document.createElement("div");
    meta.className = "news-popover-meta";
    meta.textContent = `${formatRelativeTime(news.time || point.time)} · ${formatNewsSource(news.source)}`;
    const headline = document.createElement("div");
    headline.className = "news-popover-headline";
    headline.textContent = news.headline || point.explanation || "-";
    const factors = document.createElement("div");
    factors.className = "news-popover-factors";
    factors.textContent = eventFactorText(point);
    row.append(meta, headline, factors);
    list.append(row);
  });

  const action = document.createElement("button");
  action.type = "button";
  action.className = "news-popover-action";
  action.textContent = t("moreEvents");
  action.addEventListener("click", () => {
    document.getElementById("market-context-panel")?.scrollIntoView({ block: "start", behavior: "smooth" });
  });

  popover.replaceChildren(head, list, action);
  popover.classList.remove("hidden");
}

function positionNewsMarkers() {
  const root = document.getElementById("chart-news-markers");
  if (!root || !chartRef || !latestPayload?.candles?.length) return;
  const width = root.clientWidth || 1;
  const timeScale = chartRef.timeScale();
  const candles = latestPayload.candles || [];
  const chartTimes = [
    ...candles.map((candle) => Number(candle.time)),
    ...(latestPayload.predicted || []).map((point) => Number(point.time)),
  ].filter((value) => Number.isFinite(value));
  const firstTime = Number(candles[0]?.time);
  const lastTime = Math.max(...chartTimes, Number(candles[candles.length - 1]?.time));
  root.querySelectorAll(".chart-news-marker").forEach((button, index) => {
    const eventTime = Number(button.dataset.time || 0);
    const nearestTime = chartTimes.reduce((best, value) => {
      if (best === null) return value;
      return Math.abs(value - eventTime) < Math.abs(best - eventTime) ? value : best;
    }, null);
    let x = null;
    if (typeof timeScale.timeToCoordinate === "function") {
      try {
        x = timeScale.timeToCoordinate(eventTime);
        if ((x === null || x === undefined || Number.isNaN(Number(x))) && nearestTime !== null) {
          x = timeScale.timeToCoordinate(nearestTime);
          button.dataset.anchorTime = String(nearestTime);
        } else {
          button.dataset.anchorTime = String(eventTime);
        }
      } catch {
        x = null;
      }
    }
    if (x === null || x === undefined || Number.isNaN(Number(x))) {
      const plottedTime = nearestTime === null ? eventTime : nearestTime;
      button.dataset.anchorTime = String(plottedTime);
      const ratio = lastTime > firstTime ? (Math.min(Math.max(plottedTime, firstTime), lastTime) - firstTime) / (lastTime - firstTime) : 0.98;
      x = 12 + ratio * Math.max(1, width - 24);
    }
    const stagger = index % 2 === 0 ? 0 : 6;
    button.style.left = `${Math.min(Math.max(Number(x), 14), width - 14)}px`;
    button.style.bottom = `${stagger}px`;
    const anchorTime = Number(button.dataset.anchorTime || eventTime);
    const anchorSuffix =
      anchorTime && Math.abs(anchorTime - eventTime) > 60
        ? currentLanguage === "ko"
          ? ` · 차트 기준 ${new Date(anchorTime * 1000).toLocaleDateString()}`
          : ` · chart anchor ${new Date(anchorTime * 1000).toLocaleDateString()}`
        : "";
    button.title = `${button.dataset.baseTitle || ""}${anchorSuffix}`;
  });
}

function renderChartNewsMarkers(contextPayload) {
  const root = document.getElementById("chart-news-markers");
  const legacyStrip = document.getElementById("news-timeline");
  if (legacyStrip) legacyStrip.replaceChildren();
  if (!root) return;
  const points = (contextPayload?.context_points || []).slice(-12);
  if (!points.length) {
    root.replaceChildren();
    closeNewsPopover();
    return;
  }
  const markers = points.map((point, idx) => {
    const bias = String(point.overall_bias || "neutral").toLowerCase();
    const impact = Math.max(0.05, Math.min(Number(point.impact_score || 0), 1));
    const count = Number(point.event_count || 0);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chart-news-marker";
    button.dataset.bias = bias;
    button.dataset.time = String(point.time || "");
    button.dataset.index = String(idx);
    button.textContent = count > 1 ? String(Math.min(count, 9)) : "N";
    button.setAttribute("aria-expanded", "false");
    const baseTitle =
      currentLanguage === "ko"
        ? `${directionLabelForBias(bias)} · 중요도 ${impact.toFixed(2)} · 뉴스 ${count}건`
        : `${directionLabelForBias(bias)} · impact ${impact.toFixed(2)} · ${count} news`;
    button.dataset.baseTitle = baseTitle;
    button.title = baseTitle;
    button.addEventListener("click", (event) => {
      showNewsPopover(point, newsForContextPoint(contextPayload, point), button);
    });
    return button;
  });
  root.replaceChildren(...markers);
  root.onclick = (event) => {
    const button = event.target instanceof Element ? event.target.closest(".chart-news-marker") : null;
    if (!button) return;
    event.stopPropagation();
    const idx = Number(button.dataset.index || -1);
    const point = points[idx];
    if (point) showNewsPopover(point, newsForContextPoint(contextPayload, point), button);
  };
  positionNewsMarkers();
}

function renderNewsTimeline(contextPayload) {
  renderChartNewsMarkers(contextPayload);
}

function renderMarketContextPanel(contextPayload) {
  latestContextPayload = contextPayload;
  const mode = document.getElementById("context-mode");
  const summary = document.getElementById("scenario-summary");
  const bull = document.getElementById("scenario-bull");
  const base = document.getElementById("scenario-base");
  const bear = document.getElementById("scenario-bear");
  const eventsRoot = document.getElementById("context-events");
  if (!eventsRoot) return;
  if (!contextPayload) {
    if (mode) mode.textContent = "-";
    if (summary) summary.textContent = t("noContext");
    if (bull) bull.textContent = "-";
    if (base) base.textContent = "-";
    if (bear) bear.textContent = "-";
    eventsRoot.replaceChildren(document.createTextNode(t("noContext")));
    renderNewsTimeline(null);
    return;
  }
  const scenario = contextPayload?.scenario_commentary || {};
  if (mode) {
    const llmMode = contextPayload?.llm_context_summary?.role || scenario.mode || "context";
    const source = localizeNewsSource(contextPayload?.news_source);
    mode.textContent = `${source} · ${localizeRole(llmMode)}`;
  }
  if (summary) summary.textContent = localizeScenarioText(scenario.summary || "-");
  if (bull) bull.textContent = localizeScenarioText(scenario.bull || "-", "bull");
  if (base) base.textContent = localizeScenarioText(scenario.base || "-", "base");
  if (bear) bear.textContent = localizeScenarioText(scenario.bear || "-", "bear");

  const contextPoints = contextPayload?.context_points || [];
  const newsByDay = new Map();
  (contextPayload?.news || []).forEach((item) => {
    const day = new Date(Number(item.time) * 1000).toISOString().slice(0, 10);
    const rows = newsByDay.get(day) || [];
    rows.push(item);
    newsByDay.set(day, rows);
  });
  const rows = contextPoints.slice(-12).reverse().map((point) => {
    const day = new Date(Number(point.time) * 1000).toISOString().slice(0, 10);
    const item = document.createElement("article");
    item.className = "context-event";
    item.dataset.bias = String(point.overall_bias || "neutral").toLowerCase();
    const matchedNews = (newsByDay.get(day) || []).slice(-5).reverse();
    const headlines = matchedNews.length
      ? matchedNews
          .map((news) => {
            const label = document.createElement("span");
            label.textContent = news.headline || formatNewsSource(news.source) || "News";
            return label;
          })
      : [document.createTextNode(point.explanation || "No matching headline on this day.")];
    const head = document.createElement("div");
    head.className = "context-event-head";
    const title = document.createElement("strong");
    title.textContent = `${day} · ${String(point.overall_bias || "neutral").toUpperCase()}`;
    const score = document.createElement("span");
    score.textContent =
      currentLanguage === "ko"
        ? `중요도 ${Number(point.impact_score || 0).toFixed(2)} · 뉴스 ${point.event_count || 0}`
        : `impact ${Number(point.impact_score || 0).toFixed(2)} · news ${point.event_count || 0}`;
    head.append(title, score);
    const body = document.createElement("p");
    headlines.forEach((node, idx) => {
      if (idx) body.append(document.createTextNode(" / "));
      body.append(node);
    });
    const factors = document.createElement("p");
    factors.className = "context-event-factors";
    factors.textContent = eventFactorText(point);
    item.append(head, body, factors);
    return item;
  });
  const warnings = (contextPayload?.news_warnings || []).filter(Boolean);
  const emptyText =
    warnings.length && currentLanguage === "ko"
      ? `실시간 뉴스 수집 실패: ${warnings.join(" ")}`
      : warnings.length
        ? `Live news fetch failed: ${warnings.join(" ")}`
        : t("noContext");
  eventsRoot.replaceChildren(...(rows.length ? rows : [document.createTextNode(emptyText)]));
  renderNewsTimeline(contextPayload);
}

async function loadMarketContext(symbol, interval, models = "", horizon = "") {
  const modelQuery = models ? `&models=${encodeURIComponent(models)}` : "";
  const horizonQuery = horizon ? `&horizon=${encodeURIComponent(horizon)}` : "";
  const response = await fetch(
    `/api/market-context?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}${horizonQuery}${modelQuery}&live=1&_ts=${Date.now()}`,
    { cache: "no-store" },
  );
  if (!response.ok) throw new Error("market context unavailable");
  return response.json();
}

async function loadModelCommentary(symbol, interval, models = "", originTime = null, horizon = "", language = currentLanguage) {
  const modelQuery = models ? `&models=${encodeURIComponent(models)}` : "";
  const originQuery = originTime ? `&origin_time=${encodeURIComponent(originTime)}` : "";
  const horizonQuery = horizon ? `&horizon=${encodeURIComponent(horizon)}` : "";
  const languageQuery = `&language=${encodeURIComponent(language)}`;
  const response = await fetch(
    `/api/model-commentary?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}${horizonQuery}${modelQuery}${originQuery}${languageQuery}&_ts=${Date.now()}`,
    { cache: "no-store" },
  );
  if (!response.ok) throw new Error("model commentary unavailable");
  return response.json();
}

async function loadAssistantChat(question) {
  const symbol = currentOilSymbol();
  const interval = document.getElementById("interval-input")?.value || "1d";
  const horizon = document.getElementById("horizon-input")?.value || "";
  const response = await fetch("/api/assistant-chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      symbol,
      interval,
      horizon: horizon ? Number(horizon) : null,
      models: forecastModelsQuery(),
      language: currentLanguage,
    }),
  });
  if (!response.ok) throw new Error("assistant chat unavailable");
  return response.json();
}

function appendChatMessage(role, text, warnings = []) {
  const log = document.getElementById("llm-chat-log");
  if (!log) return;
  if (log.querySelector("[data-i18n='chatEmpty']")) log.replaceChildren();
  const item = document.createElement("article");
  item.className = `llm-chat-message ${role}`;
  const label = document.createElement("strong");
  label.textContent = role === "user" ? "Q" : "LLM";
  const body = document.createElement("p");
  body.textContent = text;
  item.append(label, body);
  warnings.filter(Boolean).slice(0, 2).forEach((warning) => {
    const note = document.createElement("span");
    note.className = "llm-chat-warning";
    note.textContent = warning;
    item.append(note);
  });
  log.append(item);
  log.scrollTop = log.scrollHeight;
}

function replaceResolvedSymbolText(text, commentary = null) {
  const source = String(text || "");
  const requested =
    commentary?.display_symbol || latestPayload?.symbol_input || latestPayload?.asset_metadata?.symbol || "";
  const resolved =
    commentary?.provider_symbol || commentary?.symbol || latestPayload?.symbol_resolved || latestPayload?.asset_metadata?.provider_symbol || "";
  if (!requested || !resolved || requested === resolved) return source;
  const protectedTokens = [];
  const protect = (value) => {
    if (!source.includes(value)) return value;
    const token = `__PROVIDER_SYMBOL_${protectedTokens.length}__`;
    protectedTokens.push([token, value]);
    return token;
  };
  let output = source
    .replaceAll(`(${resolved} provider 기준)`, protect(`(${resolved} provider 기준)`))
    .replaceAll(`(provider symbol ${resolved})`, protect(`(provider symbol ${resolved})`));
  output = output.replaceAll(resolved, requested);
  protectedTokens.forEach(([token, value]) => {
    output = output.replaceAll(token, value);
  });
  return output;
}

function bindAssistantChat() {
  const form = document.getElementById("llm-chat-form");
  const input = document.getElementById("llm-chat-input");
  const button = document.getElementById("llm-chat-submit");
  if (!form || !input || !button) return;
  const submitQuestion = async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    appendChatMessage("user", question);
    button.disabled = true;
    button.textContent = t("chatLoading");
    try {
      const answer = await loadAssistantChat(question);
      appendChatMessage("assistant", answer.answer || "-", answer.warnings || []);
    } catch (error) {
      appendChatMessage("assistant", t("chatError"), [String(error?.message || error)]);
    } finally {
      button.disabled = false;
      button.textContent = t("chatAsk");
    }
  };
  form.addEventListener("submit", submitQuestion);
  button.addEventListener("click", (event) => {
    if (button.disabled) return;
    submitQuestion(event);
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") submitQuestion(event);
  });
}

function renderModelCommentaryLoading() {
  latestCommentaryPayload = null;
  const status = document.getElementById("commentary-status");
  const summary = document.getElementById("commentary-summary");
  const risks = document.getElementById("commentary-risks");
  if (status) status.textContent = t("loadingCommentary");
  if (summary) summary.textContent = t("commentaryGenerating");
  if (risks) risks.replaceChildren();
}

function renderModelCommentary(commentary) {
  latestCommentaryPayload = commentary;
  const status = document.getElementById("commentary-status");
  const summary = document.getElementById("commentary-summary");
  const risks = document.getElementById("commentary-risks");
  if (!summary || !risks) return;
  if (!commentary) {
    if (status) status.textContent = "-";
    summary.textContent = "-";
    risks.replaceChildren();
    return;
  }
  if (status) {
    const mode = commentary.llm_used ? "LLM" : "fallback";
    const modeLabel = commentary.llm_used ? t("llm") : t("fallback");
    status.textContent = commentary.cached
      ? (currentLanguage === "ko" ? "최근 해설" : "recent")
      : modeLabel;
  }
  summary.textContent = replaceResolvedSymbolText(commentary.summary || "-", commentary);
  const notes = [
    commentary.model_interpretation,
    ...(commentary.risk_notes || []),
  ].filter(Boolean);
  risks.replaceChildren(
    ...(notes.length ? notes : ["No additional notes."]).map((note) => {
      const li = document.createElement("li");
      li.textContent = replaceResolvedSymbolText(note, commentary);
      return li;
    }),
  );
}

function refreshModelCommentary(symbol, interval, models = "", originTime = null, reqId = requestVersion, horizon = "") {
  const commentaryReqId = ++commentaryRequestVersion;
  const languageAtRequest = currentLanguage;
  commentaryRequestInFlight = true;
  setLoadingState("commentary", true);
  renderModelCommentaryLoading();
  loadModelCommentary(symbol, interval, models, originTime, horizon, languageAtRequest)
    .then((commentary) => {
      if (reqId !== requestVersion || commentaryReqId !== commentaryRequestVersion || currentLanguage !== languageAtRequest) return;
      renderModelCommentary(commentary);
    })
    .catch(() => {
      if (reqId !== requestVersion || commentaryReqId !== commentaryRequestVersion || currentLanguage !== languageAtRequest) return;
      renderModelCommentary({
        summary: currentLanguage === "ko" ? "모델 해설을 불러오지 못했습니다." : "Model commentary unavailable.",
        risk_notes: [],
        warnings: [],
        llm_used: false,
      });
    })
    .finally(() => {
      if (commentaryReqId !== commentaryRequestVersion) return;
      commentaryRequestInFlight = false;
      setLoadingState("commentary", false);
    });
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
  const values = (messages || []).filter(Boolean).map(localizeNotice);
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
  const messages = [];
  (payload.info_messages || [])
    .filter((message) => !String(message || "").includes("Quantile bands are residual-volatility adapters"))
    .filter((message) => !String(message || "").includes("Forecast bands are built from"))
    .forEach((message) => messages.push(message));
  setInfoMessages(messages);
}

async function loadModelCatalog(interval = "", horizon = "") {
  void interval;
  void horizon;
  modelCatalog = new Map();
}

function forecastModelsQuery() {
  return "";
}

function visibleForecastModels(models) {
  void models;
  return [];
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

function candleAtOrBefore(time) {
  const ts = toUnixTime(time);
  if (ts === null || !latestPayload) return null;
  const candles = latestPayload.candles || [];
  if (candles.length && ts > Number(candles[candles.length - 1].time)) return null;
  let match = null;
  for (const candle of candles) {
    if (Number(candle.time) <= ts) {
      match = candle;
    } else {
      break;
    }
  }
  return match;
}

function setBacktestStatus(message = "", severity = "info") {
  const status = document.getElementById("backtest-status");
  if (!status) return;
  status.textContent = message;
  status.dataset.severity = severity;
}

function updateBacktestControls() {
  const originValue = document.getElementById("backtest-origin-value");
  const modeToggle = document.getElementById("backtest-mode-toggle");
  if (originValue) {
    originValue.textContent = selectedBacktestTime ? toDisplayTime(selectedBacktestTime) : "-";
  }
  if (modeToggle) {
    modeToggle.checked = !activeBacktestPayload;
    modeToggle.setAttribute("aria-checked", modeToggle.checked ? "true" : "false");
    const shell = modeToggle.closest(".chart-mode-toggle");
    shell?.setAttribute("data-disabled", !selectedBacktestTime && !activeBacktestPayload ? "true" : "false");
  }
}

function selectBacktestOrigin(time) {
  const candle = candleAtOrBefore(time);
  if (!candle) return;
  selectedBacktestTime = candle.time;
  setBacktestStatus("");
  updateBacktestControls();
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
    { text: `${currentLanguage === "ko" ? "예측" : "PRED"} ${formatPrice(pred)}`, className: "pred" },
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

function createFallbackChart(container) {
  container.replaceChildren();
  const canvas = document.createElement("canvas");
  canvas.className = "fallback-chart-canvas";
  container.appendChild(canvas);

  const ctx = canvas.getContext("2d");
  const seriesRefs = [];
  let visibleRange = null;

  const resizeCanvas = () => {
    const rect = container.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * ratio));
    canvas.height = Math.max(1, Math.floor(rect.height * ratio));
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  };

  const numeric = (value) => Number.isFinite(Number(value)) ? Number(value) : null;
  const colorOf = (series, fallback) => series.options?.color || series.options?.lineColor || fallback;

  const draw = () => {
    resizeCanvas();
    const width = container.clientWidth || 1;
    const height = container.clientHeight || 460;
    const pad = { left: 12, right: 58, top: 16, bottom: 54 };
    const plotWidth = Math.max(1, width - pad.left - pad.right);
    const plotHeight = Math.max(1, height - pad.top - pad.bottom);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#151b23";
    ctx.fillRect(0, 0, width, height);

    const timeSet = new Set();
    const prices = [];
    seriesRefs.forEach((series) => {
      (series.data || []).forEach((point) => {
        if (point?.time === undefined) return;
        timeSet.add(String(point.time));
        if (series.type === "candlestick") {
          ["open", "high", "low", "close"].forEach((key) => {
            const value = numeric(point[key]);
            if (value !== null) prices.push(value);
          });
        } else {
          const value = numeric(point.value);
          if (value !== null) prices.push(value);
        }
      });
    });
    const times = Array.from(timeSet).map(Number).filter(Number.isFinite).sort((a, b) => a - b);
    if (!times.length || !prices.length) {
      ctx.fillStyle = "#8b949e";
      ctx.font = "600 13px system-ui, -apple-system, Segoe UI, sans-serif";
      ctx.fillText(currentLanguage === "ko" ? "표시할 차트 데이터가 없습니다." : "No chart data to display.", 18, 32);
      return;
    }

    const fromIndex = visibleRange ? Math.max(0, Math.floor(visibleRange.from)) : 0;
    const toIndex = visibleRange ? Math.min(times.length - 1, Math.ceil(visibleRange.to)) : times.length - 1;
    const visibleTimes = times.slice(fromIndex, toIndex + 1);
    const visibleTimeSet = new Set(visibleTimes.map(String));
    const visiblePrices = [];
    seriesRefs.forEach((series) => {
      (series.data || []).forEach((point) => {
        if (!visibleTimeSet.has(String(point?.time))) return;
        if (series.type === "candlestick") {
          ["open", "high", "low", "close"].forEach((key) => {
            const value = numeric(point[key]);
            if (value !== null) visiblePrices.push(value);
          });
        } else {
          const value = numeric(point.value);
          if (value !== null) visiblePrices.push(value);
        }
      });
    });
    const minPrice = Math.min(...visiblePrices);
    const maxPrice = Math.max(...visiblePrices);
    const pricePad = Math.max((maxPrice - minPrice) * 0.08, 0.5);
    const yMin = minPrice - pricePad;
    const yMax = maxPrice + pricePad;
    const indexOf = new Map(times.map((time, index) => [String(time), index]));
    const xForIndex = (index) => {
      if (toIndex <= fromIndex) return pad.left + plotWidth;
      return pad.left + ((index - fromIndex) / (toIndex - fromIndex)) * plotWidth;
    };
    const yForPrice = (price) => pad.top + ((yMax - price) / (yMax - yMin)) * plotHeight;

    ctx.strokeStyle = "#21262d";
    ctx.lineWidth = 1;
    ctx.font = "11px system-ui, -apple-system, Segoe UI, sans-serif";
    ctx.fillStyle = "#8b949e";
    for (let i = 0; i <= 5; i += 1) {
      const y = pad.top + (plotHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
      ctx.stroke();
      const price = yMax - ((yMax - yMin) * i) / 5;
      ctx.fillText(price.toFixed(2), width - pad.right + 8, y + 4);
    }
    for (let i = 0; i <= 5; i += 1) {
      const index = Math.round(fromIndex + ((toIndex - fromIndex) * i) / 5);
      const x = xForIndex(index);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, height - pad.bottom);
      ctx.stroke();
    }

    seriesRefs
      .filter((series) => series.type === "area")
      .forEach((series) => {
        const points = (series.data || [])
          .map((point) => ({ index: indexOf.get(String(point.time)), value: numeric(point.value) }))
          .filter((point) => point.index !== undefined && point.value !== null && point.index >= fromIndex && point.index <= toIndex);
        if (points.length < 2) return;
        ctx.beginPath();
        points.forEach((point, idx) => {
          const x = xForIndex(point.index);
          const y = yForPrice(point.value);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = colorOf(series, "rgba(88, 166, 255, 0.32)");
        ctx.lineWidth = series.options?.lineWidth || 1;
        ctx.stroke();
      });

    seriesRefs
      .filter((series) => series.type === "candlestick")
      .forEach((series) => {
        const candleWidth = Math.max(3, Math.min(9, (plotWidth / Math.max(visibleTimes.length, 1)) * 0.62));
        (series.data || []).forEach((point) => {
          const index = indexOf.get(String(point.time));
          if (index === undefined || index < fromIndex || index > toIndex) return;
          const open = numeric(point.open);
          const high = numeric(point.high);
          const low = numeric(point.low);
          const close = numeric(point.close);
          if ([open, high, low, close].some((value) => value === null)) return;
          const up = close >= open;
          const color = up ? series.options?.upColor || "#2dd4bf" : series.options?.downColor || "#ff7b72";
          const x = xForIndex(index);
          const yOpen = yForPrice(open);
          const yClose = yForPrice(close);
          ctx.strokeStyle = color;
          ctx.fillStyle = color;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x, yForPrice(high));
          ctx.lineTo(x, yForPrice(low));
          ctx.stroke();
          const bodyTop = Math.min(yOpen, yClose);
          const bodyHeight = Math.max(2, Math.abs(yClose - yOpen));
          ctx.fillRect(x - candleWidth / 2, bodyTop, candleWidth, bodyHeight);
        });
      });

    seriesRefs
      .filter((series) => series.type === "line")
      .forEach((series) => {
        const points = (series.data || [])
          .map((point) => ({ index: indexOf.get(String(point.time)), value: numeric(point.value) }))
          .filter((point) => point.index !== undefined && point.value !== null && point.index >= fromIndex && point.index <= toIndex);
        if (points.length < 2) return;
        ctx.beginPath();
        points.forEach((point, idx) => {
          const x = xForIndex(point.index);
          const y = yForPrice(point.value);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = colorOf(series, "#58a6ff");
        ctx.lineWidth = series.options?.lineWidth || 1.5;
        ctx.stroke();
      });
  };

  const makeSeries = (type, options = {}) => {
    const series = {
      type,
      options: { ...options },
      data: [],
      markers: [],
      setData(nextData) {
        this.data = Array.isArray(nextData) ? nextData : [];
        draw();
      },
      applyOptions(nextOptions) {
        this.options = { ...this.options, ...(nextOptions || {}) };
        draw();
      },
      setMarkers(nextMarkers) {
        this.markers = Array.isArray(nextMarkers) ? nextMarkers : [];
      },
    };
    seriesRefs.push(series);
    return series;
  };

  return {
    addLineSeries: (options) => makeSeries("line", options),
    addCandlestickSeries: (options) => makeSeries("candlestick", options),
    addAreaSeries: (options) => makeSeries("area", options),
    removeSeries(series) {
      const index = seriesRefs.indexOf(series);
      if (index >= 0) seriesRefs.splice(index, 1);
      draw();
    },
    applyOptions(options) {
      if (options?.width || options?.height) draw();
    },
    timeScale() {
      return {
        fitContent() {
          visibleRange = null;
          draw();
        },
        setVisibleLogicalRange(range) {
          visibleRange = range;
          draw();
        },
        getVisibleLogicalRange() {
          return visibleRange;
        },
      };
    },
    subscribeCrosshairMove() {},
    subscribeClick(callback) {
      canvas.addEventListener("click", () => {
        const candles = latestPayload?.candles || [];
        const last = candles[candles.length - 1];
        if (last) callback({ time: last.time });
      });
    },
  };
}

function ensureChart() {
  if (chartRef) return;
  const container = document.getElementById("oilChart");
  const hasLightweightCharts = typeof LightweightCharts !== "undefined";
  const lineStyle = hasLightweightCharts && LightweightCharts.LineStyle ? LightweightCharts.LineStyle : {};
  if (hasLightweightCharts) {
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
      height: 460,
    });
  } else {
    chartRef = createFallbackChart(container);
    setStatus("차트 라이브러리를 불러오지 못해 로컬 캔버스 차트로 표시합니다.", "warning");
  }

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
    priceLineVisible: false,
    lastValueVisible: false,
  });
  // Mask lower half so only [lower, upper] band stays visible.
  predBandMaskRef = createAreaSeries({
    lineColor: "rgba(21,27,35,0.0)",
    topColor: "rgba(21,27,35,1.0)",
    bottomColor: "rgba(21,27,35,1.0)",
    lineWidth: 1,
    priceLineVisible: false,
    lastValueVisible: false,
  });
  predTailFillRef = createAreaSeries({
    title: "P05-P95 Tail Band",
    lineColor: "rgba(188, 140, 255, 0.0)",
    topColor: "rgba(188, 140, 255, 0.12)",
    bottomColor: "rgba(188, 140, 255, 0.02)",
    lineWidth: 1,
    priceLineVisible: false,
    lastValueVisible: false,
  });
  predTailMaskRef = createAreaSeries({
    lineColor: "rgba(21,27,35,0.0)",
    topColor: "rgba(21,27,35,0.88)",
    bottomColor: "rgba(21,27,35,0.88)",
    lineWidth: 1,
    priceLineVisible: false,
    lastValueVisible: false,
  });

  candleSeriesRef = createCandlestickSeries({
    title: "실제 OHLC",
    upColor: "#2dd4bf",
    downColor: "#ff7b72",
    borderUpColor: "#2dd4bf",
    borderDownColor: "#ff7b72",
    wickUpColor: "#2dd4bf",
    wickDownColor: "#ff7b72",
  });
  backtestActualSeriesRef = createCandlestickSeries({
    title: "백테스트 실제",
    upColor: "rgba(126, 231, 135, 0.28)",
    downColor: "rgba(255, 123, 114, 0.28)",
    borderUpColor: "rgba(126, 231, 135, 0.52)",
    borderDownColor: "rgba(255, 123, 114, 0.52)",
    wickUpColor: "rgba(126, 231, 135, 0.62)",
    wickDownColor: "rgba(255, 123, 114, 0.62)",
    priceLineVisible: false,
    lastValueVisible: false,
  });
  predSeriesRef = createLineSeries({
    title: "Predicted",
    color: "#58a6ff",
    lineWidth: 2.2,
    lineStyle: lineStyle.Dashed !== undefined ? lineStyle.Dashed : 2,
  });
  predUpperSeriesRef = createLineSeries({
    title: "Pred Upper",
    color: "rgba(88, 166, 255, 0.56)",
    lineWidth: 1.2,
    priceLineVisible: false,
    lastValueVisible: false,
  });
  predLowerSeriesRef = createLineSeries({
    title: "Pred Lower",
    color: "rgba(88, 166, 255, 0.56)",
    lineWidth: 1.2,
    priceLineVisible: false,
    lastValueVisible: false,
  });

  if (!isResizeBound) {
    window.addEventListener("resize", () => {
      if (chartRef) {
        chartRef.applyOptions({ width: container.clientWidth });
        positionNewsMarkers();
      }
    });
    isResizeBound = true;
  }
  const timeScale = chartRef.timeScale();
  if (timeScale && typeof timeScale.subscribeVisibleLogicalRangeChange === "function") {
    timeScale.subscribeVisibleLogicalRangeChange(positionNewsMarkers);
  }

  if (!isCrosshairBound && typeof chartRef.subscribeCrosshairMove === "function") {
    chartRef.subscribeCrosshairMove(updateLegendOnCrosshair);
    isCrosshairBound = true;
  }
  if (!isClickBound && typeof chartRef.subscribeClick === "function") {
    chartRef.subscribeClick((param) => {
      if (param?.time) {
        selectBacktestOrigin(param.time);
      }
    });
    isClickBound = true;
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
  void models;
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
  const lineStyle =
    typeof LightweightCharts !== "undefined" && LightweightCharts.LineStyle
      ? LightweightCharts.LineStyle
      : {};
  const overlays = (models || []).filter((model) => model.id !== latestPayload?.primary_model);
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
        lineStyle: lineStyle.Solid !== undefined ? lineStyle.Solid : 0,
        priceLineVisible: false,
        lastValueVisible: false,
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

function renderChart(payload, resetView = false, options = {}) {
  ensureChart();
  const isBacktest = options.mode === "backtest";
  if (!isBacktest) {
    activeBacktestPayload = null;
    activeBacktestOriginMarker = null;
    if (backtestActualSeriesRef) {
      backtestActualSeriesRef.setData([]);
    }
    updateBacktestControls();
  }
  latestPayload = payload;
  setChartDataWarning(payload.data_status);
  const visibleModels = visibleForecastModels(payload.forecast_models || []);
  predictedByTime = new Map((payload.predicted || []).map((p) => [String(p.time), p.value]));
  rebuildForecastLookup(visibleModels);
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
  if (backtestActualSeriesRef) {
    backtestActualSeriesRef.setData(isBacktest ? payload.actual_future_candles || [] : []);
  }
  predSeriesRef.setData(payload.predicted || []);
  predUpperSeriesRef.setData(upper);
  predLowerSeriesRef.setData(lower);
  renderForecastModelSeries(visibleModels);
  renderContextMarkers(latestContextPayload);
  renderNewsTimeline(latestContextPayload);
  refreshLegendDefault();

  if (resetView) {
    setInitialForecastView(payload);
    positionNewsMarkers();
    return;
  }

  if (savedRange && typeof timeScale.setVisibleLogicalRange === "function") {
    timeScale.setVisibleLogicalRange(savedRange);
    positionNewsMarkers();
    return;
  }
  timeScale.fitContent();
  positionNewsMarkers();
}

function renderBacktestChart(payload, savedRange = null) {
  activeBacktestPayload = payload;
  activeBacktestOriginMarker = {
    time: payload.origin_time,
    position: "aboveBar",
    color: "#f2cc60",
    shape: "circle",
    text: t("origin"),
  };
  renderChart(payload, false, { mode: "backtest" });
  if (savedRange && typeof chartRef.timeScale().setVisibleLogicalRange === "function") {
    chartRef.timeScale().setVisibleLogicalRange(savedRange);
  }
  renderMarketContextPanel(latestContextPayload);
  setBacktestStatus(`${t("actual")} ${payload.backtest?.actual_future_rows || 0}/${payload.backtest?.horizon || 0}`, "active");
  updateBacktestControls();
}

async function runSelectedBacktest() {
  if (!selectedBacktestTime) return;
  const intervalInput = document.getElementById("interval-input");
  const horizonInput = document.getElementById("horizon-input");
  const symbol = currentOilSymbol();
  const interval = intervalInput?.value || "1d";
  const horizon = horizonInput?.value || "";
  const selectedModels = forecastModelsQuery();
  const savedRange =
    chartRef && typeof chartRef.timeScale().getVisibleLogicalRange === "function"
      ? chartRef.timeScale().getVisibleLogicalRange()
      : null;
  setBacktestStatus(t("loading"), "loading");
  setLoadingState("backtest", true);
  try {
    const payload = await loadBacktestVisualization(symbol, interval, selectedBacktestTime, selectedModels, horizon);
    renderBacktestChart(payload, savedRange);
    refreshModelCommentary(symbol, interval, selectedModels, selectedBacktestTime, requestVersion, horizon);
    setMetrics(payload.metrics, payload.updated_at, payload.forecast_horizon, payload.confidence_level);
    setDataStatusBadge(payload.data_status);
    setForecastBadges(payload);
    setForecastNotices(payload);
  } catch (error) {
    console.error(error);
    setBacktestStatus(error?.message || "Failed", "error");
  } finally {
    setLoadingState("backtest", false);
    updateBacktestControls();
  }
}

function exitBacktestMode() {
  activeBacktestPayload = null;
  activeBacktestOriginMarker = null;
  setBacktestStatus("");
  if (latestPayload) {
    const symbol = currentOilSymbol();
    const interval = document.getElementById("interval-input").value || "1d";
    activeDataKey = null;
    initDashboard(symbol, interval, { force: true });
  }
  updateBacktestControls();
}

function refreshDashboardPanels(symbol, interval, selectedModels, selectedHorizon, reqId = requestVersion, options = {}) {
  const panelKey = `${symbol}|${interval}|${selectedModels}|${selectedHorizon}`;
  const now = Date.now();
  const shouldRefreshContext =
    options.forceContext || panelKey !== lastContextKey || now - lastContextLoadMs > CONTEXT_REFRESH_MS;
  if (shouldRefreshContext && !contextRequestInFlight) {
    lastContextKey = panelKey;
    lastContextLoadMs = now;
    contextRequestInFlight = true;
    setLoadingState("context", true);
    const mode = document.getElementById("context-mode");
    if (mode) mode.textContent = t("loadingNews");
    loadMarketContext(symbol, interval, selectedModels, selectedHorizon)
      .then((contextPayload) => {
        if (reqId !== requestVersion) return;
        renderMarketContextPanel(contextPayload);
        renderContextMarkers(contextPayload);
      })
      .catch(() => {
        if (latestContextPayload) {
          renderMarketContextPanel(latestContextPayload);
          renderContextMarkers(latestContextPayload);
          return;
        }
        renderMarketContextPanel(null);
        renderContextMarkers(null);
      })
      .finally(() => {
        contextRequestInFlight = false;
        setLoadingState("context", false);
      });
  } else {
    renderMarketContextPanel(latestContextPayload);
    renderContextMarkers(latestContextPayload);
  }

  const commentaryKey = `${panelKey}|${currentLanguage}`;
  const shouldRefreshCommentary =
    options.forceCommentary || commentaryKey !== lastCommentaryKey || now - lastCommentaryLoadMs > COMMENTARY_REFRESH_MS;
  if (shouldRefreshCommentary) {
    lastCommentaryKey = commentaryKey;
    lastCommentaryLoadMs = now;
    const commentaryReqId = ++commentaryRequestVersion;
    const languageAtRequest = currentLanguage;
    commentaryRequestInFlight = true;
    setLoadingState("commentary", true);
    renderModelCommentaryLoading();
    loadModelCommentary(symbol, interval, selectedModels, null, selectedHorizon, languageAtRequest)
      .then((commentary) => {
        if (reqId !== requestVersion || commentaryReqId !== commentaryRequestVersion || currentLanguage !== languageAtRequest) return;
        renderModelCommentary(commentary);
      })
      .catch(() => {
        if (reqId !== requestVersion || commentaryReqId !== commentaryRequestVersion || currentLanguage !== languageAtRequest) return;
        if (latestCommentaryPayload) {
          renderModelCommentary(latestCommentaryPayload);
          return;
        }
        renderModelCommentary({
          summary: currentLanguage === "ko" ? "모델 해설을 불러오지 못했습니다." : "Model commentary unavailable.",
          risk_notes: [],
          warnings: [],
          llm_used: false,
        });
      })
      .finally(() => {
        if (commentaryReqId !== commentaryRequestVersion) return;
        commentaryRequestInFlight = false;
        setLoadingState("commentary", false);
      });
  } else {
    renderModelCommentary(latestCommentaryPayload);
  }
}

async function initDashboard(symbol, interval, options = {}) {
  if (chartRequestInFlight && !options.force) return;
  chartRequestInFlight = true;
  setLoadingState("chart", true);
  const reqId = ++requestVersion;
  const horizonInput = document.getElementById("horizon-input");
  const selectedHorizon = horizonInput ? horizonInput.value || "" : "";
  await loadModelCatalog(interval, selectedHorizon);
  if (reqId !== requestVersion) {
    chartRequestInFlight = false;
    setLoadingState("chart", false);
    return;
  }
  const selectedModels = forecastModelsQuery();
  try {
    const payload = await loadData(symbol, interval, selectedModels, selectedHorizon);
    if (reqId !== requestVersion) return;
    const canonicalSymbol = DEFAULT_SYMBOL;
    if (payload.symbol_resolved) {
      const symbolInput = document.getElementById("symbol-input");
      if (symbolInput && symbolInput.value !== DEFAULT_SYMBOL) {
        symbolInput.value = DEFAULT_SYMBOL;
      }
    }
    setMetrics(
      payload.metrics,
      payload.updated_at,
      payload.forecast_horizon,
      payload.confidence_level,
    );
    setDataStatusBadge(payload.data_status);
    setForecastBadges(payload);
    setForecastNotices(payload);
    refreshDashboardPanels(canonicalSymbol, interval, selectedModels, selectedHorizon, reqId);
    try {
      const nextDataKey = `${payload.symbol_input}|${payload.interval_resolved}`;
      const shouldResetView = activeDataKey !== nextDataKey;
      renderChart(payload, shouldResetView);
      activeDataKey = nextDataKey;
    } catch (chartError) {
      console.error(chartError);
      document.getElementById("chart-updated-value").textContent = `${new Date(
        payload.updated_at,
      ).toLocaleString()} (chart render failed)`;
      setStatus(`차트 렌더링 실패: ${chartError?.message || "브라우저 콘솔을 확인하세요."}`, "error");
    }
  } catch (error) {
    if (reqId !== requestVersion) return;
    console.error(error);
    document.getElementById("chart-updated-value").textContent = "Updated -";
    setStatus(`API 요청 실패: ${error?.message || "서버 로그를 확인하세요."}`, "warning");
    setInfoMessages([]);
  } finally {
    // Requests are triggered by control changes and periodic refresh; no manual refresh button is shown.
    chartRequestInFlight = false;
    setLoadingState("chart", false);
  }
}

function bindControls() {
  const intervalInput = document.getElementById("interval-input");
  const horizonInput = document.getElementById("horizon-input");
  const languageToggle = document.getElementById("language-toggle");
  const languageToggleShell = document.querySelector(".language-toggle");
  const backtestModeToggle = document.getElementById("backtest-mode-toggle");
  const triggerSearch = () => {
    const symbol = currentOilSymbol();
    const interval = intervalInput?.value || "1d";
    const catalogEntry = null;
    if (catalogEntry && catalogEntry.status === "artifact_missing") {
      const label = MODEL_LABELS[catalogEntry.id] || catalogEntry.id;
      setStatus(
        `${label} requires training before dashboard use. Command: ${catalogEntry.training_command}`,
        "warning",
      );
      setInfoMessages([]);
      return;
    }
    // Force new chart request even for same symbol/interval.
    activeDataKey = null;
    selectedBacktestTime = null;
    activeBacktestPayload = null;
    activeBacktestOriginMarker = null;
    lastCommentaryKey = "";
    renderModelCommentaryLoading();
    setBacktestStatus("");
    updateBacktestControls();
    initDashboard(symbol, interval, { force: true });
  };

  backtestModeToggle?.addEventListener("change", () => {
    if (backtestModeToggle.checked) {
      exitBacktestMode();
      return;
    }
    if (!selectedBacktestTime) {
      backtestModeToggle.checked = true;
      setBacktestStatus(t("selectOriginFirst"), "error");
      updateBacktestControls();
      return;
    }
    runSelectedBacktest();
  });
  document.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    if (target && !target.closest("#news-detail-popover") && !target.closest(".chart-news-marker")) {
      closeNewsPopover();
    }
  });
  intervalInput.addEventListener("change", triggerSearch);
  horizonInput?.addEventListener("change", triggerSearch);
  languageToggle?.addEventListener("change", () => {
    setLanguage(languageToggle.checked ? "en" : "ko");
  });
  languageToggleShell?.addEventListener("click", (event) => {
    if (event.target === languageToggle) return;
    event.preventDefault();
    setLanguage(currentLanguage === "ko" ? "en" : "ko");
  });
  bindAssistantChat();
  updateBacktestControls();
  try {
    const savedLanguage = window.localStorage?.getItem("dashboard.language");
    if (savedLanguage === "en" || savedLanguage === "ko") currentLanguage = savedLanguage;
  } catch {
    // Keep the default Korean UI when localStorage is unavailable.
  }
  applyLanguage();
}

function startAutoRefresh() {
  if (refreshTimer) {
    clearInterval(refreshTimer);
  }
  refreshTimer = setInterval(() => {
    if (activeBacktestPayload) return;
    const symbol = currentOilSymbol();
    const interval = document.getElementById("interval-input").value || "1d";
    initDashboard(symbol, interval, { force: false });
  }, PRICE_REFRESH_MS);
}

async function bootDashboard() {
  bindControls();
  await loadModelCatalog();
  initDashboard(
    currentOilSymbol(),
    document.getElementById("interval-input").value || "1d",
    { force: true },
  );
  startAutoRefresh();
}

bootDashboard();
