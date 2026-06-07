from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_DIR / "frontend"


def test_report_panel_uses_download_action_and_print_pdf_flow() -> None:
    index = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")

    assert 'id="report-download-button"' in index
    assert 'id="report-generate-button"' not in index
    assert 'id="report-note"' not in index
    assert "window.print()" in script
    assert "forecast-report-print" in script


def test_desktop_layout_uses_chart_insight_report_columns_without_scenario_cards() -> None:
    index = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert 'class="chart-column"' in index
    assert 'class="insight-column"' in index
    assert 'class="report-column"' in index
    assert index.index('id="model-commentary-panel"') < index.index('id="forecast-report-panel"')
    assert "@media (min-width: 1180px)" in styles
    assert "@media (min-width: 1180px) and (orientation: landscape)" in styles
    assert "@media (min-width: 1180px) and (max-width: 1519px) and (orientation: portrait)" in styles
    assert "grid-template-columns: minmax(0, 1fr) minmax(300px, 380px);" in styles
    assert "@media (min-width: 1520px)" in styles
    assert "grid-template-columns: minmax(640px, 1fr) minmax(260px, 330px) minmax(320px, 380px);" in styles
    assert 'class="scenario-grid"' not in index
    assert 'id="scenario-bull"' not in index


def test_report_refresh_tracks_language_like_commentary() -> None:
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")

    assert "lastReportKey = \"\";" in script
    assert "reportRequestVersion += 1;" in script
    assert "forceReport: true" in script
    assert "currentLanguage !== languageAtRequest" in script


def test_commentary_risk_text_has_higher_contrast() -> None:
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert ".commentary-risks" in styles
    assert "color: #cbd7e5;" in styles
    assert "font-size: 13.5px;" in styles


def test_metric_labels_include_accessible_help_tooltips() -> None:
    index = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert 'data-term="mae"' in index
    assert 'data-term="rmse"' in index
    assert 'data-term="mape"' in index
    assert "const TERM_HELP" in script
    assert "평균 절대 오차" in script
    assert ".term-help::after" in styles
    assert "content: attr(data-tooltip);" in styles
    assert ".chart-bottom-panel" in styles
    assert "z-index: 900;" in styles


def test_model_commentary_highlights_directional_keywords() -> None:
    index = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert "20260608-chat-glass-context-dedupe" in index
    assert "const COMMENTARY_KEYWORDS" in script
    assert "highlightCommentaryText(summary" in script
    assert "highlightCommentaryText(li" in script
    assert '"횡보"' in script
    assert '"우세"' not in script
    assert "data-tone=\"bullish\"" in styles
    assert "data-tone=\"bearish\"" in styles
    assert "data-tone=\"neutral\"" in styles


def test_stale_data_warning_stays_inside_chart_with_market_session_copy() -> None:
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert "setStatus(null);" in script
    assert "oilFuturesSessionState" in script
    assert "The crude oil futures market is likely closed or in its daily break" in script
    assert ".chart-wrap[data-warning-visible=\"true\"] .tv-legend" in styles


def test_chart_language_updates_dates_and_mode_styles() -> None:
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert "formatDateTimeValue" in script
    assert '"en-US"' in script
    assert "candleSeriesRef.applyOptions({ title: \"\" });" in script
    assert "backtestActualSeriesRef.applyOptions({ title: \"\" });" in script
    assert "toLocaleString" not in script
    assert ".chart-mode-toggle:has(input:checked)" in styles
    assert "--live: #34d399;" in styles
    assert "--backtest: #f2cc60;" in styles


def test_header_controls_removed_for_fixed_daily_30d_forecast() -> None:
    index = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert 'class="controls" aria-label="Forecast controls"' not in index
    assert 'id="interval-input"' not in index
    assert 'id="horizon-input"' not in index
    assert 'id="symbol-input"' not in index
    assert "1H" not in index
    assert 'const DEFAULT_INTERVAL = "1d";' in script
    assert "const DEFAULT_HORIZON = 30;" in script
    assert "function currentInterval()" in script
    assert "function currentHorizon()" in script
    assert "grid-template-columns: auto minmax(0, 1fr) auto;" in styles
    assert ".language-mode-toggle" in styles
    assert ".language-mode-toggle:has(input:checked)" in styles
    assert "--radius-panel: 18px;" in styles


def test_backtest_mode_waits_for_chart_click_with_guide_copy() -> None:
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert 'let chartMode = "live";' in script
    assert "backtestClickGuide" in script
    assert 'chartMode = "backtest";' in script
    assert 'requestVersion += 1;\n    chartMode = "backtest";' in script
    assert 'setBacktestStatus(t("backtestClickGuide"), "guide");' in script
    assert '#backtest-status[data-severity="guide"]' in styles


def test_glass_panels_use_sticky_headers_and_stack_metrics_before_overlap() -> None:
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert "backdrop-filter: blur(22px) saturate(150%);" in styles
    assert ".explanation-head" in styles
    assert "position: sticky;" in styles
    assert "border-bottom: 0;" in styles
    assert "padding: 0 16px 16px;" in styles
    assert "margin: 0 -16px 12px;" in styles
    assert ".explanation-head::after" in styles
    assert "@media (max-width: 900px)" in styles
    assert "grid-template-columns: repeat(4, minmax(0, 1fr));" in styles
    assert ".backtest-toolbar {\n  display: contents;" in styles
    assert ".metrics {\n  display: contents;" in styles


def test_chat_panel_lives_under_chart_column() -> None:
    index = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert index.index('class="chart-column"') < index.index('id="llm-chat-panel"')
    assert index.index('id="llm-chat-panel"') < index.index('class="insight-column"')
    assert "AI에게 묻기" in index
    assert "전송" in index
    assert 'id="llm-chat-submit" type="submit"' in index
    assert 'label.textContent = role === "user" ? "Q" : "LLM";' not in script
    assert 'form.addEventListener("submit", submitQuestion);' in script
    assert "submitQuestion(event);" not in script
    assert "event.isComposing || isComposing" in script
    assert "chatRequestInFlight" in script
    assert "aiUnavailable" in script
    assert "userFacingLlmError" in script
    assert "appendChatTypingIndicator" in script
    assert "llm-chat-typing-dots" in script
    assert "scrollChatLogToEnd(log);" in script
    assert ".llm-chat-panel .explanation-head" in styles
    assert ".explanation-head" in styles
    assert ".llm-chat-panel::before" in styles
    assert "z-index: 0;" in styles
    assert "rgba(88, 166, 255, 0.14) 0%" not in styles
    assert "rgba(88, 166, 255, 0.12) 0%" not in styles
    chat_head_block = styles[
        styles.index(".llm-chat-panel .explanation-head {") : styles.index(
            ".llm-chat-panel .explanation-head::after"
        )
    ]
    assert "background: linear-gradient(180deg, rgba(21, 27, 35, 0.66)" in chat_head_block
    assert "backdrop-filter: blur(22px) saturate(165%);" in chat_head_block
    assert "background: transparent;" not in chat_head_block
    assert "backdrop-filter: none;" not in chat_head_block
    assert ".llm-chat-panel .explanation-head::after" in styles
    assert ".forecast-report-panel,\n.llm-chat-panel" in styles
    assert "backdrop-filter: blur(22px) saturate(150%);" in styles
    assert "max-height: clamp(220px, 28vh, 320px);" in styles
    assert "grid-template-rows: minmax(0, 2fr) minmax(260px, 1fr);" in styles
    assert "overflow-y: auto;" in styles
    assert "margin: 0 -16px 12px;" in styles
    assert "padding: 16px 16px 22px;" in styles
    assert "rgba(21, 27, 35, 0.4) 58%" in styles
    assert ".llm-chat-log" in styles
    assert "background: transparent;" in styles
    assert "max-height: inherit;" in styles
    assert "height: 100%;" in styles
    assert ".llm-chat-form" in styles
    chat_form_block = styles[
        styles.index(".llm-chat-form {") : styles.index(".llm-chat-form input")
    ]
    assert "background: linear-gradient(0deg, rgba(21, 27, 35, 0.66)" in chat_form_block
    assert "backdrop-filter: blur(22px) saturate(165%);" in chat_form_block
    assert "background: transparent;" not in chat_form_block
    assert "backdrop-filter: none;" not in chat_form_block
    assert "position: absolute;" in styles
    assert "bottom: 0;" in styles
    assert "margin: 0;" in styles
    assert "padding: var(--chat-head-space) 16px var(--chat-form-space);" in styles
    assert "@keyframes chatTypingGrow" in styles
    assert ".llm-chat-message.user" in styles
    assert "align-self: flex-end;" in styles


def test_backtest_metrics_panel_is_collapsed_in_live_mode() -> None:
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert '.chart-panel[data-mode="live"] .chart-bottom-panel' in styles
    assert "max-height: 0;" in styles
    assert "max-height: 260px;" in styles
    assert ".chart-panel[data-mode=\"live\"] .chart-bottom-panel" in styles
    assert "overflow: visible;" in styles
    assert "justify-content: center;" in styles
    assert ".chart-panel,\n  .insight-column" not in styles
    assert "height: max(560px, calc(100dvh - 118px));" in styles
    assert "ResizeObserver" in (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")


def test_chart_header_removes_nonessential_meta_badges() -> None:
    index = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert 'id="data-status-badge"' in index
    assert 'id="calibration-badge"' not in index
    assert 'id="confidence-badge"' not in index
    assert 'id="regime-badge"' not in index
    assert 'data-i18n="band"' not in index
    assert 'data-i18n="conf"' not in index
    assert 'data-i18n="regime"' not in index
    assert ".data-status-badge[data-status=\"stale\"]" in styles
    assert "rgba(251, 113, 133" in styles
    assert "border: 1px solid rgba(88, 166, 255, 0.38);" in styles
    assert "color: #dbeafe;" in styles
    assert "dataStatusTooltipText" in script
    assert 'badge.dataset.tooltip = tooltip;' in script
    assert '.data-status-badge[data-tooltip]::after' in styles
    assert 'if (!["mock", "fallback", "error"].includes(status))' in script


def test_loading_state_uses_chart_updated_slot() -> None:
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert 'const node = document.getElementById("chart-updated-value");' in script
    assert 'node.dataset.loading = "true";' in script
    assert 'banner.classList.add("hidden");' in script
    assert ".chart-panel-head p[data-loading=\"true\"]" in styles
    assert ".chart-panel-head p[data-loading=\"true\"]::after" in styles
    assert "animation: loadingPulse 1.25s ease-out infinite;" in styles


def test_news_markers_show_forecast_context_titles_inside_chart() -> None:
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert 'if (chartMode !== "backtest" && !activeBacktestPayload)' not in script
    assert "function markerPointsFromContext(contextPayload)" in script
    assert "contextPayload?.chart_context_points" in script
    assert "function aggregateMarkerPoints(points)" in script
    assert "const FORECAST_CONTEXT_MARKER_LIMIT = 8;" in script
    assert "function spreadPointsByTime(points, limit = FORECAST_CONTEXT_MARKER_LIMIT)" in script
    assert "return spreadPointsByTime(aggregateMarkerPoints(markerPoints));" in script
    assert "return spreadPointsByTime(aggregateMarkerPoints(contextPoints));" in script
    assert "slice(-6)" not in script
    assert "const headline = String(news?.headline || \"\").trim();" in script
    assert "newsPopoverTitle(point)" in script
    assert "context-news-list" in script
    assert "koreanNewsTopic(headline)" in script
    assert "news-popover-action" not in script
    assert "news-popover-action" not in styles
    assert "right: 70px;" in styles
    assert "forecastSegmentEndpoint" in script
    assert "forecastSegmentMarker" in script
    assert 'position: "inBar"' in script
    assert "series.setMarkers(forecastSegmentMarker(predicted, segment));" in script
    assert "bottom: 54px;" in styles
    assert '.news-detail-popover[data-bias="bullish"]' in styles


def test_forecast_is_rendered_as_fixed_30_day_segments() -> None:
    index = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert 'id="forecast-segment-legend"' in index
    assert "const FORECAST_SEGMENTS = [" in script
    assert '{ id: "week1", start: 1, end: 7' in script
    assert '{ id: "week2", start: 8, end: 14' in script
    assert '{ id: "month", start: 15, end: 30' in script
    assert "forecastWeek1" in script
    assert "forecastWeek2" in script
    assert "forecastMonth" in script
    assert '"1d": 128' in script
    assert "function renderForecastSegmentSeries(predicted)" in script
    assert "function positionForecastSegmentLabels()" in script
    assert "Forecast horizon labels are native chart markers now" in script
    assert 'predSeriesRef.setData([]);' in script
    assert '#a78bfa' in script
    assert "lineStyle.Dotted !== undefined ? lineStyle.Dotted : 1" in script
    assert "renderForecastSegmentLegend(payload);" in script
    assert ".forecast-segment-legend" in styles
    assert "root.classList.add(\"hidden\");" in script
    assert "var(--segment-color)" in styles
    assert "top: 12px;\n  right: 76px;" not in styles


def test_news_context_hides_internal_placeholder_text() -> None:
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert "NON_PUBLIC_CONTEXT_PATTERNS" in script
    assert "deterministic local event context encoder" in script
    assert "structured context only" in script
    assert "function isPublicContextText(text)" in script
    assert "function aggregateContextEventPoints(contextPoints, newsItems)" in script
    assert "function uniqueDisplayNewsItems(newsItems)" in script
    assert "aggregateContextEventPoints(contextPoints, newsItems)" in script
    assert "직접 표시할 뉴스 제목은 부족하지만" not in script
    assert "There are not enough direct headlines" not in script
    assert "news-popover-factors" not in script
    assert "news-popover-factors" not in styles


def test_dashboard_panels_use_single_combined_llm_request() -> None:
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")
    index = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    assert "/api/dashboard-analysis" in script
    assert "loadDashboardAnalysis(symbol, interval, selectedModels, selectedHorizon, originTime, languageAtRequest)" in script
    assert "renderMarketContextPanel(analysis.market_context || null);" in script
    assert "renderModelCommentary(analysis.commentary || unavailableCommentary());" in script
    assert "renderForecastReport(analysis.report || null);" in script
    assert "DASHBOARD_ANALYSIS_REFRESH_MS" in script
    assert "pendingAnalysisRefresh" in script
    assert "activeAnalysisRequestId" in script
    assert "latestAnalysisPayloadKey" in script
    assert "isDashboardAnalysisRequestCurrent" in script
    assert "clearDashboardAnalysisPanels();" in script
    assert "renderContextMarkers(null);" in script
    assert "dashboardPanelKey(" in script
    assert "originTime," in script
    assert "chat-glass-context-dedupe" in index


def test_vertical_side_panels_scroll_internally() -> None:
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert "@media (max-width: 1179px), (max-width: 1519px) and (orientation: portrait)" in styles
    assert "max-height: clamp(300px, 38vh, 480px);" in styles
    assert "overflow-y: auto;" in styles
