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
    assert "grid-template-columns: minmax(520px, 1fr) minmax(240px, 300px) minmax(300px, 360px);" in styles
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


def test_model_commentary_highlights_directional_keywords() -> None:
    index = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    script = (FRONTEND_DIR / "src" / "main.js").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "src" / "dashboard.css").read_text(encoding="utf-8")

    assert "20260604-commentary-tones" in index
    assert "const COMMENTARY_KEYWORDS" in script
    assert "highlightCommentaryText(summary" in script
    assert "highlightCommentaryText(li" in script
    assert '"횡보"' in script
    assert '"우세"' not in script
    assert "data-tone=\"bullish\"" in styles
    assert "data-tone=\"bearish\"" in styles
    assert "data-tone=\"neutral\"" in styles
