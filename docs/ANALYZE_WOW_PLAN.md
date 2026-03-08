# Plan: bnnr analyze → wow feature v0.2.0

## 1. Assessment obecnego stanu

### Co jest
- **analyze.py**: jeden moduł ~370 linii. `AnalysisReport` (dataclass): metrics, per_class_accuracy, confusion, xai_*, data_quality_result, worst_predictions, failure_patterns (lista dict), recommendations (lista string).
- **HTML**: `_render_analysis_html()` — minimalny raport: `<pre>` z JSON dla metrics, per_class, confusion, failure_patterns, worst_predictions, `<ul>` dla recommendations. Brak executive summary, brak kart/wykresów, wygląd debug.
- **Failure patterns**: tylko `confused_pair` (z confusion matrix) i `low_xai_quality` (z xai_diagnoses). Brak taxonomii (zero_recall, class_collapse, background_focus, etc.).
- **Recommendations**: 3–4 szablony tekstowe w `_build_recommendations()`; brak powiązania z findings, brak priorytetu/scope/action.
- **XAI**: wyniki w report (xai_insights, xai_diagnoses, xai_quality_summary); w HTML nie wyeksponowane; brak galerii overlay, brak wyjaśnienia score.
- **Worst predictions**: lista dict (index, true_label, pred_label, confidence, loss); bez grupowania (per class, per pair), bez path do overlay w raporcie (overlay zapisywany tylko gdy output_dir + XAI).
- **Evaluation**: per_class ma tylko accuracy + support; brak precision/recall/F1 per class, brak rozkładu predicted vs true.

### Dashboard (wzór wizualny)
- **Kolory**: dark default `--bg: #080810`, `--accent: #f0a069`, `--card: #131322`, `--border`, `--green`, `--red`. Light: `--bg: #faf7f2`, `--accent: #c96a35`.
- **Komponenty**: `.card`, `.kpi-card`, `.kpi-value`, `.kpi-label`, badges (selected/evaluated/rejected), tabele (`.ledger-table`), class bars, confusion matrix (`.cm-grid`), XAI insight cards (`.xai-insight-card`).
- **Typografia**: Inter/system-ui, 14px body, nagłówki accent, muted dla drugoplanu.

### Ryzyka
- Rozrost analyze.py — trzeba wydzielić: findings engine, recommendations engine, failure taxonomy, report schema, HTML renderer.
- HTML bez React — raport to pojedynczy plik HTML z inline/embedded CSS; trzeba skopiować tokeny z dashboard (CSS variables lub inline wartości).
- Testy: nowe struktury (findings, recommendations) muszą być testowalne jednostkowo; HTML — testy strukturalne (sekcje, klasy).

---

## 2. Plan plików

| Akcja | Ścieżka |
|-------|--------|
| Nowy | `src/bnnr/analysis/__init__.py` (pusty lub re-export wewnętrzny) |
| Nowy | `src/bnnr/analysis/schema.py` — wersjonowany schemat raportu, dataclassy: ExecutiveSummary, Finding, Recommendation, ClassDiagnostic, FailurePattern (rozszerzony) |
| Nowy | `src/bnnr/analysis/findings.py` — wykrywanie patterns (zero_recall, low_xai, confused_pair, over_predicted, …), budowa listy Finding z evidence/interpretation/confidence |
| Nowy | `src/bnnr/analysis/recommendations.py` — build_recommendations(findings, metrics, report) → lista Recommendation (title, scope, why, action, impact, confidence, linked_evidence) |
| Nowy | `src/bnnr/analysis/class_diagnostics.py` — per-class precision/recall/F1, true/pred distribution, ranking klas krytycznych |
| Modyfikacja | `src/bnnr/analyze.py` — import z analysis.*, rozszerzenie AnalysisReport o executive_summary, findings, recommendations (structured), class_diagnostics, distribution; orkiestracja wywołań findings + recommendations |
| Nowy | `src/bnnr/analysis/html_report.py` — jedna funkcja `render_analysis_html(report, options?)` zwracająca str; użycie tokenów BNNR (embedded CSS), sekcje: header, executive summary, KPIs, class diagnostics, failure analysis, XAI, worst predictions, recommendations, method/caveats |
| Modyfikacja | `src/bnnr/analyze.py` — `to_html()` wywołuje `render_analysis_html()` z analysis.html_report |
| Opcjonalnie | `src/bnnr/analysis/xai_evidence.py` — helper do zapisu overlay paths w report i listy plików dla galerii |
| Testy | `tests/test_analyze.py` — rozszerzenie; `tests/test_analysis_findings.py`, `tests/test_analysis_recommendations.py`, `tests/test_analysis_html.py` (strukturalne) |

---

## 3. Kolejność wdrożenia

1. **Schema + class diagnostics** — schema.py (ExecutiveSummary, Finding, Recommendation, ClassDiagnostic), class_diagnostics.py (precision/recall/F1, distributions). Integracja w evaluation lub analyze (dodanie pól do reportu).
2. **Findings engine** — findings.py (taxonomy patterns, build_findings()), podłączenie w analyze.py.
3. **Recommendations engine** — recommendations.py (structured), podłączenie w analyze.py.
4. **Executive summary** — budowa z findings + metrics + top recommendations w analyze.py.
5. **HTML report** — html_report.py z pełnym layoutem i CSS; analyze.to_html() używa tego.
6. **XAI evidence** — zapis overlay paths w worst_predictions, sekcja XAI w HTML (galeria, wyjaśnienie score).
7. **Worst predictions** — grupowanie (opcjonalnie), karty w HTML (obraz + overlay + etykiety).
8. **CLI/docs** — brak zmian API/CLI poza ewentualnym --no-html; docs zaktualizować.
9. **Testy** — jednostkowe findings/recommendations/schema, strukturalne HTML, regresja.

---

## 4. Ograniczenia / odłożone na później

- **bnnr compare** (pre vs post fine-tuning) — poza v0.2.0; schemat raportu ma być gotowy pod compare.
- **Interaktywne filtry** w HTML (filter by class/pattern) — v0.2.0: statyczny layout; struktura danych gotowa pod przyszłe filtry.
- **Detection** w analyze — bez zmian; tylko classification w pełni wspierany w raporcie wow.
