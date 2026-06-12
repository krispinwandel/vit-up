import os
import json
import html
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _get_pck_value(pck_statistics: Dict, pck_alpha: float) -> Optional[float]:
    target = float(pck_alpha)
    if str(target) in pck_statistics:
        return float(pck_statistics[str(target)])
    for k, v in pck_statistics.items():
        try:
            if abs(float(k) - target) < 1e-12:
                return float(v)
        except Exception:
            continue
    return None


def _escape_latex(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _render_text_table_png(table_lines: List[str], out_path: str) -> None:
    from PIL import ImageDraw, ImageFont

    font = ImageFont.load_default()
    padding = 20
    line_spacing = 6

    max_width = 0
    line_heights = []
    dummy_img = Image.new("RGB", (10, 10), color="white")
    draw = ImageDraw.Draw(dummy_img)
    for line in table_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        max_width = max(max_width, w)
        line_heights.append(h)

    height = (
        padding * 2 + sum(line_heights) + line_spacing * max(0, len(table_lines) - 1)
    )
    width = padding * 2 + max_width

    img = Image.new("RGB", (max(1, width), max(1, height)), color="white")
    draw = ImageDraw.Draw(img)
    y = padding
    for i, line in enumerate(table_lines):
        draw.text((padding, y), line, fill="black", font=font)
        y += line_heights[i] + line_spacing

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def build_table(
    save_dir: str,
    eval_id: str,
    pck_alphas: Optional[List[float]] = None,
    metric_file: str = "pck_statistics.json",
    float_digits: int = 3,
) -> Dict[str, str]:
    """
    Aggregate SPAIR-71k eval outputs into a cross-category model comparison table
    that includes multiple PCK alpha levels in one table.

    The function scans eval folders with structure:
    save_dir/<eval_id>/<category>/<model>/<metric_file>
    and writes:
    - a LaTeX table
    - a modern HTML table

    Returns a dictionary with output file paths.
    """

    save_dir = os.path.abspath(save_dir)
    save_dir_path = Path(save_dir)
    # if out_dir is None:
    out_dir = os.path.join(
        save_dir,
        "tables",
        eval_id,
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if pck_alphas is None:
        pck_alphas = [0.1, 0.05, 0.01]

    # Discover eval IDs if not provided
    eval_ids = [eval_id]
    if eval_ids is None:
        eval_ids = []
        for p in sorted(save_dir_path.iterdir()):
            if not p.is_dir():
                continue
            if (p / "eval_config.yaml").exists():
                eval_ids.append(p.name)

    if len(eval_ids) == 0:
        raise ValueError(f"No eval directories found in: {save_dir}")

    # Collect records
    # key: (eval_id, category, model, alpha) -> value
    records: Dict[Tuple[str, str, str, float], float] = {}
    all_categories = set()
    all_models = set()

    for eval_id in eval_ids:
        eval_root = save_dir_path / eval_id
        if not eval_root.exists():
            continue

        for category_dir in sorted(eval_root.iterdir()):
            if not category_dir.is_dir():
                continue
            if category_dir.name == "tables":
                continue
            category = category_dir.name

            for model_dir in sorted(category_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                metric_path = model_dir / metric_file
                if not metric_path.exists():
                    continue

                with open(metric_path, "r") as f:
                    metric_json = json.load(f)
                model = model_dir.name
                all_categories.add(category)
                all_models.add(model)
                for alpha in pck_alphas:
                    val = _get_pck_value(metric_json, pck_alpha=alpha)
                    if val is None:
                        continue
                    records[(eval_id, category, model, float(alpha))] = float(val)

    if len(records) == 0:
        raise ValueError(
            f"No valid records found for metric '{metric_file}' and alphas={pck_alphas}."
        )

    categories_sorted = sorted(all_categories)
    models_sorted = sorted(all_models)
    eval_ids_sorted = sorted(eval_ids)
    alphas_sorted = sorted([float(a) for a in pck_alphas], reverse=True)

    # Build row-wise data for table
    alpha_labels = [f"{a:.2f}" if a < 0.1 else f"{a:.1f}" for a in alphas_sorted]
    headers = ["eval_id", "category"] + [
        f"PCK@{alpha_label}:{model}"
        for alpha_label in alpha_labels
        for model in models_sorted
    ]
    rows: List[List[str]] = []
    for eval_id in eval_ids_sorted:
        for category in categories_sorted:
            row = [eval_id, category]
            for alpha in alphas_sorted:
                for model in models_sorted:
                    v = records.get((eval_id, category, model, alpha), None)
                    row.append("-" if v is None else f"{v:.{float_digits}f}")
            # only keep rows that have at least one observed metric
            if any(c != "-" for c in row[2:]):
                rows.append(row)

        # Add mean-over-categories row for each eval_id
        mean_row = [eval_id, "mean"]
        for alpha in alphas_sorted:
            for model in models_sorted:
                vals = []
                for category in categories_sorted:
                    v = records.get((eval_id, category, model, alpha), None)
                    if v is not None:
                        vals.append(v)
                if vals:
                    mean_row.append(f"{(sum(vals) / len(vals)):.{float_digits}f}")
                else:
                    mean_row.append("-")
        if any(c != "-" for c in mean_row[2:]):
            rows.append(mean_row)

    # Build best markers per (row eval_id/category, alpha) across models for LaTeX + HTML
    row_best_by_alpha_idx: Dict[Tuple[str, str, int], float] = {}
    row_second_by_alpha_idx: Dict[Tuple[str, str, int], float] = {}
    n_alpha = len(alphas_sorted)
    for row in rows:
        eval_id, category = row[0], row[1]
        for alpha_idx in range(n_alpha):
            vals = []
            for model_idx in range(len(models_sorted)):
                data_col_idx = 2 + alpha_idx * len(models_sorted) + model_idx
                cell = row[data_col_idx]
                if cell == "-":
                    continue
                vals.append(float(cell))
            if vals:
                row_best_by_alpha_idx[(eval_id, category, alpha_idx)] = max(vals)
                uniq_desc = sorted(set(vals), reverse=True)
                if len(uniq_desc) >= 2:
                    row_second_by_alpha_idx[(eval_id, category, alpha_idx)] = uniq_desc[
                        1
                    ]

    # LaTeX output
    latex_lines = []
    latex_lines.append(
        "\\begin{tabular}{ll" + "c" * (len(models_sorted) * len(alphas_sorted)) + "}"
    )
    latex_lines.append("\\toprule")
    latex_header = " & ".join([_escape_latex(h) for h in headers]) + " \\\\"
    latex_lines.append(latex_header)
    latex_lines.append("\\midrule")
    row_suffix = " " + chr(92) + chr(92)
    for row in rows:
        eval_id, category = row[0], row[1]
        escaped_cells = [_escape_latex(eval_id), _escape_latex(category)]
        data_idx = 2
        for alpha_idx, _alpha in enumerate(alphas_sorted):
            for _model_idx, _model in enumerate(models_sorted):
                cell = row[data_idx]
                data_idx += 1
                if cell == "-":
                    escaped_cells.append(cell)
                    continue
                v = float(cell)
                best = row_best_by_alpha_idx.get((eval_id, category, alpha_idx), None)
                if best is not None and abs(v - best) < 1e-12:
                    escaped_cells.append(f"\\\\textbf{{{cell}}}")
                else:
                    escaped_cells.append(cell)
        latex_lines.append(" & ".join(escaped_cells) + row_suffix)
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")

    alpha_tag = "-".join(str(a).replace(".", "p") for a in alphas_sorted)
    latex_path = os.path.join(out_dir, f"spair71k_pck_{alpha_tag}_table.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")

    # HTML output (modern, readable, sticky header, best per-alpha highlighted)
    html_path = os.path.join(out_dir, f"spair71k_pck_{alpha_tag}_table.html")
    level_divider_cols = [
        3 + i * len(models_sorted) for i in range(1, len(alphas_sorted))
    ]
    col_divider_selector = ", ".join(
        [f"tbody td:nth-child({c})" for c in level_divider_cols]
        + [
            f"thead tr:nth-child(2) th:nth-child({1 + i * len(models_sorted)})"
            for i in range(1, len(alphas_sorted))
        ]
    )

    css = """
body { font-family: Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }
.card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px; box-shadow: 0 8px 24px rgba(2, 6, 23, 0.06); }
h1 { margin: 0 0 8px 0; font-size: 22px; }
.meta { margin: 0 0 14px 0; color: #475569; font-size: 13px; }
.tbl-wrap { overflow: auto; border-radius: 10px; border: 1px solid #e2e8f0; }
table { border-collapse: separate; border-spacing: 0; width: max-content; min-width: 100%; background: white; }
th, td { padding: 8px 10px; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; box-sizing: border-box; }
thead th { border-bottom: 1px solid #cbd5e1; }
tbody td { border-bottom: 1px solid #e2e8f0; }
tbody tr:last-child td { border-bottom: 0; }
td:first-child, td:nth-child(2) { text-align: left; position: sticky; background: white; z-index: 2; }
td:first-child { left: 0; }
td:nth-child(2) { left: 180px; }
thead tr:first-child th:first-child,
thead tr:first-child th:nth-child(2) {
    text-align: left;
    position: sticky;
    background: #f1f5f9;
    z-index: 4;
}
thead tr:first-child th:first-child { left: 0; }
thead tr:first-child th:nth-child(2) { left: 180px; }
thead th { position: sticky; top: 0; z-index: 3; background: #f1f5f9; font-weight: 600; }
thead th.group { background: #e2e8f0; text-align: center; }
thead th.group.level-divider { border-left: 2px solid #cbd5e1; }
td:not(:nth-child(-n+2)), thead tr:nth-child(2) th { width: 108px; min-width: 108px; max-width: 108px; }
tbody td:nth-child(3),
thead tr:nth-child(2) th:nth-child(1),
thead tr:first-child th.group:first-of-type { border-left: 2px solid #cbd5e1; }
td.best { font-weight: 700; }
td.second { text-decoration: underline; text-underline-offset: 2px; }
td.miss { color: #94a3b8; }
"""
    if col_divider_selector:
        css += f"\n{col_divider_selector} {{ border-left: 2px solid #cbd5e1; }}\n"

    # grouped headers
    head_top = [
        "<tr><th rowspan='2' style='min-width:160px'>eval_id</th><th rowspan='2' style='min-width:120px'>category</th>"
    ]
    for alpha_idx, alpha in enumerate(alphas_sorted):
        group_cls = "group level-divider" if alpha_idx > 0 else "group"
        head_top.append(
            f"<th class='{group_cls}' colspan='{len(models_sorted)}'>PCK@{html.escape(str(alpha))}</th>"
        )
    head_top.append("</tr>")
    head_bottom = ["<tr>"]
    for _alpha in alphas_sorted:
        for model in models_sorted:
            head_bottom.append(f"<th>{html.escape(model)}</th>")
    head_bottom.append("</tr>")

    body_lines = []
    for row in rows:
        eval_id, category = row[0], row[1]
        tds = [f"<td>{html.escape(eval_id)}</td>", f"<td>{html.escape(category)}</td>"]
        data_idx = 2
        for alpha_idx, _alpha in enumerate(alphas_sorted):
            for _model_idx, _model in enumerate(models_sorted):
                cell = row[data_idx]
                data_idx += 1
                if cell == "-":
                    tds.append("<td class='miss'>-</td>")
                    continue
                v = float(cell)
                best = row_best_by_alpha_idx.get((eval_id, category, alpha_idx), None)
                second = row_second_by_alpha_idx.get(
                    (eval_id, category, alpha_idx), None
                )
                if best is not None and abs(v - best) < 1e-12:
                    cls = "best"
                elif second is not None and abs(v - second) < 1e-12:
                    cls = "second"
                else:
                    cls = ""
                tds.append(f"<td class='{cls}'>{cell}</td>")
        body_lines.append("<tr>" + "".join(tds) + "</tr>")

    html_doc = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>SPAIR-71k PCK Comparison</title>",
            f"<style>{css}</style>",
            "</head><body>",
            "<div class='card'>",
            "<h1>SPAIR-71k model comparison</h1>",
            f"<p class='meta'>Alphas: {html.escape(', '.join(map(str, alphas_sorted)))} · Rows: {len(rows)} · Metric file: {html.escape(metric_file)}</p>",
            "<div class='tbl-wrap'><table><thead>",
            "".join(head_top),
            "".join(head_bottom),
            "</thead><tbody>",
            "\n".join(body_lines),
            "</tbody></table></div>",
            "</div></body></html>",
        ]
    )
    with open(html_path, "w") as f:
        f.write(html_doc)

    return {
        "latex_path": latex_path,
        "html_path": html_path,
        "rows": str(len(rows)),
    }
