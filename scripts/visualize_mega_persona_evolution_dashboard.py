"""Build a static HTML dashboard for MegaPersona OpenEvolve runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import statistics
from typing import Any


@dataclass
class CandidateRow:
    candidate_id: str
    generation: int
    score: float
    operator_id: str
    mutation_mode: str
    schema: float
    consistency: float
    axis_alignment: float
    validation_alignment: float
    shadow_mae: float
    axis_target_mae: float
    consistency_issue_rate: float
    strict_consistency_error: float
    validation_coverage: float
    balanced_diversity: float
    avg_dist: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a MegaPersona evolution dashboard.")
    parser.add_argument("--input", required=True, help="MegaPersona evolution output directory.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path. Defaults to <input>/evolution_dashboard.html.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output = Path(args.output) if args.output else input_dir / "evolution_dashboard.html"
    output.write_text(build_dashboard(input_dir), encoding="utf-8")
    print(f"Saved MegaPersona evolution dashboard: {output}")


def build_dashboard(input_dir: Path) -> str:
    rows = load_candidates(input_dir)
    if not rows:
        raise FileNotFoundError(f"No candidate JSON files found under {input_dir}")

    final_summary = read_json(input_dir / "final_summary.json")
    final_test = final_summary.get("final_test_report", {}).get("metrics", {})
    log_counts = count_log_events(input_dir / "run.log")
    baseline = next((row for row in rows if row.generation == 0), rows[0])
    best = max(rows, key=lambda row: row.score)
    generations = summarize_generations(rows)
    operators = summarize_operators(rows)
    modes = summarize_modes(rows)

    gain_abs = best.score - baseline.score
    gain_pct = (best.score / baseline.score - 1.0) * 100.0 if baseline.score else 0.0
    plateau_generation = first_generation_at_score(generations, best.score)

    fitness_chart = line_chart(
        "每代适应度变化",
        [
            ("generation best", [(item["generation"], item["best"]) for item in generations]),
            ("generation mean", [(item["generation"], item["mean"]) for item in generations]),
            ("best so far", [(item["generation"], item["best_so_far"]) for item in generations]),
        ],
        y_label="fitness",
    )
    metrics_chart = line_chart(
        "每代最优候选的分指标变化",
        [
            ("行为覆盖率 coverage", [(item["generation"], item["best_row"].validation_coverage) for item in generations]),
            ("旧行为对齐分 alignment", [(item["generation"], item["best_row"].validation_alignment) for item in generations]),
            ("schema 合法性", [(item["generation"], item["best_row"].schema) for item in generations]),
            ("旧一致性分 consistency", [(item["generation"], item["best_row"].consistency) for item in generations]),
            ("行为多样性 diversity", [(item["generation"], item["best_row"].balanced_diversity) for item in generations]),
        ],
        y_label="metric",
    )
    strict_metrics_chart = line_chart(
        "严格误差指标变化（越低越好）",
        [
            ("shadow 行为 MAE", [(item["generation"], item["best_row"].shadow_mae) for item in generations]),
            ("axis target MAE", [(item["generation"], item["best_row"].axis_target_mae) for item in generations]),
            ("consistency 问题率", [(item["generation"], item["best_row"].consistency_issue_rate) for item in generations]),
            ("strict consistency error", [(item["generation"], item["best_row"].strict_consistency_error) for item in generations]),
        ],
        y_label="error",
    )
    operator_mean_max_chart = grouped_bar_chart(
        "算子平均表现与最佳表现",
        [
            (item["operator_id"], item["mean"], item["max"])
            for item in operators
            if item["operator_id"] != "baseline"
        ],
        labels=("mean", "best"),
        baseline=baseline.score,
    )
    delta_chart = bar_chart(
        "全局最优相对 Baseline 的提升",
        [
            (metric_label("fitness", baseline.score, best.score), best.score - baseline.score),
            (
                metric_label("coverage", baseline.validation_coverage, best.validation_coverage),
                best.validation_coverage - baseline.validation_coverage,
            ),
            (
                metric_label("shadow MAE", baseline.shadow_mae, best.shadow_mae),
                best.shadow_mae - baseline.shadow_mae,
            ),
            (
                metric_label("axis target MAE", baseline.axis_target_mae, best.axis_target_mae),
                best.axis_target_mae - baseline.axis_target_mae,
            ),
            (
                metric_label("consistency issue rate", baseline.consistency_issue_rate, best.consistency_issue_rate),
                best.consistency_issue_rate - baseline.consistency_issue_rate,
            ),
            (metric_label("schema", baseline.schema, best.schema), best.schema - baseline.schema),
            (
                metric_label("consistency", baseline.consistency, best.consistency),
                best.consistency - baseline.consistency,
            ),
            (
                metric_label("axis alignment", baseline.axis_alignment, best.axis_alignment),
                best.axis_alignment - baseline.axis_alignment,
            ),
            (
                metric_label("balanced diversity", baseline.balanced_diversity, best.balanced_diversity),
                best.balanced_diversity - baseline.balanced_diversity,
            ),
            (metric_label("avg dist", baseline.avg_dist, best.avg_dist), best.avg_dist - baseline.avg_dist),
        ],
        baseline=0.0,
        zero_center=True,
    )
    sealed_test_pairs = sealed_test_metric_pairs(best, final_test)
    test_chart = grouped_bar_chart(
        "验证集最优与封闭测试集对比",
        sealed_test_pairs,
        labels=("validation", "sealed test"),
    )
    candidate_chart = scatter_chart("候选生成器适应度散点", rows)

    config = final_summary.get("config", {})
    cards = [
        ("Baseline 综合 fitness", f"{baseline.score:.6f}"),
        ("Best 综合 fitness", f"{best.score:.6f}"),
        ("Fitness 提升", f"{gain_abs:+.6f} ({gain_pct:+.2f}%)"),
        ("Best 所在代数", f"gen {best.generation}"),
        ("Shadow 行为 MAE", f"{best.shadow_mae:.6f}"),
        ("Axis target MAE", f"{best.axis_target_mae:.6f}"),
        ("一致性问题率", f"{best.consistency_issue_rate:.3f}/persona"),
        ("候选数", str(len(rows))),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MegaPersona 进化实验可视化 - {escape(input_dir.name)}</title>
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #18202a; }}
header {{ padding: 22px 28px; background: #202734; color: white; }}
h1 {{ margin: 0 0 6px; font-size: 24px; }}
h2 {{ margin: 28px 0 12px; font-size: 18px; }}
h3 {{ margin: 0 0 10px; font-size: 16px; }}
.sub {{ color: #b9c2d0; font-size: 13px; }}
main {{ padding: 20px 28px 40px; max-width: 1440px; margin: 0 auto; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
.card {{ background: white; border: 1px solid #dfe3ea; border-radius: 8px; padding: 14px; }}
.card .label {{ color: #627085; font-size: 12px; text-transform: uppercase; letter-spacing: .02em; }}
.card .value {{ margin-top: 7px; font-size: 22px; font-weight: 700; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 680px), 1fr)); gap: 18px; align-items: start; }}
.panel {{ background: white; border: 1px solid #dfe3ea; border-radius: 8px; padding: 16px; overflow-x: auto; }}
.caption {{ margin: 10px 2px 0; color: #475569; font-size: 13px; line-height: 1.55; }}
.caption strong {{ color: #1f2937; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf0f4; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: #475569; background: #f8fafc; }}
.note {{ background: #fff7e6; border: 1px solid #f3d19e; border-radius: 8px; padding: 12px 14px; color: #5f430c; }}
svg {{ width: 100%; height: auto; display: block; overflow: visible; }}
.axis {{ stroke: #9aa4b2; stroke-width: 1; }}
.gridline {{ stroke: #e5e9f0; stroke-width: 1; }}
.legend {{ font-size: 12px; fill: #334155; }}
.tick {{ font-size: 11px; fill: #64748b; }}
</style>
</head>
<body>
<header>
  <h1>MegaPersona 进化实验可视化</h1>
  <div class="sub">{escape(str(input_dir))}</div>
</header>
<main>
  <section class="cards">{''.join(card(label, value) for label, value in cards)}</section>

  <h2>实验逻辑</h2>
  <div class="note">
    本页面按“Baseline 生成器 -> OpenEvolve 变异候选 -> 多指标评估 -> 验证集选择 -> 封闭测试集复核”的实验链路组织。
    旧的 alignment / consistency 分数保留为 sanity check；论文解释优先看严格误差指标：
    shadow 行为 MAE、axis target MAE 和 consistency 问题率。
    图 1 看进化是否累积改进，图 2 看旧高分指标是否稳定，图 3 看严格误差是否下降，图 4 看最优候选相对 baseline 的变化，
    图 5 检查验证集收益是否能迁移到封闭测试集，图 6 比较不同 v3 算子的稳定性与上限，图 7 展示每代候选的搜索分布。
    当前最优候选出现在 gen {best.generation}，相对 baseline 提升 {gain_pct:.2f}%，LLM mutator fallback 次数为 {log_counts.get("LLM mutator failed", 0)}。
  </div>

  <h2>指标中文意义</h2>
  <div class="panel">{metric_glossary()}</div>

  <h2>核心图表</h2>
  <section class="grid">
    {chart_panel(fitness_chart, "图 1：进化是否真的推进", "红线 best-so-far 是判断进化是否有效的主证据；蓝线表示每一代最优候选，绿线表示每代平均水平。如果只有蓝线偶尔冲高而绿线不动，说明更像随机发现；如果红线阶梯式上升，说明搜索确实找到了可继承的改进。")}
    {chart_panel(metrics_chart, "图 2：旧高分指标是否稳定", "该图展示历史兼容指标。alignment 和 consistency 往往初始偏高，所以它们适合作为 sanity check：证明生成器没有崩；不宜单独作为论文里的主要提升证据。")}
    {chart_panel(strict_metrics_chart, "图 3：严格误差是否下降", "该图是论文解释的核心。shadow 行为 MAE 衡量人格声明轴与 shadow survey 行为轴之间的误差；axis target MAE 衡量 persona 生成结果与目标 slot 的偏差；consistency 问题率表示每个人格平均触发多少条一致性问题。这里越低越好。")}
    {chart_panel(delta_chart, "图 4：最优候选相对 baseline 改了多少", "每个柱子都是全局最优候选减去 baseline 的绝对变化量。对 fitness、coverage、schema 等高分指标，正值更好；对 shadow MAE、axis target MAE、issue rate 等误差指标，负值更好。")}
    {chart_panel(test_chart, "图 5：验证集收益是否能迁移", "进化期间使用 validation shadow surveys 选择候选，sealed test 只在最终评估时使用。新版 final test 会同时复核 schema、strict consistency error、shadow MAE、coverage 和 diversity；旧实验若缺少完整 sealed 指标，则只展示已有指标。")}
    {chart_panel(operator_mean_max_chart, "图 6：不同 v3 算子的贡献", "蓝柱是算子的平均 fitness，代表稳定性；橙柱是该算子产生过的最高 fitness，代表探索上限。高均值算子适合作为默认搜索方向，高上限但低均值算子适合作为探索型算子。")}
    {chart_panel(candidate_chart, "图 7：候选生成器搜索分布", "每个点是一个被评估的候选生成器。纵向离散度反映同一代搜索的方差；某一代出现孤立高点，说明该代有有效发现，但也需要结合后续代和封闭测试判断是否稳定。")}
  </section>

  <h2>Best vs Baseline</h2>
  <div class="panel">{best_vs_baseline_table(baseline, best)}</div>

  <h2>Generation Summary</h2>
  <div class="panel">{generation_table(generations, baseline)}</div>

  <h2>Operator Summary</h2>
  <div class="panel">{operator_table(operators, baseline)}</div>

  <h2>Mutation Mode Summary</h2>
  <div class="panel">{mode_table(modes)}</div>

  <h2>Final Sealed Test</h2>
  <div class="panel">{dict_table(final_test)}</div>

  <h2>Config</h2>
  <div class="panel">{dict_table(config)}</div>
</main>
</body>
</html>
"""


def load_candidates(input_dir: Path) -> list[CandidateRow]:
    candidates_dir = input_dir / "mega_eval" / "candidates"
    rows: list[CandidateRow] = []
    for path in sorted(candidates_dir.glob("*.json")):
        payload = read_json(path)
        metrics = payload.get("metrics", {})
        genome = payload.get("genome", {})
        mutation = genome.get("openevolve_mutation") or genome.get("last_mutation") or {}
        operator = genome.get("last_evolution_operator") or genome.get("last_operator")
        if isinstance(mutation, dict) and mutation.get("operator_id"):
            operator_id = str(mutation["operator_id"])
        elif isinstance(operator, dict):
            operator_id = str(operator.get("id") or "unknown")
        else:
            operator_id = str(operator or "baseline")
        mode = str(mutation.get("mode") or "baseline") if isinstance(mutation, dict) else "baseline"
        score = payload.get("fitness")
        if not isinstance(score, (int, float)):
            continue
        validation_alignment = float(metrics.get("validation_shadow_alignment.mean", 0.0))
        axis_alignment = float(metrics.get("axis_alignment.mean", 0.0))
        shadow_mae = float(
            metrics.get("validation_shadow_mae.mean", max(0.0, 1.0 - validation_alignment))
        )
        axis_target_mae = float(
            metrics.get("axis_target_mae.mean", max(0.0, 1.0 - axis_alignment))
        )
        consistency_issue_rate = float(metrics.get("consistency_issue_rate.mean", 0.0))
        strict_consistency_error = float(
            metrics.get("strict_consistency_error.mean", axis_target_mae)
        )
        rows.append(
            CandidateRow(
                candidate_id=str(payload.get("candidate_id") or path.stem),
                generation=int(payload.get("generation") or 0),
                score=float(score),
                operator_id=operator_id,
                mutation_mode=mode,
                schema=float(metrics.get("schema_fitness.mean", 0.0)),
                consistency=float(metrics.get("internal_consistency.mean", 0.0)),
                axis_alignment=axis_alignment,
                validation_alignment=validation_alignment,
                shadow_mae=shadow_mae,
                axis_target_mae=axis_target_mae,
                consistency_issue_rate=consistency_issue_rate,
                strict_consistency_error=strict_consistency_error,
                validation_coverage=float(metrics.get("validation_behavior_coverage.mean", 0.0)),
                balanced_diversity=float(metrics.get("validation_behavior_balanced_diversity.mean", 0.0)),
                avg_dist=float(metrics.get("validation_behavior_avg_dist.mean", 0.0)),
            )
        )
    return rows


def summarize_generations(rows: list[CandidateRow]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    best_so_far = float("-inf")
    for generation in sorted({row.generation for row in rows}):
        group = [row for row in rows if row.generation == generation]
        best = max(group, key=lambda row: row.score)
        best_so_far = max(best_so_far, best.score)
        result.append(
            {
                "generation": generation,
                "n": len(group),
                "mean": statistics.mean(row.score for row in group),
                "best": best.score,
                "min": min(row.score for row in group),
                "best_so_far": best_so_far,
                "best_row": best,
            }
        )
    return result


def summarize_operators(rows: list[CandidateRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[CandidateRow]] = defaultdict(list)
    for row in rows:
        grouped[row.operator_id].append(row)
    result = []
    for operator_id, group in grouped.items():
        result.append(
            {
                "operator_id": operator_id,
                "n": len(group),
                "mean": statistics.mean(row.score for row in group),
                "max": max(row.score for row in group),
                "min": min(row.score for row in group),
                "alignment": statistics.mean(row.validation_alignment for row in group),
                "coverage": statistics.mean(row.validation_coverage for row in group),
                "schema": statistics.mean(row.schema for row in group),
            }
        )
    return sorted(result, key=lambda item: item["mean"], reverse=True)


def summarize_modes(rows: list[CandidateRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[CandidateRow]] = defaultdict(list)
    for row in rows:
        grouped[row.mutation_mode].append(row)
    result = []
    for mode, group in grouped.items():
        result.append(
            {
                "mode": mode,
                "n": len(group),
                "mean": statistics.mean(row.score for row in group),
                "max": max(row.score for row in group),
                "min": min(row.score for row in group),
            }
        )
    return sorted(result, key=lambda item: item["mean"], reverse=True)


def count_log_events(log_path: Path) -> dict[str, int]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "ERROR": text.count("ERROR"),
        "Traceback": text.count("Traceback"),
        "LLM mutator failed": text.count("LLM mutator failed"),
        "validation failed; revision attempt": text.count("validation failed; revision attempt"),
        "validation failed after revisions": text.count("validation failed after revisions"),
        "NoneType": text.count("NoneType"),
    }


def first_generation_at_score(generations: list[dict[str, Any]], score: float) -> int:
    for item in generations:
        if abs(item["best_so_far"] - score) < 1e-12:
            return int(item["generation"])
    return int(generations[-1]["generation"])


def line_chart(title: str, series: list[tuple[str, list[tuple[float, float]]]], y_label: str) -> str:
    width, height = 920, 430
    legend_cols = 3
    legend_rows = (len(series) + legend_cols - 1) // legend_cols
    margin = {"l": 64, "r": 24, "t": 66 + legend_rows * 18, "b": 48}
    points = [point for _, values in series for point in values]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        y_min -= 0.01
        y_max += 0.01
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#f97316", "#0891b2"]

    def sx(x: float) -> float:
        if x_max == x_min:
            return margin["l"]
        return margin["l"] + (x - x_min) / (x_max - x_min) * (width - margin["l"] - margin["r"])

    def sy(y: float) -> float:
        return height - margin["b"] - (y - y_min) / (y_max - y_min) * (height - margin["t"] - margin["b"])

    body = [svg_header(width, height, title)]
    body.extend(
        legend_items(
            [(name, colors[idx % len(colors)]) for idx, (name, _) in enumerate(series)],
            x=64,
            y=44,
            col_width=250,
            cols=legend_cols,
        )
    )
    for i in range(5):
        y = margin["t"] + i * (height - margin["t"] - margin["b"]) / 4
        value = y_max - i * (y_max - y_min) / 4
        body.append(f'<line class="gridline" x1="{margin["l"]}" y1="{y:.1f}" x2="{width-margin["r"]}" y2="{y:.1f}"/>')
        body.append(f'<text class="tick" x="8" y="{y+4:.1f}">{value:.3f}</text>')
    body.append(f'<line class="axis" x1="{margin["l"]}" y1="{height-margin["b"]}" x2="{width-margin["r"]}" y2="{height-margin["b"]}"/>')
    body.append(f'<line class="axis" x1="{margin["l"]}" y1="{margin["t"]}" x2="{margin["l"]}" y2="{height-margin["b"]}"/>')
    body.append(f'<text class="tick" x="{width/2:.1f}" y="{height-10}">generation</text>')
    body.append(f'<text class="tick" x="8" y="28">{escape(y_label)}</text>')
    for idx, (name, values) in enumerate(series):
        color = colors[idx % len(colors)]
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
        body.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{path}"/>')
        for x, y in values:
            body.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"><title>{escape(name)} gen {x}: {y:.6f}</title></circle>')
    body.append("</svg>")
    return "".join(body)


def bar_chart(
    title: str,
    values: list[tuple[str, float]],
    baseline: float | None = None,
    zero_center: bool = False,
) -> str:
    width = 920
    bar_h = 24
    margin_l, margin_r, margin_t, margin_b = 250, 72, 64, 32
    height = margin_t + margin_b + max(1, len(values)) * 34
    vals = [value for _, value in values]
    x_min = min(vals + ([baseline] if baseline is not None else []))
    x_max = max(vals + ([baseline] if baseline is not None else []))
    if x_min == x_max:
        x_min -= 0.01
        x_max += 0.01
    if zero_center:
        span = max(abs(x_min), abs(x_max), 0.01)
        x_min, x_max = -span, span
    else:
        x_min = min(0.0, x_min)

    def sx(value: float) -> float:
        return margin_l + (value - x_min) / (x_max - x_min) * (width - margin_l - margin_r)

    body = [svg_header(width, height, title)]
    if baseline is not None:
        x = sx(baseline)
        body.append(f'<line x1="{x:.1f}" y1="{margin_t-8}" x2="{x:.1f}" y2="{height-margin_b}" stroke="#dc2626" stroke-dasharray="4 4"/>')
        body.append(f'<text class="legend" x="{min(x+4, width-86):.1f}" y="{margin_t-18}">baseline</text>')
    for idx, (label, value) in enumerate(values):
        y = margin_t + idx * 34
        x0 = sx(0.0)
        x1 = sx(value)
        rect_x = min(x0, x1)
        w = max(1.0, abs(x1 - x0))
        color = "#2563eb" if baseline is None or value >= baseline else "#94a3b8"
        body.append(f'<text class="tick" x="8" y="{y+17}">{escape(short_label(label, 30))}</text>')
        body.append(f'<rect x="{rect_x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{color}" rx="3"/>')
        label_x = clamp(x1 + 5 if value >= 0 else x1 - 58, margin_l + 4, width - 66)
        body.append(f'<text class="tick" x="{label_x:.1f}" y="{y+16}">{value:+.4f}</text>')
    body.append("</svg>")
    return "".join(body)


def grouped_bar_chart(
    title: str,
    values: list[tuple[str, float, float]],
    labels: tuple[str, str],
    baseline: float | None = None,
) -> str:
    width = 920
    group_h = 44
    bar_h = 15
    margin_l, margin_r, margin_t, margin_b = 270, 72, 72, 32
    height = margin_t + margin_b + max(1, len(values)) * group_h
    vals = [value for _, first, second in values for value in (first, second)]
    if baseline is not None:
        vals.append(baseline)
    x_min = min(0.0, min(vals))
    x_max = max(vals)
    if x_min == x_max:
        x_max += 0.01

    def sx(value: float) -> float:
        return margin_l + (value - x_min) / (x_max - x_min) * (width - margin_l - margin_r)

    body = [svg_header(width, height, title)]
    body.extend(legend_items([(labels[0], "#2563eb"), (labels[1], "#f97316")], x=64, y=44, col_width=150, cols=2))
    if baseline is not None:
        x = sx(baseline)
        body.append(f'<line x1="{x:.1f}" y1="{margin_t-8}" x2="{x:.1f}" y2="{height-margin_b}" stroke="#dc2626" stroke-dasharray="4 4"/>')
        body.append(f'<text class="legend" x="{min(x+4, width-86):.1f}" y="{margin_t-18}">baseline</text>')
    for idx, (label, first, second) in enumerate(values):
        y = margin_t + idx * group_h
        body.append(f'<text class="tick" x="8" y="{y+21}">{escape(short_label(label, 34))}</text>')
        first_w = max(1.0, sx(first) - sx(0.0))
        second_w = max(1.0, sx(second) - sx(0.0))
        body.append(f'<rect x="{sx(0.0):.1f}" y="{y}" width="{first_w:.1f}" height="{bar_h}" fill="#2563eb" rx="3"><title>{escape(labels[0])}: {first:.6f}</title></rect>')
        body.append(f'<rect x="{sx(0.0):.1f}" y="{y+18}" width="{second_w:.1f}" height="{bar_h}" fill="#f97316" rx="3"><title>{escape(labels[1])}: {second:.6f}</title></rect>')
        body.append(f'<text class="tick" x="{clamp(sx(max(first, second))+5, margin_l + 4, width - 62):.1f}" y="{y+30}">{max(first, second):.4f}</text>')
    body.append("</svg>")
    return "".join(body)


def scatter_chart(title: str, rows: list[CandidateRow]) -> str:
    width, height = 920, 410
    margin = {"l": 64, "r": 24, "t": 54, "b": 48}
    x_min, x_max = min(row.generation for row in rows), max(row.generation for row in rows)
    y_min, y_max = min(row.score for row in rows), max(row.score for row in rows)
    pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.01
    y_min -= pad
    y_max += pad

    def sx(x: float) -> float:
        if x_max == x_min:
            return margin["l"]
        return margin["l"] + (x - x_min) / (x_max - x_min) * (width - margin["l"] - margin["r"])

    def sy(y: float) -> float:
        return height - margin["b"] - (y - y_min) / (y_max - y_min) * (height - margin["t"] - margin["b"])

    body = [svg_header(width, height, title)]
    for i in range(5):
        y = margin["t"] + i * (height - margin["t"] - margin["b"]) / 4
        body.append(f'<line class="gridline" x1="{margin["l"]}" y1="{y:.1f}" x2="{width-margin["r"]}" y2="{y:.1f}"/>')
    body.append(f'<line class="axis" x1="{margin["l"]}" y1="{height-margin["b"]}" x2="{width-margin["r"]}" y2="{height-margin["b"]}"/>')
    body.append(f'<line class="axis" x1="{margin["l"]}" y1="{margin["t"]}" x2="{margin["l"]}" y2="{height-margin["b"]}"/>')
    body.append(f'<text class="tick" x="{width/2:.1f}" y="{height-12}">generation</text>')
    body.append(f'<text class="tick" x="8" y="32">fitness</text>')
    for row in rows:
        color = "#2563eb" if row.operator_id != "baseline" else "#dc2626"
        body.append(
            f'<circle cx="{sx(row.generation):.1f}" cy="{sy(row.score):.1f}" r="4" fill="{color}" opacity="0.70">'
            f'<title>{escape(row.candidate_id)} gen {row.generation} score={row.score:.6f} op={escape(row.operator_id)}</title></circle>'
        )
    body.append("</svg>")
    return "".join(body)


def svg_header(width: int, height: int, title: str) -> str:
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'


def legend_items(
    items: list[tuple[str, str]],
    x: int,
    y: int,
    col_width: int,
    cols: int,
) -> list[str]:
    body: list[str] = []
    for idx, (label, color) in enumerate(items):
        col = idx % cols
        row = idx // cols
        lx = x + col * col_width
        ly = y + row * 18
        body.append(f'<rect x="{lx}" y="{ly-9}" width="10" height="10" fill="{color}"/>')
        body.append(f'<text class="legend" x="{lx+16}" y="{ly}">{escape(short_label(label, 28))}</text>')
    return body


def short_label(label: str, max_chars: int) -> str:
    text = label.replace("op", "op ").replace("_v3_", " / ").replace("_", " ")
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def sealed_test_metric_pairs(
    best: CandidateRow,
    final_test: dict[str, Any],
) -> list[tuple[str, float, float]]:
    candidates = [
        ("schema", best.schema, "test_schema_fitness.mean"),
        ("strict consistency error", best.strict_consistency_error, "test_strict_consistency_error.mean"),
        ("axis target MAE", best.axis_target_mae, "test_axis_target_mae.mean"),
        ("behavior coverage", best.validation_coverage, "test_behavior_coverage.mean"),
        ("shadow MAE", best.shadow_mae, "test_shadow_mae.mean"),
        ("balanced diversity", best.balanced_diversity, "test_behavior_balanced_diversity.mean"),
        ("avg behavior dist", best.avg_dist, "test_behavior_avg_dist.mean"),
    ]
    pairs = [
        (label, validation_value, float(final_test[key]))
        for label, validation_value, key in candidates
        if key in final_test and isinstance(final_test[key], (int, float))
    ]
    return pairs or [
        ("behavior coverage", best.validation_coverage, 0.0),
        ("shadow alignment", best.validation_alignment, 0.0),
        ("avg behavior dist", best.avg_dist, 0.0),
    ]


def percent_delta(base: float, value: float) -> float | None:
    if abs(base) < 1e-12:
        return None
    return (value / base - 1.0) * 100.0


def percent_delta_text(base: float, value: float) -> str:
    pct = percent_delta(base, value)
    return "n/a" if pct is None else f"{pct:+.2f}%"


def metric_label(name: str, base: float, value: float) -> str:
    return f"{name} ({percent_delta_text(base, value)})"


def metric_with_pct(base: float, value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f} ({percent_delta_text(base, value)})"


def best_vs_baseline_table(baseline: CandidateRow, best: CandidateRow) -> str:
    rows = [
        ("fitness", baseline.score, best.score),
        ("validation coverage", baseline.validation_coverage, best.validation_coverage),
        ("shadow behavior MAE", baseline.shadow_mae, best.shadow_mae),
        ("axis target MAE", baseline.axis_target_mae, best.axis_target_mae),
        ("consistency issue rate", baseline.consistency_issue_rate, best.consistency_issue_rate),
        ("strict consistency error", baseline.strict_consistency_error, best.strict_consistency_error),
        ("legacy validation alignment", baseline.validation_alignment, best.validation_alignment),
        ("schema fitness", baseline.schema, best.schema),
        ("legacy internal consistency", baseline.consistency, best.consistency),
        ("legacy axis alignment", baseline.axis_alignment, best.axis_alignment),
        ("balanced diversity", baseline.balanced_diversity, best.balanced_diversity),
        ("avg dist", baseline.avg_dist, best.avg_dist),
    ]
    html = ["<table><tr><th>Metric</th><th>Baseline</th><th>Best</th><th>Delta</th><th>Delta %</th></tr>"]
    for name, b, v in rows:
        html.append(
            f"<tr><td>{escape(name)}</td><td>{b:.6f}</td><td>{v:.6f}</td>"
            f"<td>{v-b:+.6f}</td><td>{percent_delta_text(b, v)}</td></tr>"
        )
    html.append("</table>")
    return "".join(html)


def generation_table(items: list[dict[str, Any]], baseline: CandidateRow) -> str:
    html = [
        "<table><tr>"
        "<th>Generation</th><th>N</th><th>Mean</th><th>Best</th><th>Best Δ%</th><th>Best So Far</th>"
        "<th>Best Operator</th><th>Coverage</th><th>Shadow MAE</th><th>Axis MAE</th><th>Issue Rate</th>"
        "<th>Schema</th><th>Legacy Consistency</th><th>Diversity</th><th>Avg Dist</th></tr>"
    ]
    for item in items:
        row = item["best_row"]
        html.append(
            f"<tr><td>gen {item['generation']:02d}</td><td>{item['n']}</td><td>{item['mean']:.6f}</td>"
            f"<td>{item['best']:.6f}</td><td>{percent_delta_text(baseline.score, item['best'])}</td>"
            f"<td>{item['best_so_far']:.6f}</td><td>{escape(row.operator_id)}</td>"
            f"<td>{metric_with_pct(baseline.validation_coverage, row.validation_coverage)}</td>"
            f"<td>{metric_with_pct(baseline.shadow_mae, row.shadow_mae)}</td>"
            f"<td>{metric_with_pct(baseline.axis_target_mae, row.axis_target_mae)}</td>"
            f"<td>{metric_with_pct(baseline.consistency_issue_rate, row.consistency_issue_rate)}</td>"
            f"<td>{metric_with_pct(baseline.schema, row.schema)}</td>"
            f"<td>{metric_with_pct(baseline.consistency, row.consistency)}</td>"
            f"<td>{metric_with_pct(baseline.balanced_diversity, row.balanced_diversity)}</td>"
            f"<td>{metric_with_pct(baseline.avg_dist, row.avg_dist)}</td></tr>"
        )
    html.append("</table>")
    return "".join(html)


def operator_table(items: list[dict[str, Any]], baseline: CandidateRow) -> str:
    html = [
        "<table><tr><th>Operator</th><th>N</th><th>Mean</th><th>Mean Δ</th><th>Mean Δ%</th>"
        "<th>Max</th><th>Max Δ%</th><th>Coverage</th><th>Alignment</th><th>Schema</th></tr>"
    ]
    for item in items:
        html.append(
            f"<tr><td>{escape(item['operator_id'])}</td><td>{item['n']}</td><td>{item['mean']:.6f}</td>"
            f"<td>{item['mean']-baseline.score:+.6f}</td><td>{percent_delta_text(baseline.score, item['mean'])}</td>"
            f"<td>{item['max']:.6f}</td><td>{percent_delta_text(baseline.score, item['max'])}</td>"
            f"<td>{metric_with_pct(baseline.validation_coverage, item['coverage'])}</td>"
            f"<td>{metric_with_pct(baseline.validation_alignment, item['alignment'])}</td>"
            f"<td>{metric_with_pct(baseline.schema, item['schema'])}</td></tr>"
        )
    html.append("</table>")
    return "".join(html)


def mode_table(items: list[dict[str, Any]]) -> str:
    html = ["<table><tr><th>Mode</th><th>N</th><th>Mean</th><th>Max</th><th>Min</th></tr>"]
    for item in items:
        html.append(
            f"<tr><td>{escape(item['mode'])}</td><td>{item['n']}</td><td>{item['mean']:.6f}</td>"
            f"<td>{item['max']:.6f}</td><td>{item['min']:.6f}</td></tr>"
        )
    html.append("</table>")
    return "".join(html)


def dict_table(payload: dict[str, Any]) -> str:
    if not payload:
        return "<p>No data.</p>"
    html = ["<table><tr><th>Key</th><th>Value</th></tr>"]
    for key, value in payload.items():
        html.append(f"<tr><td>{escape(str(key))}</td><td>{escape(fmt(value))}</td></tr>")
    html.append("</table>")
    return "".join(html)


def card(label: str, value: str) -> str:
    return f'<div class="card"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>'


def chart_panel(chart_html: str, title: str, description: str) -> str:
    return (
        '<div class="panel">'
        f"<h3>{escape(title)}</h3>"
        f"{chart_html}"
        f'<p class="caption"><strong>读图方式：</strong>{escape(description)}</p>'
        "</div>"
    )


def metric_glossary() -> str:
    rows = [
        (
            "综合 fitness",
            "旧主优化分数",
            "越高越好",
            "用于和历史实验对比；由 schema、consistency、coverage、alignment、generation_rate 叠乘得到。",
        ),
        (
            "shadow behavior MAE",
            "人格声明轴与 shadow survey 行为轴的平均绝对误差",
            "越低越好",
            "论文主解释指标。比 1-MAE 形式的 alignment 更直观，能避免 baseline 初始分数过高的问题。",
        ),
        (
            "axis target MAE",
            "生成 persona 的 primary axes 与目标 slot axes 的平均绝对误差",
            "越低越好",
            "衡量生成器是否真的按目标轴生成，而不是只看一个容易偏高的 axis_alignment 分数。",
        ),
        (
            "consistency issue rate",
            "每个人格平均触发的一致性问题数",
            "越低越好",
            "衡量跨字段心理机制是否自洽。比 internal_consistency 高分更容易被解释和复核。",
        ),
        (
            "strict consistency error",
            "axis target MAE 与加权一致性问题率的组合误差",
            "越低越好",
            "用于诊断一致性问题，暂不替代历史 fitness。",
        ),
        (
            "behavior coverage",
            "shadow behavior 空间覆盖率",
            "越高越好",
            "衡量生成群体是否覆盖更多行为区域。",
        ),
        (
            "balanced diversity",
            "覆盖、均匀性、平均距离、最小距离的平衡多样性",
            "越高越好",
            "防止只靠少数离群点拉高距离。",
        ),
        (
            "legacy alignment / consistency",
            "历史兼容指标",
            "越高越好",
            "保留作 sanity check；由于初始值常偏高，不建议作为论文主结论。",
        ),
    ]
    html = ["<table><tr><th>指标</th><th>中文意义</th><th>方向</th><th>使用建议</th></tr>"]
    for name, meaning, direction, note in rows:
        html.append(
            f"<tr><td>{escape(name)}</td><td>{escape(meaning)}</td>"
            f"<td>{escape(direction)}</td><td>{escape(note)}</td></tr>"
        )
    html.append("</table>")
    return "".join(html)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    main()
