"""Figures of the report, computed from results/ only (no model, no network).

    uv run --no-project --with matplotlib==3.9.2 --with numpy python analysis/make_figures.py

Writes figures/fig_*.pdf (the figures of the report) and figures/fig_*.png (same drawing, for the README).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
    }
)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = ROOT / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID = "#d9d8d4"
MUTED = "#52514e"
NAMES = {"gemma": "Gemma 4", "nemotron": "Nemotron", "qwen36": "Qwen3.6"}
BANK = {"gemma": 128, "nemotron": 128, "qwen36": 256}
FIRST = {"gemma": 24, "nemotron": 16, "qwen36": 24}


def save(fig: plt.Figure, name: str) -> None:
    """Write the figure as PDF (the report) and as PNG (the README)."""
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def bar_line(ax: plt.Axes) -> None:
    ax.axhline(90, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_budget() -> None:
    """Kept against the share of experts, items 0 to 299, with Lead-3 as a reference on summarize."""
    h31 = load("selrule_h31.json")["models"]
    lead = load("baseline_h50.json")["h50"]["start0"]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.35), sharey=True, constrained_layout=True)
    series = (
        ("sum_rougeL_ref", "Summarize, ROUGE-L", BLUE, "o"),
        ("qa_f1", "QA, F1", ORANGE, "s"),
        ("cls_acc", "Classify", AQUA, "^"),
    )
    for ax, m in zip(axes, NAMES, strict=True):
        ks = sorted(int(k) for k in h31[m])
        share = [100 * k / BANK[m] for k in ks]
        for key, label, color, marker in series:
            y = [100 * h31[m][str(k)][key]["mean"] for k in ks]
            lo = [100 * h31[m][str(k)][key]["lo"] for k in ks]
            hi = [100 * h31[m][str(k)][key]["hi"] for k in ks]
            ax.fill_between(share, lo, hi, color=color, alpha=0.12, linewidth=0)
            ax.plot(share, y, color=color, marker=marker, markersize=4, linewidth=1.4, label=label)
        ax.axhline(
            100 * lead[m]["lead3"]["mean"], color=BLUE, linewidth=0.9, linestyle=":", label="Lead-3, no model, ROUGE-L"
        )
        bar_line(ax)
        ax.set_title(NAMES[m], fontsize=9, color="#0b0b0b")
        ax.set_xlabel("Experts kept per layer (%)")
        ax.set_ylim(0, 112)
    axes[0].set_ylabel("Kept, % of the full model")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    save(fig, "fig_budget")


def fig_readings() -> None:
    """Same answers, three readings: ROUGE-L, Lead-3 and the jury, items 300 to 599."""
    repeat = load("selrule_h31_repeat.json")
    lead = load("baseline_h50.json")["h50"]["start300"]
    jury = load("jury_h44_blind.json")["models"]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.15), sharey=True, constrained_layout=True)
    for ax, m in zip(axes, NAMES, strict=True):
        cells = (
            ("Masked,\nROUGE-L", repeat[m][str(FIRST[m])]["sum_rougeL_ref"], BLUE),
            ("Lead-3,\nROUGE-L", lead[m]["lead3"], "#86b6ef"),
            ("Masked,\njury", jury[m]["kept"]["accepted"], ORANGE),
        )
        for i, (_, c, color) in enumerate(cells):
            mean = 100 * c["mean"]
            err = [[mean - 100 * c["lo"]], [100 * c["hi"] - mean]]
            ax.bar(i, mean, width=0.62, color=color, zorder=2)
            ax.errorbar(i, mean, yerr=err, color="#0b0b0b", linewidth=0.8, capsize=2, zorder=3)
            ax.text(i, 100 * c["hi"] + 2, f"{mean:.1f}", ha="center", va="bottom", fontsize=8, color="#0b0b0b")
        ax.set_xticks(range(3), [c[0] for c in cells], fontsize=8)
        bar_line(ax)
        ax.set_title(f"{NAMES[m]}, K$'$ = {FIRST[m]}", fontsize=9, color="#0b0b0b")
        ax.set_ylim(0, 128)
    axes[0].set_ylabel("Kept, % of the full model")
    save(fig, "fig_readings")


def fig_math() -> None:
    """Math kept against the share of experts."""
    h41 = load("mathrule_h41.json")["models"]
    colors = {"gemma": BLUE, "nemotron": ORANGE, "qwen36": AQUA}
    marks = {"gemma": "o", "nemotron": "s", "qwen36": "^"}
    fig, ax = plt.subplots(figsize=(3.3, 2.3), constrained_layout=True)
    for m in NAMES:
        cells = h41[m]["cells"]
        ks = sorted(int(k) for k in cells)
        x = [100 * k / BANK[m] for k in ks]
        y = [100 * cells[str(k)]["mean"] for k in ks]
        lo = [100 * cells[str(k)]["lo"] for k in ks]
        hi = [100 * cells[str(k)]["hi"] for k in ks]
        ax.fill_between(x, lo, hi, color=colors[m], alpha=0.12, linewidth=0)
        ax.plot(x, y, color=colors[m], marker=marks[m], markersize=4, linewidth=1.4, label=NAMES[m])
    bar_line(ax)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Experts kept per layer (%)")
    ax.set_ylabel("GSM8K accuracy kept (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "fig_math")


def _pct(c: dict) -> tuple[float, float, float]:
    return 100 * c["mean"], 100 * c["lo"], 100 * c["hi"]


def fig_gap() -> None:
    """The same masked summaries read by ROUGE-L and by the jury, against the budget: CNN/DailyMail held-out, XSum."""
    rounds = (
        (
            "CNN/DailyMail, held-out items",
            load("final_heldout.json")["models"],
            load("jury_heldout_h57.json")["models"],
        ),
        ("XSum, first 200", load("final_xsum.json")["models"], None),
    )
    fig, grid = plt.subplots(2, 3, figsize=(6.5, 4.5), sharey=True, sharex=True, constrained_layout=True)
    for row, (name, final, three) in zip(grid, rounds, strict=True):
        row[0].set_ylabel(f"{name}\nKept, % of the full model")
        _gap_row(row, final, three)
    grid[0][1].text(40, 40, "Gap hidden\nby ROUGE-L", fontsize=7.5, color=ORANGE, ha="left")
    for ax in grid[1]:
        ax.set_xlabel("Experts kept per layer (%)")
    h, lab = grid[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    save(fig, "fig_gap")


def _gap_row(axes: list, final: dict, three: dict | None) -> None:
    for ax, m in zip(axes, NAMES, strict=True):
        ks = sorted(int(k) for k in final[m]["cells"])
        x = [100 * k / BANK[m] for k in ks] + [100.0]
        rouge = [_pct(final[m]["cells"][str(k)]["rougeL_kept"]) for k in ks] + [(100.0, 100.0, 100.0)]
        jury = [_pct(final[m]["cells"][str(k)]["accepted_kept"]) for k in ks] + [(100.0, 100.0, 100.0)]
        ax.fill_between(x, [r[0] for r in rouge], [j[0] for j in jury], color=ORANGE, alpha=0.10, linewidth=0)
        ax.fill_between(x, [j[1] for j in jury], [j[2] for j in jury], color=ORANGE, alpha=0.18, linewidth=0)
        ax.plot(x, [r[0] for r in rouge], color=BLUE, marker="o", markersize=4, linewidth=1.4, label="ROUGE-L")
        ax.plot(
            x,
            [j[0] for j in jury],
            color=ORANGE,
            marker="s",
            markersize=4,
            linewidth=1.4,
            label="Two-member jury",
        )
        if three is not None:
            vend = [_pct(three[m][str(k)]["kept"]) for k in ks]
            ax.plot(
                x[:-1],
                [v[0] for v in vend],
                linestyle="none",
                marker="D",
                markersize=4,
                markerfacecolor="white",
                markeredgecolor=ORANGE,
                markeredgewidth=1.0,
                label="Jury, three vendors",
            )
        ax.axhline(
            100 * final[m]["leads"]["lead3"]["mean"],
            color=BLUE,
            linewidth=0.9,
            linestyle=":",
            label="Lead-3, no model, ROUGE-L",
        )
        for xi, (j, _, _) in zip(x[:-1], jury[:-1], strict=True):
            ax.annotate(f"{j:.0f}", (xi, j), xytext=(7, -10), textcoords="offset points", fontsize=7.5, color="#0b0b0b")
        bar_line(ax)
        ax.set_title(NAMES[m], fontsize=9, color="#0b0b0b")
        ax.set_xlim(0, 104)
        ax.set_ylim(0, 115)


def fig_fingerprint() -> None:
    """What the jury rejects: kept on each question, items 300 to 599, first budgets."""
    jury = load("jury_h44_blind.json")["models"]
    rows = (("supported", "Supported"), ("condensed", "Condensed"), ("covers", "Covers"), ("accepted", "Accepted"))
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.55), sharey=True, constrained_layout=True)
    for ax, m in zip(axes, NAMES, strict=True):
        for i, (key, _) in enumerate(rows):
            mean, lo, hi = _pct(jury[m]["kept"][key])
            color = ORANGE if key == "accepted" else BLUE
            ax.plot([lo, hi], [i, i], color=color, linewidth=1.0, alpha=0.6)
            ax.plot(mean, i, "o", color=color, markersize=5)
            ax.text(min(hi, 104) + 3, i, f"{mean:.0f}", va="center", fontsize=7.5, color="#0b0b0b")
        ax.axvline(90, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 118)
        ax.set_title(f"{NAMES[m]}, K$'$ = {FIRST[m]}", fontsize=9, color="#0b0b0b")
        ax.set_xlabel("Kept, % of the full model")
    axes[0].set_yticks(range(len(rows)), [r[1] for r in rows])
    axes[0].invert_yaxis()
    save(fig, "fig_fingerprint")


def fig_lists() -> None:
    """Random, fixed and both-ends lists at the same budget, items 300 to 599."""
    why = load("why_report.json")
    rows = [
        (
            "Gemma 4, K$'$ = 24",
            why["h45"]["models"]["gemma"]["random"],
            why["h55"]["models"]["gemma"]["fixed"],
            why["h55"]["models"]["gemma"]["ends64"],
        ),
        ("Gemma 4, K$'$ = 12", None, why["h60"]["models"]["gemma"]["fixed"], why["h60"]["models"]["gemma"]["ends64"]),
        (
            "Nemotron, K$'$ = 16",
            why["h45"]["models"]["nemotron"]["random"],
            why["h55"]["models"]["nemotron"]["fixed"],
            why["h55"]["models"]["nemotron"]["ends64"],
        ),
        (
            "Qwen3.6, K$'$ = 24",
            why["h45"]["models"]["qwen36"]["random"],
            why["h55"]["models"]["qwen36"]["fixed"],
            why["h55"]["models"]["qwen36"]["ends64"],
        ),
    ]
    tasks = (("sum_rougeL_ref", "Summarize, ROUGE-L"), ("qa_f1", "QA, F1"), ("cls_acc", "Classify"))
    kinds = (("Random list", "#9a9993", "o"), ("Fixed list", ORANGE, "s"), ("Both-ends rule", BLUE, "D"))
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.1), sharey=True, constrained_layout=True)
    for ax, (key, title) in zip(axes, tasks, strict=True):
        for i, (_, *lists) in enumerate(rows):
            vals = [100 * lst[key]["mean"] if lst else None for lst in lists]
            present = [v for v in vals if v is not None]
            ax.plot([min(present), max(present)], [i, i], color=GRID, linewidth=2.2, zorder=1)
            for v, (lab, color, marker) in zip(vals, kinds, strict=True):
                if v is not None:
                    ax.plot(v, i, marker, color=color, markersize=5, zorder=3, label=lab if i == 0 else None)
        ax.axvline(90, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(-3, 108)
        ax.set_title(title, fontsize=9, color="#0b0b0b")
        ax.set_xlabel("Kept, % of the full model")
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[0].invert_yaxis()
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    save(fig, "fig_lists")


def fig_alwayson() -> None:
    """A 0.75 cut to the always-on path against the same cut to the routed experts, masked and unmasked."""
    why = load("why_report.json")
    colors = {"gemma": BLUE, "nemotron": ORANGE, "qwen36": AQUA}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.35), constrained_layout=True)
    for ax, (meas, title) in zip(
        axes, (("loops", "Looping summaries (%)"), ("at_cap", "Summaries at the cap (%)")), strict=True
    ):
        for m in NAMES:
            r = why["h43"]["models"][m]["rates"]
            ax.plot(
                [0, 1],
                [100 * r["intact_u"][meas], 100 * r["abl_u"][meas]],
                color="#9a9993",
                linewidth=0.9,
                linestyle=(0, (2, 2)),
                marker="o",
                markersize=3,
            )
            a, b = 100 * r["intact_m"][meas], 100 * r["abl_m"][meas]
            ax.plot([0, 1], [a, b], color=colors[m], linewidth=1.6, marker="o", markersize=4)
            ax.annotate(
                f"{NAMES[m]} {b:.0f}",
                (1, b),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color="#0b0b0b",
            )
            if m in why["h56"]["models"]:
                c = 100 * why["h56"]["models"][m]["rates"]["cut_m"][meas]
                ax.plot(
                    [0, 1],
                    [a, c],
                    color=colors[m],
                    linewidth=1.0,
                    linestyle=(0, (4, 2)),
                    marker="o",
                    markersize=4,
                    markerfacecolor="white",
                )
        ax.set_xticks([0, 1], ["Intact", "Cut to 0.75"])
        ax.set_xlim(-0.15, 1.55)
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=9, color="#0b0b0b")
    handles = [
        plt.Line2D([], [], color="#0b0b0b", linewidth=1.6, marker="o", markersize=4, label="Masked, always-on cut"),
        plt.Line2D(
            [],
            [],
            color="#0b0b0b",
            linewidth=1.0,
            linestyle=(0, (4, 2)),
            marker="o",
            markersize=4,
            markerfacecolor="white",
            label="Masked, routed cut",
        ),
        plt.Line2D(
            [],
            [],
            color="#9a9993",
            linewidth=0.9,
            linestyle=(0, (2, 2)),
            marker="o",
            markersize=3,
            label="Full model, always-on cut",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    save(fig, "fig_alwayson")


def fig_teaser() -> None:
    """The main finding: at the same short list, short answers survive and long answers break."""
    final = load("final_heldout.json")
    math = load("mathrule_h41.json")["models"]
    colors = {"gemma": BLUE, "nemotron": ORANGE, "qwen36": AQUA}
    marks = {"gemma": "o", "nemotron": "s", "qwen36": "^"}
    caps = [8, 32, 120, 512]
    fig, ax = plt.subplots(figsize=(6.5, 2.6), constrained_layout=True)
    for i, m in enumerate(NAMES):
        k = str(FIRST[m])
        cells = [
            final["suite"][m]["cls_acc"],
            final["suite"][m]["qa_f1"],
            final["models"][m]["cells"][k]["accepted_kept"],
            math[m]["cells"][k],
        ]
        x = [c * (1 + 0.06 * (i - 1)) for c in caps]
        y = [100 * c["mean"] for c in cells]
        err = [[100 * (c["mean"] - c["lo"]) for c in cells], [100 * (c["hi"] - c["mean"]) for c in cells]]
        ax.errorbar(
            x,
            y,
            yerr=err,
            color=colors[m],
            marker=marks[m],
            markersize=5,
            linewidth=1.5,
            capsize=2,
            label=f"{NAMES[m]}, {100 * FIRST[m] / BANK[m]:.1f}% of experts",
        )
        rouge = 100 * final["models"][m]["cells"][k]["rougeL_kept"]["mean"]
        ax.plot(
            x[2],
            rouge,
            marker=marks[m],
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=colors[m],
            linestyle="none",
        )
    ax.annotate(
        "same summaries,\nscored by ROUGE-L",
        (125, 96),
        xytext=(170, 60),
        fontsize=7.5,
        color=MUTED,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.7},
    )
    bar_line(ax)
    ax.set_xscale("log")
    ax.set_xticks(caps, ["Classify\n8 tokens", "QA\n32 tokens", "Summarize, judged\n120 tokens", "Math\n512 tokens"])
    ax.minorticks_off()
    ax.set_xlim(6, 700)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Kept, % of the full model")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    save(fig, "fig_teaser")


def _toy_layer() -> tuple[list[list[float]], list[set[int]], list[int]]:
    """Made-up router weights for one small layer, the lists they give, and the prompt columns that vote.

    Same numbers as the interactive drawing in walkthrough/: 24 experts, a list of 6, ends of 4 tokens, refresh every 4.
    """
    import math

    kinds = {
        "tmpl": [(0, 3.1), (1, 2.7)],
        "instr": [(2, 3.2), (3, 2.8), (4, 2.5)],
        "art": [(8, 3.1), (9, 2.8), (10, 2.6), (11, 2.3)],
        "num": [(12, 3.2), (9, 2.6)],
        "ques": [(5, 3.1), (6, 2.8), (2, 2.2)],
        "a": [(5, 3.0), (8, 2.9), (13, 2.6)],
        "b": [(14, 3.1), (15, 2.8), (8, 2.3)],
        "c": [(16, 3.1), (17, 2.8), (14, 2.3)],
    }
    seq = ["tmpl"] + ["instr"] * 3 + ["art"] * 4 + ["num", "ques", "ques", "tmpl"]
    seq += ["a", "a", "b", "a", "num", "b", "b", "c", "c", "c", "c", "tmpl"]
    weights = []
    for c, kind in enumerate(seq):
        logits = [math.sin((c + 1) * 12.9898 + (e + 1) * 78.233) * 43758.5453 % 1 * 0.9 for e in range(24)]
        for e, bonus in kinds[kind]:
            logits[e] += bonus
        z = [math.exp(v) for v in logits]
        weights.append([v / sum(z) for v in z])

    def pick(cols: list[int]) -> set[int]:
        total = [sum(weights[c][e] for c in cols) for e in range(24)]
        return set(sorted(range(24), key=lambda e: -total[e])[:6])

    ends = [0, 1, 2, 3, 8, 9, 10, 11]
    lists = [pick(ends), pick([12, 13, 14, 15]), pick([16, 17, 18, 19])]
    return weights, lists, ends


def fig_method() -> None:
    """One small layer over a whole run: which experts are in memory, which tokens pick them, what each token uses."""
    weights, lists, ends = _toy_layer()
    prompt = ["<user>", "Answer", "in", "brief", "The", "bridge", "opened", "in", "1932", "When", "opened", "?"]
    answer = ["The", "bridge", "opened", "in", "1932", ",", "and", "it", "still", "carries", "trains", "."]
    n_exp, n_prompt = 24, 12
    wmax = max(max(row) for row in weights)
    ink = "#0b0b0b"

    def x_of(c: int) -> float:
        return c if c < n_prompt else c + 0.8 + 0.35 * ((c - n_prompt) // 4)

    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    ax.set_xlim(-1.2, x_of(23) + 1.1)
    ax.set_ylim(n_exp + 3.6, -7.6)
    ax.axis("off")
    lo, hi = (0.925, 0.937, 0.957), (0.153, 0.220, 0.310)
    dots: dict[str, list[tuple[float, float]]] = {BLUE: [], ORANGE: [], "ring": []}
    for c in range(24):
        allowed = lists[0 if c < n_prompt else (c - n_prompt) // 4]
        free = sorted(range(n_exp), key=lambda e: -weights[c][e])[:2]
        used = sorted(allowed, key=lambda e: -weights[c][e])[:2]
        for e in range(n_exp):
            s = 0.06 + 0.84 * weights[c][e] / wmax
            face = tuple(u + (v - u) * s for u, v in zip(lo, hi, strict=True))
            ax.add_patch(plt.Rectangle((x_of(c) + 0.05, e + 0.07), 0.9, 0.86, facecolor=face, edgecolor="none"))
            if e in used:
                dots[BLUE if c < n_prompt else ORANGE].append((x_of(c) + 0.5, e + 0.5))
            elif e in free:
                dots["ring"].append((x_of(c) + 0.5, e + 0.5))
        word = prompt[c] if c < n_prompt else answer[c - n_prompt]
        ax.text(x_of(c) + 0.5, -1.0, word, rotation=90, ha="center", va="bottom", fontsize=6.5, color=ink)
    for color in (BLUE, ORANGE):
        xs, ys = zip(*dots[color], strict=True)
        ax.scatter(xs, ys, s=11, color=color, edgecolors="white", linewidths=0.4, zorder=3)
    xs, ys = zip(*dots["ring"], strict=True)
    ax.scatter(xs, ys, s=11, facecolors="none", edgecolors=ink, linewidths=0.8, zorder=3)
    # experts in memory: one frame per expert on a list, over the tokens that the list serves
    spans = [[(0, 11), (12, 15)], [(16, 19)], [(20, 23)]]
    for parts, members, color in zip(spans, lists, (BLUE, ORANGE, ORANGE), strict=True):
        for e in members:
            for c0, c1 in parts:
                width = x_of(c1) + 1 - x_of(c0)
                ax.add_patch(plt.Rectangle((x_of(c0), e), width, 1, facecolor="none", edgecolor=color, linewidth=0.9))
    # the tokens that pick each list
    for cols, color in ((ends[:4], BLUE), (ends[4:], BLUE), ([12, 13, 14, 15], ORANGE), ([16, 17, 18, 19], ORANGE)):
        xs = [x_of(cols[0]) + 0.1, x_of(cols[-1]) + 0.9]
        ax.plot(xs, [-0.5, -0.5], color=color, linewidth=2.2, solid_capstyle="butt")
    ax.text(6, -7.3, "Prefill: the prompt, in one pass", ha="center", va="top", fontsize=7.5, color=MUTED)
    decode = "Decode: the answer, one token at a time"
    ax.text(x_of(17) + 1.2, -7.3, decode, ha="center", va="top", fontsize=7.5, color=MUTED)
    for x0, x1, name, color in (
        (0, 16.8, "list 0 in memory", BLUE),
        (17.15, 21.15, "list 1", ORANGE),
        (21.5, 25.5, "list 2", ORANGE),
    ):
        ax.text((x0 + x1) / 2, n_exp + 0.4, name, ha="center", va="top", fontsize=7, color=color)
    ax.text(-0.3, n_exp / 2, "24 experts", rotation=90, ha="right", va="center", fontsize=7, color=MUTED)
    # legend
    y = n_exp + 2.7
    for k, s in enumerate((0.1, 0.4, 0.7, 0.9)):
        face = tuple(u + (v - u) * s for u, v in zip(lo, hi, strict=True))
        ax.add_patch(plt.Rectangle((0.05 + 0.6 * k, y - 0.43), 0.55, 0.86, facecolor=face, edgecolor="none"))
    ax.text(2.7, y, "router weight, low to high", va="center", fontsize=6.5, color=MUTED)
    ax.plot([9.3, 10.3], [y, y], color=BLUE, linewidth=2.2, solid_capstyle="butt")
    ax.text(10.6, y, "tokens that pick a list", va="center", fontsize=6.5, color=MUTED)
    ax.add_patch(plt.Rectangle((16.2, y - 0.5), 1.0, 1, facecolor="none", edgecolor=BLUE, linewidth=0.9))
    ax.text(17.5, y, "in memory", va="center", fontsize=6.5, color=MUTED)
    ax.scatter([20.6], [y], s=11, color=ORANGE, edgecolors="white", linewidths=0.4)
    ax.text(20.95, y, "used", va="center", fontsize=6.5, color=MUTED)
    ax.scatter([22.9], [y], s=11, facecolors="none", edgecolors=ink, linewidths=0.8)
    ax.text(23.25, y, "wanted, not on the list", va="center", fontsize=6.5, color=MUTED)
    save(fig, "fig_method")


if __name__ == "__main__":
    fig_budget()
    fig_readings()
    fig_math()
    fig_gap()
    fig_fingerprint()
    fig_lists()
    fig_alwayson()
    fig_teaser()
    fig_method()
