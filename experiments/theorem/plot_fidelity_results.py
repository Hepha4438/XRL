#!/usr/bin/env python3
"""
plot_fidelity_results.py -- Figures and Table 5 for the Soft--Hard Fidelity theorem.

Reads the outputs of verify_fidelity.py (theorem2_analytics.npz + theorem2_summary.json).
Needs only numpy + matplotlib (no torch).

--results_file accepts, for each environment:
  * .../theorem2_analytics.npz
  * .../theorem2_analytics.pt   (old path; the sibling .npz is used)
  * the results directory itself

Single environment:
  python experiments/theorem/plot_fidelity_results.py \
      --results_file ./experiments/theorem/results_pong/theorem2_analytics.npz \
      --output_folder ./experiments/theorem/results_pong/figures

Several environments (per-env figures in sub-folders + combined Table 5 / overview):
  python experiments/theorem/plot_fidelity_results.py \
      --results_file ./experiments/theorem/results_{dyobs,doorkey,cartpole,pong,boxing} \
      --output_folder ./experiments/theorem/figures_all

Per environment:
  fig1_scatter_gamma_delta.png   gamma(x) vs Delta_total(x) on X_trig
  fig2_pairwise_slack.png        distribution of the pairwise certificate slack
  fig3_binarization_hist.png     c_soft histogram (all / rule concepts) with theta
  fig4_delta_decomposition.png   true-clause drag vs false-clause residual (cert vs uncert)
  fig5_theta_sweep.png           coverage / agreement / AF_hard vs concept threshold theta
  fig6_triangle_bound.png        teacher-level triangle inequality
  fig7_fragile_concepts.png      concepts contributing most to Delta on uncertified states
  fig8_bounds.png                actual disagreement vs Theorem 1 / Corollary bounds (at --clean_eps)
  fig9_eps_sweep.png             Corollary bounds as a function of eps_bar
Combined (also written for a single environment):
  table5_fidelity.tex            main Table 5: clean states, ties, actual disagreement, 3 bounds
  table_fidelity_details.tex     appendix: coverage, covered agreement, implementation checks,
                                 tau/theta invariance counts, teacher triangle bound
  fig_overview_bounds.png        (several environments only)
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENV_ORDER = ["Dynamic-Obs-5x5", "DoorKey-6x6", "PixelCartPole", "Pong", "Boxing"]


# ============================================================================
# Loading
# ============================================================================

def resolve(path):
    if os.path.isdir(path):
        path = os.path.join(path, "theorem2_analytics.npz")
    elif path.endswith(".pt"):
        path = path[:-3] + ".npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Re-run the new verify_fidelity.py "
                                f"(old .pt analytics are not compatible).")
    return path


def load(path):
    npz_path = resolve(path)
    arr = dict(np.load(npz_path, allow_pickle=False))
    js = os.path.join(os.path.dirname(npz_path), "theorem2_summary.json")
    with open(js) as f:
        summary = json.load(f)
    if "corollary" not in summary:
        raise ValueError(f"{js} was produced by an older verify_fidelity.py (no Cleanliness corollary). Re-run it.")
    return arr, summary


def b(arr, k):
    return arr[k].astype(bool)


# ============================================================================
# Per-environment figures
# ============================================================================

def fig_scatter(arr, s, out, max_points=20000, log=False, seed=0):
    trig = b(arr, "trig")
    g = arr["gamma"][trig]; d = arr["delta_total"][trig]
    agree = b(arr, "agree")[trig]; cert = b(arr, "cert")[trig]; tie = b(arr, "tie")[trig]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(g))
    keep = idx[agree]
    if len(keep) > max_points:
        keep = rng.choice(keep, max_points, replace=False)
    keep = np.concatenate([keep, idx[~agree]])          # always show every disagreement

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    m1 = np.isin(idx, keep) & agree & cert
    m2 = np.isin(idx, keep) & agree & ~cert
    m3 = ~agree
    ax.scatter(g[m2], d[m2], s=8, alpha=0.25, c="tab:blue", label=r"agree, not certified")
    ax.scatter(g[m1], d[m1], s=8, alpha=0.25, c="tab:green", label=r"agree, certified ($x\in\mathcal{X}_{cert}$)")
    ax.scatter(g[m3], d[m3], s=22, alpha=0.9, c="tab:red", marker="x", label=r"$\pi^{soft}\neq\pi^{hard}$")

    hi = max(float(np.max(g)) if len(g) else 1.0, float(np.max(d)) if len(d) else 1.0) * 1.05
    lo = 1e-4 if log else 0.0
    xs = np.linspace(lo, hi, 200) if not log else np.logspace(np.log10(lo), np.log10(hi), 200)
    ax.plot(xs, xs, "k--", lw=1, label=r"$\Delta_{total}=\gamma$")
    ax.fill_between(xs, lo, xs, color="green", alpha=0.07, label=r"$\Delta_{total}<\gamma$ (sufficient)")
    if log:
        ax.set_xscale("symlog", linthresh=1e-3); ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlim(left=lo); ax.set_ylim(bottom=lo)
    ax.set_xlabel(r"Hard action margin $\gamma(x)$", fontsize=12)
    ax.set_ylabel(r"$\Delta_{total}(x)=\Delta_{a^*}(x)+\max_{a\neq a^*}\Delta_a(x)$", fontsize=12)
    ax.set_title(f"{s['env_short']}: margin vs. perturbation on $\\mathcal{{X}}_{{trig}}$ "
                 f"({tie.sum()} ties at $\\gamma=0$)", fontsize=12)
    txt = (f"coverage (pairwise) {s['fidelity_coverage_pairwise']:.2f}%\n"
           f"coverage (simple)   {s['fidelity_coverage_simple']:.2f}%\n"
           f"covered agreement   {s['covered_agreement_pairwise']:.2f}%\n"
           f"overall agreement   {s['overall_soft_hard_agreement']:.2f}%")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom", family="monospace",
            fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(os.path.join(out, "fig1_scatter_gamma_delta.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_slack(arr, s, out):
    valid = b(arr, "trig") & ~b(arr, "tie")
    sl = arr["pair_slack"][valid]; ag = b(arr, "agree")[valid]
    if len(sl) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lo, hi = np.percentile(sl, 0.5), np.percentile(sl, 99.5)
    bins = np.linspace(min(lo, -1e-3), max(hi, 1e-3), 120)
    ax.hist([np.clip(sl[ag], bins[0], bins[-1]), np.clip(sl[~ag], bins[0], bins[-1])], bins=bins,
            stacked=True, color=["tab:green", "tab:red"], label=["agree", "disagree"], log=True)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel(r"pairwise slack $\min_{a\neq a^*}\,[(s^{hard}_{a^*}-s^{hard}_a)-(\Delta_{a^*}+\Delta_a)]$")
    ax.set_ylabel("#states (log)")
    ax.set_title(f"{s['env_short']}: certified iff slack > 0 (ties excluded)")
    ax.legend(); ax.grid(alpha=0.2)
    fig.savefig(os.path.join(out, "fig2_pairwise_slack.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_binarization(arr, s, out):
    edges = arr["hist_edges"]; centers = 0.5 * (edges[:-1] + edges[1:]); w = edges[1] - edges[0]
    panels = [("all concepts", arr["hist_all"], s["binarization_cleanliness_all"])]
    if arr["hist_rule"].sum() > 0:
        panels.append(("rule concepts", arr["hist_rule"], s["binarization_cleanliness_rule"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.3), squeeze=False)
    for ax, (name, h, clean) in zip(axes[0], panels):
        ax.bar(centers, np.maximum(h, 0.5), width=w, color="purple", alpha=0.75, log=True)
        ax.axvspan(0, 0.05, color="green", alpha=0.1); ax.axvspan(0.95, 1, color="green", alpha=0.1)
        ax.axvline(s["theta"], color="red", ls="--", lw=1.2, label=rf"hard_threshold $\theta={s['theta']:.2f}$")
        ax.set_xlabel(r"$c^{soft}_j(x)$"); ax.set_ylabel("count (log)")
        ax.set_title(f"{s['env_short']} -- {name}: clean {clean:.2f}%")
        ax.legend(loc="upper center"); ax.grid(alpha=0.2)
    fig.savefig(os.path.join(out, "fig3_binarization_hist.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_decomposition(arr, s, out):
    trig = b(arr, "trig"); cert = b(arr, "cert")
    groups = [("certified", trig & cert), ("uncertified", trig & ~cert)]
    comps = [("dstar_true", r"$\Delta_{a^*}$ true-clause drag", "tab:blue"),
             ("dstar_false", r"$\Delta_{a^*}$ false-clause residual", "tab:cyan"),
             ("dother_true", r"$\max\Delta_a$ true-clause drag", "tab:orange"),
             ("dother_false", r"$\max\Delta_a$ false-clause residual", "gold")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(groups)); bottom = np.zeros(len(groups))
    for key, lab, col in comps:
        vals = np.array([arr[key][m].mean() if m.any() else 0.0 for _, m in groups])
        ax.bar(x, vals, bottom=bottom, color=col, label=lab, width=0.55)
        bottom += vals
    gam = [arr["gamma"][m].mean() if m.any() else 0.0 for _, m in groups]
    ax.scatter(x, gam, marker="D", color="k", zorder=5, label=r"mean $\gamma(x)$")
    ax.set_xticks(x); ax.set_xticklabels([f"{n}\n(n={m.sum()})" for n, m in groups])
    ax.set_ylabel(r"mean $\Delta_{total}$ components")
    ax.set_title(f"{s['env_short']}: where does the perturbation come from?")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.2, axis="y")
    fig.savefig(os.path.join(out, "fig4_delta_decomposition.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_theta_sweep(arr, s, out):
    th = arr["sweep_theta"]
    if len(th) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(th, arr["sweep_cov_pairwise"], "o-", label="coverage (pairwise)")
    ax.plot(th, arr["sweep_cov_simple"], "s--", label=r"coverage ($\Delta_{total}<\gamma$)")
    ax.plot(th, arr["sweep_agreement"], "^-", label=r"$\pi^{soft}=\pi^{hard}$")
    if np.isfinite(arr["sweep_af_hard"]).any():
        ax.plot(th, arr["sweep_af_hard"], "v-", label=r"AF$_{hard}$ (vs teacher)")
    ax.plot(th, arr["sweep_tie"], ":", color="0.4", label=r"ties ($\gamma=0$)")
    ax.axvline(s["theta"], color="red", ls="--", lw=1, label=rf"used hard_threshold $\theta={s['theta']:.2f}$")
    ax.set_xlabel(r"hard_threshold $\theta$  ($c^{hard}=\mathbf{1}[c^{soft}>\theta]$)")
    ax.set_ylabel("%"); ax.set_ylim(-2, 102)
    ax.set_title(f"{s['env_short']}: sensitivity to hard_threshold $\\theta$ (threshold $\\tau={s['tau']}$ fixed)")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.savefig(os.path.join(out, "fig5_theta_sweep.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_triangle(s, out):
    t = s.get("teacher")
    if not t:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = [r"$1-\mathrm{AF}_{hard}$", "imitation\n+ discretization", "imitation\n+ (1 - coverage)"]
    ax.bar(0, t["hard_error"], color="tab:red", width=0.6)
    ax.bar(1, t["imitation_error"], color="tab:blue", width=0.6, label=r"imitation $1-\mathrm{AF}_{soft}$")
    ax.bar(1, t["discretization_error"], bottom=t["imitation_error"], color="tab:orange", width=0.6,
           label=r"$\Pr[\pi^{soft}\neq\pi^{hard}]$")
    ax.bar(2, t["imitation_error"], color="tab:blue", width=0.6)
    ax.bar(2, 100 - s["fidelity_coverage_pairwise"], bottom=t["imitation_error"], color="0.6", width=0.6,
           label=r"$1-\Pr[\mathcal{X}_{cert}]$")
    for i, v in enumerate([t["hard_error"], t["triangle_middle"], t["triangle_bound"]]):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels)
    ax.set_ylabel("error (%)")
    ax.set_title(f"{s['env_short']}: teacher-level triangle bound")
    ax.legend(fontsize=8); ax.grid(alpha=0.2, axis="y")
    fig.savefig(os.path.join(out, "fig6_triangle_bound.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_fragile(s, out, top=15):
    fc = s.get("fragile_concepts", [])[:top]
    if not fc:
        return
    names = [f"f_{c['concept']}" + ("" if c["in_rules"] else "*") for c in fc]
    fal = [c["false_clause_delta"] for c in fc]; drag = [c["true_clause_drag"] for c in fc]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(fc) + 1.5))
    y = np.arange(len(fc))[::-1]
    ax.barh(y, fal, color="tab:cyan", label="false-clause residual (via $j^*$)")
    ax.barh(y, drag, left=fal, color="tab:blue", label=r"true-clause drag $-\log(\ell+\epsilon)$")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("accumulated contribution on uncertified states")
    ax.set_title(f"{s['env_short']}: concepts driving uncertified states (* = not in any rule)")
    ax.legend(fontsize=8); ax.grid(alpha=0.2, axis="x")
    fig.savefig(os.path.join(out, "fig7_fragile_concepts.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Combined outputs
# ============================================================================

def fig_bounds(s, out):
    """Actual soft/hard disagreement vs the three upper bounds (log scale)."""
    c = s["corollary"]
    labels = ["actual\n$\\Pr[\\pi^{soft}\\neq\\pi^{hard}]$", "Theorem 1\n$1-\\Pr[\\mathcal{X}_{cert}]$",
              "Corollary\n(state-wise)", "Corollary\n(global)"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(0, max(c["actual_disagreement"], 1e-3), color="tab:red", width=0.6)
    ax.bar(1, max(c["bound_theorem"], 1e-3), color="tab:blue", width=0.6)
    for x, key, unc in ((2, "bound_statewise", "clean_valid_uncertified_statewise"),
                        (3, "bound_global", "clean_valid_uncertified_global")):
        parts = [(c["not_clean"], "0.55", r"not $\bar{\epsilon}$-clean"),
                 (c["clean_but_tie_or_fallback"], "0.8", "clean, tie/fallback"),
                 (c[unc], "tab:orange", r"clean, $\gamma\leq\bar{\Delta}$")]
        bottom = 0.0
        for v, col, lab in parts:
            ax.bar(x, v, bottom=bottom, color=col, width=0.6, label=lab if x == 2 else None)
            bottom += v
    for x, v in enumerate([c["actual_disagreement"], c["bound_theorem"], c["bound_statewise"], c["bound_global"]]):
        ax.text(x, max(v, 1e-3) * 1.15, f"{v:.2f}%", ha="center", fontsize=9)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 250)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% of held-out states (log)")
    ax.set_title(f"{s['env_short']}: soft/hard disagreement and its upper bounds "
                 f"($\\bar{{\\epsilon}}={c['clean_eps']}$, clean states {c['clean_state_rate']:.1f}%)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0)); ax.grid(alpha=0.2, axis="y")
    fig.savefig(os.path.join(out, "fig8_bounds.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_eps_sweep(s, out):
    """Corollary bound vs eps_bar: fewer clean states vs a tighter per-clause bound."""
    g = s["corollary"]["eps_grid"]
    if len(g) < 2:
        return
    e = np.array([r["eps_bar"] for r in g])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(e, [100 - r["clean_state_rate"] for r in g], "o--", color="0.5", label="not clean")
    ax.plot(e, [r["bound_statewise"] for r in g], "s-", color="tab:orange", label="bound, state-wise")
    ax.plot(e, [r["bound_global"] for r in g], "^-", color="tab:purple", label="bound, global")
    ax.axhline(s["corollary"]["actual_disagreement"], color="tab:red", lw=1.2, label="actual disagreement")
    ax.axhline(s["corollary"]["bound_theorem"], color="tab:blue", lw=1, ls=":", label="bound, Theorem 1")
    ax.axvline(s["corollary"]["e_minus_bmax"], color="k", lw=0.8, ls="--", label=r"$e^{-b_{\max}}$")
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.1); ax.set_ylim(0, 105)
    ax.set_xlabel(r"cleanliness level $\bar{\epsilon}$"); ax.set_ylabel("% of states")
    ax.set_title(f"{s['env_short']}: Cleanliness corollary vs. $\\bar{{\\epsilon}}$")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.savefig(os.path.join(out, "fig9_eps_sweep.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _thr_caption(summaries):
    taus = sorted({s["tau"] for s in summaries}); thetas = sorted({s["theta"] for s in summaries})
    return (f"selector threshold $\\tau={', '.join(map(str, taus))}$, concept threshold "
            f"$\\theta={', '.join(map(str, thetas))}$")


def latex_table(summaries):
    """Main Table 5: actual disagreement vs Theorem 1 and the Cleanliness corollary (all in %)."""
    rows = []
    for s in summaries:
        c = s["corollary"]
        mark = "$^{\\ast}$" if c["global_condition_holds_on_data"] else ""
        bs, bg = c["best_statewise"], c["best_global"]
        clean_at = next(r["clean_state_rate"] for r in c["eps_grid"] if r["eps_bar"] == bs["eps_bar"])
        rows.append(f"\\textbf{{{s['env_short']}}} & {clean_at:.2f} & {s['tie_rate']:.2f} & "
                    f"{c['actual_disagreement']:.2f} & {c['bound_theorem']:.2f} & "
                    f"{bs['bound']:.2f} ({bs['eps_bar']:g}) & {bg['bound']:.2f} ({bg['eps_bar']:g}){mark} \\\\")
    return "\n".join([
        "\\begin{table}[tp]",
        "\\centering",
        "\\caption{Soft--hard disagreement on held-out rollouts and its upper bounds, in \\% of states "
        f"({_thr_caption(summaries)}). \\emph{{Clean}}: states whose rule concepts all satisfy "
        "$\\min(c_j,1-c_j)<\\bar\\epsilon$, at the $\\bar\\epsilon$ of the state-wise bound. Corollary bounds are the "
        "minimum over a fixed grid of $\\bar\\epsilon$ (value in parentheses); each grid point is a valid bound. "
        "\\emph{Ties}: $\\gamma(x)=0$. Theorem~\\ref{thm:fidelity} uses the "
        "exact soft activations; the corollary uses only the hard truth pattern, cleanliness and "
        "data-independent constants (state-wise: strongest violated literal; global: worst case per clause). "
        "$^{\\ast}$: the smallest non-zero hard margin in the data exceeds $\\max_{a\\neq a'}(\\bar\\Delta^G_a+\\bar\\Delta^G_{a'})$, "
        "so the global bound reduces to $\\Pr[\\text{not clean}]+\\Pr[\\text{tie}]$.}",
        "\\label{tab:fidelity_validation}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Environment}} & \\textbf{Clean} & \\textbf{Ties} & "
        "\\textbf{Actual} & \\textbf{Bound} & \\multicolumn{2}{c}{\\textbf{Bound, Corollary}} \\\\",
        "\\cmidrule(lr){6-7}",
        " & \\textbf{states} & $\\gamma=0$ & $\\Pr[\\pi^{\\mathrm{soft}}\\neq\\pi^{\\mathrm{hard}}]$ & "
        "\\textbf{Thm.~\\ref{thm:fidelity}} & state-wise & global \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
    ])


def latex_table_appendix(summaries):
    """Appendix: implementation checks, certificate details and threshold invariance."""
    rows = []
    for s in summaries:
        ch, cc, iv = s["checks"], s["corollary"]["checks"], s["invariance"]
        t = s.get("teacher")
        tri = f"${t['hard_error']:.2f} \\le {t['triangle_bound']:.2f}$" if t else "--"
        rows.append(
            f"\\textbf{{{s['env_short']}}} & {s['n_states']} & {s['binarization_cleanliness_all']:.2f} & "
            f"{s['gamma_on_trig']['median']:.2f} & {s['delta_total_on_trig']['median']:.2f} & "
            f"{s['fidelity_coverage_simple']:.2f} & {s['fidelity_coverage_pairwise']:.2f} & "
            f"{s['covered_agreement_pairwise']:.2f} & "
            f"{ch['lemma1_violations']}/{ch['lemma2_violations']}/{cc['dbar_violations']} & "
            f"{iv['selectors_in_range']} & {iv['states_theta_sensitive']} & {tri} \\\\")
    lo, hi = summaries[0]["invariance"]["range"]
    return "\n".join([
        "\\begin{table}[tp]",
        "\\centering",
        f"\\caption{{Details of the fidelity certificate ({_thr_caption(summaries)}). Cleanliness is the fraction of "
        "concept activations with $\\min(c,1-c)<0.05$; $\\gamma$ and $\\Delta_{\\text{total}}$ are medians over "
        "$\\mathcal{X}_{\\mathrm{trig}}$; coverage and covered agreement in \\%. \\emph{Violations}: "
        "(state, clause) pairs breaking Lemma~\\ref{lem:clause_error} / (state, action) pairs breaking "
        "Lemma~\\ref{lem:score_perturbation} / clean (state, clause) pairs with $\\Delta>\\bar\\Delta$ -- implementation "
        f"checks, all must be $0$. \\emph{{Sel.}}: selectors with $p$ or $n$ in $({lo},{hi}]$ (0 $\\Rightarrow$ identical "
        f"rules for every $\\tau\\in[{lo},{hi}]$). \\emph{{$\\theta$-sens.}}: states with a rule-concept activation in "
        f"$({lo},{hi}]$ (the only states whose hard action can change for $\\theta\\in[{lo},{hi}]$). Last column: "
        "$1-\\mathrm{AF}_{\\mathrm{hard}} \\le (1-\\mathrm{AF}_{\\mathrm{soft}})+(1-\\Pr[\\mathcal{X}_{\\mathrm{cert}}])$.}",
        "\\label{tab:fidelity_details}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrccccccccrc}",
        "\\toprule",
        "\\textbf{Env.} & $N$ & \\textbf{Clean.} & $\\gamma$ & $\\Delta_{\\text{total}}$ & "
        "\\textbf{Cov.} $\\Delta_{\\text{total}}<\\gamma$ & \\textbf{Cov.} pairwise & \\textbf{Cov. agr.} & "
        "\\textbf{Violations} & \\textbf{Sel.} & $\\theta$\\textbf{-sens.} & \\textbf{Teacher} \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
    ])


def fig_overview(summaries, out):
    names = [s["env_short"] for s in summaries]
    x = np.arange(len(names)); w = 0.2
    series = [("actual disagreement", [s["corollary"]["actual_disagreement"] for s in summaries], "tab:red"),
              ("bound, Theorem 1", [s["corollary"]["bound_theorem"] for s in summaries], "tab:blue"),
              ("bound, Corollary (state-wise, best eps)", [s["corollary"]["best_statewise"]["bound"] for s in summaries], "tab:orange"),
              ("bound, Corollary (global, best eps)", [s["corollary"]["best_global"]["bound"] for s in summaries], "0.5")]
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(names)), 4.5))
    for k, (lab, vals, col) in enumerate(series):
        ax.bar(x + (k - 1.5) * w, np.maximum(vals, 1e-3), w, label=lab, color=col)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 250)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("% of held-out states (log)")
    ax.set_title("Soft/hard disagreement vs. its upper bounds")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.2, axis="y")
    fig.savefig(os.path.join(out, "fig_overview_bounds.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(s):
    ch, cc = s["checks"], s["corollary"]["checks"]
    ok = (ch["theorem_violations"] == 0 and ch["lemma1_violations"] == 0 and ch["lemma2_violations"] == 0
          and cc["dbar_violations"] == 0 and cc["certS_disagree"] == 0 and cc["certG_disagree"] == 0)
    c = s["corollary"]
    print(f"[{s['env_short']}] N={s['n_states']} tau={s['tau']} theta={s['theta']} | "
          f"actual {c['actual_disagreement']:.3f}% <= Thm1 {c['bound_theorem']:.2f}% | "
          f"Cor. state-wise {c['best_statewise']['bound']:.2f}% (eps {c['best_statewise']['eps_bar']:g}) | "
          f"Cor. global {c['best_global']['bound']:.2f}% (eps {c['best_global']['eps_bar']:g}) | "
          f"self-checks {'OK' if ok else 'FAILED'}")


# ============================================================================
# Main
# ============================================================================

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Plot Soft-Hard Fidelity (Theorem 1) results")
    ap.add_argument("--results_file", type=str, nargs="+",
                    default=[os.path.join(base, "results", "theorem2_analytics.npz")],
                    help="One or more theorem2_analytics.npz files / result directories")
    ap.add_argument("--output_folder", type=str, default=os.path.join(base, "results", "figures"))
    ap.add_argument("--max_points", type=int, default=20000, help="Max agreeing points in the scatter")
    ap.add_argument("--log", action="store_true", help="symlog axes for the scatter")
    args = ap.parse_args()

    loaded = [load(p) for p in args.results_file]
    loaded.sort(key=lambda t: ENV_ORDER.index(t[1]["env_short"]) if t[1]["env_short"] in ENV_ORDER else 99)
    multi = len(loaded) > 1
    os.makedirs(args.output_folder, exist_ok=True)

    seen = {}
    for arr, s in loaded:
        name = s["env_short"]
        seen[name] = seen.get(name, 0) + 1
        sub = name if seen[name] == 1 else f"{name}_{seen[name]}"
        out = os.path.join(args.output_folder, sub) if multi else args.output_folder
        os.makedirs(out, exist_ok=True)
        fig_scatter(arr, s, out, args.max_points, args.log)
        fig_slack(arr, s, out)
        fig_binarization(arr, s, out)
        fig_decomposition(arr, s, out)
        fig_theta_sweep(arr, s, out)
        fig_triangle(s, out)
        fig_fragile(s, out)
        fig_bounds(s, out)
        fig_eps_sweep(s, out)
        print_summary(s)
        print(f"   figures -> {out}")

    summaries = [s for _, s in loaded]
    if len({(s["tau"], s["theta"]) for s in summaries}) > 1:
        print("[WARN] Environments were verified with different (threshold, hard_threshold): "
              + ", ".join(f"{s['env_short']}=({s['tau']}, {s['theta']})" for s in summaries))
    with open(os.path.join(args.output_folder, "table5_fidelity.tex"), "w") as f:
        f.write(latex_table(summaries) + "\n")
    with open(os.path.join(args.output_folder, "table_fidelity_details.tex"), "w") as f:
        f.write(latex_table_appendix(summaries) + "\n")
    if multi:
        fig_overview(summaries, args.output_folder)
    print(f"\nLaTeX tables -> {os.path.join(args.output_folder, 'table5_fidelity.tex')} (main), "
          f"{os.path.join(args.output_folder, 'table_fidelity_details.tex')} (appendix)")


if __name__ == "__main__":
    main()
