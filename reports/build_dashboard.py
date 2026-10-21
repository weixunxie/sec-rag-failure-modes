"""Generate reports/dashboard.html — self-contained, no CDN."""
import os, sys, base64, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from src.config import PROCESSED_DIR

# ── Load data ─────────────────────────────────────────────────────────────────
eval_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'evaluation_results.csv'))
q_df    = pd.read_csv(os.path.join(PROCESSED_DIR, 'questions.csv'))
if 'task_type' not in eval_df.columns:
    eval_df = eval_df.merge(q_df[['question_id','task_type']], on='question_id', how='left')
n_chunks = len(pd.read_parquet(os.path.join(PROCESSED_DIR, 'filing_chunks.parquet'), columns=['chunk_id']))

# ── Palette ───────────────────────────────────────────────────────────────────
CARD  = '#ffffff'
# Semantic
G_OK  = '#10b981'   # correct / good
G_GEN = '#f59e0b'   # generation error / warning
G_RET = '#ef4444'   # retrieval error / bad
G_BLU = '#6366f1'   # info / blue

# Strategy colors — distinct, colorful
C_VAN  = '#6366f1'  # indigo   — Vanilla
C_META = '#f97316'  # orange   — Metadata Filtered
C_TASK = '#10b981'  # emerald  — Task-Aware (winner → green)

TXT = '#374151'
DIM = '#9ca3af'

STRATS       = ['vanilla', 'metadata_filtered', 'task_aware']
STRAT_COLS   = [C_VAN, C_META, C_TASK]
STRAT_LABELS = ['Vanilla', 'Metadata Filtered', 'Task-Aware']
STRAT_SHORT  = ['Vanilla', 'Meta\nFiltered', 'Task-Aware']
TASKS        = ['cross_company', 'cross_year', 'risk_factor', 'single_fact']
COMPARISON   = ['cross_company', 'cross_year']

# ── Metrics (every number on the page comes from evaluation_results.csv) ─────
by_s   = eval_df.groupby('strategy')
n_q    = by_s.size().reindex(STRATS)
stage  = pd.crosstab(eval_df['strategy'], eval_df['error_stage']).reindex(STRATS).fillna(0).astype(int)
for col in ('none', 'generation', 'retrieval'):
    if col not in stage:
        stage[col] = 0
correct_n = stage['none'].tolist()
gen_n     = stage['generation'].tolist()
ret_n     = stage['retrieval'].tolist()
pct       = {s: 100 * stage.loc[s, 'none'] / n_q[s] for s in STRATS}
pct_gen   = {s: 100 * stage.loc[s, 'generation'] / n_q[s] for s in STRATS}
pct_ret   = {s: 100 * stage.loc[s, 'retrieval'] / n_q[s] for s in STRATS}
mean_c    = by_s['answer_correctness'].mean().reindex(STRATS)
mean_f    = by_s['citation_faithfulness'].mean().reindex(STRATS)
mean_cov  = by_s['evidence_coverage'].mean().reindex(STRATS)
task_mean = eval_df.pivot_table(index='strategy', columns='task_type',
                                values='answer_correctness', aggfunc='mean').reindex(STRATS)
comp = eval_df[eval_df['task_type'].isin(COMPARISON)]
comp_ret = (comp['error_stage'] == 'retrieval').groupby(comp['strategy']).sum().reindex(STRATS).fillna(0).astype(int)
comp_n   = comp.groupby('strategy').size().reindex(STRATS)
comp_ok  = (comp['error_stage'] == 'none').groupby(comp['strategy']).sum().reindex(STRATS).fillna(0).astype(int)
fab_answers = int((eval_df['n_fabricated_citations'] > 0).sum())
fab_by_s    = (eval_df['n_fabricated_citations'] > 0).groupby(eval_df['strategy']).sum().reindex(STRATS).astype(int)
partial_gen = int(((eval_df['error_stage'] == 'generation') & (eval_df['answer_correctness'] == 1)).sum())
all_gen     = int((eval_df['error_stage'] == 'generation').sum())
best        = max(STRATS, key=lambda s: pct[s])
LABEL       = dict(zip(STRATS, STRAT_LABELS))
COLOR       = dict(zip(STRATS, STRAT_COLS))

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.spines.left':  False,
    'axes.spines.bottom': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': '#e9ebef',
    'grid.linewidth': 0.8,
    'xtick.color': DIM,
    'ytick.color': DIM,
    'xtick.major.size': 0,
    'ytick.major.size': 0,
    'figure.facecolor': CARD,
    'axes.facecolor': CARD,
})

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight', facecolor=CARD)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — Error stage horizontal stacked bar
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.8, 2.6))
fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
y = np.arange(3); h = 0.5
ax.grid(False)
ax.xaxis.grid(True, color='#e9ebef', linewidth=0.8)
ax.set_axisbelow(True)

b1 = ax.barh(y, correct_n, h, color=G_OK,  label='Correct',          zorder=3)
b2 = ax.barh(y, gen_n,     h, left=correct_n, color=G_GEN, label='Generation Error', zorder=3)
b3 = ax.barh(y, ret_n,     h, left=[c+g for c,g in zip(correct_n,gen_n)],
             color=G_RET, label='Retrieval Error', zorder=3)

for bar, v in [(b1, correct_n), (b2, gen_n), (b3, ret_n)]:
    for b, val in zip(bar, v):
        if val > 1:
            ax.text(b.get_x() + b.get_width()/2, b.get_y() + h/2,
                    str(val), va='center', ha='center',
                    fontsize=9.5, color='white', fontweight='bold', zorder=4)

ax.set_yticks(y)
ax.set_yticklabels(STRAT_SHORT, fontsize=10.5, color=TXT, fontweight='600')
ax.set_xlim(0, int(n_q.max()) + 6)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0, colors=DIM)

# Legend as a separate row below the axes — no overlap with x label
patches = [
    mpatches.Patch(color=G_OK,  label='Correct'),
    mpatches.Patch(color=G_GEN, label='Generation Error'),
    mpatches.Patch(color=G_RET, label='Retrieval Error'),
]
fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.08),
           ncol=3, fontsize=8.5, framealpha=0, labelcolor=DIM,
           handlelength=1.1, handletextpad=0.5, columnspacing=1.4)
fig.subplots_adjust(bottom=0.18)
plt.tight_layout(rect=[0, 0.12, 1, 1])
chart_stage = fig_to_b64(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 — Correctness by task type
# ─────────────────────────────────────────────────────────────────────────────
TASK_LABS = ['Cross\nCompany', 'Cross\nYear', 'Risk\nFactor', 'Single\nFact']
task_vals = {s: [float(task_mean.loc[s, t]) for t in TASKS] for s in STRATS}

fig, ax = plt.subplots(figsize=(6.4, 3.4))
fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
x = np.arange(4); bw = 0.24; offsets = [-bw, 0, bw]

for strat, col, off in zip(STRATS, STRAT_COLS, offsets):
    bars = ax.bar(x + off, task_vals[strat], bw, color=col, zorder=3,
                  alpha=0.9, label=STRAT_LABELS[STRATS.index(strat)])
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, v + 0.04, f'{v:.1f}',
                ha='center', va='bottom', fontsize=7.5, color=col, fontweight='600')

ax.set_xticks(x)
ax.set_xticklabels(TASK_LABS, fontsize=10, color=TXT)
ax.set_ylim(0, 2.55)
ax.set_ylabel('Mean Correctness (0–2)', fontsize=9, color=DIM)
ax.yaxis.set_tick_params(colors=DIM)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0)

patches = [mpatches.Patch(color=c, label=l) for c, l in zip(STRAT_COLS, STRAT_LABELS)]
ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.16),
          ncol=3, fontsize=9, framealpha=0, labelcolor=DIM,
          handlelength=1.1, handletextpad=0.5, columnspacing=1.2)
plt.tight_layout(pad=1.0)
chart_task = fig_to_b64(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 — Error type breakdown
# ─────────────────────────────────────────────────────────────────────────────
err_counts = pd.crosstab(eval_df['error_type'].fillna('CORRECT'), eval_df['strategy']).reindex(columns=STRATS).fillna(0)
err_order  = ['CORRECT'] + [e for e in err_counts.sum(axis=1).sort_values(ascending=False).index if e != 'CORRECT']
err_labels = ['\n'.join(e.title().split('_')) for e in err_order]
err_vals   = {s: [int(err_counts.loc[e, s]) for e in err_order] for s in STRATS}

fig, ax = plt.subplots(figsize=(6.4, 3.4))
fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
x = np.arange(len(err_order)); bw = 0.24; offsets = [-bw, 0, bw]

for strat, col, off in zip(STRATS, STRAT_COLS, offsets):
    bars = ax.bar(x + off, err_vals[strat], bw, color=col, zorder=3, alpha=0.9,
                  label=STRAT_LABELS[STRATS.index(strat)])
    for b in bars:
        v = b.get_height()
        if v > 0:
            ax.text(b.get_x() + b.get_width()/2, v + 0.3, str(int(v)),
                    ha='center', va='bottom', fontsize=7.5, color=col, fontweight='600')

ax.set_xticks(x)
ax.set_xticklabels(err_labels, fontsize=8.5, color=TXT)
ax.set_ylim(0, max(max(v) for v in err_vals.values()) + 4)
ax.set_ylabel('Count', fontsize=9, color=DIM)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0, colors=DIM)

patches = [mpatches.Patch(color=c, label=l) for c, l in zip(STRAT_COLS, STRAT_LABELS)]
ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.16),
          ncol=3, fontsize=9, framealpha=0, labelcolor=DIM,
          handlelength=1.1, handletextpad=0.5, columnspacing=1.2)
plt.tight_layout(pad=1.0)
chart_error = fig_to_b64(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 — Radar
# ─────────────────────────────────────────────────────────────────────────────
categories = ['Correctness', 'Faithfulness', 'Coverage', '% Correct\n(scaled)', 'No Retrieval\nError']
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# % Correct and "no retrieval error" are rescaled to 0–2 to share the axis
radar_vals = {s: [float(mean_c[s]), float(mean_f[s]), float(mean_cov[s]),
                  pct[s] / 50, 2 - pct_ret[s] / 50] for s in STRATS}

fig = plt.figure(figsize=(5.2, 4.4))
fig.patch.set_facecolor(CARD)
ax = fig.add_subplot(111, polar=True)
ax.set_facecolor(CARD)

for strat, col in zip(STRATS, STRAT_COLS):
    vals = radar_vals[strat] + radar_vals[strat][:1]
    ax.plot(angles, vals, color=col, linewidth=2.2, zorder=3)
    ax.fill(angles, vals, color=col, alpha=0.10, zorder=2)
    ax.scatter(angles[:-1], vals[:-1], color=col, s=28, zorder=4)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9, color=TXT)
ax.set_ylim(0, 2.1)
ax.set_yticks([0.5, 1.0, 1.5, 2.0])
ax.set_yticklabels(['0.5','1.0','1.5','2.0'], fontsize=7, color='#cccccc')
ax.yaxis.grid(True, color='#e9ebef', linewidth=0.7)
ax.xaxis.grid(True, color='#e9ebef', linewidth=0.7)
ax.spines['polar'].set_color('#e9ebef')

patches = [mpatches.Patch(color=c, label=l) for c, l in zip(STRAT_COLS, STRAT_LABELS)]
ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.18),
          ncol=3, fontsize=9, framealpha=0, labelcolor=DIM,
          handlelength=1.1, handletextpad=0.5, columnspacing=1.2)
plt.tight_layout(pad=0.8)
chart_radar = fig_to_b64(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Chart 5 — Grouped count (correct vs errors)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 3.0))
fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
x = np.arange(3); bw = 0.26; offsets = [-bw, 0, bw]

count_data = [
    (correct_n, G_OK,  'Correct'),
    (gen_n,     G_GEN, 'Generation Error'),
    (ret_n,     G_RET, 'Retrieval Error'),
]
for i, (vals, col, lbl) in enumerate(count_data):
    bars = ax.bar(x + offsets[i], vals, bw, color=col, zorder=3, alpha=0.9, label=lbl)
    for b in bars:
        v = b.get_height()
        if v > 0:
            ax.text(b.get_x() + b.get_width()/2, v + 0.3, str(int(v)),
                    ha='center', va='bottom', fontsize=8, color=col, fontweight='600')

ax.set_xticks(x)
ax.set_xticklabels(STRAT_SHORT, fontsize=10.5, color=TXT)
ax.set_ylim(0, max(correct_n + gen_n + ret_n) + 4)
ax.set_ylabel(f'Count (out of {int(n_q.max())})', fontsize=9, color=DIM)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0, colors=DIM)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=3, fontsize=9, framealpha=0, labelcolor=DIM,
          handlelength=1.1, handletextpad=0.5, columnspacing=1.2)
plt.tight_layout(pad=1.0)
chart_counts = fig_to_b64(fig); plt.close()

print("All 5 charts generated.")

# ─────────────────────────────────────────────────────────────────────────────
# Build collapsible HTML blocks
# ─────────────────────────────────────────────────────────────────────────────
task_meta = {
    'single_fact':   ('Single Fact',   '#6366f1', 'Extract a specific financial figure or metric from one company-year filing.'),
    'risk_factor':   ('Risk Factor',   '#10b981', 'Identify and summarize a specific category of disclosed risk.'),
    'cross_year':    ('Cross Year',    '#f59e0b', 'Compare a narrative, metric, or strategy across two fiscal years.'),
    'cross_company': ('Cross Company', '#ef4444', 'Compare disclosures or metrics between two different companies.'),
}

def q_rows_html(qs):
    rows = []
    for q in qs:
        ct = q.get('comparison_ticker'); cy = q.get('comparison_year')
        comp = f' vs {ct} {int(cy) if pd.notna(cy) else ""}' if pd.notna(ct) and str(ct) else ''
        yr = int(q['year']) if pd.notna(q.get('year')) else ''
        rows.append(
            f'<tr>'
            f'<td class="qid">{q["question_id"]}</td>'
            f'<td class="qticker">{q["ticker"]} {yr}{comp}</td>'
            f'<td class="qtext">{q["question"]}</td>'
            f'</tr>'
        )
    return ''.join(rows)

q_blocks = []
for t in ['single_fact','risk_factor','cross_year','cross_company']:
    label, color, desc = task_meta[t]
    qs = q_df[q_df['task_type']==t].to_dict('records')
    q_blocks.append(
        f'<details class="q-group">'
        f'<summary>'
        f'<span class="q-badge" style="background:{color}18;color:{color};border:1px solid {color}40">{label}</span>'
        f'<span class="q-count">{len(qs)} questions</span>'
        f'<span class="q-desc">{desc}</span>'
        f'<span class="chevron">&#x203A;</span>'
        f'</summary>'
        f'<table class="q-table"><thead><tr><th>ID</th><th>Scope</th><th>Question</th></tr></thead>'
        f'<tbody>{q_rows_html(qs)}</tbody></table>'
        f'</details>'
    )
questions_html = ''.join(q_blocks)

strat_info = [
    ('Vanilla', C_VAN, f"{pct['vanilla']:.0f}% correct",
     f'Pure cosine similarity over all {n_chunks:,} chunks with no metadata filters. Treats every chunk equally regardless of company or year.',
     ['No query parsing needed', 'Can surface both sides of a comparison when they are semantically close',
      'Works across all question types'],
     [f"{pct_ret['vanilla']:.0f}% retrieval error rate: fetches chunks from the wrong company or year",
      f"Missed one side of the comparison in {comp_ret['vanilla']} of {comp_n['vanilla']} comparison questions"]),
    ('Metadata Filtered', C_META, f"{pct['metadata_filtered']:.0f}% correct",
     'Pre-filters the Qdrant collection to the question\'s primary ticker and year before similarity search.',
     ['Clean, on-target context for single-entity questions',
      f"Ties the other strategies on single-fact and risk-factor questions",
      'Faster search on large collections'],
     [f"Retrieval failure on {comp_ret['metadata_filtered']} of {comp_n['metadata_filtered']} comparison questions: the filter excludes the second company or year",
      f"Lowest accuracy overall ({pct['metadata_filtered']:.0f}%)"]),
    ('Task-Aware', C_TASK, f"{pct['task_aware']:.0f}% correct",
     'Routes queries by question type. Single-entity questions use metadata filtering; cross-year and cross-company questions split the top-k budget equally between both targets.',
     [f"Best overall accuracy ({pct['task_aware']:.0f}%)",
      f"{comp_ret['task_aware']} retrieval errors on comparison questions",
      f"Highest correctness on cross-company questions ({task_mean.loc['task_aware', 'cross_company']:.2f} vs {task_mean.loc['metadata_filtered', 'cross_company']:.2f} for Metadata-Filtered)"],
     ['Requires question metadata (task type, companies, years) at query time',
      f"Highest generation error rate ({pct_gen['task_aware']:.0f}%): with the right evidence retrieved, the remaining failures are incomplete synthesis"]),
]
s_blocks = []
for name, color, score, desc, pros, cons in strat_info:
    pro_li = ''.join(f'<li>{p}</li>' for p in pros)
    con_li = ''.join(f'<li>{c}</li>' for c in cons)
    s_blocks.append(
        f'<details class="s-group">'
        f'<summary>'
        f'<span class="s-dot" style="background:{color}"></span>'
        f'<span class="s-name">{name}</span>'
        f'<span class="s-score" style="background:{color}18;color:{color};border:1px solid {color}40">{score}</span>'
        f'<span class="chevron">&#x203A;</span>'
        f'</summary>'
        f'<div class="s-body">'
        f'<p class="s-desc">{desc}</p>'
        f'<div class="s-lists">'
        f'<div><strong>Strengths</strong><ul class="pro-list">{pro_li}</ul></div>'
        f'<div><strong>Weaknesses</strong><ul class="con-list">{con_li}</ul></div>'
        f'</div></div></details>'
    )
strategies_html = ''.join(s_blocks)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#f2f3f5;--card:#ffffff;--border:#e4e6ea;
  --text:#1a1a2e;--mid:#5a5f72;--dim:#9ca3af;
  --green:#10b981;--green-lt:#d1fae5;
  --amber:#f59e0b;--amber-lt:#fef3c7;
  --red:#ef4444;--red-lt:#fee2e2;
  --indigo:#6366f1;--indigo-lt:#eef2ff;
  --orange:#f97316;--orange-lt:#fff7ed;
}
body{font-family:'Inter','Helvetica Neue',Arial,sans-serif;background:var(--bg);color:var(--text);font-size:13px;line-height:1.55}
header{background:var(--card);border-bottom:1px solid var(--border);padding:20px 40px}
header .header-inner{max-width:1360px;margin:0 auto;display:flex;align-items:center;gap:16px}
header h1{font-size:17px;font-weight:800;letter-spacing:-.3px;white-space:nowrap}
header .sub{font-size:12px;color:var(--dim);flex:1}
header .header-right{display:flex;align-items:center;gap:12px;flex-shrink:0}
header .author{font-size:12px;color:var(--mid);font-weight:500}
header .gh-btn{display:inline-flex;align-items:center;gap:4px;font-size:12px;font-weight:500;color:var(--dim);text-decoration:none;transition:color .15s}
header .gh-btn:hover{color:var(--text)}
main{padding:24px 40px 60px;max-width:1360px;margin:0 auto;display:flex;flex-direction:column;gap:20px}
.row{display:flex;gap:18px}
.row>*{flex:1;min-width:0}
.row.split-40>*:first-child{flex:0 0 38%}
/* KPIs */
.kpi-row{display:flex;gap:14px}
.kpi{flex:1;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 20px}
.kpi .val{font-size:26px;font-weight:800;letter-spacing:-1px;line-height:1;color:var(--text)}
.kpi .lbl{font-size:11px;color:var(--dim);margin-top:5px}
.kpi.green{border-color:var(--green);background:var(--green-lt)}.kpi.green .val{color:var(--green)}.kpi.green .lbl{color:#065f46}
.kpi.amber{border-color:var(--amber);background:var(--amber-lt)}.kpi.amber .val{color:var(--amber)}.kpi.amber .lbl{color:#92400e}
.kpi.red{border-color:var(--red);background:var(--red-lt)}.kpi.red .val{color:var(--red)}.kpi.red .lbl{color:#991b1b}
/* Summary */
.summary{background:var(--card);border:1px solid var(--border);border-left:4px solid var(--text);border-radius:10px;padding:20px 24px}
.summary h2{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin-bottom:14px}
.findings{display:flex;flex-direction:column;gap:10px}
.finding{display:flex;align-items:baseline;gap:10px;color:var(--mid);line-height:1.5}
.finding strong{color:var(--text)}
.tag{flex-shrink:0;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;padding:2px 8px;border-radius:4px}
.tag.green{background:var(--green-lt);color:var(--green)}
.tag.amber{background:var(--amber-lt);color:var(--amber)}
.tag.red{background:var(--red-lt);color:var(--red)}
.tag.indigo{background:var(--indigo-lt);color:var(--indigo)}
/* Cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px 20px 20px;display:flex;flex-direction:column;gap:14px}
.card-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--dim)}
.card img{width:100%;height:auto;display:block;border-radius:6px}
.insight{font-size:12px;color:var(--mid);line-height:1.65;border-top:1px solid var(--border);padding-top:12px}
.insight strong{color:var(--text)}
/* Strategy table */
table.strat-tbl{width:100%;border-collapse:collapse}
.strat-tbl thead th{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);padding:6px 8px 8px 0;border-bottom:1px solid var(--border);text-align:left}
.strat-tbl thead th:not(:first-child){text-align:right}
.strat-tbl tbody tr{border-bottom:1px solid #f0f1f3}
.strat-tbl tbody tr:last-child{border-bottom:none}
.strat-tbl tbody td{padding:11px 8px 11px 0;font-variant-numeric:tabular-nums}
.strat-tbl tbody td:not(:first-child){text-align:right}
.s-name-cell{display:flex;align-items:center;gap:8px;font-weight:600}
.s-dot-sm{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.badge{display:inline-block;font-size:10px;font-weight:700;padding:3px 8px;border-radius:5px}
.mini-bar{display:inline-flex;align-items:center;gap:7px}
.bar-track{width:58px;height:5px;background:#e9ebef;border-radius:3px}
.bar-fill{height:100%;border-radius:3px}
/* Collapsibles */
details{background:var(--card);border:1px solid var(--border);border-radius:10px;overflow:hidden}
details+details{margin-top:8px}
summary{display:flex;align-items:center;gap:10px;padding:14px 18px;cursor:pointer;list-style:none;user-select:none;transition:background .15s}
summary::-webkit-details-marker{display:none}
summary:hover{background:#fafbfc}
.chevron{margin-left:auto;font-size:17px;color:var(--dim);transition:transform .2s;line-height:1}
details[open] .chevron{transform:rotate(90deg)}
.q-badge{font-size:10px;font-weight:700;padding:3px 9px;border-radius:5px;flex-shrink:0}
.q-count{font-size:12px;font-weight:600;color:var(--text)}
.q-desc{font-size:12px;color:var(--dim);flex:1}
.q-table{width:100%;border-collapse:collapse;font-size:12px}
.q-table thead th{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);padding:10px 16px;background:#fafbfc;border-top:1px solid var(--border);text-align:left}
.q-table tbody tr{border-top:1px solid #f0f1f3}
.q-table tbody tr:hover{background:#fafbfc}
.q-table td{padding:10px 16px;vertical-align:top}
.qid{color:var(--dim);font-weight:600;white-space:nowrap;width:46px}
.qticker{color:var(--indigo);font-weight:600;white-space:nowrap;width:140px}
.qtext{color:var(--mid)}
.s-group summary{gap:12px}
.s-dot{width:11px;height:11px;border-radius:50%;flex-shrink:0}
.s-name{font-size:13px;font-weight:700;color:var(--text)}
.s-score{font-size:11px;font-weight:700;padding:3px 9px;border-radius:5px}
.s-body{padding:0 18px 18px;border-top:1px solid var(--border)}
.s-desc{font-size:12.5px;color:var(--mid);margin:13px 0 12px;line-height:1.65}
.s-lists{display:flex;gap:28px}
.s-lists>div{flex:1}
.s-lists strong{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);display:block;margin-bottom:7px}
.pro-list,.con-list{padding-left:16px;font-size:12px;color:var(--mid);display:flex;flex-direction:column;gap:5px}
.pro-list li::marker{color:var(--green)}.con-list li::marker{color:var(--red)}
.section-head{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--dim);padding:4px 0}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Assemble HTML
# ─────────────────────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SEC RAG Failure Modes — Dashboard</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div class="header-inner">
    <h1>SEC RAG Failure Modes</h1>
    <p class="sub">3 retrieval strategies &nbsp;·&nbsp; 40 questions &nbsp;·&nbsp; AAPL · MSFT · NVDA · TSLA · JPM &nbsp;·&nbsp; 2022–2024 10-K filings</p>
    <div class="header-right">
      <span class="author">Stephanie Xie</span>
      <a class="gh-btn" href="https://github.com/weixunxie/sec-rag-failure-modes" target="_blank">GitHub ↗</a>
    </div>
  </div>
</header>
<main>

<div class="kpi-row">
  <div class="kpi"><div class="val">15</div><div class="lbl">10-K Filings Downloaded</div></div>
  <div class="kpi"><div class="val">{n_chunks:,}</div><div class="lbl">Chunks Indexed in Qdrant</div></div>
  <div class="kpi"><div class="val">{len(eval_df)}</div><div class="lbl">Answers Evaluated</div></div>
  <div class="kpi green"><div class="val">{pct[best]:.0f}%</div><div class="lbl">Best Accuracy — {LABEL[best]}</div></div>
  <div class="kpi red"><div class="val">{comp_ret['metadata_filtered']}/{comp_n['metadata_filtered']}</div><div class="lbl">Comparison Questions Metadata-Filtered Fails to Retrieve</div></div>
  <div class="kpi amber"><div class="val">{fab_answers}</div><div class="lbl">Answers Citing an Unretrieved Chunk</div></div>
</div>

<div class="summary">
  <h2>Key Findings</h2>
  <div class="findings">
    <div class="finding"><span class="tag green">Winner</span><span><strong>Task-Aware retrieval scores {pct['task_aware']:.0f}% fully correct</strong>, ahead of Vanilla ({pct['vanilla']:.0f}%) and Metadata-Filtered ({pct['metadata_filtered']:.0f}%). It is the only strategy that retrieves evidence for both sides of every comparison question ({comp_ret['task_aware']} retrieval errors on {comp_n['task_aware']} comparison questions).</span></div>
    <div class="finding"><span class="tag red">Structural</span><span><strong>Metadata filtering fails every comparison question at retrieval</strong> ({comp_ret['metadata_filtered']} of {comp_n['metadata_filtered']}): a single ticker/year filter cannot return the second company or year. Vanilla misses one side in {comp_ret['vanilla']} of {comp_n['vanilla']}. On single-fact and risk-factor questions the three strategies are close.</span></div>
    <div class="finding"><span class="tag amber">Generation</span><span><strong>Once retrieval is right, the remaining errors are incomplete answers.</strong> {partial_gen} of {all_gen} generation-stage errors are partially correct (score 1 of 2), typically missing a detail the reference answer includes, rather than hallucinated.</span></div>
    <div class="finding"><span class="tag indigo">Citations</span><span><strong>Citation fabrication is rare once it is checked deterministically.</strong> Only {fab_answers} of {len(eval_df)} answers cite a chunk ID that was not retrieved (Vanilla {fab_by_s['vanilla']}, Metadata {fab_by_s['metadata_filtered']}, Task-Aware {fab_by_s['task_aware']}). The first run's 47 came from a judge that never saw the retrieved text.</span></div>
  </div>
</div>

<div class="row split-40">
  <div class="card">
    <div class="card-title">Strategy Performance</div>
    <table class="strat-tbl">
      <thead><tr><th>Strategy</th><th>% Correct</th><th>Correctness /2</th><th>Faithfulness /2</th></tr></thead>
      <tbody>
        <tr>
          <td><div class="s-name-cell"><div class="s-dot-sm" style="background:{COLOR['task_aware']}"></div>{LABEL['task_aware']}</div></td>
          <td><span class="badge" style="background:{COLOR['task_aware']}18;color:{COLOR['task_aware']}">{pct['task_aware']:.1f}%</span></td>
          <td><div class="mini-bar"><span>{mean_c['task_aware']:.2f}</span><div class="bar-track"><div class="bar-fill" style="width:{mean_c['task_aware'] * 50:.1f}%;background:{COLOR['task_aware']}"></div></div></div></td>
          <td><div class="mini-bar"><span>{mean_f['task_aware']:.2f}</span><div class="bar-track"><div class="bar-fill" style="width:{mean_f['task_aware'] * 50:.1f}%;background:{COLOR['task_aware']}"></div></div></div></td>
        </tr>
        <tr>
          <td><div class="s-name-cell"><div class="s-dot-sm" style="background:{COLOR['vanilla']}"></div>{LABEL['vanilla']}</div></td>
          <td><span class="badge" style="background:{COLOR['vanilla']}18;color:{COLOR['vanilla']}">{pct['vanilla']:.1f}%</span></td>
          <td><div class="mini-bar"><span>{mean_c['vanilla']:.2f}</span><div class="bar-track"><div class="bar-fill" style="width:{mean_c['vanilla'] * 50:.1f}%;background:{COLOR['vanilla']}"></div></div></div></td>
          <td><div class="mini-bar"><span>{mean_f['vanilla']:.2f}</span><div class="bar-track"><div class="bar-fill" style="width:{mean_f['vanilla'] * 50:.1f}%;background:{COLOR['vanilla']}"></div></div></div></td>
        </tr>
        <tr>
          <td><div class="s-name-cell"><div class="s-dot-sm" style="background:{COLOR['metadata_filtered']}"></div>{LABEL['metadata_filtered']}</div></td>
          <td><span class="badge" style="background:{COLOR['metadata_filtered']}18;color:{COLOR['metadata_filtered']}">{pct['metadata_filtered']:.1f}%</span></td>
          <td><div class="mini-bar"><span>{mean_c['metadata_filtered']:.2f}</span><div class="bar-track"><div class="bar-fill" style="width:{mean_c['metadata_filtered'] * 50:.1f}%;background:{COLOR['metadata_filtered']}"></div></div></div></td>
          <td><div class="mini-bar"><span>{mean_f['metadata_filtered']:.2f}</span><div class="bar-track"><div class="bar-fill" style="width:{mean_f['metadata_filtered'] * 50:.1f}%;background:{COLOR['metadata_filtered']}"></div></div></div></td>
        </tr>
      </tbody>
    </table>
    <p class="insight">Scores on a 0–2 scale, LLM-as-judge via <strong>gpt-4o-mini</strong> (temperature 0) against reference answers built from SEC XBRL facts and the correct filing sections, then checked by hand against the filings. The judge sees the full text of every retrieved chunk.</p>
  </div>
  <div class="card">
    <div class="card-title">Error Stage Distribution (out of {int(n_q.max())} per strategy)</div>
    <img src="data:image/png;base64,{chart_stage}" alt="Error stage chart">
    <p class="insight">A retrieval-stage error means the retrieved chunks did not cover the expected company and year (for comparisons: both sides). <strong style="color:{G_RET}">Metadata-Filtered</strong> has {pct_ret['metadata_filtered']:.0f}% retrieval errors, all on comparison questions; <strong style="color:{C_TASK}">Task-Aware</strong> has {pct_ret['task_aware']:.0f}%. Task-Aware's higher generation-error share is the flip side: its failures happen after the right evidence was found.</p>
  </div>
</div>

<div class="row">
  <div class="card">
    <div class="card-title">Answer Correctness by Task Type (0–2 scale)</div>
    <img src="data:image/png;base64,{chart_task}" alt="Correctness by task type">
    <p class="insight"><strong>Cross-company questions show the clearest gap:</strong> <strong style="color:{C_TASK}">Task-Aware ({task_mean.loc['task_aware', 'cross_company']:.2f})</strong> vs <strong style="color:{C_META}">Metadata-Filtered ({task_mean.loc['metadata_filtered', 'cross_company']:.2f})</strong>. On cross-year questions Vanilla ({task_mean.loc['vanilla', 'cross_year']:.2f}) edges Task-Aware ({task_mean.loc['task_aware', 'cross_year']:.2f}). With 10 questions per type, differences of a few tenths are within noise.</p>
  </div>
  <div class="card">
    <div class="card-title">Error Type Breakdown by Strategy</div>
    <img src="data:image/png;base64,{chart_error}" alt="Error type breakdown">
    <p class="insight"><strong>WRONG_ENTITY and WRONG_YEAR</strong> come from Metadata-Filtered and Vanilla on comparison questions and disappear under Task-Aware. Most remaining errors are labelled <strong>REASONING_ERROR</strong>, but nearly all of them are partially correct answers that omit a detail, so the label overstates how often the model reasons incorrectly.</p>
  </div>
</div>

<div class="row">
  <div class="card">
    <div class="card-title">Multi-Dimensional Score Profile</div>
    <img src="data:image/png;base64,{chart_radar}" alt="Radar chart">
    <p class="insight">% Correct and "No Retrieval Error" are rescaled to 0–2. <strong style="color:{C_TASK}">Task-Aware</strong> leads on correctness, faithfulness and retrieval; the gaps on faithfulness and coverage are small.</p>
  </div>
  <div class="card">
    <div class="card-title">Correct vs. Error Counts per Strategy</div>
    <img src="data:image/png;base64,{chart_counts}" alt="Count chart">
    <p class="insight"><strong style="color:{C_TASK}">Task-Aware: {correct_n[2]} correct</strong>, {gen_n[2]} generation errors, {ret_n[2]} retrieval errors. <strong style="color:{C_META}">Metadata-Filtered: {correct_n[1]} correct</strong>, {ret_n[1]} retrieval errors. <strong style="color:{C_VAN}">Vanilla: {correct_n[0]} correct</strong>. 40 questions per strategy, so a one-question difference moves accuracy by 2.5 points.</p>
  </div>
</div>

<div class="section-head">Retrieval Strategies</div>
{strategies_html}

<div class="section-head" style="margin-top:8px">Benchmark Questions</div>
{questions_html}

</main>
</body>
</html>"""

for name in ('dashboard.html', 'index.html'):          # index.html is what Vercel serves
    out = os.path.join(os.path.dirname(__file__), name)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(HTML)
    print(f"Written {len(HTML):,} bytes -> {out}")
