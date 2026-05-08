"""Build notebooks/eda_phase1.ipynb from inline cell content.

Run: uv run python notebooks/_build_nb.py
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "eda_phase1.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s))


md("""# Wellbore Prediction — EDA Phase 1

EDA only. No feature engineering for modeling, no model training. Goal: gather
enough evidence about the data to inform v1 architecture choices in the next
session.

Heavy lifting is in `notebooks/eda_pipeline.py`; the per-well summary is cached
to `eda_outputs/well_summary.parquet` after the first run.""")

code("""%matplotlib inline
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Allow `import notebooks.eda_pipeline` from the notebooks dir.
ROOT = Path.cwd().resolve()
if (ROOT / 'notebooks').exists():
    sys.path.insert(0, str(ROOT))
elif (ROOT.parent / 'notebooks').exists():
    sys.path.insert(0, str(ROOT.parent))

from notebooks.eda_pipeline import (
    build_summary, file_inventory, leakage_audit,
    TRAIN_DIR, TEST_DIR, list_wells, FIGS_DIR,
)

mpl.rcParams.update({'figure.dpi': 110, 'figure.figsize': (8, 4), 'axes.grid': True})
pd.set_option('display.max_columns', 80)
pd.set_option('display.width', 200)""")

# ---------------- Inventory ----------------
md("""## 0. Inventory & schema sanity""")
code("""inv = file_inventory()
print('Wells per split:')
print(inv['split'].value_counts())
print()
print('Files present per split:')
print(inv.groupby('split')[['has_horizontal','has_typewell','has_png']].sum())
print()
print('Test well names (note: shared with train):')
print(inv[inv.split=='test']['well'].tolist())
print()
print('Train well names that overlap with test:')
overlap = sorted(set(inv[inv.split=='train']['well']) & set(inv[inv.split=='test']['well']))
print(overlap)""")

code("""# Schema sanity on a handful of train wells
import csv
sample_wells = inv[inv.split=='train']['well'].sample(5, random_state=0).tolist()
for w in sample_wells:
    h = pd.read_csv(TRAIN_DIR / f'{w}__horizontal_well.csv', nrows=3)
    print(w, '-> horizontal cols:', list(h.columns))
    print('  dtypes:', h.dtypes.astype(str).to_dict())
    tw = pd.read_csv(TRAIN_DIR / f'{w}__typewell.csv', nrows=3)
    print('  typewell cols:', list(tw.columns))""")

code("""# Test-side schema sanity
for w in inv[inv.split=='test']['well']:
    h = pd.read_csv(TEST_DIR / f'{w}__horizontal_well.csv', nrows=3)
    tw = pd.read_csv(TEST_DIR / f'{w}__typewell.csv', nrows=3)
    print(w, 'horizontal:', list(h.columns))
    print('  typewell:', list(tw.columns))""")

code("""# Build / load the per-well summary
summary = build_summary()
print('summary shape:', summary.shape)
summary.head(3)""")

# ---------------- Q1 ----------------
md("""## Q1 — Hidden vs known zone sizes

For every train well, we computed `mask_start_idx` (first NaN in `TVT_input`),
known-prefix length, hidden length, and hidden ratio.""")

code("""train = summary[summary.split=='train'].copy()
test  = summary[summary.split=='test'].copy()

print('Train hidden_ratio (n=', len(train), '):')
print(train['hidden_ratio'].describe(percentiles=[.05,.25,.5,.75,.95]).round(3))
print()
print('Wells with no NaN at all (no hidden zone):',
      int((train['mask_start_idx'] == -1).sum()))
print('Wells with hidden_ratio > 0.95:',
      int((train['hidden_ratio'] > 0.95).sum()))
print('Wells with hidden_ratio < 0.05:',
      int((train['hidden_ratio'] < 0.05).sum()))""")

code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(train['known_len'], bins=60)
axes[0].set_title('Known-prefix length (rows) — train')
axes[0].set_xlabel('rows')
axes[1].hist(train['hidden_len'], bins=60, color='C1')
axes[1].set_title('Hidden-zone length (rows) — train')
axes[1].set_xlabel('rows')
axes[2].hist(train['hidden_ratio'], bins=40, color='C2')
axes[2].set_title('Hidden ratio — train')
axes[2].set_xlabel('hidden_len / n_rows')
plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/q1_hidden_known_dists.png', bbox_inches='tight')
plt.show()""")

code("""# Tails
print('Top 5 by hidden_ratio:')
print(train.nlargest(5, 'hidden_ratio')[['well','n_rows','known_len','hidden_len','hidden_ratio']])
print()
print('Bottom 5 by hidden_ratio:')
print(train.nsmallest(5, 'hidden_ratio')[['well','n_rows','known_len','hidden_len','hidden_ratio']])
print()
print('Same view in MD-span (feet):')
print(train[['md_known_span','md_hidden_span']].describe(percentiles=[.05,.5,.95]).round(1))""")

# ---------------- Q2 ----------------
md("""## Q2 — TVT trajectory smoothness (train only)

Per-row `dTVT/dMD` (rate of stratigraphic change per unit MD) and
`d²TVT/dMD²` (curvature/jumps) on the **known prefix** of each train well.
Per-well summary is in the cached parquet; here we recompute global
distributions on a representative subsample of rows.""")

code("""rng = np.random.default_rng(42)
sample = train.sample(min(150, len(train)), random_state=0)['well'].tolist()
all_dtvt, all_d2 = [], []
for w in sample:
    h = pd.read_csv(TRAIN_DIR / f'{w}__horizontal_well.csv',
                    usecols=['MD','TVT','TVT_input'])
    nan_mask = h['TVT_input'].isna().to_numpy()
    first_nan = int(np.argmax(nan_mask)) if nan_mask.any() else len(h)
    if first_nan < 3:
        continue
    md_v = h['MD'].iloc[:first_nan].to_numpy()
    tvt_v = h['TVT'].iloc[:first_nan].to_numpy()
    if not np.all(np.diff(md_v) > 0):
        continue
    dtvt = np.diff(tvt_v) / np.diff(md_v)
    d2 = np.diff(dtvt)
    all_dtvt.append(dtvt)
    all_d2.append(d2)
all_dtvt = np.concatenate(all_dtvt)
all_d2 = np.concatenate(all_d2)
print(f'rows in sample: dtvt={len(all_dtvt):,}  d2={len(all_d2):,}')""")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(all_dtvt, bins=200, range=(np.percentile(all_dtvt,1), np.percentile(all_dtvt,99)))
axes[0].set_title('dTVT/dMD distribution (known prefix, sample)')
axes[0].set_xlabel('dTVT/dMD')
axes[0].axvline(0, color='k', lw=0.5)

axes[1].hist(all_d2, bins=200, range=(np.percentile(all_d2,0.5), np.percentile(all_d2,99.5)))
axes[1].set_title('d²TVT/dMD² distribution (known prefix, sample)')
axes[1].set_xlabel('d²TVT/dMD²')
axes[1].set_yscale('log')
plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/q2_tvt_derivatives.png', bbox_inches='tight')
plt.show()""")

code("""# Per-well discontinuity flag based on the per-well jumps count from the pipeline.
# n_jumps_3sigma counts |d2| > 8 * MAD(d2). >0 means the well shows step-like behavior.
print('Train n_jumps_3sigma (per well) describe:')
print(train['n_jumps_3sigma'].describe().round(2))
print()
print('Wells with >=1 detected jump:', int((train['n_jumps_3sigma'] >= 1).sum()),
      f"({(train['n_jumps_3sigma'] >= 1).mean()*100:.1f}%)")
print('Wells with >=5 detected jumps:', int((train['n_jumps_3sigma'] >= 5).sum()))
print()
print('Top jumpy wells (potential faults):')
print(train.nlargest(10, 'n_jumps_3sigma')[
    ['well','n_rows','known_len','n_jumps_3sigma','d2tvtdmd2_max_abs','abs_dtvtdmd_max']
])""")

code("""# Visual sanity: plot TVT trace for the most jumpy well
worst = train.nlargest(1, 'n_jumps_3sigma').iloc[0]['well']
h = pd.read_csv(TRAIN_DIR / f'{worst}__horizontal_well.csv', usecols=['MD','TVT','TVT_input'])
nan_mask = h['TVT_input'].isna().to_numpy()
first_nan = int(np.argmax(nan_mask)) if nan_mask.any() else len(h)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(h['MD'], h['TVT'], lw=0.8, label='TVT')
ax.axvline(h['MD'].iloc[first_nan], color='r', ls='--', label=f'mask_start (idx={first_nan})')
ax.set_title(f'Most-jumpy well: {worst}')
ax.set_xlabel('MD'); ax.set_ylabel('TVT')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/q2_jumpy_{worst}.png', bbox_inches='tight')
plt.show()""")

# ---------------- Q3 ----------------
md("""## Q3 — GR scale consistency

Per-well GR stats for train wells, test wells, typewells. Then per-well
horizontal-vs-typewell offset/ratio to see whether they are calibrated.""")

code("""def desc(d, label):
    rows = []
    for col, name in [('mean','mean'), ('std','std'), ('p10','p10'), ('p50','p50'), ('p90','p90')]:
        rows.append({'stat': name, 'value_mean': d[col].mean(), 'value_std': d[col].std()})
    print(label, '— per-well GR stats summary across wells:')
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print()

desc(train.assign(mean=train['gr_mean'], std=train['gr_std'],
                  p10=train['gr_p10'], p50=train['gr_p50'], p90=train['gr_p90']),
     'TRAIN horizontal')
desc(test.assign(mean=test['gr_mean'], std=test['gr_std'],
                 p10=test['gr_p10'], p50=test['gr_p50'], p90=test['gr_p90']),
     'TEST horizontal (known + hidden)')
desc(train.assign(mean=train['tw_gr_mean'], std=train['tw_gr_std'],
                  p10=train['tw_gr_p10'], p50=train['tw_gr_p50'], p90=train['tw_gr_p90']),
     'TRAIN typewells')
desc(test.assign(mean=test['tw_gr_mean'], std=test['tw_gr_std'],
                 p10=test['tw_gr_p10'], p50=test['tw_gr_p50'], p90=test['tw_gr_p90']),
     'TEST typewells')""")

code("""# Horizontal vs own typewell calibration (offset & ratio of std)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(train['gr_scale_offset'], bins=60)
axes[0].axvline(0, color='r', lw=1)
axes[0].set_title('GR offset (horizontal_pref_mean - typewell_mean) — train')
axes[0].set_xlabel('mean offset')

ratio = train['gr_scale_ratio'].clip(0, 5)
axes[1].hist(ratio, bins=60, color='C1')
axes[1].axvline(1, color='r', lw=1)
axes[1].set_title('GR std ratio (horizontal_pref_std / typewell_std) — train, clipped at 5')
axes[1].set_xlabel('ratio')
plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/q3_gr_offset_ratio.png', bbox_inches='tight')
plt.show()

print('Train offset describe:')
print(train['gr_scale_offset'].describe(percentiles=[.05,.5,.95]).round(2))
print()
print('Train std-ratio describe:')
print(train['gr_scale_ratio'].describe(percentiles=[.05,.5,.95]).round(2))""")

code("""# Train vs test GR distributions (use horizontal-known prefix mean for fair compare)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(train['gr_pref_mean'], bins=60, alpha=0.6, label=f'train (n={len(train)})', density=True)
ax.hist(test['gr_pref_mean'], bins=60, alpha=0.7, label=f'test (n={len(test)})', density=True)
ax.set_xlabel('GR mean over known prefix')
ax.set_title('Train vs test: GR mean (known prefix)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/q3_train_vs_test_gr.png', bbox_inches='tight')
plt.show()

print('Test wells GR detail:')
print(test[['well','gr_pref_mean','gr_pref_std','gr_pref_p10','gr_pref_p50','gr_pref_p90',
            'tw_gr_mean','tw_gr_std','gr_scale_offset','gr_scale_ratio','gr_ks_proxy']].round(2))""")

# ---------------- Q4 ----------------
md("""## Q4 — Typewell coverage

For each well, does the horizontal's `TVT_input` range fit inside the typewell's
TVT range?""")

code("""print('Train wells whose horizontal TVT_input is fully inside typewell TVT range:')
print(' ', int(train['cov_inside'].sum()), '/', len(train),
      f"({train['cov_inside'].mean()*100:.1f}%)")
print()
print('Train cov_low_margin (TVT_input_min - typewell_min) describe (>=0 means inside):')
print(train['cov_low_margin'].describe(percentiles=[.05,.5,.95]).round(2))
print()
print('Train cov_high_margin describe:')
print(train['cov_high_margin'].describe(percentiles=[.05,.5,.95]).round(2))""")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(train['cov_low_margin'], bins=80)
axes[0].axvline(0, color='r', lw=1)
axes[0].set_title('TVT_input_min - typewell_TVT_min  (train)')
axes[0].set_xlabel('low-side margin')
axes[1].hist(train['cov_high_margin'], bins=80, color='C1')
axes[1].axvline(0, color='r', lw=1)
axes[1].set_title('typewell_TVT_max - TVT_input_max  (train)')
axes[1].set_xlabel('high-side margin')
plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/q4_typewell_coverage.png', bbox_inches='tight')
plt.show()""")

code("""# Wells where the lateral exits the typewell
exits = train[~train['cov_inside']]
print(f'Wells where lateral exits typewell: {len(exits)}')
print(exits[['well','tvt_input_min','tvt_input_max','tw_tvt_min','tw_tvt_max',
             'cov_low_margin','cov_high_margin']].head(10).round(2))
print()
print('Test wells coverage:')
print(test[['well','tvt_input_min','tvt_input_max','tw_tvt_min','tw_tvt_max',
            'cov_inside','cov_low_margin','cov_high_margin']].round(2))""")

# ---------------- Q5 ----------------
md("""## Q5 — Per-well difficulty proxies & clustering

Using train-only difficulty features (we do NOT use TVT for test, so the
clustering here is descriptive of train wells only).""")

code("""from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

feat_cols = ['hidden_len','known_len','abs_dtvtdmd_mean','abs_dtvtdmd_max',
             'd2tvtdmd2_std','n_jumps_3sigma','gr_scale_offset','gr_scale_ratio',
             'gr_ks_proxy','cov_low_margin','cov_high_margin']
X = train[feat_cols].fillna(train[feat_cols].median())
Xs = StandardScaler().fit_transform(X)

# pick k by silhouette quickly
from sklearn.metrics import silhouette_score
scores = {}
for k in (2,3,4,5,6):
    lbl = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
    scores[k] = silhouette_score(Xs, lbl)
print('silhouette by k:', {k: round(v,3) for k,v in scores.items()})
best_k = max(scores, key=scores.get)
print('chosen k =', best_k)

km = KMeans(n_clusters=best_k, n_init=20, random_state=0).fit(Xs)
train['cluster'] = km.labels_
pca = PCA(n_components=2, random_state=0).fit(Xs)
P = pca.transform(Xs)

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(P[:,0], P[:,1], c=km.labels_, cmap='tab10', s=12, alpha=0.7)
ax.set_title(f'Train difficulty proxies — KMeans k={best_k} (PCA 2D)')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
plt.colorbar(sc, ax=ax, label='cluster')
plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/q5_difficulty_clusters.png', bbox_inches='tight')
plt.show()

cluster_profile = train.groupby('cluster')[feat_cols].median().round(2)
print()
print('Cluster medians (rows = cluster id):')
print(cluster_profile)
print()
print('Cluster sizes:')
print(train['cluster'].value_counts().sort_index())""")

# ---------------- Q6 ----------------
md("""## Q6 — Spatial layout

Centroids (X, Y), pad/field clustering, and nearest-neighbor distances.""")

code("""fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(train['x_mean'], train['y_mean'], s=10, alpha=0.5, label='train')
ax.scatter(test['x_mean'], test['y_mean'], s=80, marker='*', color='red', label='test')
ax.set_xlabel('X'); ax.set_ylabel('Y')
ax.set_title('Well centroids — train vs test')
ax.legend()
ax.set_aspect('equal', adjustable='datalim')
plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/q6_centroids.png', bbox_inches='tight')
plt.show()

print('X range train:', round(train.x_mean.min(),1), '→', round(train.x_mean.max(),1),
      ' span:', round(train.x_mean.max() - train.x_mean.min(),1))
print('Y range train:', round(train.y_mean.min(),1), '→', round(train.y_mean.max(),1),
      ' span:', round(train.y_mean.max() - train.y_mean.min(),1))
print('Test centroids:')
print(test[['well','x_mean','y_mean']].round(1))""")

code("""from sklearn.cluster import DBSCAN
# Pick eps by visual inspection: use 2% of x-span as a starting heuristic
xy = train[['x_mean','y_mean']].to_numpy()
span = max(train.x_mean.max() - train.x_mean.min(),
           train.y_mean.max() - train.y_mean.min())
eps = span * 0.005   # 0.5% of bbox span — tight pad-scale
db = DBSCAN(eps=eps, min_samples=3).fit(xy)
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int((labels == -1).sum())
print(f'DBSCAN eps={eps:.1f}: clusters={n_clusters}, noise={n_noise}')

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(xy[labels==-1,0], xy[labels==-1,1], s=8, c='lightgray', label=f'noise ({n_noise})')
mask = labels != -1
ax.scatter(xy[mask,0], xy[mask,1], s=10, c=labels[mask], cmap='tab20')
ax.scatter(test['x_mean'], test['y_mean'], s=120, marker='*', color='red', label='test')
ax.set_aspect('equal', adjustable='datalim')
ax.set_title(f'DBSCAN pad clusters (eps={eps:.1f}) — {n_clusters} clusters')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/q6_dbscan.png', bbox_inches='tight')
plt.show()""")

code("""from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=6).fit(train[['x_mean','y_mean']].to_numpy())
test_xy = test[['x_mean','y_mean']].to_numpy()
dists, idx = nn.kneighbors(test_xy)
# col 0 is the nearest train neighbor — same well by name presumably
for i, w in enumerate(test['well'].tolist()):
    nearest_idx = idx[i, 0]
    nn_well = train.iloc[nearest_idx]['well']
    same_name = (nn_well == w)
    print(f'test {w}: nearest train = {nn_well} (dist={dists[i,0]:.2f}, same name: {same_name})')
    print(f'  next 5 distances: {[round(d,1) for d in dists[i,1:6]]}')""")

code("""# Are typewell files ever shared (i.e., identical content) across multiple wells?
import hashlib
hashes = {}
for split, d in [('train', TRAIN_DIR), ('test', TEST_DIR)]:
    for w in list_wells(d):
        with open(d / f'{w}__typewell.csv', 'rb') as f:
            h = hashlib.md5(f.read()).hexdigest()
        hashes.setdefault(h, []).append(f'{split}/{w}')
shared = {h: ws for h, ws in hashes.items() if len(ws) > 1}
print(f'Distinct typewell content hashes: {len(hashes)}')
print(f'Hashes shared by >1 well: {len(shared)}')
if shared:
    for h, ws in list(shared.items())[:8]:
        print(f'  hash {h[:8]}: {ws}')""")

# ---------------- Q7 ----------------
md("""## Q7 — Train/test distribution alignment""")

code("""def quick_compare(col, label=None):
    label = label or col
    a = train[col].dropna()
    b = test[col].dropna()
    print(f'{label:25s}  train: mean={a.mean():9.2f} std={a.std():9.2f} '
          f'min={a.min():9.2f} max={a.max():9.2f}    test: '
          f'mean={b.mean():9.2f} std={b.std():9.2f} min={b.min():9.2f} max={b.max():9.2f}')

for c, lbl in [
    ('known_len',     'known_len (rows)'),
    ('hidden_len',    'hidden_len (rows)'),
    ('hidden_ratio',  'hidden_ratio'),
    ('md_known_span', 'md_known_span'),
    ('md_hidden_span','md_hidden_span'),
    ('gr_pref_mean',  'GR mean (pref)'),
    ('gr_pref_std',   'GR std (pref)'),
    ('z_min',         'Z min'),
    ('z_max',         'Z max'),
    ('x_mean',        'X mean'),
    ('y_mean',        'Y mean'),
]:
    quick_compare(c, lbl)""")

code("""# Visual side-by-side for the most operationally important: hidden_ratio + known_len + GR mean
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, ttl in zip(axes,
                         ['hidden_ratio', 'known_len', 'gr_pref_mean'],
                         ['hidden_ratio', 'known_len', 'GR mean (pref)']):
    bins = 40
    ax.hist(train[col], bins=bins, alpha=0.6, density=True, label=f'train (n={len(train)})')
    for v in test[col]:
        ax.axvline(v, color='red', lw=1.2, alpha=0.8)
    ax.set_title(ttl + ' — train hist + test (red lines)')
plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/q7_train_vs_test.png', bbox_inches='tight')
plt.show()""")

# ---------------- Sanity / Leakage ----------------
md("""## Sanity / leakage audit""")

code("""audit = leakage_audit()
for w, info in audit.items():
    print(w, info)""")

code("""# Confirm no test PNGs
test_pngs = [w for w in list_wells(TEST_DIR) if (TEST_DIR / f'{w}.png').exists()]
print(f'Test PNGs present: {len(test_pngs)} (expected 0)')

# Spot-check 2 train PNGs
from PIL import Image
for w in list_wells(TRAIN_DIR)[:2]:
    p = TRAIN_DIR / f'{w}.png'
    if p.exists():
        with Image.open(p) as im:
            print(f'{w}.png  size={im.size}  mode={im.mode}')""")

# ---------------- Wrap-up ----------------
md("""## Notes for the next session

Key takeaways are summarised in `eda_findings.md`. Numbers above feed directly
into the journal entry for this session.""")

nb["cells"] = cells
OUT.write_text(nbf.writes(nb))
print(f"wrote {OUT}")
