import os
import matplotlib as mpl
import matplotlib.font_manager as fm


# ── Register Arial (auto-detects on the system) ──────────────────────────────
def _register_arial():
    fonts = [f.name for f in fm.fontManager.ttflist]
    if "Arial" in fonts:
        return
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        os.path.join(os.path.dirname(__file__), "Arial.ttf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            return
    import urllib.request
    dest = os.path.join(os.path.expanduser("~"), "Arial.ttf")
    if not os.path.exists(dest):
        url = "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf"
        urllib.request.urlretrieve(url, dest)
    fm.fontManager.addfont(dest)


_register_arial()
mpl.rcParams["font.family"] = "Arial"

import seaborn as sns
import matplotlib.pyplot as plt

# ── Style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="talk")
mpl.rcParams["font.family"] = "Arial"  # reassert after set_theme (seaborn overrides it)

# ── Figure ────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 7))

# ── Area chart ────────────────────────────────────────────────────────────
sns.lineplot(
    x='Idade',
    y='Dias_permanência',
    hue='Óbito',
    data=df1,
    palette={0: 'green', 1: 'orange'},
    estimator='mean',
    errorbar=None,
    linewidth=3,
)
for line in plt.gca().lines:
    x = line.get_xdata()
    y = line.get_ydata()
    plt.fill_between(x, y, alpha=0.25)

# ── Legend ────────────────────────────────────────────────────────────────
handles, _ = plt.gca().get_legend_handles_labels()
leg = plt.legend(
    handles,
    ['Discharge', 'Death'],
    title='Outcome',
    frameon=True,
    prop={'family': 'Arial', 'size': 18},
)
leg.get_title().set_fontfamily('Arial')
leg.get_title().set_fontsize(20)

# ── Title and labels ──────────────────────────────────────────────────────
plt.title(
    'Trend in Length of Stay by Age',
    fontsize=25,
    weight='bold',
    pad=20,
    fontfamily='Arial',
)
plt.xlabel('Age', fontsize=18, fontfamily='Arial')
plt.ylabel('Length of stay', fontsize=18, fontfamily='Arial')

# ── Tick labels ───────────────────────────────────────────────────────────
ax = plt.gca()
for lbl in ax.get_xticklabels():
    lbl.set_fontfamily('Arial')
for lbl in ax.get_yticklabels():
    lbl.set_fontfamily('Arial')

# ── Finalize ──────────────────────────────────────────────────────────────
sns.despine()
plt.tight_layout()
plt.savefig('tendencia_dias_idade.png', dpi=180, bbox_inches='tight')
plt.show()
