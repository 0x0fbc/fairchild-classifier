from pathlib import Path
import sys

OUTPUT = Path(__file__).resolve().parent
LOCAL_PACKAGES = OUTPUT.parent / "work" / "python_plot"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.path import Path as MplPath

FONT_NAMES = {font.name for font in font_manager.fontManager.ttflist}
FONT = "Palatino Linotype" if "Palatino Linotype" in FONT_NAMES else "DejaVu Serif"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

INK = "#202B34"
LINE = "#526370"
BLUE = "#315B79"
PALE_BLUE = "#EDF3F7"
GRAY = "#F5F6F7"

fig = plt.figure(figsize=(5.2, 6.3), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 5.2)
ax.set_ylim(0, 6.3)
ax.set_axis_off()


def box(x, y, w, h, title, body=None, *, blue=False, title_size=9.6,
        body_size=9.3, title_offset=None):
    patch = FancyBboxPatch(
        (x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.045",
        linewidth=0.95, edgecolor=BLUE if blue else LINE,
        facecolor=PALE_BLUE if blue else "white", zorder=3,
    )
    ax.add_patch(patch)
    if body:
        offset = title_offset if title_offset is not None else h/2 - 0.16
        ax.text(x, y+offset, title, ha="center", va="center", color=INK,
                fontsize=title_size, fontweight="bold", zorder=4)
        ax.text(x, y-0.095, body, ha="center", va="center", color=INK,
                fontsize=body_size, linespacing=1.22, zorder=4)
    else:
        ax.text(x, y, title, ha="center", va="center", color=INK,
                fontsize=title_size, linespacing=1.23, zorder=4)
    return patch


def arrow(points, *, dashed=False, color=LINE):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO]*(len(points)-1))
    patch = FancyArrowPatch(
        path=path, arrowstyle="-|>", mutation_scale=9,
        linewidth=0.92, linestyle=(0, (3, 2)) if dashed else "-",
        color=color, zorder=1,
    )
    ax.add_patch(patch)


CX, CW = 3.22, 3.45

box(CX, 5.84, CW, 0.64, "Approved scenario",
    "Record the choice and initial explanation")
arrow([(CX, 5.505), (CX, 5.14)])

diamond = Polygon([(CX, 5.12), (4.48, 4.77), (CX, 4.42), (1.88, 4.77)],
                  closed=True, facecolor="white", edgecolor=LINE,
                  linewidth=0.95, zorder=3)
ax.add_patch(diamond)
ax.text(CX, 4.77, "Clarification needed?", ha="center", va="center",
        fontsize=9.4, fontweight="bold", color=INK, zorder=4)
box(0.76, 4.77, 1.29, 0.94, "Neutral probe",
    "Only if needed\nUp to 5 follow-ups\nStop if declined",
    title_size=8.9, body_size=8.5, title_offset=0.31)
arrow([(1.865, 4.77), (1.425, 4.77)])
ax.text(1.655, 4.90, "Yes", ha="center", va="center", fontsize=8.5, color=INK)
arrow([(0.76, 5.255), (0.76, 5.34), (CX, 5.34), (CX, 5.14)])
ax.text(1.71, 5.43, "Add answer and reassess", ha="center", va="center", fontsize=8.1, color=INK)
arrow([(CX, 4.407), (CX, 4.065)])
ax.text(CX+0.09, 4.25, "No, or probe limit reached", ha="left", va="center",
        fontsize=8.4, color=INK)

box(CX, 3.72, CW, 0.65, "Code the expressed reasons*",
    "Link characteristics to supporting quotations;\nretain scope, prompt source and revisions", blue=True)
arrow([(CX, 3.383), (CX, 3.11)])
box(CX, 2.78, CW, 0.62, "Check eligibility and apply rules*",
    "Only usable responses in enabled pairs;\nrequire each complete template conjunction", blue=True)
arrow([(CX, 2.457), (CX, 2.225)])

box(0.76, 3.05, 1.29, 0.91, "Processing review",
    "Unresolved model\nor evidence\nuncertainty",
    title_size=8.4, body_size=8.6, title_offset=0.295)
arrow([(1.481, 3.72), (0.76, 3.72), (0.76, 3.518)], dashed=True)
ax.text(0.74, 3.89, "If unresolved", ha="center", va="center",
        fontsize=8.2, color=INK)

box(CX, 1.77, CW, 0.88, "Assign the outcome",
    "A-only  |  B-only  |  Mixed\nIndeterminate: neither / insufficient\nInvalid: no usable scenario response",
    body_size=9.0, title_offset=0.29)
arrow([(CX, 1.318), (CX, 1.085)])
box(CX, 0.72, CW, 0.70, "Evidence-based profile for clinician review",
    "Separate domains, evidence and coverage\nNo single diagnostic score",
    title_size=9.1, body_size=9.3, title_offset=0.19)
arrow([(0.76, 2.582), (0.76, 0.72), (1.481, 0.72)], dashed=True)
ax.text(0.76, 1.66, "Report as a\nseparate status", ha="center", va="center",
        fontsize=8.6, color=INK, linespacing=1.25,
        bbox=dict(facecolor="white", edgecolor="none", pad=2.5))

ax.text(2.6, 0.14,
        "* Prototype tested classification and rule matching only.\n"
        "Evidence extraction and live administration remain proposed.",
        ha="center", va="center", fontsize=8.1, color=LINE, linespacing=1.25)

OUTPUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT / "figure-2-workflow.png", dpi=400, facecolor="white")
fig.savefig(OUTPUT / "figure-2-workflow.pdf", facecolor="white",
            metadata={"Title": "Figure 2. Proposed response-to-profile workflow",
                      "Author": "", "Subject": "Conceptual assessment workflow"})
plt.close(fig)
print(f"Font: {FONT}; size: 5.2 x 6.3 inches; PNG: 2080 x 2520 pixels")
