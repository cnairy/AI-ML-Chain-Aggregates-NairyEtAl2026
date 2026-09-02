#!/usr/bin/env python3
"""
Compact JTECH-style descriptor table containing all descriptors used in the
descriptor-based workflow.

Version 2 changes:
- adds extra vertical spacing for the last 4 rows:
  Area Ratio, Curl, Fine Detail, Compactness
- reduces extra empty space on the far right by slightly narrowing the overall
  table width and the description column

Outputs:
    descriptor_table_JTECH_all_v2.png
    descriptor_table_JTECH_all_v2.pdf
"""

from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
from matplotlib import font_manager


TIMES_FONT_PATH = None
FONT_NAME = "Times New Roman"


def configure_times_new_roman():
    candidates = []

    if TIMES_FONT_PATH:
        candidates.append(Path(TIMES_FONT_PATH).expanduser())

    for root in [
        Path.home() / ".fonts",
        Path.home() / ".local/share/fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
    ]:
        if root.exists():
            for p in root.rglob("*.ttf"):
                if "times" in p.name.lower():
                    candidates.append(p)

    for p in candidates:
        try:
            if p.exists():
                font_manager.fontManager.addfont(str(p))
        except Exception:
            pass

    try:
        font_path = font_manager.findfont(FONT_NAME, fallback_to_default=False)
        print(f"Using Times New Roman: {font_path}")
    except ValueError:
        raise RuntimeError(
            "\nTimes New Roman is not installed or Matplotlib cannot see it.\n\n"
            "Ubuntu/Debian:\n"
            "    sudo apt update\n"
            "    sudo apt install ttf-mscorefonts-installer\n"
            "    fc-cache -f -v\n"
            "    rm -f ~/.cache/matplotlib/fontlist-v*.json\n\n"
            "Restart Python and rerun the script.\n"
        )


configure_times_new_roman()

plt.rcParams.update({
    "font.family": FONT_NAME,
    "font.size": 7.5,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_NAME,
    "mathtext.it": f"{FONT_NAME}:italic",
    "mathtext.bf": f"{FONT_NAME}:bold",
})


headers = ["Descriptor", "Equation", "Unit", "Description"]

rows = [
    {
        "descriptor": r"$D_{\max}$",
        "equation": "",
        "unit": "mm",
        "description": (
            "Diameter of the minimum enclosing circle fitted to the segmented "
            "particle contour."
        ),
    },
    {
        "descriptor": "Circularity",
        "equation": r"$\dfrac{4\pi A}{P^{2}}$",
        "unit": "Unitless",
        "description": (
            r"Measure of how closely the projected particle shape approaches a "
            r"circle, with values $\approx 1$ indicating a circular shape; values "
            r"decrease as the boundary becomes more irregular or elongated."
        ),
    },
    {
        "descriptor": "Solidity",
        "equation": r"$\dfrac{A}{A_{\mathrm{hull}}}$",
        "unit": "Unitless",
        "description": (
            "Ratio of particle projected area to convex-hull area. Values near 1 "
            "indicate a more convex, filled particle shape."
        ),
    },
    {
        "descriptor": "Complexity",
        "equation": r"$\dfrac{P(1+\sigma)}{\pi D_{\mathrm{eq}}}$",
        "unit": "Unitless",
        "description": (
            "Combines particle perimeter and local grayscale variability. Values "
            "near 1 indicate simpler, more circular particles; larger values indicate "
            "greater morphological complexity."
        ),
    },
    {
        "descriptor": "Area Ratio",
        "equation": r"$\dfrac{A}{\pi\left[r(2.3\times10^{-3})\right]^{2}}$",
        "unit": "Unitless",
        "description": (
            "Ratio of projected particle area to the area of its minimum enclosing "
            "circle; larger values indicate that the particle more completely fills "
            "the enclosing circle."
        ),
    },
    {
        "descriptor": "Curl",
        "equation": (
            r"$\dfrac{8r(2.3\times10^{-3})}"
            r"{P-\sqrt{P^{2}-16A}}$"
        ),
        "unit": "Unitless",
        "description": (
            'Measures the degree to which an object is "curled up." As curl '
            'decreases, the degree to which the object is "curled up" increases.'
        ),
    },
    {
        "descriptor": "Fine Detail",
        "equation": r"$\dfrac{P\left[2r(2.3\times10^{-3})\right]}{A}$",
        "unit": "Unitless",
        "description": (
            "Measure of ice crystal complexity (Holroyd 1987). Higher values "
            "indicate greater perimeter/detail relative to projected area."
        ),
    },
    {
        "descriptor": "Compactness",
        "equation": r"$\dfrac{P^{2}}{4\pi A}$",
        "unit": "Unitless",
        "description": (
            "Takes a minimum value of 1 for a circle; particles with more "
            "complicated or irregular boundaries have larger compactness."
        ),
    },
]


OUTPUT_DIR = Path(__file__).resolve().parent
PNG_NAME = OUTPUT_DIR / "descriptor_table_JTECH_all_v2.png"
PDF_NAME = OUTPUT_DIR / "descriptor_table_JTECH_all_v2.pdf"

# Slightly narrower overall width to cut the unused space on the right.
FIGSIZE = (5.5, 3.75)
DPI = 600

# Descriptor / Equation / Unit / Description
COL_WIDTHS = [0.145, 0.145, 0.15, 0.48]

HEADER_SIZE = 7.9
BODY_SIZE = 7.3
EQUATION_SIZE = 9.1

TOP_BOTTOM_LINEWIDTH = 0.75
MID_LINEWIDTH = 0.55
LEFT_PAD = 0.0025
DESCRIPTION_WRAP_CHARS = 72
DESCRIPTION_LINESPACING = 0.92


def horizontal_rule(ax, y, linewidth):
    ax.plot(
        [0.0, 1.0],
        [y, y],
        color="black",
        linewidth=linewidth,
        solid_capstyle="butt",
        clip_on=False,
    )


def wrapped_lines(text, width):
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def main():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = [0.0]
    for width in COL_WIDTHS:
        xs.append(xs[-1] + width)

    # Build row heights dynamically, then add extra height to the last 4 rows.
    header_weight = 1.0
    row_weights = []
    wrapped_descriptions = []

    for idx, row in enumerate(rows):
        lines = wrapped_lines(row["description"], DESCRIPTION_WRAP_CHARS)
        wrapped_descriptions.append("\n".join(lines))
        n = max(1, len(lines))

        # Base compact height from number of wrapped lines.
        weight = 0.82 + 0.42 * n

        # Add extra spacing for the last 4 rows:
        # Area Ratio, Curl, Fine Detail, Compactness.
        if idx == 5:
            weight += 0.9

        row_weights.append(weight)
        

    total_weight = header_weight + sum(row_weights)

    y_positions = []
    y_top = 1.0

    header_h = header_weight / total_weight
    y_bottom = y_top - header_h
    y_positions.append((y_bottom, y_top))
    y_top = y_bottom

    for weight in row_weights:
        h = weight / total_weight
        y_bottom = y_top - h
        y_positions.append((y_bottom, y_top))
        y_top = y_bottom

    horizontal_rule(ax, y_positions[0][1], TOP_BOTTOM_LINEWIDTH)
    horizontal_rule(ax, y_positions[0][0], MID_LINEWIDTH)
    horizontal_rule(ax, y_positions[-1][0], TOP_BOTTOM_LINEWIDTH)

    # Header
    y0, y1 = y_positions[0]
    yc = (y0 + y1) / 2
    for col, header in enumerate(headers):
        ax.text(
            xs[col] + COL_WIDTHS[col] / 2,
            yc,
            header,
            ha="center",
            va="center",
            fontsize=HEADER_SIZE,
        )

    # Body
    for i, row in enumerate(rows, start=1):
        y0, y1 = y_positions[i]
        yc = (y0 + y1) / 2

        ax.text(
            xs[0] + LEFT_PAD,
            yc,
            row["descriptor"],
            ha="left",
            va="center",
            fontsize=BODY_SIZE,
        )

        if row["equation"]:
            ax.text(
                xs[1] + COL_WIDTHS[1] / 2,
                yc,
                row["equation"],
                ha="center",
                va="center",
                fontsize=EQUATION_SIZE,
            )

        ax.text(
            xs[2] + COL_WIDTHS[2] / 2,
            yc,
            row["unit"],
            ha="center",
            va="center",
            fontsize=BODY_SIZE,
        )

        ax.text(
            xs[3] + LEFT_PAD,
            yc,
            wrapped_descriptions[i - 1],
            ha="left",
            va="center",
            fontsize=BODY_SIZE,
            linespacing=DESCRIPTION_LINESPACING,
        )

    plt.subplots_adjust(left=0.002, right=0.998, top=0.995, bottom=0.005)

    fig.savefig(
        PNG_NAME,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.001,
        facecolor="white",
    )
    fig.savefig(
        PDF_NAME,
        bbox_inches="tight",
        pad_inches=0.001,
        facecolor="white",
    )
    plt.close(fig)

    print(f"Created: {PNG_NAME}")
    print(f"Created: {PDF_NAME}")


if __name__ == "__main__":
    main()
