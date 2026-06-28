"""Coin-token drawing helpers for the renderer.

A coin is drawn as a filled disc whose *face* color identifies the unit type
(approximated from the original card art, see docs/UNITS.md) and whose *border*
color identifies the owning player. This mirrors the physical game, where the
coin art names the unit and only the player's side of the board tells you whose
it is.
"""
import matplotlib.patches as patches

from .game_state import DECK
from .roster import COIN_BY_ID

# Per-unit-type coin face colors and glyphs, from the roster (docs/UNITS.md).
COIN_FACE_COLORS = {c.id: c.color for c in COIN_BY_ID.values()}
COIN_GLYPHS = {c.id: c.icon for c in COIN_BY_ID.values()}

# Player identity rides on the coin border (face = type, rim = owner).
PLAYER_EDGE_COLORS = {1: 'darkred', 2: 'midnightblue'}


def _glyph_color(face_hex: str) -> str:
    """Black or white glyph, whichever contrasts the face color's luminance."""
    r = int(face_hex[1:3], 16) / 255
    g = int(face_hex[3:5], 16) / 255
    b = int(face_hex[5:7], 16) / 255
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black' if lum > 0.55 else 'white'


def draw_coin(ax, x, y, coin, player_id, radius=0.16, fontsize=10, glyph=True):
    """Draw a single coin disc centered at (x, y)."""
    face = COIN_FACE_COLORS[coin]
    ax.add_patch(patches.Circle(
        (x, y), radius=radius, facecolor=face,
        edgecolor=PLAYER_EDGE_COLORS[player_id], linewidth=2.2, zorder=3,
    ))
    if glyph:
        text = COIN_GLYPHS[coin]
        # Shrink multi-character codes (e.g. 'Sw') so they fit inside the disc.
        fs = fontsize / (1 + 0.55 * (len(text) - 1))
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fs, color=_glyph_color(face), zorder=4)


def draw_zone(ax, x, y, label, counter, player_id, radius=0.16,
              stack_dx=0.055, stack_dy=0.055, col_gap=0.5):
    """Draw a labeled zone: a caption, then one leaning pile per coin type.

    Coins of the same type are stacked into an overlapping pile (ordered S -> K ->
    R); only the top coin carries the glyph. An empty zone shows a faint
    placeholder so the layout stays stable across frames.
    """
    ax.text(x, y + 0.32, label, ha='left', va='center', fontsize=8,
            color='dimgray', fontweight='bold')
    types = [c for c in DECK if counter[c] > 0]
    if not types:
        ax.text(x + radius, y, '·', ha='center', va='center',
                fontsize=12, color='lightgray')
        return
    cx = x + radius
    for c in types:
        n = counter[c]
        for i in range(n):
            draw_coin(ax, cx + i * stack_dx, y + i * stack_dy, c, player_id,
                      radius=radius, fontsize=radius * 56, glyph=(i == n - 1))
        cx += (n - 1) * stack_dx + col_gap
