"""Unit-type embeddings for the board planes and the per-type global vectors
(docs/IDEAS.md A1).

The observation addresses a unit type by *index*: board plane `base + (id - 1)`
holds that type's stack count, and every per-type global vector (hand, bag,
discard, supply, ...) has one slot per type. An index carries no information
about the unit — plane 6 (Swordsman) and plane 7 (Knight) are as unrelated to
the network as any two integers — so the first conv layer keeps 16 disjoint
kernel slices per side, each of which only receives gradient in the ~1/4 of
games where its type was drafted. `probe_costs.py` Table B measures the
consequence: 62 % of board planes and 82 % of global dims are exactly zero on
any given forward pass, and *which* ones changes every game.

This module replaces the index with a 16-dim vector per type, so the layer
downstream reads a fixed number of channels whose meaning does not depend on
which types happen to be on the board, and its weights therefore take gradient
from every game:

    embedded[d] = sum_t count[t] * E[t, d]

`E` is `[frozen | learned]`:

* **frozen** (`N_FROZEN_ATTRS` columns) — a fixed function of `roster.py`, a
  registered non-persistent buffer that never takes gradient. This is what
  transfers between types: a column set for several units (`tactic_ranged_
  strike` for Archer + Crossbowman, `gives_extra_tempo` for five units) is one
  shared coordinate those units move together, and a type the run has barely
  seen is still legible through the rules rather than through an under-trained
  slot. Freezing is load-bearing — if these columns were learnable per type the
  parameterisation would degenerate back to (a rotation of) the one-hot it
  replaces.
* **learned** (`learned_dim` columns) — one free row per type, for whatever the
  rules attributes do not capture. This is where a type's *identity* lives, and
  it is why the frozen block does not need to be injective: Swordsman /
  Berserker / Mercenary share a frozen row (5 coins, attacks normally, gains
  extra tempo, no tactic) and are separated here, which is the intended prior —
  start them together, let the data pull them apart.

Note what this is *not*: with `learned_dim = 6` the table is 16x16, so nothing
is compressed and no plane count changes. The effective per-type weight the
conv sees, `W_eff[o,t] = sum_d W[o,d] E[t,d]`, is still full rank. The change is
that 10 of its 16 degrees of freedom per type are tied to shared, rules-derived
columns instead of being free.
"""
import numpy as np
import torch
import torch.nn as nn

from ..environment.roster import UNIT_TYPES, NUM_UNIT_TYPES

# The frozen columns, in order. Every one of them is set for at least two units
# (see `tests/test_unit_embedding.py`, which pins the exact membership): a column
# belonging to a single type would share nothing and is just a one-hot slot with
# a friendlier name, which is the thing being removed here.
FROZEN_ATTR_NAMES = (
    'coin_count',              # 4-coin types (scarcer) vs 5-coin
    'can_normal_attack',
    'gives_extra_tempo',       # acts more than once per coin, by any mechanism
    'has_defensive_trait',     # costly or conditional to attack
    'has_tactic',
    'tactic_deals_damage',     # the unit itself strikes via its tactic
    'tactic_ranged_strike',    # hits without closing (THREAT_KINDS 'ranged')
    'tactic_charge_strike',    # moves, then hits (THREAT_KINDS 'charge')
    'tactic_relocates_self',
    'tactic_targets_friendly',
)
N_FROZEN_ATTRS = len(FROZEN_ATTR_NAMES)
DEFAULT_LEARNED_DIM = 6

# Tactic mechanics grouped by what they *do*, rather than one-hot by name. The
# ranged/charge split deliberately mirrors `obs_encoders.v11.THREAT_KINDS`, which
# already spends six planes separating "hits from two hexes away" (Archer,
# Crossbowman) from "moves in and hits" (Cavalry, Lancer) — merging them here
# would contradict a distinction the rest of the observation draws.
_TEMPO_TACTICS = frozenset({'maneuver_each'})
_DAMAGE_TACTICS = frozenset({'move_then_attack', 'line_charge', 'ranged_attack'})
_RANGED_TACTICS = frozenset({'ranged_attack'})
_CHARGE_TACTICS = frozenset({'move_then_attack', 'line_charge'})
_RELOCATE_TACTICS = frozenset({'move_then_attack', 'line_charge', 'move_to', 'royal_move'})
_FRIENDLY_TACTICS = frozenset({'grant_move', 'grant_attack'})

# Per-type coin totals span 4..5 today; rescaling to [0,1] across the observed
# range keeps the column on the same scale as the binary ones (a raw /5 would
# give a 0.8-vs-1.0 split with almost no variance for the layer to use).
_MIN_COINS = min(u.total_coins for u in UNIT_TYPES)
_MAX_COINS = max(u.total_coins for u in UNIT_TYPES)

# Init scale for the learned block. The frozen columns have a per-column spread
# of roughly 0.3-0.5 across the 16 types (a k-of-16 indicator has std
# sqrt(p(1-p)), maximal 0.5), so matching that keeps neither block dominant at
# step 0.
LEARNED_INIT_STD = 0.3


def _unit_attr_row(u):
    """The frozen row for one `UnitType`, in `FROZEN_ATTR_NAMES` order."""
    tactic = u.tactic
    span = (_MAX_COINS - _MIN_COINS) or 1
    return [
        (u.total_coins - _MIN_COINS) / span,
        float(u.can_normal_attack),
        float(u.move_after_attack
              or u.extra_maneuvers_from_stack
              or u.maneuver_after_recruit
              or u.bonus_action_after_attack_or_control
              or tactic in _TEMPO_TACTICS),
        float(u.counter_when_attacked
              or u.only_attackable_when_bolstered
              or u.absorb_from_supply),
        float(tactic is not None),
        float(tactic in _DAMAGE_TACTICS),
        float(tactic in _RANGED_TACTICS),
        float(tactic in _CHARGE_TACTICS),
        float(tactic in _RELOCATE_TACTICS),
        float(tactic in _FRIENDLY_TACTICS),
    ]


def build_unit_attrs():
    """The frozen attribute table. -> float32 [NUM_UNIT_TYPES, N_FROZEN_ATTRS]

    Row order is `UNIT_TYPES` order, i.e. ascending unit id — the same order as
    the board's unit planes (`base + id - 1`) and the per-type global vectors
    (`UNIT_COINS`). `tests/test_unit_embedding.py` pins that agreement.
    """
    return np.array([_unit_attr_row(u) for u in UNIT_TYPES], dtype=np.float32)


class UnitTypeEmbedding(nn.Module):
    """The `[NUM_UNIT_TYPES, dim]` table plus the two contractions that use it.

    Deliberately has no royal-coin row. The royal coin is a single bag-only coin
    with no board unit (`roster.py`), so it has no unit behaviour for the frozen
    columns to describe; `contract_deck_vector` passes its count through as a raw
    scalar instead of inventing an all-zero row for it.
    """

    def __init__(self, learned_dim=DEFAULT_LEARNED_DIM):
        super().__init__()
        if learned_dim < 0:
            raise ValueError(f'learned_dim must be >= 0, got {learned_dim}')
        self.learned_dim = learned_dim
        self.dim = N_FROZEN_ATTRS + learned_dim
        self.register_buffer('frozen', torch.from_numpy(build_unit_attrs()), persistent=False)
        self.learned = nn.Parameter(
            torch.randn(NUM_UNIT_TYPES, learned_dim) * LEARNED_INIT_STD)

    def table(self):
        """-> [NUM_UNIT_TYPES, dim]; the frozen half never takes gradient."""
        return torch.cat([self.frozen, self.learned], dim=1)

    def contract_planes(self, planes):
        """planes: [B, NUM_UNIT_TYPES, H, W] stack counts -> [B, dim, H, W]"""
        return torch.einsum('bthw,td->bdhw', planes, self.table())

    def contract_vector(self, counts):
        """counts: [..., NUM_UNIT_TYPES] per-type quantities -> [..., dim]"""
        return counts @ self.table()


class ObsTypeEmbedding(nn.Module):
    """Rewrites one encoder's observation from per-type slots into embedding space.

    Owns a `UnitTypeEmbedding` and the layout metadata of the encoder it is built
    for, and exposes the two transforms a net needs:

    * `board(board)` — replaces the own/opponent unit-stack plane blocks with their
      contractions, leaving every other plane (bases, threat, coords, base-reach)
      in place and in order.
    * `globals(g)` — replaces each per-type block with its contraction, carrying the
      royal-coin slot of a DECK block through as a raw scalar.

    `globals` **regroups** its output (all DECK blocks, then all unit blocks, then
    the remaining scalars in their original order) rather than rewriting each block
    in situ. The consumers are `nn.Linear`/FiLM MLPs, to which any fixed permutation
    of the input is invisible, and gathering the blocks in bulk keeps this to three
    concatenations per forward instead of one per block.

    Sizes are recomputed, never assumed: with the default `learned_dim` the table is
    16-wide and both `board_channels` and `global_dim` happen to come out unchanged,
    but that is a coincidence of `10 + 6 == NUM_UNIT_TYPES`, not an invariant.
    """

    def __init__(self, enc, learned_dim=DEFAULT_LEARNED_DIM):
        super().__init__()
        self.emb = UnitTypeEmbedding(learned_dim)
        dim = self.emb.dim

        own, opp = enc.own_unit_channels, enc.opp_unit_channels
        if not (0 <= own.start < own.stop <= opp.start < opp.stop <= enc.board_channels):
            raise ValueError(
                f'expected disjoint, ascending own/opp unit-plane blocks, got {own} and {opp}')
        for name, sl in (('own_unit_channels', own), ('opp_unit_channels', opp)):
            if sl.stop - sl.start != NUM_UNIT_TYPES:
                raise ValueError(
                    f'{name} covers {sl.stop - sl.start} planes, expected {NUM_UNIT_TYPES}')
        self._own = own
        self._opp = opp
        self.board_channels = enc.board_channels - 2 * NUM_UNIT_TYPES + 2 * dim

        try:
            deck_offsets = enc.deck_block_offsets
            unit_offsets = enc.unit_block_offsets
            deck_len = enc.deck_block_len
            unit_len = enc.unit_block_len
            unit_pos = enc.deck_unit_positions
            royal_pos = enc.deck_royal_position
        except AttributeError as exc:
            raise ValueError(
                f'{type(enc).__name__} does not publish the per-type global layout '
                f'(deck_block_offsets/unit_block_offsets/...) that the unit-type embedding '
                f'needs; only obs v11 and later do (docs/IDEAS.md A1)'
            ) from exc

        spans = ([(o, deck_len) for o in deck_offsets]
                 + [(o, unit_len) for o in unit_offsets])
        covered = set()
        for start, length in spans:
            block = range(start, start + length)
            if block.stop > enc.global_dim or covered & set(block):
                raise ValueError(
                    f'per-type global block [{start}, {block.stop}) overlaps another block '
                    f'or runs past global_dim {enc.global_dim}')
            covered |= set(block)
        passthrough = [i for i in range(enc.global_dim) if i not in covered]

        self.register_buffer(
            '_deck_idx',
            torch.tensor([[o + i for i in range(deck_len)] for o in deck_offsets],
                         dtype=torch.long),
            persistent=False)
        self.register_buffer(
            '_unit_idx',
            torch.tensor([[o + i for i in range(unit_len)] for o in unit_offsets],
                         dtype=torch.long),
            persistent=False)
        self.register_buffer('_deck_unit_pos', torch.tensor(unit_pos, dtype=torch.long),
                             persistent=False)
        self.register_buffer('_passthrough', torch.tensor(passthrough, dtype=torch.long),
                             persistent=False)
        self._royal_pos = royal_pos
        self.global_dim = (len(deck_offsets) * (dim + 1)
                           + len(unit_offsets) * dim
                           + len(passthrough))

    def board(self, board):
        """board: [B, enc.board_channels, H, W] -> [B, self.board_channels, H, W]"""
        own = self.emb.contract_planes(board[:, self._own])
        opp = self.emb.contract_planes(board[:, self._opp])
        return torch.cat(
            [board[:, :self._own.start], own,
             board[:, self._own.stop:self._opp.start], opp,
             board[:, self._opp.stop:]],
            dim=1)

    def globals(self, g):
        """g: [B, enc.global_dim] -> [B, self.global_dim]  (blocks regrouped, see class doc)"""
        deck = g[:, self._deck_idx]  # [B, n_deck, deck_len]
        deck_units = self.emb.contract_vector(deck[:, :, self._deck_unit_pos])
        royal = deck[:, :, self._royal_pos].unsqueeze(-1)
        deck_out = torch.cat([deck_units, royal], dim=-1).flatten(1)
        unit_out = self.emb.contract_vector(g[:, self._unit_idx]).flatten(1)
        return torch.cat([deck_out, unit_out, g[:, self._passthrough]], dim=-1)
