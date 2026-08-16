"""Unit-type embedding + FiLM: the A1/A3 pair (docs/IDEAS.md).

What these pin, and why each one is here rather than being obvious:

**A1 — the frozen table.** Its whole value is that its columns are *shared*, so a column
belonging to exactly one unit would be a one-hot slot with a nicer name and would quietly
undo the change. `test_every_frozen_column_is_shared` is the design rule as an assertion,
and `test_frozen_memberships` spells out the intended membership so a roster edit that
silently changes a grouping fails loudly instead of shifting the prior under a training run.
The frozen half must also *stay* frozen: if it ever took gradient the parameterisation would
drift back to a rotation of the per-type one-hot it replaces (`test_frozen_half_is_not_a_
parameter`).

Three collisions are deliberate — Swordsman/Berserker/Mercenary, Knight/Pikeman and
Ensign/Marshall share a frozen row. The frozen block does not need to be injective; the
learned block carries identity, and starting genuinely-similar types together is the prior
being bought. `test_expected_frozen_collisions` records exactly which ones, so a *new* one
appearing (which would mean two unrelated types had been merged) is caught.

**A1 — the observation rewrite.** `ObsTypeEmbedding` slices a 245-wide vector by offsets
published by the encoder. Getting one offset wrong would not raise — it would silently train
on scrambled features — so `test_global_block_offsets_match_a_real_encode` recomputes three
blocks straight from game state and `test_globals_passthrough_is_exact` checks that every
slot outside the per-type blocks (including the royal coin's, which has no embedding row)
arrives untouched.

**A3 — FiLM.** Two properties. It is the identity at init, so a fresh v2 net starts from v1's
trunk behaviour rather than a random perturbation of it; and it can *multiply*, which is the
entire argument for it over the broadcast-concat it replaces — an additive per-cell bias
cannot switch a channel off when the hand makes it irrelevant.

**The v5 critic deliberately has no FiLM.** `board_only_head` exists so that its loss cannot
be satisfied from the globals (docs/next_iteration.md §3.4 — this is the fix that took the
positional-sibling tie rate 93 % -> 0 %). Conditioning the trunk on globals would put them
back inside that path. `test_v5_board_only_value_ignores_globals` is that invariant.
"""
import numpy as np
import pytest
import torch

from src.services.environment.obs_encoders import latest_encoder
from src.services.environment.obs_encoders.v11 import (
    _TOTAL_COINS_VEC, _SUPPLY_CAP_VEC, _counter_to_deck_vec, _counter_to_unit_vec,
    DECK_BLOCK_OFFSETS, UNIT_BLOCK_OFFSETS,
)
from src.services.environment.roster import NUM_UNIT_TYPES, UNIT_TYPES, ROYAL_ID
from src.services.environment.warchest_env import WarChestEnv
from src.services.policy.checkpoint import CURRENT_ARCH, POLICY_ARCHS
from src.services.policy.policy import (
    CRITIC_ARCH_V4, CRITIC_ARCH_V5, POLICY_ARCH_V1, POLICY_ARCH_V2, Critic, FiLM, Policy,
)
from src.services.policy.unit_embedding import (
    FROZEN_ATTR_NAMES, N_FROZEN_ATTRS, ObsTypeEmbedding, UnitTypeEmbedding, build_unit_attrs,
)

DEV = torch.device('cpu')
HIDDEN = 32  # divisible by CRITIC_GROUPS=8
NAME_OF = {u.name: i for i, u in enumerate(UNIT_TYPES)}


def _col(name):
    attrs = build_unit_attrs()
    return attrs[:, FROZEN_ATTR_NAMES.index(name)]


def _members(name):
    return {u.name for u, v in zip(UNIT_TYPES, _col(name)) if v > 0}


def _obs():
    env = WarChestEnv(save_game_history=False, debug_mode=False)
    env.reset(seed=11)
    return env


# --------------------------------------------------------------------------- #
# A1 — the frozen table
# --------------------------------------------------------------------------- #
def test_table_shape_and_row_order():
    attrs = build_unit_attrs()
    assert attrs.shape == (NUM_UNIT_TYPES, N_FROZEN_ATTRS)
    assert attrs.dtype == np.float32
    # Row i must be unit id i+1: that is the board plane index (`base + id - 1`) and the
    # position in every per-type global vector. A mismatch would attribute each unit's
    # counts to its neighbour's row.
    assert [u.id for u in UNIT_TYPES] == list(range(1, NUM_UNIT_TYPES + 1))


def test_every_frozen_column_is_shared():
    """The design rule: a column owned by one unit shares nothing and does not belong here."""
    attrs = build_unit_attrs()
    for i, name in enumerate(FROZEN_ATTR_NAMES):
        n = int((attrs[:, i] > 0).sum())
        assert n >= 2, f'{name} is set for {n} unit(s); a singleton column is a one-hot slot'


def test_frozen_memberships():
    assert _members('can_normal_attack') == {u.name for u in UNIT_TYPES} - {'Lancer', 'Archer'}
    assert _members('gives_extra_tempo') == {
        'Swordsman', 'Berserker', 'Mercenary', 'Footman', 'Warrior Priest'}
    assert _members('has_defensive_trait') == {'Pikeman', 'Knight', 'Royal Guard'}
    assert _members('tactic_deals_damage') == {'Cavalry', 'Lancer', 'Archer', 'Crossbowman'}
    assert _members('tactic_ranged_strike') == {'Archer', 'Crossbowman'}
    assert _members('tactic_charge_strike') == {'Cavalry', 'Lancer'}
    assert _members('tactic_relocates_self') == {
        'Cavalry', 'Light Cavalry', 'Lancer', 'Royal Guard'}
    assert _members('tactic_targets_friendly') == {'Ensign', 'Marshall'}
    assert _members('has_tactic') == {u.name for u in UNIT_TYPES if u.tactic is not None}
    # 5-coin types read 1.0, 4-coin 0.0 — rescaled across the roster's range so the column
    # has the same spread as the binary ones rather than a 0.8-vs-1.0 sliver.
    assert _members('coin_count') == {u.name for u in UNIT_TYPES if u.total_coins == 5}


def test_ranged_and_charge_stay_split_like_the_threat_planes():
    """v11's THREAT_KINDS spends six planes separating 'ranged' from 'charge'; the frozen
    columns must not merge Cavalry into the Archer group and contradict that."""
    assert _members('tactic_ranged_strike').isdisjoint(_members('tactic_charge_strike'))
    assert 'Cavalry' not in _members('tactic_ranged_strike')


def test_expected_frozen_collisions():
    """Identity is the learned block's job — these three groups start together on purpose."""
    attrs = build_unit_attrs()
    groups = {}
    for u, row in zip(UNIT_TYPES, attrs):
        groups.setdefault(row.tobytes(), []).append(u.name)
    collided = {frozenset(v) for v in groups.values() if len(v) > 1}
    assert collided == {
        frozenset({'Swordsman', 'Berserker', 'Mercenary'}),
        frozenset({'Knight', 'Pikeman'}),
        frozenset({'Ensign', 'Marshall'}),
    }


def test_frozen_half_is_not_a_parameter():
    emb = UnitTypeEmbedding(learned_dim=6)
    assert not emb.frozen.requires_grad
    # The only trainable tensor is the learned block: 16 rows x 6.
    params = list(emb.parameters())
    assert len(params) == 1 and params[0].shape == (NUM_UNIT_TYPES, 6)
    assert sum(p.numel() for p in emb.parameters()) == NUM_UNIT_TYPES * 6

    before = emb.frozen.clone()
    emb.table().sum().backward()
    torch.optim.SGD(emb.parameters(), lr=1.0).step()
    assert torch.equal(emb.frozen, before)
    assert emb.learned.grad is not None and emb.learned.grad.abs().sum() > 0


def test_contractions_are_the_weighted_sum_of_rows():
    emb = UnitTypeEmbedding(learned_dim=6)
    table = emb.table()

    counts = torch.zeros(1, NUM_UNIT_TYPES, 7, 7)
    counts[0, 5, 2, 3] = 2.0   # 2 coins of unit id 6 at (2,3)
    counts[0, 8, 2, 3] = 1.0   # + 1 of unit id 9 on the same cell
    out = emb.contract_planes(counts)
    assert out.shape == (1, emb.dim, 7, 7)
    assert torch.allclose(out[0, :, 2, 3], 2.0 * table[5] + table[8], atol=1e-6)
    assert torch.allclose(out[0, :, 0, 0], torch.zeros(emb.dim))

    vec = torch.zeros(2, NUM_UNIT_TYPES)
    vec[0, 3] = 1.5
    vec[1, 15] = 1.0
    got = emb.contract_vector(vec)
    assert torch.allclose(got[0], 1.5 * table[3], atol=1e-6)
    assert torch.allclose(got[1], table[15], atol=1e-6)


def test_no_royal_row():
    """The royal coin is bag-only with no board unit, so it has no unit behaviour to
    describe; its count rides through `globals()` as a raw scalar instead."""
    assert UnitTypeEmbedding().table().shape[0] == NUM_UNIT_TYPES
    assert ROYAL_ID not in {u.id for u in UNIT_TYPES}


# --------------------------------------------------------------------------- #
# A1 — rewriting a real observation
# --------------------------------------------------------------------------- #
def test_global_block_offsets_match_a_real_encode():
    """Recompute three blocks from game state; a wrong offset would train on noise."""
    env = _obs()
    enc = latest_encoder()
    g = enc.encode(env)['global']
    active = env.active_player
    hand = _counter_to_deck_vec(env.state.hands[active]) / _TOTAL_COINS_VEC
    bag = _counter_to_deck_vec(env.state.bags[active]) / _TOTAL_COINS_VEC
    assert np.allclose(g[DECK_BLOCK_OFFSETS[0]:DECK_BLOCK_OFFSETS[0] + 17], hand)
    assert np.allclose(g[DECK_BLOCK_OFFSETS[1]:DECK_BLOCK_OFFSETS[1] + 17], bag)
    supply = _counter_to_unit_vec(env.state.supply[active]) / _SUPPLY_CAP_VEC
    assert np.allclose(g[UNIT_BLOCK_OFFSETS[0]:UNIT_BLOCK_OFFSETS[0] + 16], supply)


def test_blocks_do_not_overlap_and_fit():
    enc = latest_encoder()
    covered = []
    for o in enc.deck_block_offsets:
        covered += list(range(o, o + enc.deck_block_len))
    for o in enc.unit_block_offsets:
        covered += list(range(o, o + enc.unit_block_len))
    assert len(covered) == len(set(covered))
    assert max(covered) < enc.global_dim


def test_globals_passthrough_is_exact():
    """Every slot outside a per-type block — scalars and the royal coin — arrives verbatim."""
    enc = latest_encoder()
    te = ObsTypeEmbedding(enc, learned_dim=6)
    g = torch.from_numpy(latest_encoder().encode(_obs())['global']).unsqueeze(0)
    out = te.globals(g)
    assert out.shape == (1, te.global_dim)

    n_deck, n_unit = len(enc.deck_block_offsets), len(enc.unit_block_offsets)
    dim = te.emb.dim
    # The royal slot is the last column of each DECK group in the output.
    for i, off in enumerate(enc.deck_block_offsets):
        expected = g[0, off + enc.deck_royal_position]
        assert torch.allclose(out[0, i * (dim + 1) + dim], expected, atol=1e-6)
    # Scalars land after the deck and unit groups, in their original relative order.
    tail = out[0, n_deck * (dim + 1) + n_unit * dim:]
    assert torch.allclose(tail, g[0, te._passthrough], atol=1e-6)


def test_board_rewrite_leaves_non_unit_planes_alone():
    enc = latest_encoder()
    te = ObsTypeEmbedding(enc, learned_dim=6)
    board = torch.randn(2, enc.board_channels, 7, 7)
    out = te.board(board)
    assert out.shape == (2, te.board_channels, 7, 7)
    own, opp = enc.own_unit_channels, enc.opp_unit_channels
    dim = te.emb.dim
    # Head (bases/empty/exploration) and tail (threat, coords, base-reach) pass through.
    assert torch.equal(out[:, :own.start], board[:, :own.start])
    assert torch.equal(out[:, own.start + 2 * dim:], board[:, opp.stop:])
    # The own block really is the contraction of the own planes.
    assert torch.allclose(out[:, own.start:own.start + dim],
                          te.emb.contract_planes(board[:, own]), atol=1e-6)


def test_default_widths_are_preserved_but_recomputed():
    """10 frozen + 6 learned == NUM_UNIT_TYPES is a coincidence, not an invariant — the
    module must size itself from the table rather than assume the observation's widths."""
    enc = latest_encoder()
    assert ObsTypeEmbedding(enc, learned_dim=6).board_channels == enc.board_channels
    assert ObsTypeEmbedding(enc, learned_dim=6).global_dim == enc.global_dim
    wide = ObsTypeEmbedding(enc, learned_dim=10)
    assert wide.board_channels == enc.board_channels + 2 * (N_FROZEN_ATTRS + 10 - NUM_UNIT_TYPES)
    assert wide.global_dim > enc.global_dim


def test_encoder_without_the_layout_is_rejected():
    class Old:
        board_channels = 48
        global_dim = 211
        own_unit_channels = slice(6, 22)
        opp_unit_channels = slice(22, 38)

    with pytest.raises(ValueError, match='per-type global layout'):
        ObsTypeEmbedding(Old())


# --------------------------------------------------------------------------- #
# A3 — FiLM
# --------------------------------------------------------------------------- #
def test_film_is_the_identity_at_init():
    film = FiLM(cond_dim=245, channels=16)
    x = torch.randn(4, 16, 7, 7)
    cond = torch.randn(4, 245)
    assert torch.allclose(film(x, cond), x, atol=1e-6)


def test_film_can_switch_a_channel_off():
    """The point of multiplying rather than adding: with the wrong hand, a channel can go
    to zero. A broadcast additive bias cannot express that at all."""
    film = FiLM(cond_dim=4, channels=3)
    with torch.no_grad():
        film.net[-1].bias[0] = -1.0  # gamma[0] = -1 -> (1 + gamma) = 0
    x = torch.randn(2, 3, 7, 7)
    out = film(x, torch.zeros(2, 4))
    assert torch.allclose(out[:, 0], torch.zeros(2, 7, 7), atol=1e-6)
    assert torch.allclose(out[:, 1:], x[:, 1:], atol=1e-6)


def test_film_output_depends_on_the_condition_per_channel():
    film = FiLM(cond_dim=4, channels=3)
    with torch.no_grad():
        for p in film.net[-1].parameters():
            p.normal_(0, 0.5)
    x = torch.randn(1, 3, 7, 7)
    a = film(x, torch.zeros(1, 4))
    b = film(x, torch.ones(1, 4))
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# Wiring: policy / critic archs
# --------------------------------------------------------------------------- #
def test_v2_is_the_default_and_both_policy_archs_build():
    assert CURRENT_ARCH == POLICY_ARCH_V2
    assert set(POLICY_ARCHS) == {POLICY_ARCH_V1, POLICY_ARCH_V2}
    assert Policy(DEV, HIDDEN).arch == POLICY_ARCH_V2
    for arch in POLICY_ARCHS:
        assert Policy(DEV, HIDDEN, arch=arch).arch == arch
    with pytest.raises(ValueError, match='unknown policy arch'):
        Policy(DEV, HIDDEN, arch='policy_v99')


def test_v1_policy_layout_is_unchanged_so_old_state_dicts_load():
    a = Policy(DEV, HIDDEN, arch=POLICY_ARCH_V1)
    Policy(DEV, HIDDEN, arch=POLICY_ARCH_V1).load_state_dict(a.state_dict())  # strict
    assert a.type_emb is None
    assert hasattr(a, 'board_encoder') and not hasattr(a, 'films')


def test_v2_policy_head_has_no_broadcast_global_block():
    """A3's arithmetic: `policy_head` shrinks by exactly global_dim input channels."""
    enc = latest_encoder()
    v1 = Policy(DEV, HIDDEN, arch=POLICY_ARCH_V1)
    v2 = Policy(DEV, HIDDEN, arch=POLICY_ARCH_V2)
    assert v1.policy_head.in_channels - v2.policy_head.in_channels == enc.global_dim
    assert v2.policy_head.in_channels == HIDDEN
    assert len(v2.films) == len(v2.conv_blocks) == 3


def test_both_policy_archs_produce_a_valid_distribution():
    env = _obs()
    obs = env._get_obs() if hasattr(env, '_get_obs') else latest_encoder().encode(env)
    for arch in POLICY_ARCHS:
        p = Policy(DEV, HIDDEN, arch=arch)
        probs = p(obs)
        assert probs.shape[-1] == obs['valid_action_mask'].shape[0]
        assert pytest.approx(1.0, abs=1e-5) == float(probs.sum().detach())
        # No mass on illegal actions.
        illegal = ~torch.from_numpy(obs['valid_action_mask']).bool()
        assert float(probs[0][illegal].sum().detach()) < 1e-6


def test_v5_critic_adds_only_the_learned_rows_over_v4():
    v4 = Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V4)
    v5 = Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V5)
    n4 = sum(p.numel() for p in v4.parameters())
    n5 = sum(p.numel() for p in v5.parameters())
    assert n5 - n4 == NUM_UNIT_TYPES * 6
    Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V5).load_state_dict(v5.state_dict())  # strict


def test_v5_board_only_value_ignores_globals():
    """The v2 auxiliary-head fix survives A1: no globals path reaches the trunk, which is
    why the v5 critic deliberately has no FiLM."""
    enc = latest_encoder()
    c = Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V5)
    board = torch.randn(3, enc.board_channels, 7, 7)
    a = c.board_only_value(board)
    b = c.board_only_value(board)
    assert torch.allclose(a, b)
    # And the full value DOES move with the globals, so the two paths are distinct.
    priv = torch.randn(3, enc.priv_dim)
    v1 = c.value_batch({'board': board, 'global': torch.zeros(3, enc.global_dim),
                        'privileged': priv})
    v2 = c.value_batch({'board': board, 'global': torch.randn(3, enc.global_dim),
                        'privileged': priv})
    assert not torch.allclose(v1, v2)


def test_v5_gathered_pool_still_reads_raw_occupancy():
    """`_gathered_pool` must keep seeing the untouched unit-stack planes; the embedding is
    applied on the way into the trunk only."""
    enc = latest_encoder()
    c = Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V5)
    board = torch.zeros(1, enc.board_channels, 7, 7)
    empty = c._pooled(board)
    board[0, enc.own_unit_channels.start + 2, 3, 3] = 1.0
    assert not torch.allclose(empty, c._pooled(board))


def test_v5_refuses_the_feature_reuse_fast_path():
    c = Critic(DEV, HIDDEN, arch=CRITIC_ARCH_V5)
    with pytest.raises(NotImplementedError, match='raw board tensor'):
        c.value_from_features(torch.randn(1, HIDDEN, 7, 7), torch.randn(1, 245), None,
                              torch.randn(1, latest_encoder().priv_dim))


def test_gradient_reaches_the_learned_rows_from_a_policy_loss():
    """End to end: the table is trained by the ordinary policy gradient, nothing special."""
    enc = latest_encoder()
    p = Policy(DEV, HIDDEN, arch=POLICY_ARCH_V2)
    batch = {
        'board': torch.randn(4, enc.board_channels, 7, 7),
        'global': torch.randn(4, enc.global_dim),
        'mask': torch.ones(4, p._group_mat.shape[1], dtype=torch.bool),
        'actions': torch.zeros(4, dtype=torch.long),
    }
    logp, _, _ = p.evaluate_actions_batch(batch)
    logp.sum().backward()
    assert p.type_emb.emb.learned.grad is not None
    assert p.type_emb.emb.learned.grad.abs().sum() > 0
    # The FiLM output layer is zero-initialised, so on step 0 it is the only layer of the
    # conditioning MLP with a gradient — the chain rule routes through its (still zero)
    # weights to reach the layers below it. It moves on the first step and the rest follow
    # from the second, which is the usual one-step lag of a zero-init adapter, not a break
    # in the path.
    assert p.films[0].net[-1].weight.grad.abs().sum() > 0
    assert p.films[0].net[0].weight.grad.abs().sum() == 0
