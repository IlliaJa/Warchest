"""Expert iteration (ExIt / AlphaZero) core: self-play data-gen + distillation.

`PuctBot` is the strongest agent in the gauntlet, so its *own* search output is a
better teacher than the raw policy that seeds it. This module closes that loop
(docs/history.md — "search moves become new training targets"):

  1. `play_selfplay_game` / `generate_selfplay` — `PuctBot` (in `value_mode='outcome'`,
     root Dirichlet noise on) plays itself; per move it records the *ego-frame*
     observation, the *ego-frame* root visit distribution (the policy target), the
     critic's privileged inputs, and the mover. After each game every sample is
     labelled with the game outcome z ∈ {+1,-1} from its mover's perspective (win = +1,
     loss or truncation/circling = -1; the critic target). `play_selfplay_game` (one game)
     is the shared unit both the serial
     path here and `services/selfplay_collector.py`'s parallel workers call, so
     sequential and multiprocess generation run byte-identical game logic.
  2. `distill` — warm-starts from the current policy+critic and minimises
     `CE(policy, visits)` and `MSE(critic_raw, z)` in two independent Adam passes
     (mirroring PPO's separate actor/critic optimisers).

The new nets are saved via the existing `save_policy_checkpoint`/`save_critic_checkpoint`
(the critic with `return_mean=0`/`return_std=1`, since it now predicts z on the [-1,1]
outcome scale directly), so both the gauntlet and `PuctBot` load them unchanged — and
the next round's `PuctBot(value_mode='outcome')` picks them up as its new prior + leaf
value. The driver in `app/expert_iteration.py` alternates gen → distill → repeat, with
`services/selfplay_collector.py` doing the same job `rollout_collector.py` does for PPO:
a persistent pool of CPU worker processes that plays games in parallel.

Frame note: the policy/critic and their obs mask are ego-centric (board rotated 180°
when the mover is player 2), while `PuctBot`'s visit counts are in the absolute action
frame. Targets are remapped absolute→ego at record time (`WarChestEnv.remap_action`,
self-inverse) so they line up index-for-index with the masked policy logits. Policy and
critic must share one `obs_version` (they always do when saved from one PPO run); this
is asserted by the caller.
"""
import copy
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import track

from .environment.warchest_env import WarChestEnv, ACTION_SPACE_SIZE
from .environment.rollout_core import OPP_TYPE_IDX

logger = logging.getLogger('warchest')

# The critic's opponent one-hot slot for self-play data (same 'pool' analogue the
# search bots condition on — see rollout_core.OPP_ONEHOT_SLOT).
_POOL_SLOT = OPP_TYPE_IDX['pool']
_OPP_DIM = len(OPP_TYPE_IDX)


class SelfPlayDataset:
    """(obs, visit-distribution, critic inputs, outcome z) samples from puct self-play.

    A plain columnar container, deliberately *not* `RolloutBuffer`: that stores a
    single scalar action per step and is entangled with GAE/PPO, whereas ExIt needs a
    full [A] target distribution per step and no advantage/log-prob machinery. Fields
    are appended per move, then `stack()`ed into arrays for training / `save`.

    Arrays (after stack), all ego-frame where applicable:
        boards [N,C,7,7] f32 · globals [N,G] f32 · masks [N,A] bool ·
        visit_targets [N,A] f32 (normalised, zero on illegal) ·
        opp_onehots [N,3] f32 · privileged [N,P] f32 · z [N] f32 ·
        game_ids [N] i64 (which game each sample came from)

    `game_ids` exists so a held-out split can be taken by *game* rather than by sample:
    ~84 samples share one game's trajectory and its outcome, so a random per-sample
    permutation puts near-duplicates on both sides of the split and the held-out number
    comes out optimistic (docs/IDEAS.md R.10.5c; `eval_board_value.py` already holds out
    by round for the same reason). Ids are assigned per game at label time and re-based
    on `concat`, so they stay unique across workers and across replay-window rounds.
    """

    def __init__(self):
        self._boards, self._globals, self._masks = [], [], []
        self._visits, self._opp, self._priv = [], [], []
        self._movers, self._z, self._game_ids = [], [], []
        self._n_games = 0
        self.boards = self.globals = self.masks = None
        self.visit_targets = self.opp_onehots = self.privileged = self.z = None
        self.game_ids = None

    def add(self, *, board, global_feats, mask, visit_target, opp_onehot, privileged, mover):
        self._boards.append(board)
        self._globals.append(global_feats)
        self._masks.append(mask)
        self._visits.append(visit_target)
        self._opp.append(opp_onehot)
        self._priv.append(privileged)
        self._movers.append(mover)

    def label_last(self, n, winner):
        """Set z for the last `n` freshly-added, still-unlabelled samples (one game).

        z is from each sample's *mover* perspective: +1 if that mover won, else -1.
        Truncation (winner == 0 — nobody reached a win before the round limit) is scored
        -1 for BOTH movers, not 0: a truncated game means the bots circled without closing
        it out, a failure for both sides rather than a neutral draw. PuctBot's outcome-mode
        search uses the same -1 truncation value so target and search agree.
        """
        movers = self._movers[len(self._z):len(self._z) + n]
        game_id = self._n_games
        self._n_games += 1
        for mover in movers:
            self._z.append(1.0 if winner == mover else -1.0)
            self._game_ids.append(game_id)

    def stack(self):
        self.boards = np.stack(self._boards).astype(np.float32)
        self.globals = np.stack(self._globals).astype(np.float32)
        self.masks = np.stack(self._masks).astype(bool)
        self.visit_targets = np.stack(self._visits).astype(np.float32)
        self.opp_onehots = np.stack(self._opp).astype(np.float32)
        self.privileged = np.stack(self._priv).astype(np.float32)
        self.z = np.asarray(self._z, dtype=np.float32)
        self.game_ids = np.asarray(self._game_ids, dtype=np.int64)
        assert len(self.z) == len(self.boards), 'unlabelled samples remain (call label_last per game)'
        # Free the per-sample lists: after stacking they're pure duplication of the
        # arrays above, and a parallel worker ships this whole object back to the main
        # process (services/selfplay_collector.py), so trimming keeps that IPC payload
        # to just the compact arrays instead of shipping both copies.
        self._boards = self._globals = self._masks = []
        self._visits = self._opp = self._priv = []
        return self

    def __len__(self):
        return len(self._z) if self.z is None else len(self.z)

    @classmethod
    def concat(cls, parts):
        """Merge already-`stack()`ed datasets (one per parallel worker, or one per
        replay-window round) into one.

        `game_ids` are re-based per part, since each part numbers its own games from 0 —
        without the offset two workers' game 0 would merge into one apparent game and the
        by-game held-out split would leak across them. A part saved before `game_ids`
        existed contributes one id per sample (the old per-sample behaviour) rather than
        failing the merge.
        """
        ds = cls()
        ds.boards = np.concatenate([p.boards for p in parts])
        ds.globals = np.concatenate([p.globals for p in parts])
        ds.masks = np.concatenate([p.masks for p in parts])
        ds.visit_targets = np.concatenate([p.visit_targets for p in parts])
        ds.opp_onehots = np.concatenate([p.opp_onehots for p in parts])
        ds.privileged = np.concatenate([p.privileged for p in parts])
        ds.z = np.concatenate([p.z for p in parts])
        ids, offset = [], 0
        for p in parts:
            part_ids = p.game_ids
            if part_ids is None:
                part_ids = np.arange(len(p.z), dtype=np.int64)
            ids.append(np.asarray(part_ids, dtype=np.int64) + offset)
            offset += int(ids[-1].max()) + 1 if len(ids[-1]) else 0
        ds.game_ids = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)
        return ds

    def save(self, path):
        if self.boards is None:
            self.stack()
        np.savez_compressed(
            path, boards=self.boards, globals=self.globals, masks=self.masks,
            visit_targets=self.visit_targets, opp_onehots=self.opp_onehots,
            privileged=self.privileged, z=self.z, game_ids=self.game_ids,
        )

    @classmethod
    def load(cls, path):
        d = np.load(path)
        ds = cls()
        ds.boards, ds.globals, ds.masks = d['boards'], d['globals'], d['masks']
        ds.visit_targets, ds.opp_onehots = d['visit_targets'], d['opp_onehots']
        ds.privileged, ds.z = d['privileged'], d['z']
        if 'game_ids' in d:
            ds.game_ids = d['game_ids']
        else:
            # Pre-2026-08-20 dataset: no game boundaries were recorded. One id per sample
            # reproduces exactly the old random per-sample split rather than silently
            # inventing groups that don't exist.
            logger.warning('%s predates game_ids; held-out split falls back to per-sample '
                           '(leaky — see SelfPlayDataset docstring).', path)
            ds.game_ids = np.arange(len(ds.z), dtype=np.int64)
        return ds

    def iter_minibatches(self, minibatch_size, device, *, shuffle=True, indices=None):
        """Yield training minibatch dicts (torch tensors on `device`).

        Keys: board, global, mask (policy) · visit_targets (policy CE target) ·
        opp_onehot, privileged (critic) · z (critic MSE target).
        """
        n = len(self.z) if indices is None else len(indices)
        order = np.asarray(indices) if indices is not None else np.arange(n)
        if shuffle:
            order = order.copy()
            np.random.shuffle(order)
        for start in range(0, n, minibatch_size):
            idx = order[start:start + minibatch_size]
            yield {
                'board': torch.from_numpy(self.boards[idx]).to(device),
                'global': torch.from_numpy(self.globals[idx]).to(device),
                'mask': torch.from_numpy(self.masks[idx]).to(device),
                'visit_targets': torch.from_numpy(self.visit_targets[idx]).to(device),
                'opp_onehot': torch.from_numpy(self.opp_onehots[idx]).to(device),
                'privileged': torch.from_numpy(self.privileged[idx]).to(device),
                'z': torch.from_numpy(self.z[idx]).to(device),
            }


class ReplayWindow:
    """A sliding window of the last `max_rounds` self-play `SelfPlayDataset`s.

    `cmd_loop` (app/expert_iteration.py) used to call `distill()` on the round it had
    just generated only — one round, ~25k samples for a 300-game round — so every
    round's fine-tune saw a single network's narrow self-play slice and nothing else
    (docs/IDEAS.md R.10.5a, R.10.8 item 2). Every AlphaZero-family trainer instead
    samples from a window over several generations; this is that window. `push` keeps
    the most recent `max_rounds` datasets (oldest dropped first) regardless of whether
    the round that produced them was later promoted — a rejected round retries
    self-play from the same checkpoint, so its data is still drawn from the same
    distribution as the retry and stays useful pool for it.
    """

    def __init__(self, max_rounds):
        self.max_rounds = max(1, max_rounds)
        self._datasets = []

    def push(self, dataset):
        self._datasets.append(dataset)
        if len(self._datasets) > self.max_rounds:
            self._datasets.pop(0)

    def concat(self):
        if len(self._datasets) == 1:
            return self._datasets[0]
        return SelfPlayDataset.concat(self._datasets)

    def __len__(self):
        return len(self._datasets)


def _ego_visit_target(visit_counts, mover):
    """Absolute-frame `{action_id: prob}` → dense ego-frame [A] target vector.

    The policy is ego-centric, so for player-2 movers each absolute id is mapped to its
    ego index via `WarChestEnv.remap_action` (self-inverse) — the inverse of what
    `PuctBot._policy_priors` does when reading priors. Already normalised on input.
    """
    target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for action_id, prob in visit_counts.items():
        idx = WarChestEnv.remap_action(action_id) if mover == 2 else action_id
        target[idx] = prob
    return target


def _sample_move(visit_counts, temperature):
    """Pick an absolute action id from the visit distribution.

    `temperature <= 0` (or a degenerate distribution) → greedy argmax visits. Otherwise
    sample ∝ p**(1/temperature) — the AlphaZero exploration schedule; since `visit_counts`
    is already normalised, p**(1/T) renormalised equals raw-count**(1/T) renormalised.
    """
    actions = list(visit_counts.keys())
    probs = np.array([visit_counts[a] for a in actions], dtype=np.float64)
    if temperature and temperature > 0 and len(actions) > 1:
        p = probs ** (1.0 / temperature)
        s = p.sum()
        if s > 0:
            return int(np.random.choice(actions, p=p / s))
    return int(actions[int(np.argmax(probs))])


def play_selfplay_game(bot, env, dataset, *, temperature, temp_moves, max_turns,
                       apprentice_frac=0.0):
    """Play one self-play game with `bot` on `env` (already built with a matching
    encoder), appending recorded samples into `dataset` and labelling them with the
    outcome. Shared by `generate_selfplay` (serial) and every worker in
    `services/selfplay_collector.py` (parallel), so both paths run identical game
    logic — only how many games run concurrently differs.

    `apprentice_frac` is the probability that a recorded ply is *played* by the raw
    policy (sampled from `last_stats['policy_priors']`) instead of by the search. The
    search still runs at every ply and its visit distribution is still the recorded
    target, so this changes only *which states end up in the dataset*, at identical
    cost. `0.0` is AlphaZero's convention (the expert plays); `1.0` is Expert
    Iteration's original one (Anthony et al. 2017 — the apprentice plays, the expert
    labels). The latter matters here because a policy trained on states only the
    stronger player reaches has no target for the positions its own weaker play walks
    into — the standard compounding-error argument against plain behaviour cloning
    (Ross & Bagnell), and a candidate explanation for the recorded pattern where a
    round's in-sample agreement with the teacher rises 0.744 -> 0.861 while its
    strength does not move at all (docs/IDEAS.md R.10.12).

    Returns a per-game stats dict with *sums*, not means (`visit_entropy_sum`,
    `legal_sum`, `agree_sum`, `apprentice_sum`), so a caller aggregating many games —
    possibly across several workers — only has to add them up and divide once at the end
    (`summarize_game_stats`).
    """
    env.reset()
    n_before = len(dataset._movers)
    winner = 0
    visit_entropy_sum = 0.0
    legal_sum = 0
    agree_sum = 0
    apprentice_sum = 0
    ply = 0
    for ply in range(max_turns):
        mover = env.active_player
        bot.act(env)  # runs the search; sets last_stats['visit_counts']
        visit_counts = bot.last_stats.get('visit_counts') or {}
        if not visit_counts:
            # No search result (e.g. a single forced legal action) — play it,
            # but don't record a target-less sample.
            legal = env.get_possible_actions()
            _, _, term, trunc, info = env.step(legal[0])
        else:
            obs = env.generate_observation()
            probs = np.fromiter(visit_counts.values(), dtype=np.float64)
            visit_entropy_sum += float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())
            legal_sum += int(obs['valid_action_mask'].sum())
            # Move-level policy/search agreement: did the raw policy's top move match the
            # search's most-visited one? The direct policy-improvement signal — if these
            # agree almost always, the search isn't finding anything the prior didn't.
            policy_argmax = bot.last_stats.get('policy_argmax')
            move_agree = int(policy_argmax is not None
                             and policy_argmax == max(visit_counts, key=visit_counts.get))
            agree_sum += move_agree
            dataset.add(
                board=obs['board'], global_feats=obs['global'],
                mask=obs['valid_action_mask'],
                visit_target=_ego_visit_target(visit_counts, mover),
                opp_onehot=_pool_onehot(), privileged=env.get_privileged_features(),
                mover=mover,
            )
            priors = bot.last_stats.get('policy_priors') or {}
            as_apprentice = bool(priors) and apprentice_frac > 0.0 \
                and np.random.random() < apprentice_frac
            if as_apprentice:
                # Sample from the policy's own distribution, exactly as `PolicyAgent`
                # does in the gauntlet — this ply's state distribution is the student's.
                action = _sample_move(priors, 1.0)
                apprentice_sum += 1
            else:
                action = _sample_move(visit_counts, temperature if ply < temp_moves else 0.0)
            _, _, term, trunc, info = env.step(action)
            if not info['action'].is_valid:
                # puct only returns legal moves, so this is a belt-and-braces guard;
                # drop the just-added (untaken-move) sample and continue randomly.
                agree_sum -= move_agree
                apprentice_sum -= int(as_apprentice)
                for lst in (dataset._boards, dataset._globals, dataset._masks,
                            dataset._visits, dataset._opp, dataset._priv, dataset._movers):
                    lst.pop()
                _, _, term, trunc, info = env.make_random_step()
        if term:
            winner = mover if info['action'].is_valid else env.active_player
            break
        if trunc:
            winner = 0
            break
    n_samples = len(dataset._movers) - n_before
    dataset.label_last(n_samples, winner)
    return {
        'turns': ply + 1, 'winner': winner, 'n_samples': n_samples,
        'visit_entropy_sum': visit_entropy_sum, 'legal_sum': legal_sum,
        'agree_sum': agree_sum, 'apprentice_sum': apprentice_sum,
    }


def summarize_game_stats(game_stats):
    """Aggregate a list of `play_selfplay_game` stats dicts into one summary — the
    numbers `app/expert_iteration.py` logs after every generation round.
    """
    n_games = len(game_stats)
    n_samples = sum(s['n_samples'] for s in game_stats)
    turns = np.array([s['turns'] for s in game_stats], dtype=np.float64)
    decisive = sum(1 for s in game_stats if s['winner'] != 0)
    entropy_sum = sum(s['visit_entropy_sum'] for s in game_stats)
    legal_sum = sum(s['legal_sum'] for s in game_stats)
    agree_sum = sum(s['agree_sum'] for s in game_stats)
    apprentice_sum = sum(s.get('apprentice_sum', 0) for s in game_stats)
    return {
        'apprentice_frac': apprentice_sum / n_samples if n_samples else 0.0,
        'n_games': n_games,
        'n_samples': n_samples,
        'turns_mean': float(turns.mean()) if n_games else 0.0,
        'turns_min': int(turns.min()) if n_games else 0,
        'turns_max': int(turns.max()) if n_games else 0,
        'decisive_frac': decisive / n_games if n_games else 0.0,
        'mean_visit_entropy': entropy_sum / n_samples if n_samples else 0.0,
        'mean_legal_actions': legal_sum / n_samples if n_samples else 0.0,
        'mean_agreement': agree_sum / n_samples if n_samples else 0.0,
    }


def generate_selfplay(bot, n_games, *, encoder, temperature=1.0, temp_moves=12,
                      max_turns=2000, seed=None, desc='self-play', apprentice_frac=0.0):
    """Self-play `bot` for `n_games`, sequentially in-process (a live `rich` progress
    bar shows games completed — see `services/selfplay_collector.py` for the
    multi-process equivalent's own live bar).

    Returns `(dataset, game_stats)` — the labelled `SelfPlayDataset` and the list of
    per-game stats dicts (`summarize_game_stats` turns these into one summary). `bot`
    should be a `PuctBot`; `encoder` is the (shared) obs encoder for the nets being
    distilled — the recording env is built with it so `generate_observation()`/
    `get_privileged_features()` match what the nets consume. For the first
    `temp_moves` plies of each game the move is temperature-sampled from the visit
    counts (exploration); after that it is greedy.
    """
    dataset = SelfPlayDataset()
    env = WarChestEnv(save_game_history=False, obs_encoder=encoder)
    if seed is not None:
        np.random.seed(seed)
    game_stats = []
    for _ in track(range(n_games), description=desc):
        stats = play_selfplay_game(bot, env, dataset, temperature=temperature,
                                   temp_moves=temp_moves, max_turns=max_turns,
                                   apprentice_frac=apprentice_frac)
        game_stats.append(stats)
    return dataset.stack(), game_stats


def _pool_onehot():
    v = np.zeros(_OPP_DIM, dtype=np.float32)
    v[_POOL_SLOT] = 1.0
    return v


def _sharpen_target(visit_targets, visit_temp):
    """Raise a normalised visit distribution to `1/visit_temp` and renormalise.

    `visit_temp < 1` sharpens (peaks), `> 1` flattens; `1.0` is a no-op returned as-is.
    Zero entries (illegal actions) stay zero under any positive power, so legality is
    preserved. Exists because at this project's search budget the raw visit counts are
    *less* decisive than the policy already distilling toward them — measured on
    `data/exit/round0.npz` (2026-08-18): mean visit entropy 0.720 nats vs a pre-distill
    policy entropy of 0.469, and one round of unsharpened distillation dragged the
    policy's own entropy up to 0.875 (a confident policy turned into a near-uniform
    one — the ExIt loop making the model measurably worse, `docs/IDEAS.md` R.10.8 item
    2 turned R.10.9). `visit_temp=0.5` (the CLI default) brought the same dataset's
    mean entropy down to 0.304, a clear margin under 0.469, without going as far as
    the near-one-hot 0.090 nats `visit_temp=0.15` produces.
    """
    if visit_temp == 1.0:
        return visit_targets
    sharpened = visit_targets.clamp_min(0) ** (1.0 / visit_temp)
    return sharpened / sharpened.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _kl_to_reference(ref_joint, joint):
    """KL(ref || new) from two joint log-prob tensors `[B, A]` (illegal ids at -1e9,
    `exp(-1e9) == 0` so they drop out of the sum on their own — no mask needed).
    Mean over the batch, matching how `ce` is already averaged in `distill`.
    """
    ref_probs = ref_joint.exp()
    return (ref_probs * (ref_joint - joint)).sum(dim=1).mean()


def _split_by_game(game_ids, val_frac):
    """`(train_idx, val_idx)` holding out whole *games*, not random samples.

    A game contributes ~84 samples that share its trajectory and its single outcome z,
    so a per-sample permutation puts near-duplicates on both sides and the held-out loss
    reads better than it is — the same leak `eval_board_value.py` fixed by holding out by
    round (docs/IDEAS.md R.10.5c). With fewer than two games (unit tests, a one-game
    smoke) there is nothing to split by, so it falls back to the per-sample behaviour.
    """
    n = len(game_ids)
    if not val_frac:
        return np.arange(n), np.empty(0, dtype=np.int64)
    games = np.unique(game_ids)
    if len(games) < 2:
        n_val = max(1, int(n * val_frac))
        perm = np.random.permutation(n)
        return perm[n_val:], perm[:n_val]
    n_val_games = min(len(games) - 1, max(1, int(round(len(games) * val_frac))))
    val_games = np.random.permutation(games)[:n_val_games]
    is_val = np.isin(game_ids, val_games)
    return np.nonzero(~is_val)[0], np.nonzero(is_val)[0]


def distill(dataset, policy, critic, *, epochs=4, minibatch_size=256, lr_policy=3e-4,
            lr_critic=3e-4, grad_clip=1.0, device='cpu', val_frac=0.1, log_every=1,
            train_critic=True, visit_temp=0.5, kl_coeff=0.0, early_stop=True, patience=2):
    """Distil `policy` (CE→visits) and, unless `train_critic=False`, `critic` (MSE→z).

    Two independent Adam passes (like PPO's separate actor/critic optimisers). The
    policy CE is `-(sharpened_target * joint_log_probs).sum(1).mean()` over the full
    action space — illegal ids sit at -1e9 in the log-probs but are zeroed by the
    target. `visit_target` is raised to `1/visit_temp` and renormalised before use
    (`_sharpen_target`; `visit_temp=1.0` is the raw recorded distribution). Returns
    per-epoch train stats plus a final held-out `(ce, mse, agreement)`, so a caller
    can confirm both losses fell and the policy's argmax move now agrees more with the
    search's.

    The held-out split is taken by **game** (`_split_by_game`), and — unless
    `early_stop=False` — it is now a *control*, not just a report: the held-out CE/MSE
    is measured after every epoch, the best epoch's weights are kept and restored at
    the end, and training stops after `patience` epochs without improvement. Before
    this, `val_frac` was computed, printed and ignored, so all `epochs` ran regardless
    of what it said (docs/IDEAS.md R.10.5c / R.10.8 item 3) — on a ~25k-sample round
    that is ~350 unconstrained Adam steps toward a target whose informative content is
    a minority of samples.

    `train_critic=False` leaves the critic bit-identical and distils the policy only.
    That is not a convenience switch — it is the configuration the record argues for
    (`IDEAS.md` R.3, `independent_opponents.md` §1 fact 2): ExIt's *only* round that made
    the policy stronger was round 0, the one round that still used the PPO shaped-return
    critic; every round that ran on the self-distilled `z`-critic got weaker, and row 2b
    independently measured `z` as the worse of the two targets. With the critic frozen the
    leaf value cannot drift round to round, so the loop has exactly one moving part.

    `kl_coeff > 0` adds `kl_coeff * KL(policy_at_round_start || policy)` to the policy
    loss (`docs/IDEAS.md` R.10.10's replay window addresses *which* data a round trains
    on; this addresses *how far* one round is allowed to move the policy on it — plain
    CE has no trust region at all, unlike PPO's clip, so `epochs=4` at `lr=3e-4` can push
    arbitrarily far from a policy that was already fine everywhere the round's narrow
    self-play didn't visit). The reference is a frozen snapshot of `policy` taken before
    training starts, so the penalty is against where the round began, not a moving
    target. `0.0` (default) is the old, unregularised behaviour.
    """
    n = len(dataset)
    game_ids = dataset.game_ids if dataset.game_ids is not None else np.arange(n)
    train_idx, val_idx = _split_by_game(game_ids, val_frac)
    n_val = len(val_idx)

    policy.to(device)
    critic.to(device)
    p_opt = torch.optim.Adam(policy.parameters(), lr=lr_policy)
    c_opt = torch.optim.Adam(critic.parameters(), lr=lr_critic) if train_critic else None

    ref_policy = None
    if kl_coeff > 0:
        ref_policy = copy.deepcopy(policy).to(device).eval()
        for p in ref_policy.parameters():
            p.requires_grad_(False)

    watch = early_stop and n_val > 0
    best = {'ce': math.inf, 'mse': math.inf}
    best_epoch = {'policy': -1, 'critic': -1}
    best_state = {'policy': None, 'critic': None}
    stale = 0

    history = []
    for epoch in range(epochs):
        policy.train()
        critic.train() if train_critic else critic.eval()
        ce_sum = mse_sum = kl_sum = 0.0
        n_mb = 0
        for batch in dataset.iter_minibatches(minibatch_size, device, indices=train_idx):
            # Policy: cross-entropy to the (sharpened) visit distribution, plus an
            # optional KL trust-region term against the round's starting policy.
            joint = policy.joint_log_probs_batch(batch)  # [B, A], illegal at -1e9
            tgt = _sharpen_target(batch['visit_targets'], visit_temp)
            ce = -(tgt * joint).sum(dim=1).mean()
            loss = ce
            kl = None
            if ref_policy is not None:
                with torch.no_grad():
                    ref_joint = ref_policy.joint_log_probs_batch(batch)
                kl = _kl_to_reference(ref_joint, joint)
                loss = ce + kl_coeff * kl
            p_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            p_opt.step()
            if kl is not None:
                kl_sum += float(kl.item())

            # Critic: MSE to the game outcome z (raw output, no return-normaliser).
            # Skipped entirely when frozen — including the forward, so a frozen run costs
            # strictly less than a trained one rather than the same.
            if train_critic:
                val = critic.value_batch(batch).reshape(-1)
                mse = F.mse_loss(val, batch['z'])
                c_opt.zero_grad()
                mse.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                c_opt.step()
                mse_sum += float(mse.item())

            ce_sum += float(ce.item())
            n_mb += 1
        stats = {'epoch': epoch, 'ce': ce_sum / max(1, n_mb), 'mse': mse_sum / max(1, n_mb),
                 'kl': kl_sum / max(1, n_mb)}
        if watch:
            held = evaluate_distillation(dataset, policy, critic, device=device,
                                         indices=val_idx, visit_temp=visit_temp)
            stats['val_ce'] = held['ce']
            stats['val_mse'] = held['mse']
            improved = False
            if held['ce'] < best['ce'] - 1e-6:
                best['ce'] = held['ce']
                best_state['policy'] = copy.deepcopy(policy.state_dict())
                best_epoch['policy'] = epoch
                improved = True
            if train_critic and held['mse'] < best['mse'] - 1e-6:
                best['mse'] = held['mse']
                best_state['critic'] = copy.deepcopy(critic.state_dict())
                best_epoch['critic'] = epoch
                improved = True
            stale = 0 if improved else stale + 1
        history.append(stats)
        if log_every and (epoch + 1) % log_every == 0:
            logger.info('distill: epoch %d/%d — ce=%.4f mse=%.4f kl=%.4f%s',
                        epoch + 1, epochs, stats['ce'], stats['mse'], stats['kl'],
                        '' if not watch else
                        f" | held-out ce={stats['val_ce']:.4f} mse={stats['val_mse']:.4f}")
        if watch and stale >= patience:
            logger.info('distill: held-out loss has not improved for %d epochs — stopping at '
                        'epoch %d/%d', stale, epoch + 1, epochs)
            break

    if watch:
        # Roll back to the best-scoring epoch. Policy and critic are scored on their own
        # objective and can peak at different epochs; they are separate nets trained by
        # separate optimisers, so restoring them independently is well defined.
        if best_state['policy'] is not None and best_epoch['policy'] != history[-1]['epoch']:
            policy.load_state_dict(best_state['policy'])
            logger.info('distill: restored the policy from epoch %d (held-out ce %.4f)',
                        best_epoch['policy'] + 1, best['ce'])
        if best_state['critic'] is not None and best_epoch['critic'] != history[-1]['epoch']:
            critic.load_state_dict(best_state['critic'])
            logger.info('distill: restored the critic from epoch %d (held-out mse %.4f)',
                        best_epoch['critic'] + 1, best['mse'])

    val = evaluate_distillation(dataset, policy, critic, device=device, indices=val_idx,
                                visit_temp=visit_temp) if n_val else {}
    return {'history': history, 'val': val, 'n_train': len(train_idx), 'n_val': n_val,
            'best_epoch': best_epoch['policy'] + 1 if watch else None,
            'epochs_run': len(history), 'n_val_games': int(len(np.unique(game_ids[val_idx])))
            if n_val else 0}


def evaluate_distillation(dataset, policy, critic, *, device='cpu', indices=None,
                          minibatch_size=512, visit_temp=1.0):
    """Held-out `(ce, mse, agreement, policy_entropy, visit_entropy)`:
    - `ce`/`mse`: CE to visits (sharpened by `visit_temp`, see `_sharpen_target`),
      critic MSE to z (the actual training objectives).
    - `agreement`: fraction where the policy's argmax legal action matches the
      search's argmax visit. Unaffected by `visit_temp` (a monotonic transform of a
      distribution never changes its argmax).
    - `policy_entropy` / `visit_entropy`: mean entropy (nats) of the policy's own
      distribution vs. the (sharpened) visit-count target it's being distilled
      toward. If `visit_entropy` sits *above* `policy_entropy`, the search target is
      less decisive than the policy already is — distilling toward it will flatten
      (not sharpen) the policy, the opposite of the intended AlphaZero effect
      (usually a sign the search ran too few simulations/move for its branching, or
      that `visit_temp` needs to be lower). Logged before and after every distill
      call precisely so this is visible each round rather than a silent regression.
    """
    policy.to(device).eval()
    critic.to(device).eval()
    ce_sum = mse_sum = pol_ent_sum = visit_ent_sum = 0.0
    n_seen = agree = 0
    with torch.inference_mode():
        for batch in dataset.iter_minibatches(minibatch_size, device, shuffle=False, indices=indices):
            joint = policy.joint_log_probs_batch(batch)
            tgt = _sharpen_target(batch['visit_targets'], visit_temp)
            ce_sum += float((-(tgt * joint).sum(dim=1)).sum().item())
            val = critic.value_batch(batch).reshape(-1)
            mse_sum += float(((val - batch['z']) ** 2).sum().item())
            agree += int((joint.argmax(dim=1) == tgt.argmax(dim=1)).sum().item())
            probs = joint.exp()
            pol_ent_sum += float((-(probs * joint)).sum(dim=1).sum().item())
            visit_ent_sum += float((-(tgt * torch.log(tgt.clamp_min(1e-12)))).sum(dim=1).sum().item())
            n_seen += joint.shape[0]
    n_seen = max(1, n_seen)
    return {
        'ce': ce_sum / n_seen, 'mse': mse_sum / n_seen, 'agreement': agree / n_seen,
        'policy_entropy': pol_ent_sum / n_seen, 'visit_entropy': visit_ent_sum / n_seen,
    }
