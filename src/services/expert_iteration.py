"""Expert iteration (ExIt / AlphaZero) core: self-play data-gen + distillation.

`PuctBot` is the strongest agent in the gauntlet, so its *own* search output is a
better teacher than the raw policy that seeds it. This module closes that loop
(docs/next_steps.md — "search moves become new training targets"):

  1. `generate_selfplay` — `PuctBot` (in `value_mode='outcome'`, root Dirichlet noise
     on) plays itself; per move it records the *ego-frame* observation, the *ego-frame*
     root visit distribution (the policy target), the critic's privileged inputs, and
     the mover. After each game every sample is labelled with the game outcome z ∈
     {+1,0,-1} from its mover's perspective (the critic target).
  2. `distill` — warm-starts from the current policy+critic and minimises
     `CE(policy, visits)` and `MSE(critic_raw, z)` in two independent Adam passes
     (mirroring PPO's separate actor/critic optimisers).

The new nets are saved via the existing `save_policy_checkpoint`/`save_critic_checkpoint`
(the critic with `return_mean=0`/`return_std=1`, since it now predicts z on the [-1,1]
outcome scale directly), so both the gauntlet and `PuctBot` load them unchanged — and
the next round's `PuctBot(value_mode='outcome')` picks them up as its new prior + leaf
value. The driver in `app/expert_iteration.py` alternates gen → distill → repeat.

Frame note: the policy/critic and their obs mask are ego-centric (board rotated 180°
when the mover is player 2), while `PuctBot`'s visit counts are in the absolute action
frame. Targets are remapped absolute→ego at record time (`WarChestEnv.remap_action`,
self-inverse) so they line up index-for-index with the masked policy logits. Policy and
critic must share one `obs_version` (they always do when saved from one PPO run); this
is asserted by the caller.
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

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
        opp_onehots [N,3] f32 · privileged [N,P] f32 · z [N] f32
    """

    def __init__(self):
        self._boards, self._globals, self._masks = [], [], []
        self._visits, self._opp, self._priv = [], [], []
        self._movers, self._z = [], []
        self.boards = self.globals = self.masks = None
        self.visit_targets = self.opp_onehots = self.privileged = self.z = None

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

        z is from each sample's *mover* perspective: +1 if that mover won, -1 if the
        opponent won, 0 on a draw/truncation.
        """
        movers = self._movers[len(self._z):len(self._z) + n]
        for mover in movers:
            if winner == 0:
                self._z.append(0.0)
            elif winner == mover:
                self._z.append(1.0)
            else:
                self._z.append(-1.0)

    def stack(self):
        self.boards = np.stack(self._boards).astype(np.float32)
        self.globals = np.stack(self._globals).astype(np.float32)
        self.masks = np.stack(self._masks).astype(bool)
        self.visit_targets = np.stack(self._visits).astype(np.float32)
        self.opp_onehots = np.stack(self._opp).astype(np.float32)
        self.privileged = np.stack(self._priv).astype(np.float32)
        self.z = np.asarray(self._z, dtype=np.float32)
        assert len(self.z) == len(self.boards), 'unlabelled samples remain (call label_last per game)'
        return self

    def __len__(self):
        return len(self._z) if self.z is None else len(self.z)

    def save(self, path):
        if self.boards is None:
            self.stack()
        np.savez_compressed(
            path, boards=self.boards, globals=self.globals, masks=self.masks,
            visit_targets=self.visit_targets, opp_onehots=self.opp_onehots,
            privileged=self.privileged, z=self.z,
        )

    @classmethod
    def load(cls, path):
        d = np.load(path)
        ds = cls()
        ds.boards, ds.globals, ds.masks = d['boards'], d['globals'], d['masks']
        ds.visit_targets, ds.opp_onehots = d['visit_targets'], d['opp_onehots']
        ds.privileged, ds.z = d['privileged'], d['z']
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


def generate_selfplay(bot, n_games, *, encoder, temperature=1.0, temp_moves=12,
                      max_turns=2000, seed=None, log_every=10):
    """Self-play `bot` for `n_games` and return a labelled `SelfPlayDataset`.

    `bot` should be a `PuctBot(value_mode='outcome', dirichlet_alpha>0)`. `encoder` is
    the (shared) obs encoder for the nets being distilled — the recording env is built
    with it so `generate_observation()`/`get_privileged_features()` match what the nets
    consume. For the first `temp_moves` plies of each game the move is temperature-
    sampled from the visit counts (exploration); after that it is greedy.
    """
    dataset = SelfPlayDataset()
    if seed is not None:
        np.random.seed(seed)
    for g in range(n_games):
        env = WarChestEnv(save_game_history=False, obs_encoder=encoder)
        env.reset()
        n_before = len(dataset._movers)
        winner = 0
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
                dataset.add(
                    board=obs['board'], global_feats=obs['global'],
                    mask=obs['valid_action_mask'],
                    visit_target=_ego_visit_target(visit_counts, mover),
                    opp_onehot=_pool_onehot(), privileged=env.get_privileged_features(),
                    mover=mover,
                )
                action = _sample_move(visit_counts, temperature if ply < temp_moves else 0.0)
                _, _, term, trunc, info = env.step(action)
                if not info['action'].is_valid:
                    # puct only returns legal moves, so this is a belt-and-braces guard;
                    # drop the just-added (untaken-move) sample and continue randomly.
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
        dataset.label_last(len(dataset._movers) - n_before, winner)
        if log_every and (g + 1) % log_every == 0:
            logger.info('selfplay: %d/%d games, %d samples', g + 1, n_games, len(dataset._movers))
    return dataset.stack()


def _pool_onehot():
    v = np.zeros(_OPP_DIM, dtype=np.float32)
    v[_POOL_SLOT] = 1.0
    return v


def distill(dataset, policy, critic, *, epochs=4, minibatch_size=256, lr_policy=3e-4,
            lr_critic=3e-4, grad_clip=1.0, device='cpu', val_frac=0.1, log_every=1):
    """Distil `policy` (CE→visits) and `critic` (MSE→z) on `dataset`, in place.

    Two independent Adam passes (like PPO's separate actor/critic optimisers). The
    policy CE is `-(visit_target * joint_log_probs).sum(1).mean()` over the full action
    space — illegal ids sit at -1e9 in the log-probs but are zeroed by the target.
    Returns per-epoch train stats plus a final held-out `(ce, mse, agreement)` from a
    `val_frac` tail split, so a caller can confirm both losses fell and the policy's
    argmax move now agrees more with the search's.
    """
    n = len(dataset)
    n_val = max(1, int(n * val_frac)) if val_frac else 0
    perm = np.random.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    policy.to(device).train()
    critic.to(device).train()
    p_opt = torch.optim.Adam(policy.parameters(), lr=lr_policy)
    c_opt = torch.optim.Adam(critic.parameters(), lr=lr_critic)

    history = []
    for epoch in range(epochs):
        ce_sum = mse_sum = 0.0
        n_mb = 0
        for batch in dataset.iter_minibatches(minibatch_size, device, indices=train_idx):
            # Policy: cross-entropy to the visit distribution.
            joint = policy.joint_log_probs_batch(batch)  # [B, A], illegal at -1e9
            ce = -(batch['visit_targets'] * joint).sum(dim=1).mean()
            p_opt.zero_grad()
            ce.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            p_opt.step()

            # Critic: MSE to the game outcome z (raw output, no return-normaliser).
            val = critic.value_batch(batch).reshape(-1)
            mse = F.mse_loss(val, batch['z'])
            c_opt.zero_grad()
            mse.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
            c_opt.step()

            ce_sum += float(ce.item())
            mse_sum += float(mse.item())
            n_mb += 1
        stats = {'epoch': epoch, 'ce': ce_sum / max(1, n_mb), 'mse': mse_sum / max(1, n_mb)}
        history.append(stats)
        if log_every and (epoch + 1) % log_every == 0:
            logger.info('distill: epoch %d/%d — ce=%.4f mse=%.4f', epoch + 1, epochs, stats['ce'], stats['mse'])

    val = evaluate_distillation(dataset, policy, critic, device=device, indices=val_idx) if n_val else {}
    return {'history': history, 'val': val, 'n_train': len(train_idx), 'n_val': n_val}


def evaluate_distillation(dataset, policy, critic, *, device='cpu', indices=None, minibatch_size=512):
    """Held-out `(ce, mse, agreement)` — CE to visits, critic MSE to z, and the
    fraction where the policy's argmax legal action matches the search's argmax visit.
    """
    policy.to(device).eval()
    critic.to(device).eval()
    ce_sum = mse_sum = 0.0
    n_seen = agree = 0
    with torch.inference_mode():
        for batch in dataset.iter_minibatches(minibatch_size, device, shuffle=False, indices=indices):
            joint = policy.joint_log_probs_batch(batch)
            tgt = batch['visit_targets']
            ce_sum += float((-(tgt * joint).sum(dim=1)).sum().item())
            val = critic.value_batch(batch).reshape(-1)
            mse_sum += float(((val - batch['z']) ** 2).sum().item())
            agree += int((joint.argmax(dim=1) == tgt.argmax(dim=1)).sum().item())
            n_seen += joint.shape[0]
    n_seen = max(1, n_seen)
    return {'ce': ce_sum / n_seen, 'mse': mse_sum / n_seen, 'agreement': agree / n_seen}
