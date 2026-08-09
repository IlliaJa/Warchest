"""`RandomEvalBot` — one bot class, a *continuum* of playstyles (docs/IDEAS.md B1).

`SimGreedyBot` plays a fixed 2-ply search whose leaf is `HeuristicEvaluator` with a
fixed coefficient vector. That vector (`theta`, 8 dims — see `evaluation.THETA_KEYS`)
is the only thing separating a base racer from a material grinder from a recruit-economy
bot: the search machinery, the legal-move enumeration and the tactic resolution are
identical. Sampling θ instead of fixing it therefore yields hundreds of distinct,
fast, **policy-independent** opponents for the price of a constructor argument — which
is what `docs/independent_opponents.md` Phase 1 asks for and proposed to hand-write, one
archetype at a time.

Two things this is *not*:

  * **Not a strength play.** The `rich_eval=True` measurement (docs/bots.md) is a
    documented negative result: turning the four new terms on costs Elo, because a
    depth-bounded leaf cannot cash a long-horizon asset. That result rules θ out as a way
    to build a *strong* bot and says nothing about θ as a way to build a *varied* one —
    an opponent-pool entrant is judged on coverage and pressure, not Elo
    (`independent_opponents.md` §3). A θ that spams `recruit` is not a broken bot, it is
    the state distribution self-play never produces.
  * **Not unboundedly diverse.** The search also replays the env's real `Action.reward`
    and the holding reward, neither of which θ scales, so every member of the family
    still wants captures and still wants to win. θ redistributes emphasis on top of that
    floor. How much behaviour actually moves is a measurement, not an assumption — run
    `src/app/eval_theta_family.py`.

**Resampling is off by default.** Per-episode θ is the point of the family in the
*training pool* (`OpponentPool`, which calls `new_episode()` once per episode), but it is
wrong in the *gauntlet*: that schedule is antithetic — the two games of a pair replay one
draft with the seats swapped so the draft cancels — and a bot that changes playstyle
between the two games breaks the pairing it exists to exploit. Gauntlet/eval entrants
therefore pin one θ each, which is also what makes a per-θ behaviour table readable.

θ is drawn from the bot's own `numpy.random.Generator`, never the global RNG, because
`gauntlet.play_game` re-seeds the global RNG per game to pin the draft.
"""
import numpy as np

from .greedy_sim_bot import SimGreedyBot
from .lookahead_bot import LookaheadBot
from .evaluation import sample_theta, normalize_theta, theta_tag, format_theta


class ThetaSampling:
    """Mixin: owns θ, its private RNG, and the episode hook. Nothing search-specific.

    Mixed in *before* the search bot so cooperative `super().__init__(**kwargs)` reaches
    it, which is what lets one θ implementation sit on two different search depths
    (`RandomEvalBot` on `SimGreedyBot`, `RandomEvalLookaheadBot` on `LookaheadBot`)
    without either copy drifting from the other.
    """

    def __init__(self, theta=None, seed=0, resample_each_episode=False, name=None,
                 **kwargs):
        """
        Args:
            theta: explicit coefficient multipliers (see `evaluation.normalize_theta`);
                None draws one from `seed`. Partial dicts are filled from `LEGACY_THETA`.
            seed: seeds this bot's private `Generator` — the whole θ sequence, so two bots
                with the same seed are the same bot and different seeds are different
                playstyles.
            resample_each_episode: draw a fresh θ on every `new_episode()` call. See the
                module docstring for why this is off by default.
            name: display name; defaults to `<prefix>_<tag>` naming θ's dominant term.
            **kwargs: passed to the search bot (`reply_branching`, `time_budget`,
                `see_opponent_hand`, ...). `rich_eval` is not accepted — θ subsumes it.
        """
        if 'rich_eval' in kwargs:
            raise TypeError(f'{type(self).__name__} does not take rich_eval; pass theta '
                            f'instead (evaluation.RICH_THETA reproduces rich_eval=True)')
        self._rng = np.random.default_rng(seed)
        self.seed = seed
        self.resample_each_episode = resample_each_episode
        theta = normalize_theta(theta) if theta is not None else sample_theta(self._rng)
        super().__init__(name=name or f'{self._NAME_PREFIX}_{theta_tag(theta)}', **kwargs)
        # Set after super(): the search bot's __init__ builds the evaluator this re-weights.
        self.theta = None
        self.set_theta(theta)
        # θ history, for the resampling case: `usage` is a lifetime counter, so with
        # resampling on it mixes every θ this instance has played and only this list says
        # which. With θ pinned (the gauntlet/eval case) it holds exactly one entry and
        # `usage` *is* that θ's behaviour profile.
        self.theta_history = [self.theta]

    def set_theta(self, theta):
        """Adopt `theta` (in place — no search env rebuild)."""
        self.theta = normalize_theta(theta)
        self._evaluator.set_theta(self.theta)

    def new_episode(self):
        """Episode-boundary hook, called by `gauntlet.play_game` and `OpponentPool.sample`.

        A no-op unless `resample_each_episode`. Duck-typed rather than declared on a base
        class: the gauntlet calls it via `getattr`, so bots with no episode state simply
        do not define it.
        """
        if self.resample_each_episode:
            self.set_theta(sample_theta(self._rng))
            self.theta_history.append(self.theta)

    def reset_usage(self):
        """Clear the lifetime verb counter (harnesses that reuse one bot across arms)."""
        self.usage.clear()

    def describe(self):
        return f'{self.name}(seed={self.seed}): {format_theta(self.theta)}'


class RandomEvalBot(ThetaSampling, SimGreedyBot):
    """`SimGreedyBot` (2 ply) whose leaf-evaluator coefficients are a sampled θ.

    The cheap member of the family, ~18-25 ms/move — the only one affordable as a rollout
    opponent. Its measured ceiling is the base bot's: in a k=24 gauntlet the arms rate
    906-1140 against `greedy_fast` at 1131 and `random` at 585, i.e. a curriculum rung
    between the pool's two weak opponents, and 0.00-0.08 against a trained policy. See
    `RandomEvalLookaheadBot` when strength matters more than throughput.
    """

    _NAME_PREFIX = 'theta'


class RandomEvalLookaheadBot(ThetaSampling, LookaheadBot):
    """The same θ family on the iterative-deepening alpha-beta search instead of 2 ply.

    Exists because the sampled-θ measurement found the *base bot*, not the coefficients,
    was the binding constraint on strength: every dial peaks at or near the default θ, so
    within one search depth diversity can only be bought with Elo. `LookaheadBot` beats
    `SimGreedyBot` 0.79 (docs/bots.md, k=24), which is a much higher place to start.

    Costs ~104 ms/move at the default 0.1 s budget (IDEAS.md Table A) against
    `SimGreedyBot`'s ~18 — roughly 6x — so this is a gauntlet/eval-grade opponent, not
    something to give a large slice of the rollout hot path. `time_budget` is the dial.

    Whether θ still separates playstyles at this depth is a genuine open question, not an
    assumption: more plies put more of the *real* env reward between root and leaf, which
    could wash the leaf coefficients out. `src/app/eval_theta_family.py --base lookahead`
    is the measurement.
    """

    _NAME_PREFIX = 'thetaLA'
