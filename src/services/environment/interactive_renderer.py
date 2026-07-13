"""Interactive human-vs-model play: click-driven board/hand UI + a live critic
eval overlay, on top of the pure click-resolution logic in `play_controller.py`.

Two-phase click model (see play_controller.py's docstring for the full rationale):
  1. select an anchor (a highlighted friendly unit on the board, or a highlighted
     hand coin) — auto-skipped when only one anchor is legal;
  2. resolve its kind — auto-skipped when the anchor has only one kind, and
     auto-committed when that kind is a single-id "immediate" action (bolster,
     claim_base, tactic-initiate, pass, claim_initiative, decline). Otherwise a
     second click (a highlighted destination cell, or a "take" button for
     recruit) commits the action.

The opponent (always P2) auto-plays via a `GauntletAgent`-compatible `act(env)`;
its moves animate with a short pause between them.
"""
import copy
import os
import time

import matplotlib
import numpy as np
import torch

from .cell_ids import INVALID_CELL_ID
from .coin_render import draw_coin
from .game_record import build_game_record, determine_result, save_game_record
from .game_renderer import PLAYER_COLORS
from .game_state import DECK
from .play_controller import (
    IMMEDIATE_KINDS, KIND_LABELS, KIND_RECRUIT, compute_anchors, group_by_kind,
    nearest_cell, nearest_key, targets_for_group,
)
from .roster import COIN_BY_ID
from .rollout_core import OPP_TYPE_IDX
from .warchest_env import WarChestEnv

# Safety net against a pathological forced-action loop (e.g. a rules bug that
# makes every state look like a single forced action forever) — real games
# never chain more than a handful of truly forced steps in a row.
MAX_AUTOCASCADE = 50
MAX_OPPONENT_STEPS_PER_TURN = 500
OPPONENT_MOVE_PAUSE = 0.35


class PlayRenderer:
    def __init__(self, env, human_player, opponent, *, critic=None, critic_encoder=None,
                opp_type='pool', value_scale=1.0, value_shift=0.0, player_labels=None,
                save_dir='data/games'):
        matplotlib.use('TkAgg')
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self._plt = plt
        self._patches = patches
        self._Button = Button

        self.env = env
        self.human_player = human_player
        self.opponent = opponent
        self.critic = critic
        self.critic_encoder = critic_encoder
        self.value_scale = value_scale
        self.value_shift = value_shift
        self._calibrated = (value_scale != 1.0 or value_shift != 0.0)
        self.player_labels = player_labels or {}
        self.save_dir = save_dir

        self.opp_onehot = None
        self._eval_env = None
        if critic is not None:
            onehot = np.zeros(len(OPP_TYPE_IDX), dtype=np.float32)
            onehot[OPP_TYPE_IDX.get(opp_type, OPP_TYPE_IDX['pool'])] = 1.0
            self.opp_onehot = torch.from_numpy(onehot).unsqueeze(0).to(critic.device)
            # Scratch env for the eval overlay's one-ply lookahead (see _reliable_value):
            # we clone the live state into it, apply the human's candidate moves, and score
            # the resulting model-to-move states. Built with the critic's own encoder so its
            # internal generate_observation (run by step) matches the critic's obs shapes.
            self._eval_env = WarChestEnv(save_game_history=False, obs_encoder=critic_encoder)

        self.eval_history = []
        # Per-point flag set by _reliable_value: True when the value was obtained in the
        # critic's well-conditioned model-to-move regime (directly, or via the one-ply
        # lookahead at the human's node). False only in the rare fallback where even the
        # lookahead lands on another human-to-move state. Unreliable points are dimmed.
        self.eval_reliable = []
        self.ui_state = {'stage': 'idle'}
        self.anchors = {}
        self._hand_positions = {}
        self._menu_widgets = []
        self._busy = False
        self._done = False
        self._truncated = False
        self._last_action = None
        self._result_text = None
        self._saved_path = None
        self._autocascade_guard = 0

        self._board_cells = list(zip(*np.where(env.board.board != INVALID_CELL_ID)))

        self.fig = plt.figure(figsize=(16, 9))
        self.ax = self.fig.add_axes([0.02, 0.16, 0.60, 0.82])
        self._hand_bbox = (0.02, 0.02, 0.26, 0.11)
        self.ax_hand = self.fig.add_axes(list(self._hand_bbox))
        self.ax_hand.set_axis_off()
        self.ax_log = self.fig.add_axes([0.66, 0.70, 0.32, 0.28])
        self.ax_log.set_axis_off()
        self._menu_bbox = (0.66, 0.44, 0.32, 0.22)
        self.ax_menu = self.fig.add_axes(list(self._menu_bbox))
        self.ax_menu.set_axis_off()
        self.ax_eval = self.fig.add_axes([0.68, 0.04, 0.28, 0.34])

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        if self.env.active_player == self.human_player:
            self.anchors = compute_anchors(self.env, self.env.get_possible_actions())
            self._autocascade_guard = 0
            self._maybe_autoselect()
            self.draw()
        else:
            self.draw()
            self._run_opponent()

        plt.show()

    # ------------------------------------------------------------------
    # Click routing
    # ------------------------------------------------------------------

    def on_click(self, event):
        if self._busy or self._done or self.env.active_player != self.human_player:
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes is self.ax:
            cell = nearest_cell(event.xdata, event.ydata, self._board_cells)
            if cell is not None:
                self._handle_click(('cell', cell))
        elif event.inaxes is self.ax_hand:
            if not self._hand_positions:
                return
            coin = nearest_key(event.xdata, event.ydata, self._hand_positions)
            if coin is not None:
                self._handle_click(('coin', coin))

    def _handle_click(self, key):
        stage = self.ui_state['stage']
        if stage in ('idle', 'kind_choice'):
            if key in self.anchors:
                self._select_anchor(key)
        elif stage == 'target_choice_board':
            if key[0] == 'cell' and key[1] in self.ui_state['targets']:
                self._commit(self.ui_state['targets'][key[1]])
        # target_choice_menu (recruit's take-list) is resolved via menu buttons only.

    # ------------------------------------------------------------------
    # Selection state machine
    # ------------------------------------------------------------------

    def _select_anchor(self, key):
        ids = self.anchors[key]
        groups = group_by_kind(ids)
        if len(groups) == 1:
            kind, kind_ids = next(iter(groups.items()))
            self._select_kind(key, kind, kind_ids)
        else:
            self.ui_state = {'stage': 'kind_choice', 'anchor': key, 'groups': groups}
            self.draw()

    def _select_kind(self, anchor, kind, ids):
        if kind in IMMEDIATE_KINDS:
            self._commit(ids[0])
            return
        targets = targets_for_group(kind, ids)
        stage = 'target_choice_menu' if kind == KIND_RECRUIT else 'target_choice_board'
        self.ui_state = {'stage': stage, 'anchor': anchor, 'kind': kind, 'targets': targets}
        self.draw()

    def _maybe_autoselect(self):
        """Skip the first click when exactly one (non-decline) anchor is legal —
        this is also what lets forced tactic-continuation steps cascade through
        without any click at all, since a lone anchor + lone immediate kind
        commits straight away via `_select_kind` -> `_commit` -> `_after_step`.
        """
        non_decline = {k: v for k, v in self.anchors.items() if k[0] != 'decline'}
        if len(non_decline) != 1:
            return
        self._autocascade_guard += 1
        if self._autocascade_guard > MAX_AUTOCASCADE:
            return
        self._select_anchor(next(iter(non_decline)))

    def _commit(self, action_id):
        obs, reward, terminated, truncated, info = self.env.step(action_id)
        self._last_action = info['action']
        self._truncated = truncated
        self._done = terminated or truncated
        self._after_step()

    def _after_step(self):
        self._update_eval()
        if self._done:
            self._finish_game()
            self.draw()
            return
        if self.env.active_player == self.human_player:
            self.anchors = compute_anchors(self.env, self.env.get_possible_actions())
            self.ui_state = {'stage': 'idle'}
            self._autocascade_guard = 0
            self._maybe_autoselect()
            self.draw()
        else:
            self.draw()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self._run_opponent()

    def _run_opponent(self):
        self._busy = True
        guard = 0
        while not self._done and self.env.active_player != self.human_player:
            guard += 1
            if guard > MAX_OPPONENT_STEPS_PER_TURN:
                break
            action_id = self.opponent.act(self.env)
            obs, reward, terminated, truncated, info = self.env.step(action_id)
            self._last_action = info['action']
            self._truncated = truncated
            self._done = terminated or truncated
            self._update_eval()
            self.draw()
            self.fig.canvas.draw_idle()
            self._plt.pause(OPPONENT_MOVE_PAUSE)
            if self._done:
                break
        self._busy = False
        if self._done:
            self._finish_game()
            self.draw()
            return
        self.anchors = compute_anchors(self.env, self.env.get_possible_actions())
        self.ui_state = {'stage': 'idle'}
        self._autocascade_guard = 0
        self._maybe_autoselect()
        self.draw()

    # ------------------------------------------------------------------
    # Critic eval overlay
    # ------------------------------------------------------------------

    def _score_state(self, env):
        """Critic value of ``env``'s current state, from the *human's* perspective.

        Returns ``(value, model_to_move)``. The critic scores whoever is about to move
        (the obs encoder rotates ego-centrically around them), so we negate when the
        mover isn't the human — the negamax convention the game's antisymmetric rewards
        make valid (same as LookaheadCriticBot._critic_root_values). ``model_to_move``
        reports whether this was the critic's well-conditioned regime (opponent = the
        passive side, i.e. the model is up): the value is trustworthy only then.
        """
        obs = self.critic_encoder.encode(env)
        priv = self.critic_encoder.encode_privileged(env)
        priv_t = torch.from_numpy(priv).unsqueeze(0).to(self.critic.device)
        with torch.no_grad():
            raw = self.critic.value_single(obs, self.opp_onehot, priv_t).item()
        value = raw * self.value_scale + self.value_shift
        model_to_move = env.active_player != self.human_player
        if model_to_move:
            value = -value
        return value, model_to_move

    def _reliable_value(self):
        """Human-perspective value evaluated at a *model-to-move* frame at every ply.

        Naively querying the critic at both players' decision nodes and sign-flipping
        produces a sawtooth: the "player to move" tempo bias flips sign each ply, and
        the human's node also mis-conditions opp_onehot (there is no "self" opponent
        label — see docs/bots.md "Known limitation"). Fix: when the model is up, keep
        the reliable direct query; when the *human* is up, look one ply ahead over the
        human's legal moves and report the best resulting (model-to-move) value. Every
        plotted point then references the same "model about to move" regime, so the
        curve tracks the real game value instead of the tempo flip.

        Returns ``(value, reliable)`` — ``reliable`` is False only in the rare fallback
        where even the lookahead lands on another human-to-move state (a pending tactic
        continuation, or the model being out of coins).
        """
        env = self._eval_env
        base = copy.deepcopy(self.env.state)
        env.set_state(base)
        if env.active_player != self.human_player:
            return self._score_state(env)  # model to move: direct, well-conditioned

        best_v, best_reliable = None, False
        for action_id in env.get_possible_actions():
            env.set_state(copy.deepcopy(base))
            _, _, terminated, _, _ = env.step(action_id)
            if terminated:
                # The human's move just ended the game in the human's favour.
                value, reliable = 1.0, True
            else:
                value, reliable = self._score_state(env)
            if best_v is None or value > best_v:
                best_v, best_reliable = value, reliable
        if best_v is None:  # no legal human move (shouldn't happen mid-game)
            env.set_state(base)
            return self._score_state(env)
        return best_v, best_reliable

    def _update_eval(self):
        if self.critic is None:
            return
        value, reliable = self._reliable_value()
        self.eval_history.append(value)
        self.eval_reliable.append(reliable)

    # ------------------------------------------------------------------
    # End of game
    # ------------------------------------------------------------------

    def _finish_game(self):
        result = determine_result(self._last_action, self._truncated)
        players = {pid: self.player_labels.get(pid, f'P{pid}') for pid in (1, 2)}
        record = build_game_record(self.env, players, result)
        os.makedirs(self.save_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d-%H%M%S')
        path = os.path.join(self.save_dir, f'game_{ts}.json')
        save_game_record(record, path)
        self._saved_path = path
        if result['winner'] is None:
            self._result_text = f"Draw ({result['reason']}) — saved to {path}"
        else:
            who = 'You' if result['winner'] == self.human_player else players[result['winner']]
            self._result_text = f"{who} won! ({result['reason']}) — saved to {path}"
        print(self._result_text)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self):
        self._draw_board()
        self._draw_hand()
        self._draw_menu()
        self._draw_log()
        self._draw_eval()
        if self._result_text:
            self.ax.text(0.5, 1.02, self._result_text, transform=self.ax.transAxes,
                         ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
        self.fig.canvas.draw_idle()

    def _add_cell_highlight(self, r, q, edgecolor='gold', linewidth=2.0):
        x, y = WarChestEnv.convert_hex_grid_to_cartesian(r, q, hex_radius=0.5)
        hexagon = self._patches.RegularPolygon(
            (x, y), numVertices=6, radius=0.5, orientation=np.pi / 2,
            fill=False, edgecolor=edgecolor, linewidth=linewidth, zorder=6,
        )
        self.ax.add_patch(hexagon)

    def _draw_board(self):
        self.ax.clear()
        self.env.render(self.ax, player_labels=self.player_labels)
        stage = self.ui_state['stage']
        if stage in ('idle', 'kind_choice'):
            current_anchor = self.ui_state.get('anchor')
            for key in self.anchors:
                if key[0] != 'cell':
                    continue
                r, q = key[1]
                is_current = (stage == 'kind_choice' and key == current_anchor)
                self._add_cell_highlight(r, q, edgecolor='deepskyblue' if is_current else 'gold',
                                         linewidth=3.2 if is_current else 2.0)
        elif stage == 'target_choice_board':
            anchor = self.ui_state['anchor']
            if anchor[0] == 'cell':
                self._add_cell_highlight(*anchor[1], edgecolor='deepskyblue', linewidth=3.2)
            attack_like = self.ui_state['kind'] == 'attack'
            for (r, q) in self.ui_state['targets']:
                self._add_cell_highlight(r, q, edgecolor='tomato' if attack_like else 'limegreen',
                                         linewidth=2.6)

    def _draw_hand(self):
        self.ax_hand.clear()
        self.ax_hand.set_axis_off()
        # `ax_hand` is a wide, short panel; without an explicit equal aspect a
        # data-unit Circle gets squashed into an ellipse to fill the box. Compute
        # the box's actual physical aspect ratio (from the figure's current size,
        # so this stays correct even if the window is resized) and set the data
        # range to match it exactly, so `set_aspect('equal')` doesn't need to
        # shrink the box — the coins use the full panel width. Pin the box back to
        # its original position first: `set_aspect('equal')` below adjusts it in
        # place, and reading it back via `get_position()` on the next draw would
        # compound that adjustment into progressive shrinkage.
        self.ax_hand.set_position(list(self._hand_bbox))
        fig_w, fig_h = self.fig.get_size_inches()
        _, _, box_w, box_h = self._hand_bbox
        aspect = (box_w * fig_w) / (box_h * fig_h)
        self.ax_hand.set_xlim(0, aspect)
        self.ax_hand.set_ylim(0, 1)
        self.ax_hand.set_aspect('equal')
        self.ax_hand.text(0.05, 0.92, 'Your hand — click a coin to play it', ha='left', va='top',
                          fontsize=7.5, fontweight='bold', color='dimgray')
        hand = self.env.state.hands[self.human_player]
        types = [c for c in DECK if hand[c] > 0]
        self._hand_positions = {}
        if not types:
            self.ax_hand.text(0.05, 0.5, '(empty)', ha='left', va='center',
                              fontsize=9, color='lightgray')
            return
        n = len(types)
        radius = 0.16
        # Compact, left-aligned row (fixed gap between coins) rather than spreading
        # across the full panel width — with 1-2 coins, stretching to fill leaves
        # one coin pinned to each edge with a large empty gap between them.
        gap = radius * 0.8
        start_x = radius + 0.12
        xs = [start_x + i * (2 * radius + gap) for i in range(n)]
        stage = self.ui_state['stage']
        current_anchor = self.ui_state.get('anchor')
        y_coin = 0.52
        for x, coin in zip(xs, types):
            draw_coin(self.ax_hand, x, y_coin, coin, self.human_player, radius=radius, fontsize=10)
            self.ax_hand.text(x, y_coin - radius - 0.09, f'x{hand[coin]}', ha='center', va='top',
                              fontsize=7, color='dimgray')
            self._hand_positions[coin] = (x, y_coin)
            if stage in ('idle', 'kind_choice') and ('coin', coin) in self.anchors:
                is_current = (stage == 'kind_choice' and current_anchor == ('coin', coin))
                self.ax_hand.add_patch(self._patches.Circle(
                    (x, y_coin), radius=radius * 1.3, fill=False,
                    edgecolor='deepskyblue' if is_current else 'gold', linewidth=2.0, zorder=5,
                ))

    def _clear_menu_widgets(self):
        for w in self._menu_widgets:
            # Button keeps its own hover/motion callbacks registered on the canvas
            # independent of our reference to it — dropping the axes without
            # disconnecting them first leaves a stale handler that crashes the
            # next mouse-move (it reaches into an axes with no parent figure).
            try:
                w.disconnect_events()
            except Exception:
                pass
            try:
                w.ax.remove()
            except Exception:
                pass
        self._menu_widgets = []

    def _add_menu_button(self, idx, total, label, callback):
        left, bottom, width, height = self._menu_bbox
        btn_h = height / total * 0.85
        gap = height / total * 0.15
        y = bottom + height - (idx + 1) * (btn_h + gap)
        bax = self.fig.add_axes([left, y, width, btn_h])
        button = self._Button(bax, label)
        button.on_clicked(callback)
        self._menu_widgets.append(button)

    def _draw_menu(self):
        self.ax_menu.clear()
        self.ax_menu.set_axis_off()
        self._clear_menu_widgets()
        buttons = []
        stage = self.ui_state['stage']

        if stage == 'kind_choice':
            for kind, ids in self.ui_state['groups'].items():
                buttons.append((KIND_LABELS[kind], self._make_kind_callback(self.ui_state['anchor'], kind, ids)))
            buttons.append(('Cancel', self._make_cancel_callback()))
        elif stage == 'target_choice_menu':
            for take_coin, aid in self.ui_state['targets'].items():
                buttons.append((f'Take {COIN_BY_ID[take_coin].name}', self._make_commit_callback(aid)))
            buttons.append(('Cancel', self._make_cancel_callback()))
        elif stage == 'target_choice_board':
            kind_label = KIND_LABELS[self.ui_state['kind']]
            self.ax_menu.text(0.5, 0.92, f'Click a highlighted cell to {kind_label}',
                              ha='center', va='top', fontsize=9, wrap=True,
                              transform=self.ax_menu.transAxes)
            buttons.append(('Cancel', self._make_cancel_callback()))
        else:
            self.ax_menu.text(0.5, 0.92, 'Click a highlighted unit or hand coin',
                              ha='center', va='top', fontsize=9, transform=self.ax_menu.transAxes)

        # Decline is always independently offered when legal, regardless of stage.
        decline_ids = self.anchors.get(('decline', None))
        if decline_ids:
            buttons.append(('Decline', self._make_commit_callback(decline_ids[0])))

        for i, (label, cb) in enumerate(buttons):
            self._add_menu_button(i, len(buttons), label, cb)

    def _make_kind_callback(self, anchor, kind, ids):
        def _cb(event, anchor=anchor, kind=kind, ids=ids):
            self._select_kind(anchor, kind, ids)
        return _cb

    def _make_commit_callback(self, action_id):
        def _cb(event, action_id=action_id):
            self._commit(action_id)
        return _cb

    def _make_cancel_callback(self):
        def _cb(event):
            self.ui_state = {'stage': 'idle'}
            self.draw()
        return _cb

    def _draw_log(self):
        self.ax_log.clear()
        self.ax_log.set_axis_off()
        self.ax_log.set_xlim(0, 1)
        self.ax_log.set_ylim(0, 1)
        self.ax_log.text(0.5, 0.99, 'Recent events', ha='center', va='top', fontsize=8,
                         fontweight='bold', color='dimgray')
        events = [e for e in (self.env.event_log or []) if e.get('ply_kind') == 'action'][-12:]
        for j, ev in enumerate(events):
            y = 0.92 - j * 0.075
            color = PLAYER_COLORS.get(ev.get('player'), 'black')
            self.ax_log.text(0.03, y, ev['text'], ha='left', va='top', fontsize=7.2, color=color)

    def _draw_eval(self):
        self.ax_eval.clear()
        if self.critic is None:
            self.ax_eval.set_axis_off()
            self.ax_eval.text(0.5, 0.5, 'No critic checkpoint loaded', ha='center', va='center',
                              fontsize=9, color='lightgray', transform=self.ax_eval.transAxes)
            return
        hist = self.eval_history
        reliable = self.eval_reliable
        label = 'Critic (your view, calibrated)' if self._calibrated else 'Critic (your view, raw units)'
        title = f'{label}: {hist[-1]:+.3f}' if hist else f'{label}: N/A'
        self.ax_eval.set_title(title, fontsize=9)
        self.ax_eval.axhline(0, color='gray', linewidth=0.8)
        if hist:
            self.ax_eval.plot(range(len(hist)), hist, color='steelblue', linewidth=1.2, alpha=0.5)
            xs = np.arange(len(hist))
            reliable_mask = np.array(reliable, dtype=bool)
            # Reliable points (queried at the model's own decision node) are solid;
            # unreliable ones (queried at the human's node, mis-conditioned opp_onehot —
            # see eval_reliable's docstring) are faded so they don't read as equally trustworthy.
            self.ax_eval.scatter(xs[reliable_mask], np.array(hist)[reliable_mask],
                                 color='steelblue', s=14, zorder=4)
            self.ax_eval.scatter(xs[~reliable_mask], np.array(hist)[~reliable_mask],
                                 facecolors='none', edgecolors='steelblue', s=14,
                                 alpha=0.5, zorder=4)
            last_color = 'limegreen' if hist[-1] >= 0 else 'tomato'
            self.ax_eval.scatter([len(hist) - 1], [hist[-1]], color=last_color, s=25, zorder=5)
        self.ax_eval.set_xlabel('ply', fontsize=7)
        self.ax_eval.tick_params(labelsize=7)
