import matplotlib
from .coin_render import COIN_GLYPHS

PLAYER_COLORS = {1: 'darkred', 2: 'midnightblue'}

ACTION_SHORT = {
    'move': 'move',
    'attack': 'attack',
    'claim_base': 'base',
    'deploy': 'deploy',
    'bolster': 'bolster',
    'claim_initiative': 'init',
    'pass': 'pass',
    'recruit': 'recruit',
    'tactic': 'tactic',
}


class GameRenderer:
    def __init__(self, env, history, player_labels=None):
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.env = env
        self.history = history
        self.idx = 0
        self.player_labels = player_labels or {}

        self.fig = plt.figure(figsize=(14, 8))
        # Main board axes: leave right margin for the side panel
        self.ax = self.fig.add_axes([0.02, 0.04, 0.73, 0.90])

        # History log panel (right side, above buttons)
        self.ax_hist = self.fig.add_axes([0.77, 0.12, 0.16, 0.58])
        self.ax_hist.set_axis_off()

        # Navigation buttons stacked below the history log
        axfirst = self.fig.add_axes([0.77, 0.74, 0.16, 0.05])
        axnext  = self.fig.add_axes([0.86, 0.05, 0.07, 0.05])
        axprev  = self.fig.add_axes([0.77, 0.05, 0.07, 0.05])
        axlast  = self.fig.add_axes([0.77, 0.80, 0.16, 0.05])
        self.bfirst = Button(axfirst, '|◀  First')
        self.bnext  = Button(axnext,  'Next ▶')
        self.bprev  = Button(axprev,  '◀ Prev')
        self.blast  = Button(axlast,  'Last  ▶|')
        self.bfirst.on_clicked(self.first)
        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)
        self.blast.on_clicked(self.last)

        # Keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.draw()
        plt.show()

    def draw(self):
        self.ax.clear()
        self.env.set_state(self.history[self.idx])
        self.env.render(self.ax, player_labels=self.player_labels)
        n = len(self.history) - 1
        self.ax.set_title(f"Action {self.idx}/{n}", fontsize=10)
        self._draw_history_panel()
        self.fig.canvas.draw_idle()

    def _draw_history_panel(self):
        self.ax_hist.clear()
        self.ax_hist.set_axis_off()
        self.ax_hist.set_xlim(0, 1)
        self.ax_hist.set_ylim(0, 1)

        self.ax_hist.text(0.5, 0.99, 'Last 10 actions', ha='center', va='top',
                          fontsize=8, fontweight='bold', color='dimgray')

        # Each state[i] records the action that produced it (last_coin / last_action_type).
        # Show the 10 most recent including the current step.
        entries = []
        for i in range(max(1, self.idx - 9), self.idx + 1):
            st = self.history[i]
            if st.last_coin is None:
                continue
            pid = st.last_coin_player
            atype = ACTION_SHORT.get(st.last_action_type or '', '?')
            if st.last_action_type == 'recruit' and st.last_recruited_coin is not None:
                icon = COIN_GLYPHS.get(st.last_recruited_coin, '?')
            else:
                icon = COIN_GLYPHS.get(st.last_coin, '?')
            entries.append((pid, atype, icon, i == self.idx))

        for j, (pid, atype, icon, is_current) in enumerate(entries[-10:]):
            y = 0.91 - j * 0.092
            weight = 'bold' if is_current else 'normal'
            prefix = '▶' if is_current else ' '
            self.ax_hist.text(
                0.05, y,
                f'{prefix} P{pid}  {atype}  {icon}',
                ha='left', va='top', fontsize=8.5,
                color=PLAYER_COLORS[pid], fontweight=weight,
            )

    def next(self, event=None):
        if self.idx < len(self.history) - 1:
            self.idx += 1
            self.draw()

    def prev(self, event=None):
        if self.idx > 0:
            self.idx -= 1
            self.draw()

    def first(self, event=None):
        self.idx = 0
        self.draw()

    def last(self, event=None):
        self.idx = len(self.history) - 1
        self.draw()

    def on_key(self, event):
        if event.key in ['right', 'd']:
            self.next()
        elif event.key in ['left', 'a']:
            self.prev()
        elif event.key == 'home':
            self.first()
        elif event.key == 'end':
            self.last()
