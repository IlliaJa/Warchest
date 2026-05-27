import matplotlib


class GameRenderer:
    def __init__(self, env, history):
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.env = env
        self.history = history
        self.idx = 0

        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        plt.subplots_adjust(left=0, right=1, top=0.97, bottom=0.1)

        # Buttons
        axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bprev = Button(axprev, 'Prev')
        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)

        # Keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.draw()
        plt.show()

    def draw(self):
        self.ax.clear()
        self.env.set_state(self.history[self.idx])
        self.env.render(self.ax)
        self.ax.set_title(f"Action {self.idx}/{len(self.history) - 1}")
        self.fig.canvas.draw_idle()

    def next(self, event=None):
        if self.idx < len(self.history) - 1:
            self.idx += 1
            self.draw()

    def prev(self, event=None):
        if self.idx > 0:
            self.idx -= 1
            self.draw()

    def on_key(self, event):
        if event.key in ['right', 'd']:
            self.next()
        elif event.key in ['left', 'a']:
            self.prev()
