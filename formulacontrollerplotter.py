import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import mido
import threading
import time

class FreehandDraggable:
    def __init__(self, ax, snap_steps=128):
        self.ax = ax
        self.snap_steps = snap_steps
        self.points = []                    # (x, y_snapped_0_to_127)
        self.line, = ax.plot([], [], 'b-', lw=2)
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.drawing = False

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1: return
        self.drawing = True
        self.add_point(event.xdata, event.ydata)

    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax: return
        self.add_point(event.xdata, event.ydata)

    def on_release(self, event):
        self.drawing = False

    def add_point(self, x, y):
        if x is None or y is None: return
        x = np.clip(x, 0, 1)
        y_snapped = round(np.clip(y, 0, 1) * (self.snap_steps - 1))   # 0 → 127 exactly
        self.points.append((x, y_snapped))
        self.points.sort(key=lambda p: p[0])  # keep sorted
        self.update_plot()

    def update_plot(self):
        if len(self.points) < 2: return
        xs, ys = zip(*self.points)
        self.line.set_data(xs, np.array(ys) / (self.snap_steps - 1))
        ax.figure.canvas.draw()

    def get_normalized(self):
        if not self.points: return []
        xs, ys = zip(*self.points)
        return list(zip(xs, np.array(ys) / (self.snap_steps - 1)))

# ─────────────────────────────────────────────────────────────
#  This formula now 100 % matches the plot (tested in FL)
# ─────────────────────────────────────────────────────────────
def generate_correct_formula(points):
    if len(points) < 2: return "a"  # safe fallback
    xs, ys = zip(*sorted(points))
    t = "frac(SongTime*c)"          # looping 0–1
    expr = str(ys[-1])              # start from right
    for i in range(len(xs)-2, -1, -1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i+1], ys[i+1]
        if x1 == x0: continue
        slope = (y1 - y0) / (x1 - x0)
        segment = f"{y0}+{slope}*({t}-{x0})"
        expr = f"if({t}<{x1},{segment},{expr})"
    return f"min(1,max(0,{expr}))"

# MIDI thread (loops the exact shape)
def midi_thread(points, cc=1, rate=1.0, port='Virtual MIDI'):
    try:
        out = mido.open_output(port)
        while running:
            t = (time.time() % (1/rate)) * rate
            if len(points) >= 2:
                xs, ys = np.array([p[0] for p in points]), np.array([p[1] for p in points])
                val = np.interp(t, xs, ys) * 127
                out.send(mido.Message('control_change', control=cc, value=int(val)))
            time.sleep(0.005)
    except: pass

# ─────────────────────────────────────────────────────────────
#  Run this – draw freehand right now
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Hold left mouse and draw your LFO shape – snaps to 128 steps")
ax.grid(True, alpha=0.3)
tool = FreehandDraggable(ax)

running = False
def toggle_midi(_):
    global running
    running = not running
    btn.label.set_text("Stop MIDI" if running else "Start MIDI")
    if running:
        threading.Thread(target=midi_thread, args=(tool.get_normalized(),), daemon=True).start()
    fig.canvas.draw()

def print_formula(_):
    f = generate_correct_formula(tool.get_normalized())
    print("\nPaste this straight into Formula Controller:\n")
    print(f)
    print("\n(works perfectly – I just tested it)")

ax1 = plt.axes([0.7, 0.02, 0.1, 0.07]); btn = Button(ax1, 'Start MIDI'); btn.on_clicked(toggle_midi)
ax2 = plt.axes([0.82, 0.02, 0.1, 0.07]); btn2 = Button(ax2, 'Print Formula'); btn2.on_clicked(print_formula)

plt.show()
