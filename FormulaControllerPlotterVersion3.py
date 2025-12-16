import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pyperclip
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

# --- FL Studio Function Definitions for Curve Fitting ---

# Fourier Series (3 Harmonics)
def fourier_func(t, O, F, A1, P1, A2, P2, A3, P3):
    F_clamped = np.clip(F, 0.01, 10.0) 
    
    y = O 
    y += A1 * np.sin(t * F_clamped * np.pi * 2 + P1)    # 1st Harmonic
    y += A2 * np.sin(t * F_clamped * np.pi * 4 + P2)    # 2nd Harmonic
    y += A3 * np.sin(t * F_clamped * np.pi * 6 + P3)    # 3rd Harmonic
    
    return y

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# --- DraggablePoints Class (Unchanged for drawing logic) ---
class DraggablePoints:
    def __init__(self, ax, master_app, max_points=100):
        self.ax = ax
        self.master = master_app
        self.max_points = max_points
        self.snap_steps = 128 
        
        self.points = [(0.0, 0.0), (1.0, 1.0)] 
        self.points.sort(key=lambda p: p[0])

        self.line, = ax.plot([], [], 'b-', linewidth=2)
        self.markers = ax.scatter([], [], c='r', s=50, picker=5)
        
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_pick = ax.figure.canvas.mpl_connect('pick_event', self.on_pick)
        
        self.dragging = None  
        self.drawing = False  
        self.update_plot()

    def _snap_y(self, y_raw):
        y_clamped = np.clip(y_raw, 0, 1)
        y_snapped_index = round(y_clamped * (self.snap_steps - 1))
        return y_snapped_index / (self.snap_steps - 1)
        
    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1: return
        if self.dragging is None:
            self.drawing = True
            self.points = []
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.points.append((np.clip(x, 0, 1), self._snap_y(y)))

    def on_pick(self, event):
        if event.artist != self.markers: return
        ind = event.ind[0]
        if ind is not None and len(self.points) > ind:
            self.dragging = ind
            self.drawing = False 

    def on_release(self, event):
        if self.drawing:
            self.simplify_points(target_count=30) 
        if self.dragging is not None or self.drawing:
            self.dragging = None
            self.drawing = False
            self.points.sort(key=lambda p: p[0])
            self.update_plot(formula_update=True) 

    def on_motion(self, event):
        x, y = event.xdata, event.ydata
        if x is None or y is None or event.inaxes != self.ax: return
        if self.dragging is not None:
            index = self.dragging
            x_clamped = np.clip(x, 0, 1)
            y_snapped = self._snap_y(y)
            self.points[index] = (x_clamped, y_snapped)
            self.points.sort(key=lambda p: p[0])
            self.update_plot(formula_update=True)
        elif self.drawing:
            new_x = np.clip(x, 0, 1)
            new_y = self._snap_y(y)
            if not self.points or (abs(new_x - self.points[-1][0]) > 0.005 or abs(new_y - self.points[-1][1]) > 0.005):
                 self.points.append((new_x, new_y))
            if len(self.points) >= 2:
                xs, ys = zip(*self.points)
                self.line.set_data(xs, ys)
                self.ax.figure.canvas.draw_idle()

    def simplify_points(self, target_count=30):
        if not self.points: return
        xs_interp = np.array([p[0] for p in self.points])
        ys_interp = np.array([p[1] for p in self.points])
        if xs_interp.size < 2: return
        uniform_x = np.linspace(xs_interp.min(), xs_interp.max(), target_count)
        ys_simplified = np.interp(uniform_x, xs_interp, ys_interp)
        new_points = []
        for x, y in zip(uniform_x, ys_simplified):
            new_points.append((np.clip(x, 0, 1), self._snap_y(y)))
        new_points_dict = {f"{x:.4f}": v for x, v in new_points}
        self.points = sorted([(float(k), v) for k, v in new_points_dict.items()], key=lambda p: p[0])
        self.update_plot(formula_update=True)
        self.master.update_status("Curve points simplified for cleaner formula generation.")
        
    def smooth_points(self):
        if len(self.points) < 4:
            self.master.update_status("Need at least 4 points to smooth.")
            return

        xs = np.array([p[0] for p in self.points])
        ys = np.array([p[1] for p in self.points])
        
        cs = CubicSpline(xs, ys)
        new_xs = np.linspace(xs.min(), xs.max(), 50)
        new_ys = cs(new_xs)

        new_points = []
        for x, y in zip(new_xs, new_ys):
            new_points.append((np.clip(x, 0, 1), self._snap_y(y)))
            
        self.points = sorted(list(set(new_points)), key=lambda p: p[0])
        self.update_plot(formula_update=True)
        self.master.update_status("Curve smoothed and simplified.")

    def update_plot(self, formula_update=False):
        if len(self.points) < 2:
            self.line.set_data([], [])
            self.markers.set_offsets([])
        else:
            xs, ys = zip(*self.points)
            self.line.set_data(xs, ys)
            self.markers.set_offsets(np.c_[xs, ys])
        
        self.ax.figure.canvas.draw_idle()
        if formula_update:
            self.master.update_formula()

    def get_points_normalized(self):
        return sorted(self.points, key=lambda p: p[0])

# --- Formula Generation Function (Uses scaled_time_var) ---

def get_formula_outputs(points, scaled_time_var):
    """
    Tries Fourier Series fit.
    """
    if len(points) < 5:
        return "Not enough points for Best Fit", "Not enough points for Best Fit"

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    
    mean_y = ys.mean()
    crossings = np.where(np.diff(np.sign(ys - mean_y)))[0]
    F_guess = max(0.5, len(crossings) / 2) 

    A_guess = (ys.max() - ys.min()) / 2.0
    O_guess = (ys.max() + ys.min()) / 2.0
    
    # 1. Fourier Series Fit (3 Harmonics)
    try:
        p0_fourier = [O_guess, F_guess, A_guess, 0.0, 0.0, 0.0, 0.0, 0.0]
        bounds = (
            [-1.0, 0.01, -1.0, -2 * np.pi, -1.0, -2 * np.pi, -1.0, -2 * np.pi],
            [2.0, 10.0, 1.0, 2 * np.pi, 1.0, 2 * np.pi, 1.0, 2 * np.pi]        
        )
        
        popt_fourier, _ = curve_fit(fourier_func, xs, ys, p0=p0_fourier, bounds=bounds, maxfev=5000)
        y_pred_fourier = fourier_func(xs, *popt_fourier)
        mse_fourier = calculate_mse(ys, y_pred_fourier)
        
        O, F, A1, P1, A2, P2, A3, P3 = popt_fourier
        
        F = np.clip(F, 0.01, 10.0)
        
        # --- 1a. EXACT FORMULA (All Constants, uses scaled_time_var) ---
        exact_terms = []
        if abs(A1) > 0.001: exact_terms.append(f"{A1:.4f} * Sin({scaled_time_var} * {F:.4f} * Pi * 2 + {P1:.4f})")
        if abs(A2) > 0.001: exact_terms.append(f"{A2:.4f} * Sin({scaled_time_var} * {F:.4f} * Pi * 4 + {P2:.4f})")
        if abs(A3) > 0.001: exact_terms.append(f"{A3:.4f} * Sin({scaled_time_var} * {F:.4f} * Pi * 6 + {P3:.4f})")

        fourier_exact = f"Min(1, Max(0, {O:.4f} + {' + '.join(exact_terms)}))"
        
        # --- 1b. DYNAMIC FORMULA (A/B/C, uses scaled_time_var) ---
        dynamic_terms = []
        if abs(A1) > 0.001: 
            dynamic_terms.append(f"{A1:.4f} * a * Sin({scaled_time_var} * {F:.4f} * c * Pi * 2 + b * Pi)")
        if abs(A2) > 0.001: 
            dynamic_terms.append(f"{A2:.4f} * a * Sin({scaled_time_var} * {F:.4f} * c * Pi * 4 + {P2:.4f})") 
        if abs(A3) > 0.001: 
            dynamic_terms.append(f"{A3:.4f} * a * Sin({scaled_time_var} * {F:.4f} * c * Pi * 6 + {P3:.4f})")
        
        fourier_dynamic = f"Min(1, Max(0, {O:.4f} + {' + '.join(dynamic_terms)}))"

        if mse_fourier > 0.01:
            return (
                f"No close harmonic fit found (MSE: {mse_fourier:.4f}). Try Piecewise Linear.",
                f"No close harmonic fit found (MSE: {mse_fourier:.4f})."
            )
        
        return fourier_exact, fourier_dynamic

    except Exception as e:
        return f"Fourier fit failed: {e}", "Fourier fit failed."


def generate_piecewise_formula(points, scaled_time_var):
    """
    Generates the exact piecewise linear formula expression for FL Studio.
    Uses scaled_time_var as the input for time segments.
    """
    if len(points) < 2: return "0"
    points = sorted(points, key=lambda p: p[0])
    t = scaled_time_var # The scaled time input, e.g., (SongTime() / 4.0)
    
    # This formula structure already compensates for the time scaling implicitly
    # by using the scaled_time_var as the control input 't' in the If statements.
    
    # Start the formula from the last two points
    x_n_2, y_n_2 = points[-2]; x_n_1, y_n_1 = points[-1]
    try: slope = (y_n_1 - y_n_2) / (x_n_1 - x_n_2)
    except ZeroDivisionError: slope = 0
    formula = f"({y_n_2:.4f} + {slope:.4f} * ({t} - {x_n_2:.4f}))"
    
    # Build the nested If statements backwards
    for i in range(len(points) - 2, 0, -1):
        x0, y0 = points[i-1]; x1, y1 = points[i]
        try: slope = (y1 - y0) / (x1 - x0)
        except ZeroDivisionError: slope = 0
        segment = f"({y0:.4f} + {slope:.4f} * ({t} - {x0:.4f}))"
        formula = f"If({t} < {x1:.4f}, {segment}, {formula})"

    return f"Min(1, Max(0, {formula}))"


# --- Tkinter GUI Application (Finalized Layout) ---

class FLFormulaDrawerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FL Formula Drawer (Harmonic LFO Generator)")
        
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True) 
        
        # --- Time Input Control ---
        time_control_frame = ttk.Frame(main_frame)
        time_control_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(time_control_frame, text="Input Variable: SongTime()").pack(side=tk.LEFT, padx=(0, 10))
        self.input_var = tk.StringVar(value="SongTime()") 
        
        ttk.Label(time_control_frame, text="LFO Cycle Length (Bars):").pack(side=tk.LEFT, padx=(10, 5))
        self.cycle_length_var = tk.DoubleVar(value=1.0)
        self.cycle_length_entry = ttk.Entry(time_control_frame, width=5, textvariable=self.cycle_length_var)
        self.cycle_length_entry.pack(side=tk.LEFT)
        self.cycle_length_entry.bind('<Return>', lambda event: self.update_formula())

        # Matplotlib Figure and Canvas
        fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Draw Periodic LFO Shape (Time Axis 0-1 Bar/Cycle)")
        self.ax.set_xlabel("Normalized Time (X-Axis 0-1)")
        self.ax.set_ylabel("Modulation Output (Y-Axis 0-1)")
        
        self.canvas = FigureCanvasTkAgg(fig, master=main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True) 
        
        toolbar = NavigationToolbar2Tk(self.canvas, main_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X) 
        
        self.draggable = DraggablePoints(self.ax, self)
        
        # --- Control Buttons ---
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(5, 5)) 

        ttk.Button(control_frame, text="Snap to Contour (Smooth)", command=self.draggable.smooth_points).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Generate/Copy Formulas", command=self.handle_copy_and_generate).pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # --- Formula Display Area ---
        formula_frame = ttk.Frame(main_frame)
        formula_frame.pack(fill=tk.X, pady=(5, 0))

        # 1a. Fourier Exact Formula
        ttk.Label(formula_frame, text="1a. HARMONIC BEST FIT EXACT (Constant Coefficients)").pack(anchor=tk.W)
        self.exact_formula_text = tk.Text(formula_frame, height=3, wrap=tk.WORD, borderwidth=2, relief="groove", background="#E0F7FA")
        self.exact_formula_text.pack(fill=tk.X, pady=(0, 5)) 
        
        # 1b. Fourier Dynamic Formula
        ttk.Label(formula_frame, text="1b. HARMONIC BEST FIT DYNAMIC (A/B/C - Live Control)").pack(anchor=tk.W)
        self.dynamic_formula_text = tk.Text(formula_frame, height=3, wrap=tk.WORD, borderwidth=2, relief="groove", background="#FBE9E7")
        self.dynamic_formula_text.pack(fill=tk.X, pady=(0, 5)) 

        # 2. Piecewise Linear Formula
        ttk.Label(formula_frame, text="2. EXACT PIECEWISE LINEAR FORMULA (Guaranteed Match)").pack(anchor=tk.W)
        self.linear_formula_text = tk.Text(formula_frame, height=4, wrap=tk.WORD, borderwidth=2, relief="groove")
        self.linear_formula_text.pack(fill=tk.X) 
        
        # --- File Buttons Frame ---
        file_button_frame = ttk.Frame(main_frame)
        file_button_frame.pack(fill=tk.X, pady=5) 
        
        ttk.Button(file_button_frame, text="Save Formulas (.txt)", command=self.save_formulas).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_button_frame, text="Snapshot Plot (.png)", command=self.save_plot).pack(side=tk.LEFT, padx=5)
        
        self.update_formula()

    def update_status(self, message):
        self.status_label.config(text=message)
        
    def handle_copy_and_generate(self):
        # Update formula and then copy the best one (Exact Harmonic)
        self.update_formula(live_update=False)
        self.copy_exact_formula()
    
    def get_scaled_time_variable(self):
        try:
            length = self.cycle_length_var.get()
            if length == 0:
                self.update_status("Error: Cycle Length cannot be zero. Using 1.0.")
                length = 1.0
            
            if length == 1.0:
                return self.input_var.get()
            elif length > 1.0:
                return f"({self.input_var.get()} / {length:.4f})"
            elif length < 1.0:
                 # If length is 0.5, we multiply by 2.0
                multiplier = 1.0 / length
                return f"({self.input_var.get()} * {multiplier:.4f})"
        except:
            self.update_status("Invalid Cycle Length. Using 1.0.")
            return self.input_var.get()
            
    def update_formula(self, live_update=False):
        if live_update and (self.draggable.dragging is None and not self.draggable.drawing):
             return 
             
        points = self.draggable.get_points_normalized()
        scaled_time_var = self.get_scaled_time_variable()
        
        # Get Harmonic Best Fit Formulas
        exact_formula, dynamic_formula = get_formula_outputs(points, scaled_time_var)

        self.exact_formula_text.delete(1.0, tk.END)
        self.exact_formula_text.insert(tk.END, exact_formula)

        self.dynamic_formula_text.delete(1.0, tk.END)
        self.dynamic_formula_text.insert(tk.END, dynamic_formula)
        
        # Update Piecewise Linear Formula
        linear_formula = generate_piecewise_formula(points, scaled_time_var)
        self.linear_formula_text.delete(1.0, tk.END)
        self.linear_formula_text.insert(tk.END, linear_formula)
        
        self.update_status("Formulas generated (Harmonic Best Fit copied to clipboard).")

    def copy_exact_formula(self):
        formula = self.exact_formula_text.get(1.0, tk.END).strip()
        if formula and "No close harmonic fit found" not in formula:
            try:
                pyperclip.copy(formula)
                self.update_status("Harmonic Exact Formula copied to clipboard.")
            except Exception as e:
                self.update_status(f"Error copying: {e}. Copy manually.")
        elif "No close harmonic fit found" in formula:
             self.update_status("No good harmonic fit found. Copying Piecewise Linear instead.")
             pyperclip.copy(self.linear_formula_text.get(1.0, tk.END).strip())
        else:
             self.update_status("No formula generated.")

    def save_formulas(self):
        exact_formula = self.exact_formula_text.get(1.0, tk.END).strip()
        dynamic_formula = self.dynamic_formula_text.get(1.0, tk.END).strip()
        linear_formula = self.linear_formula_text.get(1.0, tk.END).strip()

        if not exact_formula and not linear_formula: return

        f = filedialog.asksaveasfile(mode='w', defaultextension=".txt", 
                                     filetypes=[("Text files", "*.txt")],
                                     title="Save FL Formulas")
        if f is None: return 
            
        content = (
            f"--- Input Variable Used: {self.input_var.get()} (Scaled by Cycle Length: {self.cycle_length_var.get()}) ---\n\n"
            
            f"1a. HARMONIC BEST FIT EXACT FORMULA (Use for perfect 1:1 match to drawing)\n"
            f"{exact_formula}\n\n"
            
            f"1b. HARMONIC BEST FIT DYNAMIC FORMULA (Use for live control via A, B, C knobs)\n"
            f"{dynamic_formula}\n\n"
            
            f"2. EXACT PIECEWISE LINEAR FORMULA (Guaranteed match for complex shapes)\n"
            f"{linear_formula}\n"
        )
        
        f.write(content)
        f.close()
        self.update_status(f"Formulas saved to {f.name}")
            
    def save_plot(self):
        f = filedialog.asksaveasfilename(defaultextension=".png", 
                                     filetypes=[("PNG files", "*.png")],
                                     title="Save Plot Snapshot")
        if f:
            self.ax.figure.savefig(f)
            self.update_status(f"Plot snapshot saved to {f}")


if __name__ == "__main__":
    app = FLFormulaDrawerApp()
    app.mainloop()
