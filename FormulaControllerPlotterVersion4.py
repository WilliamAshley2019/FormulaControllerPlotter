import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pyperclip
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import math
import os

# --- FL Studio Helper Functions and Curve Fitting Definitions ---

# FL Studio helper functions
def Frac(x): return x - math.floor(x) # FL Studio's Frac function

# Waveform Library (normalized time t: 0-1)
# Note: These now accept Frequency (F), Amplitude (A), and Offset (O) parameters
def wf_sine(t, F, A, O): 
    return O + A * math.sin(t * F * np.pi * 2)

def wf_sawtooth(t, F, A, O): 
    # Sawtooth typically centered around 0.5
    return O + A * (Frac(t * F) - 0.5)

def wf_square(t, F, A, O): 
    # O controls the duty cycle (offset for FL-style square)
    return O if Frac(t * F) < A else (1.0 - O) # Simple High/Low based on Amplitude

def wf_triangle(t, F, A, O): 
    # Triangle starts at O, goes up/down by A
    val = 1 - 2 * abs(Frac(t * F + 0.25) - 0.5) 
    return O + A * (val - 0.5) # Center around O

def fourier_func(t, O, F, A1, P1, A2, P2, A3, P3):
    """Fourier Series function for curve_fit."""
    F_clamped = np.clip(F, 0.01, 10.0)
    
    y = O
    y += A1 * np.sin(t * F_clamped * np.pi * 2 + P1)     # 1st Harmonic
    y += A2 * np.sin(t * F_clamped * np.pi * 4 + P2)     # 2nd Harmonic
    y += A3 * np.sin(t * F_clamped * np.pi * 6 + P3)     # 3rd Harmonic
    
    return y

def calculate_mse(y_true, y_pred):
    """Mean Squared Error calculation."""
    return np.mean((y_true - y_pred)**2)

# --- GLOBAL FORMULA GENERATION FUNCTIONS (RESTORED) ---

def fit_fourier_series(x_data, y_data):
    """
    Fits the data to the 3-harmonic Fourier function.
    Returns (Optimal Parameters, Mean Squared Error) or (None, infinity) on failure.
    """
    if len(x_data) < 10:
        return None, float('inf')

    # Initial Guesses
    y_range = y_data.max() - y_data.min()
    p0 = [
        np.mean(y_data),  # O (Offset)
        1.0,              # F (Frequency)
        y_range / 2,      # A1 (Amplitude 1)
        0.0,              # P1 (Phase 1)
        y_range / 4,      # A2 (Amplitude 2)
        0.0,              # P2 (Phase 2)
        y_range / 8,      # A3 (Amplitude 3)
        0.0               # P3 (Phase 3)
    ]

    # Bounds: [O, F, A1, P1, A2, P2, A3, P3]
    # Restrict F (Frequency) to be close to 1.0 (since the drawing is 0-1 time)
    bounds = (
        [0.0, 0.01, 0.0, -np.pi*2, 0.0, -np.pi*2, 0.0, -np.pi*2], # Lower bounds
        [1.0, 10.0, 1.0, np.pi*2,  1.0, np.pi*2,  1.0, np.pi*2]  # Upper bounds
    )

    try:
        popt, pcov = curve_fit(fourier_func, x_data, y_data, p0=p0, bounds=bounds, maxfev=5000)
        y_fit = fourier_func(x_data, *popt)
        mse = calculate_mse(y_data, y_fit)
        return popt, mse
    except RuntimeError:
        return None, float('inf')
    except ValueError:
        return None, float('inf')

def get_formula_outputs(points, scaled_time_var):
    """
    Generates both the exact constant-coefficient and the dynamic A/B/C knob formulas.
    """
    if len(points) < 10:
        return "Not enough data points drawn.", "Not enough data points drawn."

    x_data = np.array([p[0] for p in points])
    y_data = np.array([p[1] for p in points])
    
    popt, mse = fit_fourier_series(x_data, y_data)

    if popt is None or mse > 0.005: # Threshold for a 'close' fit
        return (
            "No close harmonic fit found (MSE: {:.4f}). Try smoothing the curve or use the Piecewise formula.".format(mse),
            "No close harmonic fit found."
        )

    # Deconstruct optimal parameters
    O, F, A1, P1, A2, P2, A3, P3 = popt
    
    # 1. EXACT Formula (Constant Coefficients)
    exact_terms = []
    if abs(A1) > 0.001:
        exact_terms.append(f"{A1:.4f} * sin({scaled_time_var} * {F:.4f} * 6.2831853 + {P1:.4f})")
    if abs(A2) > 0.001:
        exact_terms.append(f"{A2:.4f} * sin({scaled_time_var} * {F*2:.4f} * 6.2831853 + {P2:.4f})")
    if abs(A3) > 0.001:
        exact_terms.append(f"{A3:.4f} * sin({scaled_time_var} * {F*3:.4f} * 6.2831853 + {P3:.4f})")
        
    exact_formula = f"{O:.4f}"
    if exact_terms:
        exact_formula += " + " + " + ".join(exact_terms).replace(" + -", " - ")
    
    # 2. DYNAMIC Formula (A/B/C Knobs in FL Studio)
    # Map parameters to A, B, C knobs. This is a creative choice.
    # Typically: A = Amplitude, B = Frequency/Speed, C = Phase/Offset
    
    # We will use the base parameters (O, F, A1, P1) and apply A, B, C for dynamic control
    
    # O -> B (Offset/Bias)
    # A1 -> A (Main Amplitude)
    # F -> C (Frequency Multiplier) - Note: C is usually Phase, but F is often a useful knob
    
    dynamic_terms = []
    # Amplitude controlled by 'A', Frequency controlled by 'C'
    # Base amplitude term (A1)
    if abs(A1) > 0.001:
        dynamic_terms.append(f"(A * {A1/0.5:.4f}) * sin({scaled_time_var} * (C * {F/0.5:.4f}) * 6.2831853 + {P1:.4f})")
    # Higher harmonics are scaled by the change in the base amplitude
    if abs(A2) > 0.001:
        dynamic_terms.append(f"(A * {A2/0.5:.4f}) * sin({scaled_time_var} * (C * {F*2/0.5:.4f}) * 6.2831853 + {P2:.4f})")
    if abs(A3) > 0.001:
        dynamic_terms.append(f"(A * {A3/0.5:.4f}) * sin({scaled_time_var} * (C * {F*3/0.5:.4f}) * 6.2831853 + {P3:.4f})")

    dynamic_formula = f"(B * {O/0.5:.4f})" # B knob controls the DC offset
    if dynamic_terms:
        dynamic_formula += " + " + " + ".join(dynamic_terms).replace(" + -", " - ")
        
    return exact_formula, dynamic_formula

def generate_piecewise_formula(points, scaled_time_var):
    """
    Generates a piecewise linear interpolation formula for perfect shape matching.
    """
    if len(points) < 2:
        return "0" 

    # Ensure points are sorted by x-coordinate
    points.sort(key=lambda p: p[0])

    formula = []
    
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        
        # Guard against zero-division (shouldn't happen with sorted, unique x points, but safer)
        if abs(x2 - x1) < 1e-6:
            continue

        # Calculate slope (m) and y-intercept (b) for y = m*x + b
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        
        # Condition: scaled_time_var < x2
        # Start the next segment with x1, ensuring the first segment is always true from x=0
        
        # The equation for the segment: (m * scaled_time_var) + b
        equation = f"({m:.4f} * {scaled_time_var} + {b:.4f})"
        
        # The condition: (scaled_time_var < x2)
        condition = f"({scaled_time_var} < {x2:.4f})"
        
        # Ternary structure: IF(condition, value_if_true, value_if_false)
        # FL Studio syntax: If(condition, value_if_true, value_if_false)
        
        # The last segment doesn't need an 'If' wrapper, it's the final value.
        if i == len(points) - 2:
            formula.append(equation)
        else:
            formula.append(f"If{condition}, {equation}, ")
    
    # Combine the formula components and close all the 'If' statements
    piecewise_formula = "".join(formula) + (")" * (len(points) - 2))
    
    # Add final clamping to ensure output stays within [0, 1]
    final_formula = f"min(1, max(0, {piecewise_formula}))"

    return final_formula


# --- DraggablePoints Class (Modified for Live Drawing/Parameter Mode) ---
class DraggablePoints:
    def __init__(self, ax, master_app, max_points=100):
        self.ax = ax
        self.master = master_app
        self.max_points = max_points
        self.snap_steps = 128 
        
        self.points = [(0.0, 0.5), (1.0, 0.5)] # Committed points
        self.temp_points = [] # Points for the live-edited shape
        self.points.sort(key=lambda p: p[0])

        self.line, = ax.plot([], [], 'b-', linewidth=2)
        self.temp_line, = ax.plot([], [], 'r--', linewidth=1) # New line for live edit
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
        
        if self.master.current_mode == 'FREEHAND':
            if self.dragging is None:
                self.drawing = True
                self.points = [] # Clear committed points for new freehand drawing
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    self.points.append((np.clip(x, 0, 1), self._snap_y(y)))

    def on_pick(self, event):
        if event.artist != self.markers: return
        if self.master.current_mode != 'FREEHAND': return # Disable dragging markers in param mode
            
        ind = event.ind[0]
        if ind is not None and len(self.points) > ind:
            self.dragging = ind
            self.drawing = False 

    def on_release(self, event):
        if self.master.current_mode == 'FREEHAND':
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
        
        if self.master.current_mode == 'FREEHAND':
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
        
        elif self.master.current_mode != 'FREEHAND':
            # In Parameter Mode, only redraw the temp line if parameters were changed via scroll
            if self.temp_points:
                self.update_plot()
                self.master.update_status(
                    f"Mode: {self.master.current_mode.upper()} | Freq (Shift): {self.master.param_freq.get():.2f} | Amp (Ctrl): {self.master.param_amp.get():.2f} | Offset/Duty (Alt): {self.master.param_offset.get():.2f}"
                )


    def simplify_points(self, target_count=30):
        # ... (Same simplification logic as before) ...
        if not self.points: return
        xs_interp = np.array([p[0] for p in self.points])
        ys_interp = np.array([p[1] for p in self.points])
        if xs_interp.size < 2: return
        
        # Sort points before interpolation
        sorted_indices = np.argsort(xs_interp)
        xs_interp = xs_interp[sorted_indices]
        ys_interp = ys_interp[sorted_indices]
        
        uniform_x = np.linspace(xs_interp.min(), xs_interp.max(), target_count)
        ys_simplified = np.interp(uniform_x, xs_interp, ys_interp)
        
        new_points = []
        for x, y in zip(uniform_x, ys_simplified):
            new_points.append((np.clip(x, 0, 1), self._snap_y(y)))
            
        self.points = sorted(list(set(new_points)), key=lambda p: p[0])
        
    def smooth_points(self):
        # ... (Same smooth logic as before) ...
        if len(self.points) < 4:
            self.master.update_status("Need at least 4 points to smooth.")
            return

        xs = np.array([p[0] for p in self.points])
        ys = np.array([p[1] for p in self.points])
        
        sorted_indices = np.argsort(xs)
        xs = xs[sorted_indices]
        ys = ys[sorted_indices]
        
        cs = CubicSpline(xs, ys)
        new_xs = np.linspace(xs.min(), xs.max(), 50)
        new_ys = cs(new_xs)

        new_points = []
        for x, y in zip(new_xs, new_ys):
            new_points.append((np.clip(x, 0, 1), self._snap_y(y)))
            
        self.points = sorted(list(set(new_points)), key=lambda p: p[0])
        self.update_plot(formula_update=True)
        self.master.update_status("Curve smoothed using Cubic Spline and simplified.")


    def update_plot(self, formula_update=False):
        """Redraws both committed (blue) and temporary (red) lines."""
        # Draw committed points
        if len(self.points) < 2:
            self.line.set_data([], [])
            self.markers.set_offsets([])
        else:
            xs, ys = zip(*self.points)
            self.line.set_data(xs, ys)
            self.markers.set_offsets(np.c_[xs, ys])
        
        # Draw temporary points (for live editing)
        if len(self.temp_points) > 1:
            txs, tys = zip(*self.temp_points)
            self.temp_line.set_data(txs, tys)
        else:
            self.temp_line.set_data([], [])

        self.ax.figure.canvas.draw_idle()
        if formula_update:
            self.master.update_formula()

    def get_points_normalized(self):
        return sorted(self.points, key=lambda p: p[0])
        
    def clear_points(self):
        self.points = [(0.0, 0.5), (1.0, 0.5)]
        self.temp_points = []
        self.update_plot(formula_update=True)
        self.master.update_status("Canvas cleared. Switched to FREEHAND mode.")
        self.master.set_mode('FREEHAND')
        
    def generate_temp_waveform(self, waveform_func, F, A, O):
        """Generates the waveform for live editing and stores it in temp_points."""
        steps = 100
        new_points = []
        
        for i in range(steps):
            t = i / (steps - 1)  # 0 to 1
            try:
                # Apply amplitude and offset parameters in a controlled way
                y_raw = waveform_func(t, F, A, O)
            except Exception as e:
                # print(f"Waveform generation error: {e}")
                continue
            
            # Clamp and snap the output for clean data
            y_snapped = self._snap_y(np.clip(y_raw, 0, 1))
            new_points.append((t, y_snapped))
        
        self.temp_points = sorted(new_points, key=lambda p: p[0])
        self.update_plot() # Only update plot, not formula, in live edit

    def commit_temp_waveform(self):
        """Commits the live-edited shape to the main points list."""
        if self.temp_points:
            # Simple copy and simplify
            self.points = self.temp_points
            self.simplify_points(target_count=30)
            self.temp_points = []
            self.update_plot(formula_update=True)
            self.master.update_status("Waveform committed and simplified.")
        else:
            self.master.update_status("No temporary waveform to commit.")


# --- Tkinter GUI Application ---

class FLFormulaDesignerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FL Formula Designer (LFO & Waveform Generator)")
        
        # --- State Variables ---
        self.current_mode = 'FREEHAND' # FREEHAND, SINE, SAWTOOTH, etc.
        self.param_freq = tk.DoubleVar(value=1.0)
        self.param_amp = tk.DoubleVar(value=0.5)
        self.param_offset = tk.DoubleVar(value=0.5)
        self.cycle_length_var = tk.DoubleVar(value=1.0)
        self.input_var = tk.StringVar(value="SongTime()")
        
        # Map modes to the corresponding function
        self.mode_to_func = {
            'SINE': wf_sine, 'SAWTOOTH': wf_sawtooth, 
            'SQUARE': wf_square, 'TRIANGLE': wf_triangle
        }
        
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True) 

        # --- Time Input Control (Row 0) ---
        time_control_frame = ttk.Frame(main_frame)
        time_control_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(time_control_frame, text="Input Var: SongTime()").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(time_control_frame, text="LFO Cycle Length (Bars):").pack(side=tk.LEFT, padx=(10, 5))
        self.cycle_length_entry = ttk.Entry(time_control_frame, width=5, textvariable=self.cycle_length_var)
        self.cycle_length_entry.pack(side=tk.LEFT)
        self.cycle_length_entry.bind('<Return>', lambda event: self.update_formula())

        # --- Waveform Presets/Modes ---
        preset_frame = ttk.LabelFrame(main_frame, text="Waveform Modes (Click, Adjust with Scroll/Modifier Keys, Press ENTER to Commit)")
        preset_frame.pack(fill=tk.X, pady=(5, 5))
        
        # Buttons to set the mode
        for mode in ['FREEHAND'] + list(self.mode_to_func.keys()):
            ttk.Button(preset_frame, text=mode.capitalize(), 
                       command=lambda m=mode: self.set_mode(m)).pack(side=tk.LEFT, padx=5)

        ttk.Button(preset_frame, text="Clear Canvas", command=self.clear_canvas).pack(side=tk.RIGHT, padx=5)

        # --- Matplotlib Figure and Canvas ---
        fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Draw Periodic LFO Shape (Time Axis 0-1 Bar/Cycle)")
        self.ax.set_xlabel("Normalized Time (X-Axis 0-1)")
        self.ax.set_ylabel("Modulation Output (Y-Axis 0-1)")
        
        self.canvas = FigureCanvasTkAgg(fig, master=main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True, pady=(5, 5)) 
        
        toolbar = NavigationToolbar2Tk(self.canvas, main_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        self.draggable = DraggablePoints(self.ax, self)
        
        # --- Event Bindings for Live Editing ---
        self.canvas_widget.bind('<MouseWheel>', self.on_scroll_wheel)
        self.canvas_widget.bind('<Control-MouseWheel>', self.on_scroll_wheel_ctrl)
        self.canvas_widget.bind('<Shift-MouseWheel>', self.on_scroll_wheel_shift)
        self.canvas_widget.bind('<Alt-MouseWheel>', self.on_scroll_wheel_alt)
        self.bind('<Return>', self.commit_and_exit_mode)
        
        # --- Control Buttons ---
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(5, 5))

        ttk.Button(control_frame, text="Snap to Contour (Smooth)", command=self.draggable.smooth_points).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Simplify Points", command=lambda: self.draggable.simplify_points(target_count=30)).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Generate/Copy Formulas", command=self.handle_copy_and_generate).pack(side=tk.LEFT, padx=(20, 5))
        
        self.status_label = ttk.Label(control_frame, text="")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # --- Formula Display Area ---
        formula_frame = ttk.Frame(main_frame)
        formula_frame.pack(fill=tk.X, pady=(5, 0))

        # Formula text areas (rest of layout is the same)
        ttk.Label(formula_frame, text="1a. HARMONIC BEST FIT EXACT (Constant Coefficients)").pack(anchor=tk.W)
        self.exact_formula_text = tk.Text(formula_frame, height=3, wrap=tk.WORD, borderwidth=2, relief="groove", background="#E0F7FA")
        self.exact_formula_text.pack(fill=tk.X, pady=(0, 5)) 
        
        ttk.Label(formula_frame, text="1b. HARMONIC BEST FIT DYNAMIC (A/B/C - Live Control)").pack(anchor=tk.W)
        self.dynamic_formula_text = tk.Text(formula_frame, height=3, wrap=tk.WORD, borderwidth=2, relief="groove", background="#FBE9E7")
        self.dynamic_formula_text.pack(fill=tk.X, pady=(0, 5)) 

        ttk.Label(formula_frame, text="2. EXACT PIECEWISE LINEAR FORMULA (Guaranteed Match)").pack(anchor=tk.W)
        self.linear_formula_text = tk.Text(formula_frame, height=4, wrap=tk.WORD, borderwidth=2, relief="groove")
        self.linear_formula_text.pack(fill=tk.X) 
        
        # --- File Buttons Frame ---
        file_button_frame = ttk.Frame(main_frame)
        file_button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_button_frame, text="Save Formulas (.txt)", command=self.save_formulas).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_button_frame, text="Snapshot Plot (.png)", command=self.save_plot).pack(side=tk.LEFT, padx=5)
        
        self.update_formula()

    # --- Mode and Parameter Control Methods ---

    def set_mode(self, mode):
        """Switches the drawing mode and initializes parameters."""
        self.current_mode = mode
        
        if mode != 'FREEHAND':
            self.param_freq.set(1.0)
            self.param_amp.set(0.5)
            self.param_offset.set(0.5)
            self.canvas_widget.config(cursor="crosshair")
            self.update_status(f"Entered {mode} Mode. Use scroll wheel with Ctrl/Shift/Alt to adjust parameters. Press ENTER to commit.")
            self.generate_live_waveform()
        else:
            self.canvas_widget.config(cursor="")
            self.draggable.temp_points = []
            self.draggable.update_plot()
            self.update_status("Switched to FREEHAND (Draw/Drag) mode.")

    def update_parameter(self, param_var, delta, min_val, max_val, step):
        """Adjusts a parameter variable based on scroll delta."""
        if self.current_mode == 'FREEHAND':
            return # Ignore scroll wheel when in freehand mode

        current = param_var.get()
        new_val = np.clip(current + delta * step, min_val, max_val)
        param_var.set(new_val)
        
        self.generate_live_waveform()
        self.update_status(
            f"Mode: {self.current_mode.upper()} | Freq (Shift): {self.param_freq.get():.2f} | Amp (Ctrl): {self.param_amp.get():.2f} | Offset/Duty (Alt): {self.param_offset.get():.2f}"
        )

    def on_scroll_wheel_ctrl(self, event):
        # Ctrl + Scroll Wheel: Amplitude (Y-Axis/Vertical Stretch)
        self.update_parameter(self.param_amp, event.delta / 120, 0.0, 1.0, 0.05)
        return "break"
        
    def on_scroll_wheel_shift(self, event):
        # Shift + Scroll Wheel: Frequency (Period/Repeats)
        self.update_parameter(self.param_freq, event.delta / 120, 0.5, 8.0, 0.25)
        return "break"

    def on_scroll_wheel_alt(self, event):
        # Alt + Scroll Wheel: Offset/Duty (Y-Offset/Duty Cycle)
        self.update_parameter(self.param_offset, event.delta / 120, 0.0, 1.0, 0.05)
        return "break"

    def on_scroll_wheel(self, event):
        # If not in a parameter mode, allow standard scrolling for the window
        if self.current_mode != 'FREEHAND':
            return "break" # Block un-modified scroll if in a parameter mode

    def generate_live_waveform(self):
        """Calculates the temporary waveform based on current parameters."""
        func = self.mode_to_func.get(self.current_mode)
        if func:
            F = self.param_freq.get()
            A = self.param_amp.get()
            O = self.param_offset.get()
            self.draggable.generate_temp_waveform(func, F, A, O)
            
    def commit_and_exit_mode(self, event):
        """Commits the temporary waveform to the main points and exits mode."""
        if self.current_mode != 'FREEHAND':
            self.draggable.commit_temp_waveform()
            self.set_mode('FREEHAND')
            self.update_formula() # Recalculate formulas after commitment
            return "break" # Prevent default key behavior

    def clear_canvas(self):
        """Clears the drawing canvas."""
        self.draggable.clear_points()
        self.set_mode('FREEHAND') # Ensure mode resets

    # --- Formula and File Methods (Same as before) ---

    def update_status(self, message):
        self.status_label.config(text=message)
        
    def handle_copy_and_generate(self):
        self.update_formula(live_update=False)
        self.copy_exact_formula()
        
    def get_scaled_time_variable(self):
        try:
            length = self.cycle_length_var.get()
            if length == 0: length = 1.0
            
            input_var = self.input_var.get()
            
            if length == 1.0:
                return input_var
            else:
                multiplier = 1.0 / length
                return f"({input_var} * {multiplier:.4f})"
        except:
            return self.input_var.get()
            
    def update_formula(self, live_update=False):
        # Only update if explicitly called (or if drawing is active, which is now FREEHAND)
        if live_update and (self.draggable.dragging is None and not self.draggable.drawing and self.current_mode == 'FREEHAND'):
             return 
             
        points = self.draggable.get_points_normalized()
        scaled_time_var = self.get_scaled_time_variable()
        
        # --- FIX: CALLING GLOBAL FUNCTIONS ---
        exact_formula, dynamic_formula = get_formula_outputs(points, scaled_time_var)
        
        self.exact_formula_text.delete(1.0, tk.END)
        self.exact_formula_text.insert(tk.END, exact_formula)

        self.dynamic_formula_text.delete(1.0, tk.END)
        self.dynamic_formula_text.insert(tk.END, dynamic_formula)
        
        linear_formula = generate_piecewise_formula(points, scaled_time_var)
        self.linear_formula_text.delete(1.0, tk.END)
        self.linear_formula_text.insert(tk.END, linear_formula)
        
        if self.current_mode == 'FREEHAND':
            self.update_status("Formulas generated.")

    def copy_exact_formula(self):
        formula = self.exact_formula_text.get(1.0, tk.END).strip()
        
        if formula and "No close harmonic fit found" not in formula and "failed" not in formula:
            try:
                pyperclip.copy(formula)
                self.update_status("Harmonic Exact Formula copied to clipboard.")
            except Exception as e:
                messagebox.showerror("Copy Error", f"Failed to copy to clipboard: {e}. Please copy manually.")
        else:
            linear_formula = self.linear_formula_text.get(1.0, tk.END).strip()
            if linear_formula and linear_formula != "0": 
                try:
                    pyperclip.copy(linear_formula)
                    self.update_status("Harmonic fit failed. Piecewise Linear Formula copied instead.")
                except Exception as e:
                    messagebox.showerror("Copy Error", f"Failed to copy to clipboard: {e}. Please copy manually.")
            else:
                 self.update_status("No valid formula generated to copy.")

    def save_formulas(self):
        # ... (File saving logic, same as before) ...
        exact_formula = self.exact_formula_text.get(1.0, tk.END).strip()
        dynamic_formula = self.dynamic_formula_text.get(1.0, tk.END).strip()
        linear_formula = self.linear_formula_text.get(1.0, tk.END).strip()

        if not exact_formula and not linear_formula: return

        f_path = filedialog.asksaveasfilename(defaultextension=".txt", 
                                     filetypes=[("Text files", "*.txt")],
                                     title="Save FL Formulas")
        if not f_path: return 
            
        content = (
            f"--- Input Variable Used: {self.input_var.get()} (Scaled by Cycle Length: {self.cycle_length_var.get()}) ---\n\n"
            
            f"1a. HARMONIC BEST FIT EXACT FORMULA (Use for perfect 1:1 match to drawing)\n"
            f"{exact_formula}\n\n"
            
            f"1b. HARMONIC BEST FIT DYNAMIC FORMULA (Use for live control via A, B, C knobs)\n"
            f"{dynamic_formula}\n\n"
            
            f"2. EXACT PIECEWISE LINEAR FORMULA (Guaranteed match for complex shapes)\n"
            f"{linear_formula}\n"
        )
        
        try:
            with open(f_path, 'w') as f:
                f.write(content)
            self.update_status(f"Formulas saved to {os.path.basename(f_path)}")
        except Exception as e:
             messagebox.showerror("Save Error", f"Failed to save file: {e}")
            
    def save_plot(self):
        # ... (Plot saving logic, same as before) ...
        f = filedialog.asksaveasfilename(defaultextension=".png", 
                                     filetypes=[("PNG files", "*.png")],
                                     title="Save Plot Snapshot")
        if f:
            self.ax.figure.savefig(f)
            self.update_status(f"Plot snapshot saved to {os.path.basename(f)}")


if __name__ == "__main__":
    app = FLFormulaDesignerApp()
    app.mainloop()
