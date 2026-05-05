"""
plot_trials.py
==============
Plots integrated mixing score vs liquid lag from trial_results.csv.
Uses only stdlib + tkinter (built into Python) — no pip installs needed.

Run:
    python plot_trials.py
    python plot_trials.py path/to/trial_results.csv   # optional custom path
"""

import csv
import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# ── resolve CSV path ─────────────────────────────────────────
if len(sys.argv) > 1:
    CSV_PATH = sys.argv[1]
else:
    CSV_PATH = "outputs/trial_results.csv"

# If file not found, open a file-picker dialog
if not os.path.exists(CSV_PATH):
    root = tk.Tk()
    root.withdraw()
    CSV_PATH = filedialog.askopenfilename(
        title="Select trial_results.csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not CSV_PATH:
        print("No file selected. Exiting.")
        sys.exit(0)

# ── read CSV ─────────────────────────────────────────────────
no_pid_tau,  no_pid_score  = [], []
pid_tau,     pid_score     = [], []

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tau   = float(row["liquid_tau"])
        score = float(row["integrated_mix_score"])
        if row["condition"] == "no_pid":
            no_pid_tau.append(tau)
            no_pid_score.append(score)
        elif row["condition"] == "pid_no_lqr":
            pid_tau.append(tau)
            pid_score.append(score)

# Sort by tau just in case
no_pid_tau,  no_pid_score  = zip(*sorted(zip(no_pid_tau,  no_pid_score)))  if no_pid_tau  else ([], [])
pid_tau,     pid_score     = zip(*sorted(zip(pid_tau,     pid_score)))     if pid_tau     else ([], [])

# ── draw with tkinter Canvas (no matplotlib) ─────────────────
W, H         = 860, 520
PAD_L        = 80    # left  (y-axis labels)
PAD_R        = 30
PAD_T        = 40    # top
PAD_B        = 60    # bottom (x-axis labels)
PLOT_W       = W - PAD_L - PAD_R
PLOT_H       = H - PAD_T  - PAD_B

all_scores = list(no_pid_score) + list(pid_score)
all_tau    = list(no_pid_tau)   + list(pid_tau)

y_min = 0.0
y_max = max(all_scores) * 1.1 if all_scores else 1.0
x_min = min(all_tau)           if all_tau    else 0.0
x_max = max(all_tau)           if all_tau    else 2.0

def tx(val):
    """data-x → canvas-x"""
    return PAD_L + (val - x_min) / (x_max - x_min) * PLOT_W

def ty(val):
    """data-y → canvas-y"""
    return PAD_T + (1.0 - (val - y_min) / (y_max - y_min)) * PLOT_H

# ── grid lines & ticks ────────────────────────────────────────
def draw_grid(canvas):
    N_Y = 6
    for i in range(N_Y + 1):
        v  = y_min + (y_max - y_min) * i / N_Y
        cy = ty(v)
        canvas.create_line(PAD_L, cy, PAD_L + PLOT_W, cy,
                           fill="#e0e0e0", dash=(3, 4))
        canvas.create_text(PAD_L - 8, cy,
                           text=f"{v:.1f}", anchor="e",
                           font=("Helvetica", 10), fill="#555")

    N_X = 5
    for i in range(N_X + 1):
        v  = x_min + (x_max - x_min) * i / N_X
        cx = tx(v)
        canvas.create_line(cx, PAD_T, cx, PAD_T + PLOT_H,
                           fill="#e0e0e0", dash=(3, 4))
        canvas.create_text(cx, PAD_T + PLOT_H + 10,
                           text=f"{v:.2f}", anchor="n",
                           font=("Helvetica", 10), fill="#555")

    # Axes
    canvas.create_line(PAD_L, PAD_T, PAD_L, PAD_T + PLOT_H,
                       fill="#888", width=1)
    canvas.create_line(PAD_L, PAD_T + PLOT_H,
                       PAD_L + PLOT_W, PAD_T + PLOT_H,
                       fill="#888", width=1)

    # Axis labels
    canvas.create_text(PAD_L + PLOT_W // 2, H - 10,
                       text="Liquid lag τ (s)",
                       font=("Helvetica", 12), fill="#333")
    canvas.create_text(14, PAD_T + PLOT_H // 2,
                       text="Integrated mixing score (°·s)",
                       font=("Helvetica", 12), fill="#333",
                       angle=90)

# ── draw a series as a polyline + dots ───────────────────────
def draw_series(canvas, tau_vals, score_vals, color, dash=()):
    if len(tau_vals) < 2:
        return
    pts = [(tx(t), ty(s)) for t, s in zip(tau_vals, score_vals)]
    flat = [coord for p in pts for coord in p]
    canvas.create_line(*flat, fill=color, width=2, smooth=True, dash=dash)
    for cx, cy in pts:
        canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                           fill=color, outline=color)

# ── legend ───────────────────────────────────────────────────
def draw_legend(canvas):
    lx, ly = PAD_L + 20, PAD_T + 16
    # no-pid
    canvas.create_line(lx, ly, lx + 28, ly, fill="#378ADD", width=2)
    canvas.create_oval(lx+11, ly-3, lx+17, ly+3,
                       fill="#378ADD", outline="#378ADD")
    canvas.create_text(lx + 34, ly, text="No PID",
                       anchor="w", font=("Helvetica", 11), fill="#333")
    # pid no lqr
    ly2 = ly + 22
    canvas.create_line(lx, ly2, lx + 28, ly2,
                       fill="#D85A30", width=2, dash=(6, 3))
    canvas.create_oval(lx+11, ly2-3, lx+17, ly2+3,
                       fill="#D85A30", outline="#D85A30")
    canvas.create_text(lx + 34, ly2, text="PID (no LQR)",
                       anchor="w", font=("Helvetica", 11), fill="#333")

# ── tooltip ───────────────────────────────────────────────────
def make_tooltip(root, canvas,
                 no_pid_tau, no_pid_score,
                 pid_tau,    pid_score):
    tip_lbl = tk.Label(root, text="", bg="#ffffcc",
                       relief="solid", bd=1,
                       font=("Helvetica", 10), padx=4, pady=2)
    tip_win = None

    def find_nearest(mx, my):
        best_dist = 12  # px threshold
        best_text = None
        for series_label, taus, scores, color in [
            ("No PID",       no_pid_tau, no_pid_score, "#378ADD"),
            ("PID (no LQR)", pid_tau,    pid_score,    "#D85A30"),
        ]:
            for t, s in zip(taus, scores):
                cx, cy = tx(t), ty(s)
                d = ((mx - cx)**2 + (my - cy)**2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_text = (f"{series_label}\n"
                                 f"τ = {t:.3f} s\n"
                                 f"score = {s:.2f} °·s")
        return best_text

    def on_motion(event):
        text = find_nearest(event.x, event.y)
        if text:
            tip_lbl.config(text=text)
            tip_lbl.place(x=event.x + 12, y=event.y - 10)
        else:
            tip_lbl.place_forget()

    canvas.bind("<Motion>", on_motion)

# ── main window ───────────────────────────────────────────────
root = tk.Tk()
root.title("Integrated mixing score vs liquid lag")
root.resizable(False, False)

canvas = tk.Canvas(root, width=W, height=H, bg="white",
                   highlightthickness=0)
canvas.pack()

# Title
canvas.create_text(W // 2, 16,
                   text="Integrated mixing score vs liquid lag",
                   font=("Helvetica", 14, "bold"), fill="#222")

draw_grid(canvas)
draw_series(canvas, no_pid_tau, no_pid_score, "#378ADD", dash=())
draw_series(canvas, pid_tau,    pid_score,    "#D85A30", dash=(8, 4))
draw_legend(canvas)
make_tooltip(root, canvas,
             no_pid_tau, no_pid_score,
             pid_tau,    pid_score)

root.mainloop()