import tkinter as tk
from tkinter import messagebox, simpledialog
import math

# =========================
# Style / Layout
# =========================
FONT = ("Consolas", 9)
FONT_S = ("Consolas", 8)

SEC_W, SEC_H = 860, 620      # main section view
PLAN_W, PLAN_H = 360, 270    # plan view
SIDE_W = 380
M = 14

SEC_V_SCALE = 35

COL_GROUND = "#cc0000"       # thick red ground line
COL_ABOVE = "#ccffcc"
COL_SEMI  = "#fff2cc"
COL_BASE  = "#cce0ff"

def clamp(v, a, b):
    return max(a, min(b, v))

def parse_floor_heights(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("Enter floor heights like: 3,3,3,3")
    hs = []
    for p in parts:
        v = float(p)
        if v <= 0:
            raise ValueError("Floor heights must be > 0.")
        hs.append(v)
    return hs

def class_color(cls: str):
    if cls == "ABOVE":
        return COL_ABOVE
    if cls == "SEMI":
        return COL_SEMI
    return COL_BASE

# =========================
# App State
# =========================
class State:
    def __init__(self, master):
        self.country = tk.StringVar(master, value="US")            # US / UK / RU
        self.floor_heights_str = tk.StringVar(master, value="3,3,3,3")

        # building footprint rectangle (m)
        self.bld_w = tk.DoubleVar(master, value=36.0)
        self.bld_d = tk.DoubleVar(master, value=18.0)

        # US mode toggle
        self.us_side_min_mode = tk.BooleanVar(master, value=True)

        # perimeter ground points: {id, s (m along perimeter), elev (m)}
        self.points = []
        self.next_id = 1
        self.selected_id = None

        # UI controls
        self.building_x_px = tk.IntVar(master, value=120)
        self.auto_n = tk.IntVar(master, value=8)

        # popup gating
        self.warn_popup_active = False

# =========================
# Perimeter geometry (rectangle)
# =========================
def perimeter_length(st: State):
    w = float(st.bld_w.get()); d = float(st.bld_d.get())
    return 2 * (w + d)

def s_to_xy(st: State, s: float):
    w = float(st.bld_w.get()); d = float(st.bld_d.get())
    P = perimeter_length(st)
    s = s % P

    if s <= w:
        return (s, 0.0)
    s -= w
    if s <= d:
        return (w, s)
    s -= d
    if s <= w:
        return (w - s, d)
    s -= w
    return (0.0, d - s)

def xy_to_s(st: State, x: float, y: float):
    w = float(st.bld_w.get()); d = float(st.bld_d.get())
    x = clamp(x, 0.0, w); y = clamp(y, 0.0, d)

    dist_bottom = abs(y - 0.0)
    dist_top    = abs(y - d)
    dist_left   = abs(x - 0.0)
    dist_right  = abs(x - w)

    m = min(dist_bottom, dist_right, dist_top, dist_left)
    if m == dist_bottom:
        return x
    if m == dist_right:
        return w + y
    if m == dist_top:
        return w + d + (w - x)
    return w + d + w + (d - y)

# =========================
# View transforms
# =========================
def plan_xy_to_px(st: State, x, y):
    w = float(st.bld_w.get()); d = float(st.bld_d.get())
    sx = (PLAN_W - 2*M) / max(w, 0.001)
    sy = (PLAN_H - 2*M) / max(d, 0.001)
    s = min(sx, sy)
    ox = M + (PLAN_W - 2*M - w*s) / 2
    oy = M + (PLAN_H - 2*M - d*s) / 2
    return (ox + x*s, oy + y*s)

def plan_px_to_xy(st: State, px, py):
    w = float(st.bld_w.get()); d = float(st.bld_d.get())
    sx = (PLAN_W - 2*M) / max(w, 0.001)
    sy = (PLAN_H - 2*M) / max(d, 0.001)
    s = min(sx, sy)
    ox = M + (PLAN_W - 2*M - w*s) / 2
    oy = M + (PLAN_H - 2*M - d*s) / 2
    x = (px - ox) / s
    y = (py - oy) / s
    return x, y

def sec_s_to_px(st: State, s):
    P = perimeter_length(st)
    usable = SEC_W - 2*M
    return M + ((s % P) / P) * usable

def sec_elev_to_py(elev, emin):
    bottom = SEC_H - M
    return bottom - (elev - emin) * SEC_V_SCALE

def sec_py_to_elev(py, emin):
    bottom = SEC_H - M
    return emin + (bottom - py) / SEC_V_SCALE

# =========================
# HARD CONSTRAINT:
# Ground must be >= 0 everywhere (building never hangs)
# =========================
def ground_limits(hs):
    roof = sum(hs)
    # IMPORTANT: min ground = 0.0 so building base is never above ground
    return 0.0, roof + 8.0

def clamp_ground(elev, hs):
    lo, hi = ground_limits(hs)
    return clamp(elev, lo, hi)

# =========================
# Reference ground + storey logic
# =========================
def height_target(st: State, hs):
    roof = sum(hs)
    if st.country.get() == "UK" and len(hs) > 1:
        return sum(hs[:-1])
    return roof

def compute_reference_ground(st: State):
    if len(st.points) < 2:
        raise ValueError("Add at least 2 ground points.")

    hs = parse_floor_heights(st.floor_heights_str.get())
    roof = sum(hs)
    elevs = [p["elev"] for p in st.points]
    c = st.country.get()

    used_ids = []

    if c == "UK":
        min_p = min(st.points, key=lambda p: p["elev"])
        ref = min_p["elev"]
        used_ids = [min_p["id"]]
        expl = "UK: reference ground = lowest perimeter point (simplified)"
        return ref, used_ids, expl, roof

    if c == "RU":
        ref = sum(elevs) / len(elevs)
        used_ids = [p["id"] for p in st.points]
        expl = "RU: reference ground = average of all perimeter points (simplified)"
        return ref, used_ids, expl, roof

    # US
    if st.us_side_min_mode.get():
        w = float(st.bld_w.get()); d = float(st.bld_d.get())
        P = perimeter_length(st)
        intervals = [
            ("bottom", 0.0, w),
            ("right",  w, w+d),
            ("top",    w+d, 2*w+d),
            ("left",   2*w+d, 2*(w+d)),
        ]
        mins = []
        used_ids = []
        for _, a, b in intervals:
            side_pts = [p for p in st.points if a <= (p["s"] % P) <= b]
            if not side_pts:
                continue
            mp = min(side_pts, key=lambda p: p["elev"])
            mins.append(mp["elev"])
            used_ids.append(mp["id"])
        ref = sum(mins)/len(mins) if mins else sum(elevs)/len(elevs)
        expl = "US: reference ground = average of lowest point per side (visual proxy)"
        return ref, used_ids, expl, roof

    ref = sum(elevs)/len(elevs)
    used_ids = [p["id"] for p in st.points]
    expl = "US: reference ground = average of all points (simplified)"
    return ref, used_ids, expl, roof

def compute_storey_classes(st: State, ref):
    hs = parse_floor_heights(st.floor_heights_str.get())
    classes = []
    floor = 0.0
    for h in hs:
        if st.country.get() == "RU":
            diff = floor - ref
            if diff >= 0:
                classes.append("ABOVE")
            else:
                buried = abs(diff)
                classes.append("BASE" if buried > 0.5*h else "SEMI")
        else:
            classes.append("ABOVE" if floor >= ref else "BASE")
        floor += h
    return classes

# =========================
# Warnings (with your hard rule)
# =========================
def compute_warnings(st: State, hs):
    warnings = []
    if not st.points:
        return warnings

    elevs = [p["elev"] for p in st.points]
    lo = min(elevs); hi = max(elevs)
    roof = sum(hs)

    # HARD RULE: no ground below 0 anywhere
    if lo < 0.0:
        warnings.append("IMPOSSIBLE: some ground points are below 0.00m. Building would hang in the air.")

    # Also warn if ground only touches at exactly 0 and everything is flat (user mistake sometimes)
    if hi == 0.0 and lo == 0.0:
        warnings.append("Note: ground is exactly 0.00m everywhere (flat grade).")

    # sanity
    if (hi - lo) > 20:
        warnings.append("Unusual: terrain variation > 20m (check inputs).")
    if lo > roof:
        warnings.append("Unusual: ground above roof everywhere (check inputs).")

    return warnings

# =========================
# Drawing helpers
# =========================
def draw_visual_legend(canvas):
    canvas.delete("all")
    canvas.create_line(10, 16, 70, 16, fill=COL_GROUND, width=5)
    canvas.create_text(80, 16, text="Ground profile (always on top)", anchor="w", font=FONT_S)

    canvas.create_oval(10, 34, 22, 46, fill="orange", outline="black")
    canvas.create_text(80, 40, text="Ground point (drag)", anchor="w", font=FONT_S)

    canvas.create_oval(10, 56, 22, 68, fill="orange", outline="black")
    canvas.create_oval(8, 54, 24, 70, outline="blue", width=2)
    canvas.create_text(80, 62, text="Used for reference ground", anchor="w", font=FONT_S)

    canvas.create_rectangle(10, 76, 26, 88, fill=COL_ABOVE, outline="#999")
    canvas.create_rectangle(30, 76, 46, 88, fill=COL_SEMI, outline="#999")
    canvas.create_rectangle(50, 76, 66, 88, fill=COL_BASE, outline="#999")
    canvas.create_text(80, 82, text="Storeys: above / semi / basement", anchor="w", font=FONT_S)

def draw_grid(sec_canvas, emin, emax):
    for mval in range(int(math.floor(emin))-1, int(math.ceil(emax))+2):
        y = sec_elev_to_py(mval, emin)
        if M <= y <= SEC_H - M:
            sec_canvas.create_line(M, y, SEC_W - M, y, fill="#eeeeee")
            sec_canvas.create_text(6, y, text=f"{mval}m", anchor="w", font=FONT_S, fill="#888888")

def draw_plan(st: State, plan_canvas):
    plan_canvas.delete("all")
    plan_canvas.create_text(PLAN_W/2, 12, text="PLAN (Secondary)", font=FONT)

    w = float(st.bld_w.get()); d = float(st.bld_d.get())
    x1, y1 = plan_xy_to_px(st, 0, 0)
    x2, y2 = plan_xy_to_px(st, w, d)
    plan_canvas.create_rectangle(x1, y1, x2, y2, outline="#222222", width=1, fill="#f8f8ff")

    for p in st.points:
        x, y = s_to_xy(st, p["s"])
        px, py = plan_xy_to_px(st, x, y)
        r = 5
        outline = "blue" if p["id"] == st.selected_id else "black"
        plan_canvas.create_oval(px-r, py-r, px+r, py+r, fill="orange", outline=outline, width=2 if p["id"]==st.selected_id else 1)

def draw_section(st: State, sec_canvas, status_label, show_popup_warning: bool):
    sec_canvas.delete("all")
    sec_canvas.create_text(SEC_W/2, 12, text="SECTION (Primary) — Ground + Height Definitions", font=FONT)

    if len(st.points) < 2:
        sec_canvas.create_text(SEC_W/2, 50, text="Use Auto Points or Add Point. Drag points left/right + up/down.", font=FONT)
        status_label.config(text="Add ≥ 2 ground points.")
        return

    hs = parse_floor_heights(st.floor_heights_str.get())
    roof = sum(hs)

    ref, used_ids, expl, _ = compute_reference_ground(st)
    classes = compute_storey_classes(st, ref)

    warnings = compute_warnings(st, hs)
    status_label.config(text=("OK" if not warnings else "⚠ " + " | ".join(warnings)))

    if show_popup_warning:
        if warnings and not st.warn_popup_active:
            st.warn_popup_active = True
            messagebox.showwarning("Invalid geometry", "\n".join(warnings))
        if not warnings:
            st.warn_popup_active = False

    elevs = [p["elev"] for p in st.points]
    emin = min(0.0, min(elevs), -1.0)     # ensure 0 line is visible
    emax = max(roof + 2.0, max(elevs) + 1.0)

    draw_grid(sec_canvas, emin, emax)

    # Draw Floor 0 line (so users see what ground must cover)
    y0 = sec_elev_to_py(0.0, emin)
    sec_canvas.create_line(M, y0, SEC_W-M, y0, fill="#444444", dash=(3,3), width=1)
    sec_canvas.create_text(M+6, y0-12, text="Floor 0 = 0.00m", font=FONT_S, fill="#444444", anchor="nw")

    # Building box
    bx0 = M + int(st.building_x_px.get())
    bx1 = bx0 + 220

    floor = 0.0
    for i, h in enumerate(hs):
        yA = sec_elev_to_py(floor, emin)
        yB = sec_elev_to_py(floor + h, emin)
        sec_canvas.create_rectangle(bx0+1, yB, bx1-1, yA, fill=class_color(classes[i]), outline="")
        sec_canvas.create_line(bx0, yA, bx1, yA, fill="#bbbbbb", width=1)
        floor += h

    sec_canvas.create_rectangle(bx0, sec_elev_to_py(roof, emin), bx1, sec_elev_to_py(0.0, emin), outline="#222222", width=1)

    # Storey height arrow
    fh = hs[0]
    yA = sec_elev_to_py(0.0, emin)
    yB = sec_elev_to_py(fh, emin)
    xh = bx0 - 26
    sec_canvas.create_line(xh, yA, xh, yB, fill="#111111", width=2)
    sec_canvas.create_text(xh-6, (yA+yB)/2, text=f"{fh:.2f}m\nstorey", anchor="e", font=FONT_S)

    # Reference ground line
    yref = sec_elev_to_py(ref, emin)
    sec_canvas.create_line(M, yref, SEC_W-M, yref, fill="red", dash=(6,4), width=2)

    label_y = yref - 18
    if label_y < M + 22:
        label_y = yref + 14
    sec_canvas.create_text(M+6, label_y, text=f"Reference ground = {ref:.2f}m", font=FONT_S, fill="red", anchor="nw")
    sec_canvas.create_text(M+6, label_y+14, text=expl, font=FONT_S, fill="#333333", anchor="nw")

    # Building height arrow
    target = height_target(st, hs)
    y_target = sec_elev_to_py(target, emin)
    x_arrow = bx1 + 26
    sec_canvas.create_line(x_arrow, yref, x_arrow, y_target, fill="#111111", width=2)
    for yy, sgn in [(y_target, 1), (yref, -1)]:
        sec_canvas.create_line(x_arrow, yy, x_arrow-6, yy+8*sgn, fill="#111111", width=2)
        sec_canvas.create_line(x_arrow, yy, x_arrow+6, yy+8*sgn, fill="#111111", width=2)
    sec_canvas.create_text(x_arrow+10, (yref+y_target)/2, text="Building height", anchor="w", font=FONT_S)

    # Ground line ON TOP (thick red)
    pts = sorted(st.points, key=lambda p: p["s"] % perimeter_length(st))
    poly = []
    for p in pts:
        x = sec_s_to_px(st, p["s"])
        y = sec_elev_to_py(p["elev"], emin)
        poly.extend([x, y])
    sec_canvas.create_line(*poly, fill=COL_GROUND, width=5, smooth=True)

    # Points + used rings
    for p in st.points:
        x = sec_s_to_px(st, p["s"])
        y = sec_elev_to_py(p["elev"], emin)
        r = 7
        if p["id"] in used_ids:
            sec_canvas.create_oval(x-(r+3), y-(r+3), x+(r+3), y+(r+3), outline="blue", width=2)
        outline = "blue" if p["id"] == st.selected_id else "black"
        sec_canvas.create_oval(x-r, y-r, x+r, y+r, fill="orange", outline=outline, width=2 if p["id"]==st.selected_id else 1)
        sec_canvas.create_text(x+9, y-9, text=f"{p['elev']:.1f}", font=FONT_S, anchor="nw")

# =========================
# Build UI
# =========================
root = tk.Tk()
root.title("Building Geometry Guide — Visual Code Logic (US / UK / RU)")
root.geometry("1300x760")

state = State(root)

top = tk.Frame(root)
top.pack(fill=tk.X, padx=8, pady=6)

tk.Label(top, text="Country", font=FONT).pack(side=tk.LEFT)
tk.OptionMenu(top, state.country, "US", "UK", "RU").pack(side=tk.LEFT, padx=6)

tk.Label(top, text="Plan W (m)", font=FONT).pack(side=tk.LEFT, padx=(10,4))
tk.Entry(top, textvariable=state.bld_w, width=6, font=FONT).pack(side=tk.LEFT)

tk.Label(top, text="Plan D (m)", font=FONT).pack(side=tk.LEFT, padx=(10,4))
tk.Entry(top, textvariable=state.bld_d, width=6, font=FONT).pack(side=tk.LEFT)

tk.Label(top, text="Floor heights (m)", font=FONT).pack(side=tk.LEFT, padx=(10,4))
tk.Entry(top, textvariable=state.floor_heights_str, width=18, font=FONT).pack(side=tk.LEFT)

tk.Checkbutton(top, text="US side-min mode", variable=state.us_side_min_mode, font=FONT).pack(side=tk.LEFT, padx=(12,4))

tk.Label(top, text="Building X", font=FONT).pack(side=tk.LEFT, padx=(12,4))
tk.Scale(top, from_=0, to=260, orient="horizontal", variable=state.building_x_px, length=170).pack(side=tk.LEFT)

tk.Label(top, text="Auto N", font=FONT).pack(side=tk.LEFT, padx=(12,4))
tk.Spinbox(top, from_=2, to=40, textvariable=state.auto_n, width=4, font=FONT).pack(side=tk.LEFT)

main = tk.Frame(root)
main.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

sec = tk.Canvas(main, width=SEC_W, height=SEC_H, bg="white")
sec.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

side = tk.Frame(main, width=SIDE_W)
side.pack(side=tk.LEFT, fill=tk.Y, padx=(10,0))

plan = tk.Canvas(side, width=PLAN_W, height=PLAN_H, bg="white")
plan.pack()

status_box = tk.LabelFrame(side, text="Status", font=FONT)
status_box.pack(fill=tk.X, pady=(10,6))
status_lbl = tk.Label(status_box, text="OK", font=FONT_S, justify="left", wraplength=340)
status_lbl.pack(anchor="w", padx=6, pady=4)

legend = tk.LabelFrame(side, text="Legend", font=FONT)
legend.pack(fill=tk.X, pady=(0,8))
leg = tk.Canvas(legend, width=350, height=92, bg="white", highlightthickness=0)
leg.pack(padx=6, pady=6)
draw_visual_legend(leg)

# Output
out_frame = tk.LabelFrame(side, text="Output", font=FONT)
out_frame.pack(fill=tk.BOTH, expand=True)
out = tk.Text(out_frame, height=12, font=FONT_S)
out.pack(fill=tk.BOTH, expand=True)

# Toolbar buttons (Generate next to others)
btns = tk.Frame(side)
btns.pack(fill=tk.X, pady=(8,6))

# =========================
# App actions
# =========================
def redraw(show_popup_warning=True):
    try:
        parse_floor_heights(state.floor_heights_str.get())
    except Exception:
        pass
    draw_section(state, sec, status_lbl, show_popup_warning=show_popup_warning)
    draw_plan(state, plan)

def apply_inputs():
    try:
        hs = parse_floor_heights(state.floor_heights_str.get())
        if float(state.bld_w.get()) <= 1 or float(state.bld_d.get()) <= 1:
            raise ValueError("Plan W/D must be > 1m.")

        # enforce hard constraint by clamping all existing points to >=0
        for p in state.points:
            p["elev"] = clamp_ground(p["elev"], hs)

    except Exception as e:
        messagebox.showerror("Invalid input", str(e))
        return
    redraw(True)

def add_point():
    elev = simpledialog.askfloat("Ground elevation (m)", "Ground elevation (m). Must be >= 0.00m:")
    if elev is None:
        return
    hs = parse_floor_heights(state.floor_heights_str.get())
    elev = clamp_ground(float(elev), hs)

    P = perimeter_length(state)
    s = P/2.0 if not state.points else (state.points[-1]["s"] + P*0.12) % P

    state.points.append({"id": state.next_id, "s": float(s), "elev": float(elev)})
    state.selected_id = state.next_id
    state.next_id += 1
    redraw(True)

def auto_points():
    n = int(state.auto_n.get())
    base = simpledialog.askfloat("Base elevation", "Base elevation (m). Must be >= 0.00m:", initialvalue=0.0)
    if base is None:
        return
    hs = parse_floor_heights(state.floor_heights_str.get())
    base = clamp_ground(float(base), hs)

    P = perimeter_length(state)
    state.points = []
    state.next_id = 1
    for i in range(n):
        s = (i/(n-1)) * P
        state.points.append({"id": state.next_id, "s": float(s), "elev": float(base)})
        state.next_id += 1
    state.selected_id = None
    redraw(True)

def delete_point():
    if state.selected_id is None:
        messagebox.showinfo("Delete", "Select a point first.")
        return
    state.points = [p for p in state.points if p["id"] != state.selected_id]
    state.selected_id = None
    redraw(True)

def generate_output():
    out.delete("1.0", tk.END)
    try:
        hs = parse_floor_heights(state.floor_heights_str.get())
        ref, used_ids, expl, roof = compute_reference_ground(state)
        classes = compute_storey_classes(state, ref)
        warnings = compute_warnings(state, hs)
    except Exception as e:
        messagebox.showerror("Generate", str(e))
        return

    if warnings:
        messagebox.showwarning("Warnings", "\n".join(warnings))

    out.insert(tk.END, f"COUNTRY: {state.country.get()}\n\n")
    out.insert(tk.END, f"Reference ground level:\n  {expl}\n")
    out.insert(tk.END, f"Computed reference level: {ref:.2f} m\n")
    out.insert(tk.END, f"Points used: {used_ids}\n\n")

    target = height_target(state, hs)
    height = target - ref
    out.insert(tk.END, f"BUILDING HEIGHT: {height:.2f} m\n")
    out.insert(tk.END, f"(from reference ground to {'top storey floor' if state.country.get()=='UK' else 'roof'})\n\n")

    out.insert(tk.END, "STOREYS:\n")
    out.insert(tk.END, "Idx | Floor elev (m) | Height (m) | Class\n")
    out.insert(tk.END, "----+---------------+-----------+------\n")
    floor = 0.0
    for i, h in enumerate(hs):
        out.insert(tk.END, f"{i:>3} | {floor:>13.2f} | {h:>9.2f} | {classes[i]}\n")
        floor += h

# Wire Apply button at top
tk.Button(top, text="Apply", font=FONT, command=lambda: apply_inputs()).pack(side=tk.LEFT, padx=10)

# Wire toolbar buttons (Generate next to others)
tk.Button(btns, text="Add Point", font=FONT, command=lambda: add_point()).pack(side=tk.LEFT, padx=2)
tk.Button(btns, text="Auto Points", font=FONT, command=lambda: auto_points()).pack(side=tk.LEFT, padx=2)
tk.Button(btns, text="Delete", font=FONT, command=lambda: delete_point()).pack(side=tk.LEFT, padx=2)
tk.Button(btns, text="Generate", font=FONT, command=lambda: generate_output()).pack(side=tk.LEFT, padx=8)

# Redraw on changes
def on_any_change(*_):
    redraw(True)

state.country.trace_add("write", on_any_change)
state.us_side_min_mode.trace_add("write", on_any_change)
state.building_x_px.trace_add("write", on_any_change)

# =========================
# Drag interactions
# =========================
drag = {"id": None, "emin": None, "P": None}

def section_hit(px, py):
    if not state.points:
        return None, None, None
    hs = parse_floor_heights(state.floor_heights_str.get())
    elevs = [p["elev"] for p in state.points]
    emin = min(0.0, min(elevs), -1.0)
    P = perimeter_length(state)

    for p in state.points:
        x = sec_s_to_px(state, p["s"])
        y = sec_elev_to_py(p["elev"], emin)
        if math.hypot(px-x, py-y) <= 12:
            return p["id"], emin, P
    return None, None, None

def sec_down(e):
    pid, emin, P = section_hit(e.x, e.y)
    state.selected_id = pid
    drag["id"], drag["emin"], drag["P"] = pid, emin, P
    redraw(False)

def sec_move(e):
    if drag["id"] is None:
        return
    hs = parse_floor_heights(state.floor_heights_str.get())
    pid = drag["id"]
    emin = drag["emin"]
    P = drag["P"]

    elev_raw = sec_py_to_elev(e.y, emin)
    elev = clamp_ground(elev_raw, hs)  # HARD constraint: never below 0

    usable = SEC_W - 2*M
    s = ((e.x - M) / max(usable, 1)) * P
    s = s % P

    for p in state.points:
        if p["id"] == pid:
            p["elev"] = float(elev)
            p["s"] = float(s)
            break
    redraw(True)

def sec_up(_e):
    drag["id"] = None

plan_drag = {"id": None}

def plan_hit(px, py):
    for p in state.points:
        x, y = s_to_xy(state, p["s"])
        xp, yp = plan_xy_to_px(state, x, y)
        if math.hypot(px-xp, py-yp) <= 10:
            return p["id"]
    return None

def plan_down(e):
    pid = plan_hit(e.x, e.y)
    state.selected_id = pid
    plan_drag["id"] = pid
    redraw(False)

def plan_move(e):
    if plan_drag["id"] is None:
        return
    pid = plan_drag["id"]
    x, y = plan_px_to_xy(state, e.x, e.y)
    s = xy_to_s(state, x, y) % perimeter_length(state)

    for p in state.points:
        if p["id"] == pid:
            p["s"] = float(s)
            break
    redraw(True)

def plan_up(_e):
    plan_drag["id"] = None

sec.bind("<Button-1>", sec_down)
sec.bind("<B1-Motion>", sec_move)
sec.bind("<ButtonRelease-1>", sec_up)

plan.bind("<Button-1>", plan_down)
plan.bind("<B1-Motion>", plan_move)
plan.bind("<ButtonRelease-1>", plan_up)

# =========================
# Seed default points (all >= 0)
# =========================
def seed():
    hs = parse_floor_heights(state.floor_heights_str.get())
    P = perimeter_length(state)
    state.points = []
    state.next_id = 1
    for i, s in enumerate([0, P*0.2, P*0.45, P*0.7, P*0.9]):
        e = clamp_ground(0.0 + 0.2*i, hs)
        state.points.append({"id": state.next_id, "s": float(s), "elev": float(e)})
        state.next_id += 1
    state.selected_id = None

seed()
redraw(False)
root.mainloop()
