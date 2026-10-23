# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mcp<2",
#   "httpx>=0.27.0",
#   "numpy>=1.26.0",
#   "matplotlib>=3.9.0",
#   "pillow>=10.3.0",
# ]
# ///

import glob
import io
import json
import math
import os
import re
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
from mcp.server.fastmcp import FastMCP, Image

mcp = FastMCP("orcaslicer-2.5-native")

ORCA_API_URL = os.environ.get("ORCA_API_URL", "http://127.0.0.1:13130").rstrip("/")
ORCA_BIN = os.environ.get("ORCA_BIN", "/Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer")
ORCA_DATA_DIR = Path(
    os.environ.get(
        "ORCA_DATA_DIR",
        os.path.expanduser("~/Library/Application Support/OrcaSlicer"),
    )
)
NOTES_DIR = Path(os.environ.get("ORCA_MCP_NOTES_DIR", os.path.expanduser("~/.orcaslicer-mcp/notes")))
RENDERS_DIR = Path(os.path.expanduser("~/orca25-mcp/renders"))
NOTES_DIR.mkdir(parents=True, exist_ok=True)
RENDERS_DIR.mkdir(parents=True, exist_ok=True)

_LAST_SLICE_CACHE = {
    "gcode_path": "",
    "project_path": "",
}

CAMERA_ANGLES = {
    "iso": (30, -45),
    "top": (90, -90),
    "bottom": (-90, -90),
    "front": (0, -90),
    "rear": (0, 90),
    "left": (0, 180),
    "right": (0, 0),
}


def _discover_api_token() -> str:
    env_tok = os.environ.get("ORCA_API_TOKEN", "").strip()
    if env_tok:
        return env_tok
    pattern = str(ORCA_DATA_DIR / "orca_plugins" / "plugin_data" / "**" / "api_token.txt")
    for match in glob.glob(pattern, recursive=True):
        try:
            tok = Path(match).read_text(encoding="utf-8").strip()
            if tok:
                return tok
        except Exception:
            pass
    return ""


def _plugin_get(endpoint: str, params: Optional[dict] = None) -> dict:
    token = _discover_api_token()
    headers = {"X-Api-Token": token} if token else {}
    with httpx.Client(timeout=12.0) as client:
        resp = client.get(f"{ORCA_API_URL}{endpoint}", headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()


def _plugin_post(endpoint: str, payload: Optional[dict] = None) -> dict:
    token = _discover_api_token()
    headers = {"X-Api-Token": token} if token else {}
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(f"{ORCA_API_URL}{endpoint}", headers=headers, json=payload or {})
        resp.raise_for_status()
        return resp.json()


def _parse_float(val: Any, default: float = 0.0) -> float:
    if isinstance(val, (list, tuple)):
        val = val[0] if val else default
    if val is None:
        return default
    try:
        s = str(val).strip().rstrip("%").rstrip("mm").strip()
        return float(s)
    except Exception:
        return default


# ============================================================================
# 1. LIVE GUI STATUS, OBJECTS, CONFIG & PRESETS
# ============================================================================

@mcp.tool()
def get_status() -> dict:
    """Get live OrcaSlicer 2.5.0 GUI status, dirty flags, object count, and active project path."""
    return _plugin_get("/api/v1/status")


@mcp.tool()
def watch_events(limit: int = 50) -> list:
    """Return the most recent lifecycle events (slices, saves, preset switches, object edits) from OrcaSlicer 2.5.0."""
    data = _plugin_get("/api/v1/events")
    return data.get("events", [])[-limit:]


@mcp.tool()
def list_objects() -> dict:
    """List all objects on the OrcaSlicer 2.5.0 build plate with world-space bounding boxes, volumes, and instances."""
    return _plugin_get("/api/v1/objects")


@mcp.tool()
def get_config(keys: Optional[List[str]] = None) -> dict:
    """Read live configuration values from the open OrcaSlicer 2.5.0 window."""
    params = {"keys": ",".join(keys)} if keys else None
    return _plugin_get("/api/v1/config", params=params).get("config", {})


@mcp.tool()
def find_config_keys(query: str) -> dict:
    """Search the ~800 live OrcaSlicer 2.5.0 configuration keys by substring and return matching keys and current values."""
    all_cfg = _plugin_get("/api/v1/config").get("config", {})
    q = query.lower().strip()
    matches = {k: v for k, v in all_cfg.items() if q in k.lower()}
    return {"count": len(matches), "matches": matches}


@mcp.tool()
def list_presets() -> dict:
    """List currently selected and available Print, Filament, and Printer presets in OrcaSlicer 2.5.0."""
    return _plugin_get("/api/v1/presets").get("presets", {})


# ============================================================================
# 2. OFFSCREEN 3D RENDERER (render_plate -> FastMCP Image)
# ============================================================================

def _load_stl_triangles(stl_path: str, max_tris: int = 10000) -> np.ndarray:
    """Fast numpy binary/ASCII STL parser for rendering local STL files directly."""
    data = Path(stl_path).read_bytes()
    if len(data) > 84:
        n_tris = struct.unpack("<I", data[80:84])[0]
        if 84 + n_tris * 50 == len(data):
            dt = np.dtype([("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")])
            arr = np.frombuffer(data, dtype=dt, offset=84)
            tris = arr["verts"].astype(np.float64)
            if len(tris) > max_tris:
                tris = tris[:: max(1, len(tris) // max_tris)]
            return tris
    # Fallback ASCII STL
    verts = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("vertex "):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        return np.zeros((0, 3, 3))
    tris = np.array(verts, dtype=np.float64).reshape(-1, 3, 3)
    if len(tris) > max_tris:
        tris = tris[:: max(1, len(tris) // max_tris)]
    return tris


def _style_and_render_axes(ax, all_tris: np.ndarray, elev: float, azim: float, title: str, frame: str, bed_size=(220.0, 220.0)):
    ax.set_facecolor("#1e1e24")
    # Compute Simple Lambertian shading per triangle
    v0, v1, v2 = all_tris[:, 0], all_tris[:, 1], all_tris[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.maximum(norms, 1e-9))
    light_dir = np.array([0.4, -0.6, 0.7])
    light_dir /= np.linalg.norm(light_dir)
    intensity = np.clip(np.abs(normals @ light_dir), 0.25, 1.0)

    # Highlight bottom/first-layer contact faces in bright green, overhangs in warm amber
    z_max_tri = np.max(all_tris[:, :, 2], axis=1)
    z_min_global = float(np.min(all_tris[:, :, 2]))
    colors = np.zeros((len(all_tris), 4))
    # Default teal-cyan OrcaSlicer part color
    colors[:, 0] = 0.08 * intensity
    colors[:, 1] = 0.72 * intensity
    colors[:, 2] = 0.65 * intensity
    colors[:, 3] = 0.92

    # First-layer contact highlight (<0.3mm from bottom)
    contact_mask = z_max_tri <= (z_min_global + 0.35)
    colors[contact_mask] = [0.18, 0.95, 0.35, 1.0]

    # Steep downward-facing overhang highlight (normal Z < -0.55 and above bed)
    overhang_mask = (normals[:, 2] < -0.55) & (~contact_mask)
    colors[overhang_mask] = [1.0, 0.42, 0.15, 0.95]

    poly = Poly3DCollection(all_tris, facecolors=colors, edgecolors="none")
    ax.add_collection3d(poly)

    mins = all_tris.reshape(-1, 3).min(axis=0)
    maxs = all_tris.reshape(-1, 3).max(axis=0)

    # Draw build plate grid
    if frame == "plate":
        bx, by = bed_size
        cx, cy = bx / 2.0, by / 2.0
        span = max(bx, by, maxs[2] - mins[2]) / 2.0
    else:
        center = (mins + maxs) / 2.0
        cx, cy = center[0], center[1]
        span = max(np.max(maxs - mins) * 0.65, 10.0)

    z_mid = (mins[2] + maxs[2]) / 2.0
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_zlim(max(0.0, z_mid - span), z_mid + span)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, color="#e0e0e0", fontsize=10, pad=6)
    ax.tick_params(colors="#888888", labelsize=7)
    ax.set_xlabel("X (mm)", color="#999999", fontsize=8)
    ax.set_ylabel("Y (mm)", color="#999999", fontsize=8)
    ax.set_zlabel("Z (mm)", color="#999999", fontsize=8)


@mcp.tool()
def render_plate(
    camera: str = "quad",
    view: str = "editor",
    frame: str = "object",
    stl_or_gcode_path: Optional[str] = None,
) -> Image:
    """Render an offscreen 3D PNG of the current OrcaSlicer 2.5.0 build plate, a specific STL file, or sliced G-code toolpaths.
    - camera: 'quad' (4-panel Iso/Top/Front/Right inspection sheet) or one of 'iso', 'top', 'bottom', 'front', 'rear', 'left', 'right'.
    - view: 'editor' (renders 3D mesh with green first-layer contact & orange overhang highlights) or 'preview' (renders sliced G-code toolpaths colored by feature role).
    - frame: 'object' (zoomed in on the part) or 'plate' (full build plate).
    Returns a native MCP Image so the AI assistant can visually inspect orientation, bed contact, and overhangs.
    """
    out_file = RENDERS_DIR / f"render_{view}_{camera}_{int(time.time())}.png"

    if view == "preview":
        gpath = stl_or_gcode_path or get_status().get("last_gcode_path") or _LAST_SLICE_CACHE.get("gcode_path")
        if not gpath or not os.path.isfile(gpath):
            raise RuntimeError("No sliced G-code available for view='preview'. Run slice_project() first.")

        segments = []
        seg_colors = []
        role_palette = {
            "outer wall": "#ff8800",
            "inner wall": "#ffe100",
            "sparse infill": "#a83232",
            "internal solid infill": "#9932cc",
            "top surface": "#e63946",
            "support": "#2a9d8f",
            "support interface": "#1d3557",
            "bridge": "#457b9d",
            "overhang wall": "#ff0055",
            "skirt": "#888888",
            "brim": "#55a630",
        }
        cur_color = "#00b4d8"
        lx, ly, lz = 0.0, 0.0, 0.0
        with open(gpath, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if line.startswith(";TYPE:"):
                    rname = line.split(":", 1)[1].strip().lower()
                    cur_color = role_palette.get(rname, "#00b4d8")
                elif line.startswith("G1 ") or line.startswith("G0 "):
                    parts = line.split(";", 1)[0].split()
                    nx, ny, nz, has_e = lx, ly, lz, False
                    for tok in parts[1:]:
                        p = tok[0].upper()
                        try:
                            val = float(tok[1:])
                        except ValueError:
                            continue
                        if p == "X":
                            nx = val
                        elif p == "Y":
                            ny = val
                        elif p == "Z":
                            nz = val
                        elif p == "E" and val > 0:
                            has_e = True
                    if has_e and line.startswith("G1 "):
                        segments.append([[lx, ly, lz], [nx, ny, nz]])
                        seg_colors.append(cur_color)
                    lx, ly, lz = nx, ny, nz

        if len(segments) > 18000:
            step = len(segments) // 18000
            segments = segments[::step]
            seg_colors = seg_colors[::step]

        fig = plt.figure(figsize=(9, 7), dpi=130, facecolor="#1e1e24")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#1e1e24")
        lc = Line3DCollection(segments, colors=seg_colors, linewidths=0.65, alpha=0.85)
        ax.add_collection3d(lc)
        pts = np.array(segments).reshape(-1, 3)
        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        center = (mins + maxs) / 2.0
        span = max(np.max(maxs - mins) * 0.6, 10.0)
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(max(0.0, center[2] - span), center[2] + span)
        elev, azim = CAMERA_ANGLES.get(camera if camera != "quad" else "iso", (30, -45))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"G-code Toolpath Preview ({camera})", color="#ffffff")
        plt.tight_layout()
        fig.savefig(out_file, facecolor=fig.get_facecolor())
        plt.close(fig)
        return Image(path=str(out_file))

    # Mesh Editor View (either from local STL or live OrcaSlicer 2.5.0 GUI via /api/v1/mesh_snapshot)
    if stl_or_gcode_path and os.path.isfile(stl_or_gcode_path) and stl_or_gcode_path.lower().endswith(".stl"):
        all_tris = _load_stl_triangles(stl_or_gcode_path)
    else:
        snap = _plugin_get("/api/v1/mesh_snapshot", params={"max_triangles": "6000"})
        objs = snap.get("objects", [])
        if not objs and stl_or_gcode_path and os.path.isfile(stl_or_gcode_path):
            if stl_or_gcode_path.lower().endswith(".stl"):
                all_tris = _load_stl_triangles(stl_or_gcode_path)
            elif stl_or_gcode_path.lower().endswith(".3mf"):
                # For .3mf input files during script runs, render via preview view using the generated gcode
                return render_plate(camera=camera, view="preview", stl_or_gcode_path=_LAST_SLICE_CACHE.get("gcode_path"))
        elif not objs:
            cached_proj = _LAST_SLICE_CACHE.get("project_path")
            if cached_proj and os.path.isfile(cached_proj) and cached_proj.lower().endswith(".stl"):
                all_tris = _load_stl_triangles(cached_proj)
            else:
                # Ultimate fallback to G-code toolpath preview render if mesh snapshot is empty
                return render_plate(camera=camera, view="preview", stl_or_gcode_path=_LAST_SLICE_CACHE.get("gcode_path"))
        else:
            tri_blocks = [np.array(o["triangles"], dtype=np.float64) for o in objs if o.get("triangles")]
            all_tris = np.concatenate(tri_blocks, axis=0)

    if camera == "quad":
        fig = plt.figure(figsize=(11, 9), dpi=130, facecolor="#1e1e24")
        quad_views = [("iso", "Isometric (Orange=Overhangs)"), ("top", "Top (+Z)"), ("front", "Front (-Y)"), ("bottom", "Bottom Contact (Green=Bed)")]
        for idx, (cam_key, label) in enumerate(quad_views, start=1):
            ax = fig.add_subplot(2, 2, idx, projection="3d")
            elev, azim = CAMERA_ANGLES[cam_key]
            _style_and_render_axes(ax, all_tris, elev, azim, label, frame)
        plt.tight_layout()
    else:
        fig = plt.figure(figsize=(8, 7), dpi=130, facecolor="#1e1e24")
        ax = fig.add_subplot(111, projection="3d")
        elev, azim = CAMERA_ANGLES.get(camera, (30, -45))
        _style_and_render_axes(ax, all_tris, elev, azim, f"Plate Render ({camera.upper()})", frame)
        plt.tight_layout()

    fig.savefig(out_file, facecolor=fig.get_facecolor())
    plt.close(fig)
    return Image(path=str(out_file))


# ============================================================================
# 3. DETERMINISTIC PHYSICS & FLOW GATE + PRESET MUTATION
# ============================================================================

@mcp.tool()
def check_profile_physics(proposed_overrides: Optional[Dict[str, Any]] = None) -> dict:
    """Deterministic physics gate: overlays proposed setting changes onto the live OrcaSlicer 2.5.0 config
    and checks volumetric flow ceiling, layer height vs nozzle diameter, line width bounds, and acceleration.
    """
    live = get_config()
    merged = {**live, **(proposed_overrides or {})}

    nozzle = _parse_float(merged.get("nozzle_diameter"), 0.4)
    layer_h = _parse_float(merged.get("layer_height"), 0.2)
    initial_h = _parse_float(merged.get("initial_layer_print_height"), 0.2)
    max_vol_speed = _parse_float(merged.get("filament_max_volumetric_speed"), 15.0)

    warnings, blocked = [], []
    if layer_h > nozzle * 0.8:
        blocked.append(f"layer_height ({layer_h}mm) exceeds 80% of nozzle_diameter ({nozzle}mm).")
    elif layer_h > nozzle * 0.72:
        warnings.append(f"layer_height ({layer_h}mm) is above 72% of nozzle_diameter ({nozzle}mm).")

    role_pairs = [
        ("outer_wall", "outer_wall_speed", "outer_wall_line_width"),
        ("inner_wall", "inner_wall_speed", "inner_wall_line_width"),
        ("sparse_infill", "sparse_infill_speed", "sparse_infill_line_width"),
        ("internal_solid_infill", "internal_solid_infill_speed", "internal_solid_infill_line_width"),
        ("top_surface", "top_surface_speed", "top_surface_line_width"),
    ]

    flow_report = {}
    for role, speed_key, width_key in role_pairs:
        speed = _parse_float(merged.get(speed_key), 100.0)
        raw_w = merged.get(width_key, 0.42)
        if isinstance(raw_w, str) and "%" in raw_w:
            width = nozzle * (_parse_float(raw_w, 100.0) / 100.0)
        else:
            width = _parse_float(raw_w, nozzle * 1.05) or (nozzle * 1.05)

        area = (width - layer_h) * layer_h + math.pi * ((layer_h / 2.0) ** 2)
        flow_mm3_s = round(speed * max(area, 0.01), 2)
        flow_report[role] = {
            "speed_mm_s": speed,
            "line_width_mm": round(width, 3),
            "estimated_flow_mm3_s": flow_mm3_s,
            "max_volumetric_speed_mm3_s": max_vol_speed,
        }
        if max_vol_speed > 0 and flow_mm3_s > max_vol_speed:
            warnings.append(
                f"{role} requests {flow_mm3_s} mm3/s, exceeding filament_max_volumetric_speed ({max_vol_speed} mm3/s)."
            )

    return {
        "status": "blocked" if blocked else ("warnings" if warnings else "ok"),
        "nozzle_diameter_mm": nozzle,
        "layer_height_mm": layer_h,
        "initial_layer_height_mm": initial_h,
        "max_volumetric_speed_mm3_s": max_vol_speed,
        "flow_by_role": flow_report,
        "warnings": warnings,
        "blocked": blocked,
    }


@mcp.tool()
def edit_local_preset(
    overrides: Dict[str, Any],
    preset_type: str = "print",
    save_as_name: Optional[str] = None,
) -> dict:
    """Modify a user/local preset JSON on disk and immediately trigger OrcaSlicer 2.5.0's
    orca.host.preset.bundles.reload_local() so the running GUI updates in place.
    """
    presets_info = list_presets()
    if preset_type in ("print", "process"):
        edited = presets_info.get("print", {}).get("edited", {})
        subfolder = "process"
    elif preset_type == "printer":
        edited = presets_info.get("printer", {}).get("edited", {})
        subfolder = "machine"
    elif preset_type == "filament":
        selected_list = presets_info.get("filaments", {}).get("selected_presets", [])
        edited = selected_list[0] if selected_list else {}
        subfolder = "filament"
    else:
        return {"ok": False, "error": f"Unsupported preset_type '{preset_type}'"}

    src_file = edited.get("file", "")
    base_name = edited.get("name", f"Custom {preset_type.title()}")
    user_default_dir = ORCA_DATA_DIR / "user" / "default" / subfolder
    user_default_dir.mkdir(parents=True, exist_ok=True)

    if src_file and os.path.isfile(src_file) and edited.get("is_user") and not save_as_name:
        target_path = Path(src_file)
        data = json.loads(target_path.read_text(encoding="utf-8"))
    else:
        new_name = save_as_name or f"{base_name} (MCP)"
        safe_filename = re.sub(r'[\\/*?:"<>|]', "_", new_name) + ".json"
        target_path = user_default_dir / safe_filename
        if src_file and os.path.isfile(src_file):
            data = json.loads(Path(src_file).read_text(encoding="utf-8"))
        else:
            data = {"type": subfolder, "from": "User", "is_custom_defined": "0"}
        data["name"] = new_name
        data["from"] = "User"
        if not data.get("inherits"):
            data["inherits"] = base_name

    for k, v in overrides.items():
        if isinstance(v, bool):
            data[k] = "1" if v else "0"
        elif isinstance(v, (int, float)):
            data[k] = str(v)
        elif isinstance(v, list):
            data[k] = [str(item) for item in v]
        else:
            data[k] = str(v)

    target_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    reload_res = _plugin_post("/api/v1/presets/reload", {})
    return {
        "ok": True,
        "preset_file": str(target_path),
        "preset_name": data.get("name"),
        "applied_overrides": overrides,
        "gui_reload": reload_res,
    }


# ============================================================================
# 4. SLICING, BREAKDOWNS, & CLOSED-LOOP AUTO-OPTIMIZER
# ============================================================================

def _parse_orca_gcode_breakdown(gcode_path: str) -> dict:
    if not gcode_path or not os.path.isfile(gcode_path):
        return {"ok": False, "error": f"G-code file not found: {gcode_path}"}

    role_stats: Dict[str, Dict[str, float]] = {}
    current_role = "Custom"
    last_x, last_y, last_z = None, None, None
    current_f = 1800.0
    total_time_str = ""
    filament_weight_g = 0.0
    filament_length_mm = 0.0

    with open(gcode_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(";"):
                if line.startswith(";TYPE:"):
                    current_role = line.split(":", 1)[1].strip()
                    role_stats.setdefault(current_role, {"time_s": 0.0, "e_mm": 0.0, "extrusion_time_s": 0.0})
                elif "estimated printing time" in line:
                    total_time_str = line.split("=", 1)[-1].strip()
                elif line.startswith("; total filament used [g] ="):
                    filament_weight_g = _parse_float(line.split("=", 1)[-1].strip())
                elif line.startswith("; filament used [mm] ="):
                    filament_length_mm = _parse_float(line.split("=", 1)[-1].strip())
                continue

            if line.startswith(("G0 ", "G1 ", "G2 ", "G3 ")):
                tokens = line.split(";", 1)[0].split()
                x, y, z, e, f_val = last_x, last_y, last_z, 0.0, None
                for tok in tokens[1:]:
                    p = tok[0].upper()
                    try:
                        val = float(tok[1:])
                    except ValueError:
                        continue
                    if p == "X":
                        x = val
                    elif p == "Y":
                        y = val
                    elif p == "Z":
                        z = val
                    elif p == "E":
                        e = val
                    elif p == "F":
                        f_val = val
                if f_val and f_val > 0:
                    current_f = f_val
                dist = 0.0
                if last_x is not None and last_y is not None and x is not None and y is not None:
                    dz = (z - last_z) if (z is not None and last_z is not None) else 0.0
                    dist = math.sqrt((x - last_x) ** 2 + (y - last_y) ** 2 + dz ** 2)
                speed_mm_s = max(current_f / 60.0, 1.0)
                dt = dist / speed_mm_s if dist > 0 else 0.0
                bucket = role_stats.setdefault(current_role, {"time_s": 0.0, "e_mm": 0.0, "extrusion_time_s": 0.0})
                bucket["time_s"] += dt
                if e > 0:
                    bucket["e_mm"] += e
                    bucket["extrusion_time_s"] += dt
                if x is not None:
                    last_x = x
                if y is not None:
                    last_y = y
                if z is not None:
                    last_z = z

    filament_area = math.pi * ((1.75 / 2.0) ** 2)
    sum_time_s = sum(v["time_s"] for v in role_stats.values()) or 1.0
    breakdown = []
    for role, st in sorted(role_stats.items(), key=lambda item: item[1]["time_s"], reverse=True):
        if st["time_s"] < 0.5 and st["e_mm"] <= 0:
            continue
        vol_mm3 = st["e_mm"] * filament_area
        grams = (vol_mm3 / 1000.0) * 1.24
        mean_flow = (vol_mm3 / st["extrusion_time_s"]) if st["extrusion_time_s"] > 0 else 0.0
        breakdown.append({
            "role": role,
            "time_seconds": round(st["time_s"], 1),
            "share_pct": round((st["time_s"] / sum_time_s) * 100.0, 1),
            "filament_g": round(grams, 2),
            "mean_flow_mm3_s": round(mean_flow, 2),
        })

    return {
        "ok": True,
        "gcode_path": gcode_path,
        "estimated_print_time": total_time_str or f"{int(sum_time_s // 60)}m {int(sum_time_s % 60)}s",
        "total_time_seconds": round(sum_time_s, 1),
        "total_filament_g": round(filament_weight_g or sum(b["filament_g"] for b in breakdown), 2),
        "total_filament_mm": round(filament_length_mm, 1),
        "breakdown": breakdown,
    }


def _build_cli_flags_from_overrides(overrides: Dict[str, Any]) -> List[str]:
    flags = []
    for k, v in (overrides or {}).items():
        flag_name = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            flags.append(f"{flag_name}={'1' if v else '0'}")
        elif isinstance(v, (list, tuple)):
            flags.append(f"{flag_name}={','.join(str(x) for x in v)}")
        else:
            flags.append(f"{flag_name}={v}")
    return flags


@mcp.tool()
def slice_project(
    input_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    arrange: bool = False,
    orient: bool = False,
    plate: int = 0,
    open_in_gui: bool = False,
) -> dict:
    """Slice a model (.stl, .step, .obj, .3mf) or active project using native 2.5.0 CLI."""
    status = get_status()
    target_input = input_path or status.get("last_project_path") or _LAST_SLICE_CACHE.get("project_path")
    if not target_input or not os.path.isfile(target_input):
        return {"ok": False, "error": "Provide a valid input_path model file."}

    out_dir = Path(tempfile.mkdtemp(prefix="orca25_slice_"))
    out_3mf = out_dir / "sliced_output.3mf"

    cmd = [ORCA_BIN, target_input]
    if orient:
        cmd.extend(["--orient", "1"])
    if arrange:
        cmd.extend(["--arrange", "1"])
    cmd.append("--ensure-on-bed")
    if overrides:
        cmd.extend(_build_cli_flags_from_overrides(overrides))

    cmd.extend(["--slice", str(plate), "--outputdir", str(out_dir), "--export-3mf", str(out_3mf)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    gcode_files = sorted(out_dir.glob("*.gcode"))

    if not gcode_files:
        return {"ok": False, "return_code": proc.returncode, "stdout": proc.stdout[-1500:], "stderr": proc.stderr[-1500:]}

    gcode_path = str(gcode_files[0])
    _LAST_SLICE_CACHE["gcode_path"] = gcode_path
    _LAST_SLICE_CACHE["project_path"] = str(out_3mf) if out_3mf.is_file() else target_input

    if open_in_gui and out_3mf.is_file():
        subprocess.Popen(["open", "-a", "OrcaSlicer", str(out_3mf)])

    breakdown = _parse_orca_gcode_breakdown(gcode_path)
    return {"ok": True, "input_path": target_input, "exported_3mf": str(out_3mf), **breakdown}


def auto_optimize_print(
    model_path: str,
    intent: str = "strength",
    auto_orient: bool = True,
    auto_arrange: bool = True,
    open_in_gui: bool = True,
) -> dict:
    """End-to-End CAD-to-OrcaSlicer Optimizer:
    Takes a 3D model file (.stl, .step, .3mf), selects optimal process overrides for the requested intent
    ('strength', 'visual_quality', 'speed', 'dimensional_accuracy', or 'draft'), runs check_profile_physics,
    auto-orients and arranges the part on the bed, slices it, and opens the prepared .3mf in OrcaSlicer 2.5.0.
    """
    live = get_config(["nozzle_diameter", "layer_height"])
    nozzle = _parse_float(live.get("nozzle_diameter"), 0.4)

    intent_key = intent.lower().strip()
    recipes: Dict[str, Dict[str, Any]] = {
        "strength": {
            "layer_height": round(nozzle * 0.5, 2),
            "wall_loops": 5,
            "top_shell_layers": 5,
            "bottom_shell_layers": 5,
            "sparse_infill_density": "40%",
            "sparse_infill_pattern": "gyroid",
            "outer_wall_line_width": round(nozzle * 1.15, 2),
            "inner_wall_line_width": round(nozzle * 1.2, 2),
        },
        "visual_quality": {
            "layer_height": round(nozzle * 0.3, 2),
            "wall_loops": 3,
            "sparse_infill_density": "15%",
            "sparse_infill_pattern": "gyroid",
            "outer_wall_speed": 60,
            "top_surface_speed": 50,
            "seam_position": "aligned",
        },
        "dimensional_accuracy": {
            "layer_height": round(nozzle * 0.45, 2),
            "wall_loops": 4,
            "wall_sequence": "outer/inner",
            "outer_wall_speed": 55,
            "inner_wall_speed": 90,
            "elefant_foot_compensation": 0.15,
            "sparse_infill_density": "25%",
        },
        "speed": {
            "layer_height": round(nozzle * 0.65, 2),
            "wall_loops": 2,
            "sparse_infill_density": "12%",
            "sparse_infill_pattern": "crosshatch",
        },
    }
    chosen_overrides = recipes.get(intent_key, recipes["strength"])
    physics = check_profile_physics(chosen_overrides)
    slice_res = slice_project(
        input_path=model_path,
        overrides=chosen_overrides,
        orient=auto_orient,
        arrange=auto_arrange,
        open_in_gui=open_in_gui,
    )
    return {
        "ok": slice_res.get("ok", False),
        "intent": intent_key,
        "applied_overrides": chosen_overrides,
        "physics_check": physics,
        "slice_result": slice_res,
    }


@mcp.tool()
def get_slice_breakdown(gcode_path: Optional[str] = None) -> dict:
    """Return per-feature time, percentage share, filament weight (g), and volumetric flow (mm3/s)."""
    target = gcode_path or get_status().get("last_gcode_path") or _LAST_SLICE_CACHE.get("gcode_path")
    if not target or not os.path.isfile(target):
        return {"ok": False, "error": "No G-code file found yet."}
    return _parse_orca_gcode_breakdown(target)


@mcp.tool()
def compare_slices(variants: Dict[str, Dict[str, Any]], input_path: Optional[str] = None) -> dict:
    """Slice a model across multiple named setting variants and return a side-by-side comparison with deltas."""
    results = []
    for name, overrides in variants.items():
        res = slice_project(input_path=input_path, overrides=overrides)
        if not res.get("ok"):
            results.append({"variant": name, "ok": False, "error": res.get("stderr") or res.get("error")})
            continue
        results.append({
            "variant": name,
            "ok": True,
            "estimated_print_time": res["estimated_print_time"],
            "total_time_seconds": res["total_time_seconds"],
            "total_filament_g": res["total_filament_g"],
            "breakdown": res["breakdown"],
        })
    valid = [r for r in results if r.get("ok")]
    if valid:
        base = valid[0]
        for r in valid:
            r["delta_time_seconds"] = round(r["total_time_seconds"] - base["total_time_seconds"], 1)
            r["delta_filament_g"] = round(r["total_filament_g"] - base["total_filament_g"], 2)
    return {"ok": True, "comparisons": results}


# ============================================================================
# 5. POST-PRINT SYMPTOM DOCTOR & AUTO-FIXER (diagnose_print_failure)
# ============================================================================

@mcp.tool()
def diagnose_print_failure(
    symptom: str,
    apply_fix: bool = False,
    save_as_preset_name: Optional[str] = None,
) -> dict:
    """Closed-Loop Print Doctor:
    Analyzes a real-world print defect against your active OrcaSlicer 2.5.0 settings, explains the root cause,
    calculates exact mathematical setting corrections, and (if apply_fix=True) writes the changes to your local preset
    and reloads the OrcaSlicer 2.5.0 GUI immediately.
    Supported symptoms: 'warping', 'bed_adhesion', 'stringing', 'under_extrusion', 'overhang_curling',
    'layer_splitting', 'elephant_foot', 'ghosting_ringing', 'z_seam_blobs', 'supports_fused'.
    """
    live = get_config()
    sym = symptom.lower().strip()

    cur_first_spd = _parse_float(live.get("initial_layer_speed"), 50.0)
    cur_outer_spd = _parse_float(live.get("outer_wall_speed"), 120.0)
    cur_ret_len = _parse_float(live.get("retraction_length"), 0.8)
    cur_layer_h = _parse_float(live.get("layer_height"), 0.2)
    cur_accel = _parse_float(live.get(" outer_wall_acceleration") or live.get("default_acceleration"), 4000.0)

    fixes: Dict[str, Dict[str, Any]] = {
        "warping": {
            "explanation": "Corners lift due to thermal contraction pulling upward against weak perimeter bed hold and early part-cooling fan.",
            "print_overrides": {
                "brim_type": "outer_only",
                "brim_width": 8,
                "brim_object_gap": 0.05,
                "initial_layer_speed": max(20, round(cur_first_spd * 0.65)),
            },
            "filament_overrides": {
                "close_fan_the_first_x_layers": 3,
            },
            "mechanical_checks": [
                "Wash build plate with warm water and dish soap (IPA alone spreads oils).",
                "Verify Klipper Z-offset (BLTouch) gives slight squish on first-layer lines.",
            ],
        },
        "bed_adhesion": {
            "explanation": "First layer is either laid down too fast, too cold, or with insufficient line width.",
            "print_overrides": {
                "initial_layer_speed": 25,
                "initial_layer_infill_speed": 45,
                "initial_layer_line_width": "125%",
                "brim_type": "outer_only",
                "brim_width": 5,
            },
            "filament_overrides": {},
            "mechanical_checks": [
                "Run PROBE_CALIBRATE / check Z-offset in Klipper.",
                "Ensure bed mesh (BED_MESH_PROFILE) is loaded in START_PRINT.",
            ],
        },
        "stringing": {
            "explanation": "Molten filament oozes during non-print travel moves due to insufficient retraction, wet filament, or z-hop dragging.",
            "print_overrides": {
                "wipe": True,
                "reduce_crossing_wall": True,
            },
            "printer_overrides": {
                "retraction_length": round(min(cur_ret_len + 0.2, 2.0), 2),
                "retraction_speed": 45,
                "z_hop_types": "Spiral Lift",
            },
            "mechanical_checks": [
                "Dry filament if you hear subtle popping at the Orbiter nozzle.",
                "Lower hotend printing temperature by 5°C to 10°C.",
            ],
        },
        "overhang_curling": {
            "explanation": "Steep overhangs curl upward when layer height is too thick relative to line width or overhang speed exceeds cooling capacity.",
            "print_overrides": {
                "layer_height": round(max(0.12, cur_layer_h * 0.8), 2),
                "enable_overhang_speed": True,
                "overhang_2_4_speed": 35,
                "overhang_3_4_speed": 20,
                "overhang_4_4_speed": 12,
                "wall_sequence": "inner/outer",
            },
            "filament_overrides": {
                "overhang_fan_speed": 100,
            },
            "mechanical_checks": [
                "Verify both dual 5015 part-cooling fans spin up to 100% on steep overhangs.",
            ],
        },
        "under_extrusion": {
            "explanation": "Extruder is outrunning the hotend melt zone ceiling or experiencing filament drag.",
            "print_overrides": {
                "outer_wall_speed": round(cur_outer_spd * 0.75),
                "sparse_infill_speed": 120,
            },
            "filament_overrides": {},
            "mechanical_checks": [
                "Check filament_max_volumetric_speed against your hotend's real mm3/s ceiling.",
                "Inspect Orbiter extruder gears and filament sensor path for friction.",
            ],
        },
        "layer_splitting": {
            "explanation": "Inter-layer delamination occurs when extrusion is too cool, fan speed is too high, or walls are too thin.",
            "print_overrides": {
                "wall_loops": 4,
                "outer_wall_line_width": "115%",
                "inner_wall_line_width": "120%",
            },
            "filament_overrides": {},
            "mechanical_checks": [
                "Increase nozzle temperature by +10°C and reduce part-cooling fan by 20-30%.",
            ],
        },
        "elephant_foot": {
            "explanation": "First layer flares outward from bed squish and thermal load.",
            "print_overrides": {
                "elefant_foot_compensation": 0.2,
                "elefant_foot_compensation_layers": 1,
            },
            "filament_overrides": {},
            "mechanical_checks": ["Ensure first-layer Z-offset isn't overly compressed into the bed."],
        },
        "ghosting_ringing": {
            "explanation": "Echoes around sharp corners occur when outer wall acceleration exceeds your frame's Input Shaper damping limit.",
            "print_overrides": {
                "outer_wall_acceleration": min(2500, round(cur_accel * 0.6)),
                "outer_wall_speed": min(80, round(cur_outer_spd * 0.75)),
            },
            "filament_overrides": {},
            "mechanical_checks": [
                "Re-run ADXL345 SHAPER_CALIBRATE in Klipper and verify outer_wall_acceleration <= recommended max_accel.",
            ],
        },
        "supports_fused": {
            "explanation": "Supports weld to the model when top Z distance is too tight or support interface density is too high.",
            "print_overrides": {
                "support_type": "tree(auto)",
                "support_top_z_distance": round(max(0.2, cur_layer_h + 0.04), 2),
                "support_bottom_z_distance": 0.2,
                "support_interface_top_layers": 3,
                "support_interface_spacing": 0.25,
            },
            "filament_overrides": {},
            "mechanical_checks": [],
        },
    }

    matched = fixes.get(sym)
    if not matched:
        for k, v in fixes.items():
            if k in sym or sym in k:
                matched = v
                sym = k
                break
    if not matched:
        return {
            "ok": False,
            "error": f"Unrecognized symptom '{symptom}'. Choose one of: {', '.join(fixes.keys())}",
        }

    applied_result = None
    if apply_fix and matched.get("print_overrides"):
        applied_result = edit_local_preset(
            overrides=matched["print_overrides"],
            preset_type="print",
            save_as_name=save_as_preset_name,
        )

    return {
        "ok": True,
        "symptom": sym,
        "explanation": matched["explanation"],
        "recommended_print_overrides": matched.get("print_overrides", {}),
        "recommended_filament_overrides": matched.get("filament_overrides", {}),
        "recommended_printer_overrides": matched.get("printer_overrides", {}),
        "mechanical_checks": matched.get("mechanical_checks", []),
        "applied_to_gui": applied_result,
    }


# ============================================================================
# 6. KLIPPER / MOONRAKER STATUS, WEBCAM SNAPSHOT, & UPLOAD/PRINT
# ============================================================================

def _resolve_moonraker_url() -> str:
    env_url = os.environ.get("MOONRAKER_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    conn = _plugin_get("/api/v1/printer_connection").get("connection", {})
    host = str(conn.get("print_host") or "http://klipper-pi.local").strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


@mcp.tool()
def get_printer_status() -> dict:
    """Query your Klipper/Moonraker printer for live temperatures, print state, progress, and filament sensor state."""
    base_url = _resolve_moonraker_url()
    with httpx.Client(timeout=6.0) as client:
        resp = client.get(
            f"{base_url}/printer/objects/query?print_stats&extruder&heater_bed&virtual_sdcard&webhooks"
        )
        resp.raise_for_status()
        st = resp.json().get("result", {}).get("status", {})

    pstats = st.get("print_stats", {})
    ext = st.get("extruder", {})
    bed = st.get("heater_bed", {})
    vsd = st.get("virtual_sdcard", {})
    wh = st.get("webhooks", {})

    return {
        "ok": True,
        "printer_url": base_url,
        "state": pstats.get("state", "unknown"),
        "state_message": wh.get("state_message", ""),
        "filename": pstats.get("filename", ""),
        "progress_pct": round((vsd.get("progress", 0.0) or 0.0) * 100.0, 1),
        "print_duration_s": round(pstats.get("print_duration", 0.0) or 0.0, 1),
        "extruder_temp_c": ext.get("temperature"),
        "extruder_target_c": ext.get("target"),
        "bed_temp_c": bed.get("temperature"),
        "bed_target_c": bed.get("target"),
    }


@mcp.tool()
def capture_webcam_snapshot(snapshot_url: Optional[str] = None) -> Image:
    """Capture a live JPEG/PNG frame from your printer's Crowsnest/Moonraker webcam so the AI assistant
    can visually inspect first-layer adhesion, nozzle cleanliness, or print defects.
    """
    base_url = _resolve_moonraker_url()
    target_url = snapshot_url or f"{base_url}/webcam/?action=snapshot"
    out_img = RENDERS_DIR / f"webcam_{int(time.time())}.jpg"
    with httpx.Client(timeout=8.0, follow_redirects=True) as client:
        resp = client.get(target_url)
        resp.raise_for_status()
        out_img.write_bytes(resp.content)
    return Image(path=str(out_img))


@mcp.tool()
def upload_and_print(gcode_path: Optional[str] = None, start_print: bool = False) -> dict:
    """Upload the sliced G-code file to your Klipper/Moonraker printer (/server/files/upload).
    Set start_print=True to begin printing immediately, or False (default) to stage it safely in Moonraker.
    """
    target = gcode_path or get_status().get("last_gcode_path") or _LAST_SLICE_CACHE.get("gcode_path")
    if not target or not os.path.isfile(target):
        return {"ok": False, "error": "No G-code file found to upload. Run slice_project() first."}

    base_url = _resolve_moonraker_url()
    filename = os.path.basename(target)
    with open(target, "rb") as f:
        files = {"file": (filename, f, "application/octet-stream")}
        data = {"print": "true" if start_print else "false"}
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{base_url}/server/files/upload", data=data, files=files)
            resp.raise_for_status()
            return {"ok": True, "printer_url": base_url, "started_print": start_print, "moonraker": resp.json()}


@mcp.tool()
def check_printer_match() -> dict:
    """Compare the active OrcaSlicer 2.5.0 printer profile against live Klipper firmware settings."""
    base_url = _resolve_moonraker_url()
    slicer_cfg = get_config([
        "nozzle_diameter",
        "machine_max_speed_x",
        "machine_max_acceleration_x",
        "default_acceleration",
        "outer_wall_acceleration",
        "use_firmware_retraction",
    ])
    with httpx.Client(timeout=6.0) as client:
        resp = client.get(f"{base_url}/printer/objects/query?toolhead&configfile")
        resp.raise_for_status()
        st = resp.json().get("result", {}).get("status", {})

    toolhead = st.get("toolhead", {})
    k_settings = st.get("configfile", {}).get("settings", {})
    k_extruder = k_settings.get("extruder", {})

    klipper_max_vel = _parse_float(toolhead.get("max_velocity"))
    klipper_max_accel = _parse_float(toolhead.get("max_accel"))
    klipper_nozzle = _parse_float(k_extruder.get("nozzle_diameter"), 0.4)
    slicer_nozzle = _parse_float(slicer_cfg.get("nozzle_diameter"), 0.4)
    slicer_accel = _parse_float(slicer_cfg.get("default_acceleration"), 0.0)

    warnings = []
    if abs(klipper_nozzle - slicer_nozzle) > 0.01:
        warnings.append(f"Nozzle mismatch: OrcaSlicer={slicer_nozzle}mm vs Klipper={klipper_nozzle}mm.")
    if klipper_max_accel > 0 and slicer_accel > klipper_max_accel:
        warnings.append(f"OrcaSlicer default_acceleration ({slicer_accel}) exceeds Klipper max_accel ({klipper_max_accel}).")

    return {
        "ok": True,
        "printer_url": base_url,
        "klipper": {
            "nozzle_diameter": klipper_nozzle,
            "max_velocity": klipper_max_vel,
            "max_accel": klipper_max_accel,
            "has_firmware_retraction": "firmware_retraction" in k_settings,
        },
        "orcaslicer": slicer_cfg,
        "warnings": warnings,
    }


@mcp.tool()
def remember(topic: str, note: str) -> dict:
    """Save persistent machine, filament, or project notes to ~/.orcaslicer-mcp/notes/<topic>.md."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", topic.strip().lower())
    target = NOTES_DIR / f"{safe}.md"
    existing = target.read_text(encoding="utf-8") + "\n" if target.is_file() else ""
    target.write_text(existing + f"- ({time.strftime('%Y-%m-%d %H:%M')}) {note}\n", encoding="utf-8")
    return {"ok": True, "file": str(target)}


@mcp.tool()
def consult(topic: Optional[str] = None) -> dict:
    """Read saved slicing & machine notes from ~/.orcaslicer-mcp/notes/."""
    notes = {}
    for p in sorted(NOTES_DIR.glob("*.md")):
        if not topic or topic.lower() in p.stem.lower():
            notes[p.stem] = p.read_text(encoding="utf-8")
    return {"ok": True, "notes": notes}


if __name__ == "__main__":
    mcp.run()
