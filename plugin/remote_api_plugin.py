# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
#
# [tool.orcaslicer.plugin]
# name = "Remote API Bridge (Native 2.5.0 MCP)"
# description = "Native HTTP bridge for OrcaSlicer 2.5.0+. Exposes live model geometry, world-space mesh snapshots, full configs, preset paths, G-code taps, and local bundle reload."
# author = "Oliver Shipp"
# version = "3.0.0"
# ///

import json
import os
import secrets
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import orca

_SERVER_LOCK = threading.Lock()
_HTTP_SERVER = None
_SERVER_THREAD = None
_STATE = {
    "running": False,
    "host": "127.0.0.1",
    "port": 13130,
    "token": "",
    "slice_status": "idle",
    "last_slice_error": "",
    "last_gcode_path": "",
    "last_project_path": "",
    "events": deque(maxlen=250),
}

REDACTED_KEYS = {"printhost_apikey", "printhost_user", "printhost_password", "printhost_cafile"}


def _record_event(event_name: str, payload: dict):
    _STATE["events"].append({
        "timestamp": time.time(),
        "event": event_name,
        **payload,
    })


def _safe_host_snapshot() -> dict:
    snap = {
        "ready": False,
        "language": None,
        "is_project_dirty": False,
        "is_presets_dirty": False,
        "inside_snapshot_capture": False,
        "object_count": 0,
        "last_project_path": _STATE["last_project_path"],
    }
    try:
        plater = orca.host.plater()
        model = orca.host.model()
        snap["ready"] = True
        snap["language"] = orca.host.app_language()
        snap["is_project_dirty"] = bool(plater.is_project_dirty())
        snap["is_presets_dirty"] = bool(plater.is_presets_dirty())
        snap["inside_snapshot_capture"] = bool(plater.inside_snapshot_capture())
        snap["object_count"] = len(model.objects())
    except Exception as exc:
        snap["error"] = str(exc)
    return snap


def _list_objects_data() -> list:
    model = orca.host.model()
    result = []
    for idx, obj in enumerate(model.objects()):
        bbox = obj.bounding_box() if hasattr(obj, "bounding_box") else None
        volumes = []
        for v_idx, vol in enumerate(obj.volumes()):
            mesh = vol.mesh()
            v_bbox = mesh.bounding_box()
            volumes.append({
                "index": v_idx,
                "name": getattr(vol, "name", f"volume_{v_idx}"),
                "vertex_count": mesh.vertex_count(),
                "triangle_count": mesh.triangle_count(),
                "volume_mm3": mesh.volume(),
                "is_manifold": mesh.is_manifold(),
                "bounding_box": {
                    "min": list(v_bbox.min),
                    "max": list(v_bbox.max),
                    "size": list(v_bbox.size),
                    "center": list(v_bbox.center),
                } if v_bbox and v_bbox.defined else None,
            })

        instances = []
        for i_idx, inst in enumerate(obj.instances()):
            instances.append({
                "index": i_idx,
                "offset": list(inst.offset) if hasattr(inst, "offset") else [0.0, 0.0, 0.0],
                "rotation": list(inst.rotation) if hasattr(inst, "rotation") else [0.0, 0.0, 0.0],
                "scaling_factor": list(inst.scaling_factor) if hasattr(inst, "scaling_factor") else [1.0, 1.0, 1.0],
                "mirror": list(inst.mirror) if hasattr(inst, "mirror") else [1.0, 1.0, 1.0],
                "is_left_handed": bool(inst.is_left_handed()) if hasattr(inst, "is_left_handed") else False,
            })

        result.append({
            "index": idx,
            "name": getattr(obj, "name", f"object_{idx}"),
            "volumes": volumes,
            "instances": instances,
            "bounding_box": {
                "min": list(bbox.min),
                "max": list(bbox.max),
                "size": list(bbox.size),
                "center": list(bbox.center),
            } if bbox and bbox.defined else None,
        })
    return result


def _get_world_mesh_snapshot(max_triangles_per_instance: int = 8000) -> dict:
    """Extracts world-space decimated triangle coordinates for every active instance on the plate
    so the MCP server can render 7-angle offscreen PNGs of the exact live GUI state.
    """
    model = orca.host.model()
    objects_out = []
    for o_idx, obj in enumerate(model.objects()):
        obj_name = getattr(obj, "name", f"object_{o_idx}")
        inst_list = obj.instances()
        if not inst_list:
            continue
        for i_idx, inst in enumerate(inst_list):
            inst_mat = np.asarray(inst.matrix(), dtype=np.float64)
            for v_idx, vol in enumerate(obj.volumes()):
                mesh = vol.mesh()
                if mesh.is_empty():
                    continue
                V = np.asarray(mesh.vertices(), dtype=np.float64)
                T = np.asarray(mesh.triangles(), dtype=np.int32)
                if len(T) == 0:
                    continue
                # Decimate uniformly if high-poly so HTTP transfer and rendering stay <200ms
                if len(T) > max_triangles_per_instance:
                    step = max(1, len(T) // max_triangles_per_instance)
                    T = T[::step]
                vol_mat = np.asarray(vol.matrix(), dtype=np.float64)
                M = inst_mat @ vol_mat
                world_V = (np.c_[V, np.ones(len(V))] @ M.T)[:, :3]
                tris = world_V[T]  # shape (K, 3, 3)
                objects_out.append({
                    "object_index": o_idx,
                    "instance_index": i_idx,
                    "volume_index": v_idx,
                    "name": f"{obj_name} (inst {i_idx})",
                    "triangle_count": int(len(tris)),
                    "bounds_min": world_V.min(axis=0).round(3).tolist(),
                    "bounds_max": world_V.max(axis=0).round(3).tolist(),
                    "triangles": np.round(tris, 2).tolist(),
                })
    # Also fetch bed size from live config
    bed_area = orca.host.preset_bundle().full_config_value("printable_area")
    return {"objects": objects_out, "printable_area": bed_area}


def _get_live_config(keys_filter=None, include_private=False) -> dict:
    bundle = orca.host.preset_bundle()
    all_keys = bundle.full_config_keys()
    target_keys = [k for k in all_keys if k in keys_filter] if keys_filter else all_keys
    config_data = {}
    for k in target_keys:
        if not include_private and k in REDACTED_KEYS:
            config_data[k] = "<REDACTED>"
        else:
            config_data[k] = bundle.full_config_value(k)
    return config_data


def _serialize_preset(p) -> dict:
    if p is None:
        return {}
    own_cfg = {}
    for k in p.config_keys():
        own_cfg[k] = "<REDACTED>" if k in REDACTED_KEYS else p.config_value(k)
    return {
        "name": p.name,
        "alias": p.alias,
        "file": p.file,
        "bundle_id": p.bundle_id,
        "is_default": bool(p.is_default),
        "is_system": bool(p.is_system),
        "is_user": bool(p.is_user()),
        "is_from_bundle": bool(p.is_from_bundle()),
        "is_dirty": bool(p.is_dirty),
        "is_compatible": bool(p.is_compatible),
        "config": own_cfg,
    }


def _get_presets_detail() -> dict:
    bundle = orca.host.preset_bundle()

    def _pack_collection(coll):
        items = []
        for i in range(coll.size()):
            p = coll.preset(i)
            items.append({
                "name": p.name,
                "file": p.file,
                "bundle_id": p.bundle_id,
                "is_user": bool(p.is_user()),
                "is_system": bool(p.is_system),
                "is_dirty": bool(p.is_dirty),
            })
        return {
            "selected_name": coll.selected_preset_name(),
            "edited": _serialize_preset(coll.edited_preset()),
            "presets": items,
        }

    return {
        "print": _pack_collection(bundle.prints),
        "printer": _pack_collection(bundle.printers),
        "filaments": {
            "selected_names": list(bundle.current_filament_preset_names()),
            "selected_presets": [_serialize_preset(p) for p in bundle.current_filament_presets() if p],
            "all_presets": _pack_collection(bundle.filaments)["presets"],
        },
        "local_bundle_ids": list(orca.host.preset.bundles.list_ids()),
    }


class RemoteAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        expected = _STATE.get("token", "")
        if not expected:
            return True
        auth = self.headers.get("Authorization", "")
        token_hdr = self.headers.get("X-Api-Token", "")
        provided = auth[7:].strip() if auth.startswith("Bearer ") else token_hdr.strip()
        if not secrets.compare_digest(provided, expected):
            self._send_json(401, {"ok": False, "error": "Unauthorized API token"})
            return False
        return True

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        if not self._check_auth():
            return
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)

        try:
            if path in ("", "/api/v1/status", "/status"):
                self._send_json(200, {
                    "ok": True,
                    "mode": "orcaslicer-2.5-native-plugin-v3",
                    "slice_status": _STATE["slice_status"],
                    "last_slice_error": _STATE["last_slice_error"],
                    "last_gcode_path": _STATE["last_gcode_path"],
                    "last_project_path": _STATE["last_project_path"],
                    "host": _safe_host_snapshot(),
                })
            elif path in ("/api/v1/objects", "/objects"):
                self._send_json(200, {"ok": True, "objects": _list_objects_data()})
            elif path in ("/api/v1/mesh_snapshot", "/mesh_snapshot"):
                max_t = int(query.get("max_triangles", ["8000"])[0])
                self._send_json(200, {"ok": True, **_get_world_mesh_snapshot(max_t)})
            elif path in ("/api/v1/config", "/config"):
                keys_param = query.get("keys", [None])[0]
                keys_filter = set(keys_param.split(",")) if keys_param else None
                self._send_json(200, {"ok": True, "config": _get_live_config(keys_filter)})
            elif path in ("/api/v1/printer_connection", "/printer_connection"):
                bundle = orca.host.preset_bundle()
                conn = {
                    "print_host": bundle.full_config_value("print_host"),
                    "host_type": bundle.full_config_value("host_type"),
                    "printhost_apikey": bundle.full_config_value("printhost_apikey"),
                    "printer_model": bundle.full_config_value("printer_model"),
                    "nozzle_diameter": bundle.full_config_value("nozzle_diameter"),
                    "printable_area": bundle.full_config_value("printable_area"),
                    "printable_height": bundle.full_config_value("printable_height"),
                    "gcode_flavor": bundle.full_config_value("gcode_flavor"),
                }
                self._send_json(200, {"ok": True, "connection": conn})
            elif path in ("/api/v1/presets", "/presets"):
                self._send_json(200, {"ok": True, "presets": _get_presets_detail()})
            elif path in ("/api/v1/events", "/events"):
                self._send_json(200, {"ok": True, "events": list(_STATE["events"])})
            elif path in ("/api/v1/gcode", "/gcode"):
                gpath = _STATE.get("last_gcode_path", "")
                if not gpath or not os.path.isfile(gpath):
                    self._send_json(404, {"ok": False, "error": "No live GUI G-code captured yet."})
                    return
                with open(gpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                self._send_json(200, {"ok": True, "gcode_path": gpath, "gcode": content})
            else:
                self._send_json(404, {"ok": False, "error": f"Unknown endpoint {path}"})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})

    def do_POST(self):
        if not self._check_auth():
            return
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            body = self._read_body()
            if path in ("/api/v1/presets/reload", "/presets/reload"):
                bundles = orca.host.preset.bundles
                target_id = body.get("bundle_id")
                reloaded = []
                if target_id:
                    bundles.reload_local(target_id)
                    reloaded.append(target_id)
                else:
                    for b_id in bundles.list_ids():
                        bundles.reload_local(b_id)
                        reloaded.append(b_id)
                orca.host.ui.push_notification(
                    orca.host.ui.NotificationLevel.Info,
                    f"MCP updated & reloaded local presets ({len(reloaded)} bundle(s))",
                )
                self._send_json(200, {"ok": True, "reloaded_bundles": reloaded})
            elif path in ("/api/v1/notify", "/notify"):
                msg = body.get("message", "MCP update")
                orca.host.ui.push_notification(orca.host.ui.NotificationLevel.Info, msg)
                self._send_json(200, {"ok": True})
            else:
                self._send_json(404, {"ok": False, "error": f"Unknown POST endpoint {path}"})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})


def _start_server(host: str, port: int, token: str):
    global _HTTP_SERVER, _SERVER_THREAD
    with _SERVER_LOCK:
        _STATE["host"] = host
        _STATE["port"] = int(port)
        _STATE["token"] = token
        if _HTTP_SERVER is not None:
            return
        srv = ThreadingHTTPServer((host, int(port)), RemoteAPIHandler)
        t = threading.Thread(target=srv.serve_forever, name="Orca25MCPBridge", daemon=True)
        t.start()
        _HTTP_SERVER = srv
        _SERVER_THREAD = t
        _STATE["running"] = True


class RemoteAPIServerCapability(orca.script.ScriptPluginCapabilityBase):
    def get_name(self) -> str:
        return "Remote API Server (Port 13130)"

    def _init_token(self) -> str:
        try:
            store_dir = orca.host.plugin.storage()
            tfile = os.path.join(store_dir, "api_token.txt")
            if os.path.isfile(tfile):
                with open(tfile, "r", encoding="utf-8") as f:
                    tok = f.read().strip()
                if tok:
                    return tok
            tok = secrets.token_urlsafe(24)
            with open(tfile, "w", encoding="utf-8") as f:
                f.write(tok)
            return tok
        except Exception:
            return _STATE["token"] or secrets.token_urlsafe(24)

    def on_load(self):
        tok = self._init_token()
        try:
            _start_server("127.0.0.1", 13130, tok)
        except Exception as exc:
            _record_event("server_error", {"error": str(exc)})

    def on_unload(self):
        global _HTTP_SERVER, _SERVER_THREAD
        with _SERVER_LOCK:
            if _HTTP_SERVER:
                _HTTP_SERVER.shutdown()
                _HTTP_SERVER.server_close()
                _HTTP_SERVER = None
                _SERVER_THREAD = None
                _STATE["running"] = False

    def on_lifecycle_event(self, event, ctx):
        evt_name = str(event).split(".")[-1]
        name = getattr(ctx, "name", "")
        msg = getattr(ctx, "msg", "")
        _record_event(evt_name, {"name": name, "msg": msg, "code": str(getattr(ctx, "code", ""))})

        if event in (orca.LifecycleEvent.ProjectOpened, orca.LifecycleEvent.ProjectAfterSave):
            if name:
                _STATE["last_project_path"] = name
        elif event == orca.LifecycleEvent.SliceStarted:
            _STATE["slice_status"] = "slicing"
            _STATE["last_slice_error"] = ""
        elif event == orca.LifecycleEvent.GCodeExportFinished:
            if msg and os.path.isfile(msg):
                _STATE["last_gcode_path"] = msg
        elif event == orca.LifecycleEvent.SlicingJobComplete:
            if getattr(ctx, "code", None) == orca.LifecycleEvtCode.Ok:
                _STATE["slice_status"] = "completed"
            elif getattr(ctx, "code", None) == orca.LifecycleEvtCode.Warn:
                _STATE["slice_status"] = "cancelled"
            else:
                _STATE["slice_status"] = "error"
                _STATE["last_slice_error"] = msg

    def execute(self):
        if not _STATE["running"]:
            _start_server("127.0.0.1", 13130, self._init_token())
        html = f"""<!DOCTYPE html>
<html><body style="font-family:sans-serif;padding:18px;">
<h3>OrcaSlicer 2.5.0 Native MCP Bridge (v3.0)</h3>
<p><b>Endpoint:</b> <code>http://{_STATE['host']}:{_STATE['port']}</code></p>
<p><b>ORCA_API_TOKEN:</b> (Auto-discovered by orca25_mcp_server.py)</p>
<pre style="background:rgba(128,128,128,0.15);padding:10px;border-radius:6px;user-select:all;">{_STATE['token']}</pre>
</body></html>"""
        orca.host.ui.show_dialog(html=html, title="OrcaSlicer 2.5.0 MCP Token", width=520, height=280)
        return orca.ExecutionResult.success()


class RemoteAPIGCodeTapCapability(orca.slicing.SlicingPipelineCapabilityBase):
    def get_name(self) -> str:
        return "Remote API G-code Tap"

    def execute(self, ctx):
        if ctx.step == orca.slicing.Step.psGCodePostProcess and ctx.gcode_path:
            _STATE["last_gcode_path"] = ctx.gcode_path
        return orca.ExecutionResult.success()


@orca.plugin
class RemoteAPIPluginPackage(orca.base):
    def register_capabilities(self):
        orca.register_capability(RemoteAPIServerCapability)
        orca.register_capability(RemoteAPIGCodeTapCapability)
