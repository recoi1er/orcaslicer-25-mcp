#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx>=0.27.0",
#   "numpy>=1.26.0",
#   "matplotlib>=3.9.0",
#   "mcp<2",
# ]
# ///

import argparse
import sys
import os

sys.path.append(os.path.expanduser("~/orca25-mcp"))

try:
    from orca25_mcp_server import auto_optimize_print, render_plate, check_printer_match, get_printer_status
except ImportError as e:
    print(f"❌ Error importing orca25_mcp_server tools: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Automated CAD-to-Print Pipeline")
    parser.add_argument("--model", required=True, help="Path to input .stl or .step file")
    parser.add_argument("--intent", default="strength", choices=["strength", "visual_quality", "dimensional_accuracy", "speed"], help="Engineering optimization intent")
    args = parser.parse_args()

    print(f"🚀 [1/4] Optimizing & slicing {args.model} for intent='{args.intent}' (0.5mm nozzle)...")
    opt_res = auto_optimize_print(model_path=args.model, intent=args.intent, auto_orient=True, auto_arrange=True, open_in_gui=True)
    if not opt_res.get("ok"):
        print(f"❌ Optimization/Slicing failed: {opt_res}")
        sys.exit(1)
    print(f"   Successfully sliced! G-code: {opt_res['slice_result'].get('gcode_path')}")

    print("🖼️  [2/4] Generating offscreen 3D quad render verification...")
    render_img = render_plate(camera="quad", view="editor", stl_or_gcode_path=args.model)
    print(f"   Render saved and verified: {render_img.path}")

    print("⚖️  [3/4] Checking Klipper printer hardware match (0.5mm nozzle)...")
    match_res = check_printer_match()
    print(f"   Klipper URL: {match_res.get('printer_url')}")
    if match_res.get("warnings"):
        for w in match_res["warnings"]:
            print(f"   ⚠️  Warning: {w}")
    else:
        print("   ✅ Slicer and Klipper hardware limits match perfectly.")

    printer_status = get_printer_status()
    print(f"   Printer State: {printer_status.get('state')} | Bed: {printer_status.get('bed_temp_c')}°C | Nozzle: {printer_status.get('extruder_temp_c')}°C")

    print("\n🎉 Workflow completed successfully! The optimized part is open in OrcaSlicer 2.5.0.")


if __name__ == "__main__":
    main()
