# OrcaSlicer 2.5.0 Native MCP Bridge & Closed-Loop Workflow

Drive **OrcaSlicer 2.5.0+** natively from Claude, Gemini, or any Model Context Protocol (MCP) client **without recompiling OrcaSlicer**. 

This package combines a lightweight embedded Python plugin with a powerful FastMCP server, bringing full object inspection, live preset mutation, deterministic physics checks, offscreen 3D/G-code rendering, and Klipper integration into your AI workflow.

# OrcaSlicer 2.5.0 Native MCP Bridge & Closed-Loop Workflow (v1.1.0)

Drive **OrcaSlicer 2.5.0+** natively from Gemini, Claude Desktop, Cursor, or any Model Context Protocol (MCP) client **without recompiling OrcaSlicer**. 

## What's New in v1.1.0
- **Interactive Multi-LLM Installer:** Automatically detects or prompts for Gemini Desktop, Claude Desktop, Cursor, or custom JSON configuration paths.
- **Side-by-Side Tool Chaining:** Seamlessly pair with `text-to-cad` for a complete design-to-print AI loop.


## Features
- **Native 2.5.0 Plugin Bridge:** Runs inside OrcaSlicer's embedded CPython interpreter via `orca.host` and lifecycle event taps.
- **Offscreen 3D Renders (`render_plate`):** Generates 7-angle 3D inspection sheets (or 4-angle quad views) highlighting bed contact and overhangs.
- **Deterministic Physics Gate (`check_profile_physics`):** Prevents volumetric flow starvation and layer-height boundary errors before slicing.
- **Closed-Loop Print Doctor (`diagnose_print_failure`):** Analyzes print defects (warping, stringing, ghosting) and auto-fixes local preset parameters.
- **Klipper / Moonraker Integration:** Directly queries print status, webcam snapshots, and validates firmware acceleration limits.

## Installation


### Prerequisites
* **OrcaSlicer 2.5.0+** installed on your Mac (`/Applications/OrcaSlicer.app`).
* **uv** package manager installed (`brew install uv`).

### Step 1: Clone the Repository
Open your terminal and clone the project to your local machine:
```bash
git clone https://github.com/recoi1er/orcaslicer-25-mcp.git
cd orcaslicer-25-mcp