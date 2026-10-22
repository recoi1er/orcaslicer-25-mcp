# OrcaSlicer 2.5.0 Native MCP Bridge & Closed-Loop Workflow (v1.2.0)

Drive **OrcaSlicer 2.5.0+** natively from Gemini, Claude Desktop, Cursor, or any Model Context Protocol (MCP) client **without recompiling OrcaSlicer**. 

This package combines a lightweight embedded Python plugin with a powerful FastMCP server, bringing full object inspection, live preset mutation, deterministic physics checks, offscreen 3D/G-code rendering, and Klipper integration into your AI workflow.

## What's New in v1.2.0
- **Optional Text-to-CAD Prompt:** The installer now interactively asks users if they want to install and configure Text-to-CAD (`cadgen`) side-by-side during setup.
- **Interactive Multi-LLM Installer:** Automatically detects or prompts for Gemini Desktop, Claude Desktop, Cursor, or custom JSON configuration paths.

## Features
- **Native 2.5.0 Plugin Bridge:** Runs inside OrcaSlicer's embedded CPython interpreter via `orca.host` and lifecycle event taps.
- **Offscreen 3D Renders (`render_plate`):** Generates 7-angle 3D inspection sheets (or 4-angle quad views) highlighting bed contact and overhangs.
- **Deterministic Physics Gate (`check_profile_physics`):** Prevents volumetric flow starvation and layer-height boundary errors before slicing.
- **Closed-Loop Print Doctor (`diagnose_print_failure`):** Analyzes print defects (warping, stringing, ghosting) and auto-fixes local preset parameters.
- **Klipper / Moonraker Integration:** Directly queries print status, webcam snapshots, and validates firmware acceleration limits.

---

## Installation

### Prerequisites
* **OrcaSlicer 2.5.0+** installed on your Mac (`/Applications/OrcaSlicer.app`).
* **uv** package manager installed (`brew install uv`).

### Step 1: Clone the Repository
Open your terminal and clone the project to your local machine:
```bash
git clone [https://github.com/recoi1er/orcaslicer-25-mcp.git](https://github.com/recoi1er/orcaslicer-25-mcp.git)
cd orcaslicer-25-mcp

###Step 2: Run the Interactive Installer
Execute the installation script. It will automatically deploy the plugin, set up your Python environment, offer to install Text-to-CAD (cadgen), and prompt you to configure your preferred AI client:
bash install.sh

###Step 3: Enable the Plugin in OrcaSlicer
Open OrcaSlicer 2.5.0.
Navigate to the top menu and select File -> Plugins.
Locate Remote API Bridge (Native 2.5.0 MCP) in the list and toggle it Enabled (click Yes to confirm).
Restart your AI client application to load the MCP server tools.
"""
with open("README.md", "w", encoding="utf-8") as f:
f.write(content)
print("✅ README.md heading hierarchy fixed successfully!")
'
* Save and exit in nano by pressing **`Ctrl + O`**, then **`Enter`**, then **`Ctrl + X`**.

---