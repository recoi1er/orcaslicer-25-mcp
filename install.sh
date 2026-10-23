#!/usr/bin/env zsh
set -euo pipefail

echo "=================================================================="
echo " OrcaSlicer 2.5.0 Native MCP Bridge & Closed-Loop Installer (v1.2.0)"
echo "=================================================================="

ORCA_SUPPORT_DIR="$HOME/Library/Application Support/OrcaSlicer"
PLUGIN_DIR="$ORCA_SUPPORT_DIR/orca_plugins/remote_api_bridge"
MCP_DIR="$HOME/orca25-mcp"
UV_BIN="/opt/homebrew/bin/uv"

if [[ ! -x "$UV_BIN" ]]; then
  UV_BIN="$(command -v uv || true)"
fi
if [[ -z "$UV_BIN" ]]; then
  echo "❌ Error: 'uv' was not found. Install it with: brew install uv"
  exit 1
fi

echo ""
echo "-> [1/3] Installing OrcaSlicer 2.5.0 plugin..."
mkdir -p "$PLUGIN_DIR"
cp plugin/remote_api_plugin.py "$PLUGIN_DIR/remote_api_plugin.py"

echo "-> [2/3] Installing MCP server & workflow scripts to $MCP_DIR..."
mkdir -p "$MCP_DIR"
mkdir -p "$HOME/.orcaslicer-mcp/notes"
mkdir -p "$MCP_DIR/renders"
cp server/orca25_mcp_server.py "$MCP_DIR/orca25_mcp_server.py"
cp workflow/run_workflow.py "$MCP_DIR/run_workflow.py"
chmod +x "$MCP_DIR/orca25_mcp_server.py"
chmod +x "$MCP_DIR/run_workflow.py"

echo "-> [3/3] Pre-warming uv Python environment..."
"$UV_BIN" run --no-config --managed-python --python 3.13 "$MCP_DIR/orca25_mcp_server.py" --help >/dev/null 2>&1 || true

echo ""
echo "=================================================================="
echo " Configuring MCP Client Integration"
echo "=================================================================="

python3 - "$UV_BIN" "$MCP_DIR/orca25_mcp_server.py" << 'EOF_PY'
import json, os, sys

uv_bin, server_script = sys.argv[1], sys.argv[2]

clients = {
    "1": ("Gemini Desktop", os.path.expanduser("~/.gemini/settings.json")),
    "2": ("Claude Desktop", os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")),
    "3": ("Cursor", os.path.expanduser("~/.cursor/mcp.json")),
    "4": ("Custom JSON Config Path", "")
}

print("Select your AI client(s) to configure (comma-separated, e.g. 1,2):")
for k, (name, path) in clients.items():
    exists_status = " (detected)" if path and os.path.isfile(path) else ""
    print(f"  [{k}] {name}{exists_status}")

selection = input("Enter choice(s) [default: 1]: ").strip()
if not selection:
    selection = "1"

selected_paths = []
for choice in selection.split(","):
    choice = choice.strip()
    if choice in clients:
        if choice == "4":
            custom_path = input("Enter full path to client config JSON: ").strip()
            if custom_path:
                selected_paths.append(os.path.expanduser(custom_path))
        else:
            selected_paths.append(clients[choice][1])

# Optional Text-to-CAD prompt
print("")
install_cad = input("Would you like to also install & configure Text-to-CAD (cadgen) for complete design-to-print capability? (y/N): ").strip().lower()

server_configs = {
    "orcaslicer-25": {
        "command": uv_bin,
        "args": ["run", "--no-config", "--managed-python", "--python", "3.13", server_script],
        "env": {
            "ORCA_BIN": "/Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer",
            "MOONRAKER_URL": "http://klipper-pi.local"
        }
    }
}

if install_cad == 'y':
    server_configs["text-to-cad"] = {
        "command": "/opt/homebrew/bin/uvx",
        "args": [
            "--no-config",
            "--managed-python",
            "--python",
            "3.13",
            "--from",
            "cadgen==0.7.24",
            "cadgen",
            "mcp"
        ]
    }
    print("   + Text-to-CAD (cadgen) queued for installation.")

for path in selected_paths:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    
    mcp_servers = data.setdefault("mcpServers", {})
    for server_name, config in server_configs.items():
        mcp_servers[server_name] = config
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Successfully configured MCP server(s) in: {path}")

EOF_PY

echo ""
echo "=================================================================="
echo " ✅ Installation Complete!"
echo " Next Steps:"
echo "   1. Open OrcaSlicer 2.5.0 -> File -> Plugins."
echo "   2. Enable 'Remote API Bridge (Native 2.5.0 MCP)' and click Yes."
echo "   3. Restart your AI client(s)."
echo "=================================================================="
