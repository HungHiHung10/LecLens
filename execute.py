import os
import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import socket
import webbrowser
import uvicorn

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def find_available_port(start_port: int = 8000) -> int:
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port

if __name__ == "__main__":
    # Ensure current directory is in sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    port = find_available_port(8000)
    url = f"http://127.0.0.1:{port}"

    print("=" * 65)
    print(" 🚀 LectureLens — Video Search Assistant & Grounding Studio")
    print(" 🎯 Thiết kế tối ưu cho CPU Intel Core i5-1135G7 (12GB RAM)")
    print(f" 🌐 Địa chỉ Web UI: {url}")
    print("=" * 65)

    # Open browser automatically after server starts
    webbrowser.open(url)

    uvicorn.run("backend.main:app", host="127.0.0.1", port=port, log_level="info")
