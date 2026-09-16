"""공장 작업자 자세 동작 분류 프로토타입 대시보드."""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, send_from_directory

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pose_action_classifier import build_demo_scenarios, render_all_figures

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=None)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/scenarios")
def api_scenarios():
    return jsonify(build_demo_scenarios())


@app.get("/outputs/<path:filename>")
def outputs(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


def serve(host: str = "0.0.0.0", port: int = 8765, out_dir: Path | None = None) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = Path(out_dir) if out_dir is not None else OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_all_figures(OUTPUT_DIR)
    print(f"\n대시보드: http://127.0.0.1:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    serve()
