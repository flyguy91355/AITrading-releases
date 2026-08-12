#!/usr/bin/env python3
"""Capture modal/toggle screenshots by driving headless Chrome via CDP.

Targets (real dashboard.html functions/selectors, verified by grep before use):
  - Manual Buy modal:  openManualTradeModal()      (button "+ Trade" in header,
                        web/templates/dashboard.html:670)
  - Deep Dive modal:   #manualDDTicker input + triggerManualDeepDive()
                        (web/templates/dashboard.html:744-748); result arrives
                        async via the 'deep_dive_report' WebSocket message and
                        showDeepDiveModal() is called automatically when
                        msg.ticker matches the pending _manualDDTicker
                        (web/templates/dashboard.html:1035-1043). We poll for
                        #deepDiveModal to gain the 'active' class rather than a
                        fixed sleep, since a real Claude API call has variable
                        latency.
  - Universe toggle:  setUniverseView('universe')  (web/templates/dashboard.html:774,
                        1945-1959) - fetches GET /api/universe and re-renders the grid.
"""
import base64
import json
import subprocess
import time
from pathlib import Path
import urllib.request

OUT_DIR = Path(__file__).parent.parent / "docs" / "manual_screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PORT = 9333
DEEP_DIVE_TICKER = "ABBV"  # real held position (confirmed via GET /api/portfolio)


def start_chrome():
    proc = subprocess.Popen([
        "google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
        f"--remote-debugging-port={PORT}", "--remote-allow-origins=*",
        "--window-size=1400,900",
        "http://localhost:8080/",
    ])
    time.sleep(2)
    return proc


def cdp_ws_url() -> str:
    # /json can list multiple targets (extension background pages, the
    # TradingView ticker-tape iframe, etc) alongside the actual dashboard tab
    # -- select explicitly by type=="page" and our target URL rather than
    # blindly taking the first entry.
    with urllib.request.urlopen(f"http://localhost:{PORT}/json") as r:
        tabs = json.loads(r.read())
    for tab in tabs:
        if tab.get("type") == "page" and tab.get("url", "").startswith("http://localhost:8080"):
            return tab["webSocketDebuggerUrl"]
    raise RuntimeError(f"dashboard tab not found among CDP targets: {tabs}")


def main():
    proc = start_chrome()
    try:
        import websocket  # websocket-client package
        ws = websocket.create_connection(cdp_ws_url())
        msg_id = 0

        def send(method, params=None):
            nonlocal msg_id
            msg_id += 1
            ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
            while True:
                resp = json.loads(ws.recv())
                if resp.get("id") == msg_id:
                    return resp

        def evaluate(expression: str):
            return send("Runtime.evaluate", {"expression": expression})

        def screenshot(filename: str):
            shot = send("Page.captureScreenshot")
            (OUT_DIR / filename).write_bytes(base64.b64decode(shot["result"]["data"]))
            print(f"OK {filename}")

        def wait_for(expression: str, timeout_s: float, interval_s: float = 2.0) -> bool:
            waited = 0.0
            while waited < timeout_s:
                resp = evaluate(expression)
                if resp.get("result", {}).get("result", {}).get("value"):
                    return True
                time.sleep(interval_s)
                waited += interval_s
            return False

        send("Page.enable")
        send("Runtime.enable")
        time.sleep(2)  # allow initial WebSocket connection + first data load

        # --- Manual Buy modal ---
        # openManualTradeModal() is wired to the "+ Trade" header button
        # (dashboard.html:670); it populates the SELL tab from live positions
        # and sets #manualTradeModal display:flex (dashboard.html:2256-2284).
        evaluate("openManualTradeModal()")
        time.sleep(1)
        screenshot("modal_manual_buy.png")
        evaluate("closeManualTradeModal()")
        time.sleep(0.5)

        # --- Deep Dive modal ---
        # triggerManualDeepDive() (dashboard.html:2112) reads ticker from
        # #manualDDTicker and POSTs /api/deep-dive. This is a REAL Claude API
        # call against the live paper-trading account. The result modal is
        # opened asynchronously by the 'deep_dive_report' WebSocket handler
        # (dashboard.html:1035-1043), so poll for #deepDiveModal to become
        # 'active' instead of a fixed sleep.
        evaluate(
            f"document.getElementById('manualDDTicker').value='{DEEP_DIVE_TICKER}';"
            "triggerManualDeepDive();"
        )
        ready = wait_for(
            "document.getElementById('deepDiveModal').classList.contains('active')",
            timeout_s=120,
            interval_s=3,
        )
        if not ready:
            print("WARN deep dive modal did not become active within timeout")
        time.sleep(1)  # let final DOM render settle
        screenshot("modal_deep_dive.png")
        evaluate("closeDeepDiveModal()")
        time.sleep(0.5)

        # --- Universe toggle view ---
        # setUniverseView('universe') (dashboard.html:1945) is bound to the
        # #viewUniverse button (dashboard.html:774) and fetches GET /api/universe.
        evaluate("setUniverseView('universe')")
        time.sleep(2.5)  # /api/universe fetch + grid re-render
        screenshot("dashboard_universe.png")

        ws.close()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
