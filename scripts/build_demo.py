#!/usr/bin/env python3
"""Builds the self-contained investor demo HTML: the real dashboard.html
plus an embedded data snapshot and a network-interception shim. See
docs/superpowers/specs/2026-08-14-interactive-demo-design.md.

Usage: python3 scripts/build_demo.py
Reads:  data/demo_snapshot.json, web/templates/dashboard.html
Writes: data/demo_output.html
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SNAPSHOT_PATH = ROOT / "data" / "demo_snapshot.json"
DASHBOARD_PATH = ROOT / "web" / "templates" / "dashboard.html"
OUTPUT_PATH = ROOT / "data" / "demo_output.html"
# Vendored copy of the exact CDN file dashboard.html normally loads via
# <script src="https://unpkg.com/lightweight-charts@4.1.3/..."> -- Claude
# Artifacts (where the demo is actually hosted) enforce a strict CSP that
# blocks every external network request, so that CDN tag silently fails to
# load there and every chart in the app breaks (2026-08-14 owner report:
# "its every chart in demp not working"). This library has no runtime network
# calls of its own (pure client-side rendering), so inlining its real source
# is a safe, exact substitute -- unlike the two TradingView *widget* scripts
# below, which need a live TradingView connection and can't be inlined at all.
LWC_VENDOR_PATH = ROOT / "scripts" / "vendor" / "lightweight-charts-4.1.3.standalone.production.js"
LWC_CDN_TAG = '<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>'

# {ticker} substitution: real fetch URLs like /api/position/OXY/history are
# captured verbatim as keys (the capture script stores the literal expanded
# URL, not a template), so the shim only ever needs an exact string match --
# no runtime templating logic required here.
_SHIM_TEMPLATE = """
<script id="demo-shim">
(function() {
  var SNAPSHOT = __DEMO_SNAPSHOT__;

  // The real dashboard's Positions/On-Deck resize handle persists its width in
  // localStorage and applies it unclamped on load -- a fresh browser has no
  // saved value, so it falls back to the page's own cramped default (owner
  // report, 2026-08-14: "the postions is all crunched up," same issue already
  // seen and fixed for the plain screenshot earlier this session, this time
  // for the actual interactive demo). Set a real default here, before the
  // real dashboard's own resize-handle script runs later in this document --
  // computed from the viewer's real window width (half of it, capped at
  // 1150px, the same width already confirmed to look right) rather than one
  // fixed pixel value, so it looks reasonable regardless of their screen size.
  try {
    var _leftCol = Math.round(Math.min(window.innerWidth * 0.5, 1150));
    localStorage.setItem('mainLayoutLeftColWidth', String(_leftCol));
  } catch (e) {}

  // Two TradingView-hosted widget scripts (the header ticker tape, and the
  // Advanced Real-Time Chart in stock modals) fetch a live embed script from
  // s3.tradingview.com and, once loaded, pull live data from TradingView's own
  // servers -- unlike the lightweight-charts library inlined below, there's no
  // static file to vendor here; the whole point of these widgets is a live
  // connection this sandbox can never have. Left alone, appending them under
  // the Artifact CSP just silently fails, leaving a bare empty box. Intercept
  // at appendChild (both real call sites -- initTickerTape/mountTradingViewChart
  // in dashboard.html -- append the <script> tag straight into its own mount
  // point) and swap in an honest placeholder instead, so the demo degrades
  // gracefully instead of looking broken (2026-08-14, same investigation as the
  // lightweight-charts CDN fix above).
  var _origAppendChild = Node.prototype.appendChild;
  Node.prototype.appendChild = function(node) {
    if (node && node.tagName === 'SCRIPT' && node.src && node.src.indexOf('s3.tradingview.com') !== -1) {
      var note = document.createElement('div');
      note.style.cssText = 'display:flex;align-items:center;justify-content:center;height:100%;min-height:60px;' +
        'color:var(--text-muted,#8a94a6);font:13px sans-serif;text-align:center;padding:12px;';
      note.textContent = 'Live TradingView chart unavailable in this demo snapshot.';
      return _origAppendChild.call(this, note);
    }
    return _origAppendChild.call(this, node);
  };

  function demoResponse(body, status) {
    return new Response(JSON.stringify(body), {
      status: status || 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  window.fetch = function(input, init) {
    var url = typeof input === 'string' ? input : input.url;
    var method = (init && init.method) ? init.method.toUpperCase() : 'GET';
    var path = url.replace(/^https?:\\/\\/[^/]+/, '');

    if (method !== 'GET') {
      // Every write action (Buy, Sell, Save Settings, Full Scan, Deep Dive,
      // manual removal, etc.) -- demo mode, never reaches a real network call.
      return Promise.resolve(demoResponse({
        status: 'demo_mode',
        message: 'Demo Mode — no real action taken.',
      }));
    }
    if (Object.prototype.hasOwnProperty.call(SNAPSHOT, path)) {
      return Promise.resolve(demoResponse(SNAPSHOT[path]));
    }
    // Unrecognized GET (e.g. an On Shore ticker outside the captured top 15)
    // -- degrade this one thing gracefully, never a hard JS error.
    console.warn('[demo] no captured data for', path);
    return Promise.resolve(demoResponse({ error: 'not available in demo' }, 404));
  };

  function FakeWebSocket(url) {
    this.url = url;
    this.readyState = 0;
    var self = this;
    setTimeout(function() {
      self.readyState = 1;
      if (self.onopen) self.onopen({});
      self._sendInit();
      // Reset the real dashboard's own 60s heartbeat-timeout/reconnect timer
      // every 45s WITHOUT resending the real 'init' message a second time
      // (fixed 2026-08-14, owner report: closing a popup got progressively
      // slower "to the point i dont think its working," and charts going
      // missing -- both traced to this same spot). handleMessage's 'init'
      // case unconditionally re-appends every AI log entry as a brand new
      // DOM node (never clears old ones) and re-fetches/re-renders the whole
      // On Deck/On Shore grid, re-mounting every chart from scratch -- fine
      // once on a genuine real connection (init only ever fires once there),
      // but resending it repeatedly here replayed that entire expensive
      // sequence on top of itself every 45s: the log grew without bound, and
      // charts got re-mounted onto containers that already had a live chart
      // on them. A message with a type the real switch statement doesn't
      // recognize (falls through, no-op) still resets the heartbeat via
      // resetHeartbeat(), which runs before handleMessage in the real
      // ws.onmessage handler, and does nothing else.
      self._heartbeat = setInterval(function() { self._sendHeartbeatOnly(); }, 45000);
    }, 50);
  }
  FakeWebSocket.prototype._sendInit = function() {
    var payload = SNAPSHOT['/api/dashboard-poll'];
    if (payload && this.onmessage) {
      this.onmessage({ data: JSON.stringify(payload) });
    }
  };
  FakeWebSocket.prototype._sendHeartbeatOnly = function() {
    if (this.onmessage) {
      this.onmessage({ data: JSON.stringify({ type: 'demo_heartbeat' }) });
    }
  };
  FakeWebSocket.prototype.send = function(data) {
    // Every WS command (execute_buy, confirm_buy, execute_sell, pause/resume,
    // remove_on_deck) -- demo mode, never actually mutates anything.
    console.log('[demo] WS send intercepted:', data);
  };
  FakeWebSocket.prototype.close = function() {
    this.readyState = 3;
    clearInterval(this._heartbeat);
    if (this.onclose) this.onclose({});
  };
  window.WebSocket = FakeWebSocket;

  window.addEventListener('DOMContentLoaded', function() {
    var banner = document.createElement('div');
    banner.textContent = 'Demo — showing a snapshot, not live. Click to dismiss.';
    banner.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:99999;' +
      'background:#5b9bf2;color:#0a0e16;font:600 13px sans-serif;text-align:center;' +
      'padding:8px;cursor:pointer;';
    banner.onclick = function() { banner.remove(); };
    document.body.prepend(banner);
  });
})();
</script>
"""


def _inline_lightweight_charts(html: str, lwc_source: str) -> str:
    """Replace the CDN <script src=unpkg...> tag with the real library source
    inlined directly -- the Artifact CSP blocks the CDN request entirely (see
    LWC_VENDOR_PATH's comment above), so every chart in the app silently broke
    on the published demo until this ran. Raises if the tag isn't found, since
    a silent no-op here would just reintroduce the exact bug this exists to fix."""
    if LWC_CDN_TAG not in html:
        raise ValueError(
            "Expected lightweight-charts CDN <script> tag not found in dashboard.html "
            "-- it may have moved or its version changed. Update LWC_CDN_TAG (and "
            "re-vendor scripts/vendor/lightweight-charts-*.js if the version changed) "
            "before rebuilding the demo, or every chart will silently break again."
        )
    inline_tag = f"<script>\n{lwc_source}\n</script>"
    return html.replace(LWC_CDN_TAG, inline_tag)


def build_demo_html(snapshot: dict, dashboard_html: str, lwc_source: str = "") -> str:
    shim = _SHIM_TEMPLATE.replace("__DEMO_SNAPSHOT__", json.dumps(snapshot))
    # Insert immediately after <head> so the shim runs before every other
    # script in the document, including the real dashboard's own inline
    # <script> block (which calls connect() as its very last statement).
    marker = "<head>"
    idx = dashboard_html.index(marker) + len(marker)
    out = dashboard_html[:idx] + shim + dashboard_html[idx:]
    if lwc_source:
        out = _inline_lightweight_charts(out, lwc_source)
    return out


def main():
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    dashboard_html = DASHBOARD_PATH.read_text(encoding="utf-8")
    lwc_source = LWC_VENDOR_PATH.read_text(encoding="utf-8")
    out = build_demo_html(snapshot, dashboard_html, lwc_source)
    OUTPUT_PATH.write_text(out, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({len(out):,} bytes)")


if __name__ == "__main__":
    main()
