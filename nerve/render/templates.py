"""State -> HTML for the three intake sample UIs.

These are the *ground-truth renderers* the neural field learns from. Each function takes a spec state and returns
the page as it would look in that state. (For a real product this is the running app under Playwright.)
"""
from __future__ import annotations

from typing import Any

_BASE = """<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:-apple-system,Helvetica,sans-serif;background:#f6f7f9;margin:0;display:grid;place-items:center;height:100vh}
.card{background:#fff;border-radius:16px;box-shadow:0 8px 30px rgba(0,0,0,.08)}
%s</style></head><body>%s</body></html>"""


def push_notifications_toggle(s: dict[str, Any]) -> str:
    css = """.card{padding:32px 40px;width:360px}h1{font-size:18px;margin:0 0 18px;color:#111}
.row{display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-top:1px solid #eee}
.label{color:#333;font-size:15px}.sub{color:#888;font-size:12px}
.switch{width:52px;height:30px;border-radius:15px;position:relative}
.knob{position:absolute;top:3px;width:24px;height:24px;border-radius:12px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.3)}"""
    enabled, grabbed, knob = s["enabled"], s["grabbed"], float(s["knob"])
    # track colour follows the knob while grabbed, the committed state otherwise
    mix = knob if grabbed else (1.0 if enabled else 0.0)
    r, g, b = int(204 + (52 - 204) * mix), int(204 + (199 - 204) * mix), int(204 + (89 - 204) * mix)
    left = 3 + 22 * knob
    sub = "Dragging…" if grabbed else ("Currently: On" if enabled else "Currently: Off")
    html = f'''<div class="card"><h1>Notifications</h1><div class="row"><div><div class="label">Push notifications</div>
<div class="sub">{sub}</div></div><div class="switch" style="background:rgb({r},{g},{b})"><div class="knob" style="left:{left:.1f}px"></div></div></div></div>'''
    return _BASE % (css, html)


def guest_stepper(s: dict[str, Any]) -> str:
    css = """.card{padding:28px 36px;width:380px}h1{font-size:18px;margin:0 0 6px;color:#111}.sub{color:#888;font-size:13px;margin-bottom:18px}
.stepper{display:flex;align-items:center;gap:18px;justify-content:center;margin:16px 0}
button{width:48px;height:48px;border-radius:12px;border:1px solid #ddd;background:#fafafa;font-size:24px;color:#222}
button[disabled]{color:#bbb;background:#f2f2f2}.val{font-size:44px;font-weight:600;min-width:70px;text-align:center;color:#111}
.foot{display:flex;justify-content:space-between;color:#666;font-size:13px;border-top:1px solid #eee;padding-top:12px;margin-top:12px}.reset{color:#0a66ff;font-weight:500}"""
    g = int(s["guests"])
    html = f'''<div class="card"><h1>Guests</h1><div class="sub">Choose between 1 and 8 guests</div>
<div class="stepper"><button {"disabled" if g <= 1 else ""}>−</button><div class="val">{g}</div><button {"disabled" if g >= 8 else ""}>+</button></div>
<div class="foot"><span>Minimum 1, maximum 8</span><span class="reset">Reset</span></div></div>'''
    return _BASE % (css, html)


def volume_slider(s: dict[str, Any]) -> str:
    css = """.card{padding:28px 36px;width:420px}h1{font-size:18px;margin:0 0 4px;color:#111}.sub{color:#888;font-size:13px;margin-bottom:22px}
.track{height:6px;border-radius:3px;background:#e5e5ea;position:relative;margin:30px 10px 10px}
.fill{position:absolute;left:0;top:0;height:6px;background:#0a66ff;border-radius:3px}
.thumb{position:absolute;top:-9px;width:24px;height:24px;margin-left:-12px;border-radius:12px;background:#fff;border:1px solid #ccc;box-shadow:0 2px 6px rgba(0,0,0,.2)}
.thumb.grab{transform:scale(1.25);border-color:#0a66ff}
.ticks{display:flex;justify-content:space-between;color:#999;font-size:12px;margin:0 6px}
.readout{text-align:center;font-size:22px;font-weight:600;color:#111;margin-top:14px}.hint{color:#888;font-size:12px;text-align:center;margin-top:6px}"""
    v, grabbed, muted = int(s["volume"]), s["grabbed"], s["muted"]
    html = f'''<div class="card"><h1>Volume</h1><div class="sub">Drag the handle to set volume, in steps of 10</div>
<div class="track"><div class="fill" style="width:{v}%"></div><div class="thumb{" grab" if grabbed else ""}" style="left:{v}%"></div></div>
<div class="ticks"><span>0</span><span>50</span><span>100</span></div>
<div class="readout">{"Muted" if muted else v}</div><div class="hint">{"Muted at 0 · Max 100" if not grabbed else "Release to set"}</div></div>'''
    return _BASE % (css, html)


TEMPLATES = {"push_notifications_toggle": push_notifications_toggle, "guest_stepper": guest_stepper, "volume_slider": volume_slider}
