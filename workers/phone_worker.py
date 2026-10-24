#!/usr/bin/env python3
"""Optional Galaxy S25+ worker via the existing headless phone link (no taps, no UI).
Guards: pauses when the SoC is hot (> 42 C on thermal_zone0) or when battery level is low/unknown.
The phone currently has no Python runtime and the app uid cannot read battery level, so this
guard reports PAUSED and the coordinator simply never assigns it shards (safe default)."""
import subprocess, json, sys
import os
PHONE = os.environ.get("GENESIS_PHONE_CMD", "phone")  # path to your own headless phone-link wrapper
def sh(cmd):
    r = subprocess.run(["timeout", "40", PHONE, "sh", cmd], capture_output=True, text=True); return r.returncode, r.stdout.strip()
def status():
    rc, out = sh("cat /sys/class/thermal/thermal_zone0/temp; cat /sys/class/power_supply/battery/capacity 2>/dev/null || echo NA; command -v python3 || echo NOPY")
    if rc == 3: return dict(state="PAUSED", reason="phone not connected")
    lines = out.splitlines(); temp = float(lines[0]) / 1000 if lines and lines[0].isdigit() else None
    batt = lines[1] if len(lines) > 1 else "NA"; py = not (len(lines) > 2 and lines[2] == "NOPY")
    if temp is None or temp > 42: return dict(state="PAUSED", reason=f"hot or unknown temp ({temp})", temp_c=temp)
    if batt == "NA" or int(batt) < 40: return dict(state="PAUSED", reason=f"battery low or unreadable ({batt})", temp_c=temp)
    if not py: return dict(state="PAUSED", reason="no python runtime on phone", temp_c=temp)
    return dict(state="READY", temp_c=temp, battery=int(batt))
if __name__ == "__main__":
    print(json.dumps(status()))
