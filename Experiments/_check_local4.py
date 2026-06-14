import json, os, time, sys
# Data-health watchdog only (NO H1 peeking): fires when local-4 reaches the
# 4500-cell target OR the checkpoint goes stale (>180s without a write = dead/hung).
try:
    n = len(json.load(open('confirmatory_responses.json')))
    age = time.time() - os.path.getmtime('confirmatory_responses.json')
except Exception:
    n, age = -1, 0.0
done = n >= 4500
stalled = (n < 4500) and (age > 1200)  # 14B writes every 10 cells @ ~30s = ~300s gaps; 1200s = real death only
sys.exit(0 if (done or stalled) else 1)
