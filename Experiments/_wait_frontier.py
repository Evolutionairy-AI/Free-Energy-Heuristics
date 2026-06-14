"""Waiter for the frontier run. Exits when total cells hit 7875 (DONE) or the
runner vanishes (DIED). Uses tasklist for liveness and a brief json.load for the
count; atomic_write retries on WinError5 so the count read can't crash the runner."""
import json, time, subprocess

JSON_PATH = "confirmatory_responses.json"
TARGET = 7875


def runner_alive():
    r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                       capture_output=True, text=True)
    return r.stdout.count("python.exe") >= 2  # waiter is also python


def cell_count():
    try:
        return len(json.load(open(JSON_PATH)))
    except Exception:
        return -1


while True:
    n = cell_count()
    if n >= TARGET:
        print(f"DONE cells={n}")
        break
    if not runner_alive():
        n = cell_count()
        print(f"DONE cells={n}" if n >= TARGET else f"DIED cells={n}")
        break
    time.sleep(20)
