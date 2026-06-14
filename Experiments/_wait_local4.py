"""Background waiter for the local-4 resume. Exits with a one-line verdict on:
  DONE  - cell count >= 5625
  DIED  - no python runner alive AND count < 5625
Polls every 20s. Uses tasklist (never opens the data JSON) for liveness and a
brief json.load only to read the count; atomic_write now retries on WinError5 so
that brief read can no longer crash the runner. Progress is confirmed by reading
the flushed confirmatory_run.log directly, not here.
"""
import json, time, subprocess

JSON_PATH = "confirmatory_responses.json"


def runner_alive():
    r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                       capture_output=True, text=True)
    # this waiter is also python.exe, so >=2 means the runner is present too
    return r.stdout.count("python.exe") >= 2


def cell_count():
    try:
        return len(json.load(open(JSON_PATH)))
    except Exception:
        return -1


while True:
    n = cell_count()
    if n >= 5625:
        print(f"DONE cells={n}")
        break
    if not runner_alive():
        n = cell_count()
        print(f"DONE cells={n}" if n >= 5625 else f"DIED cells={n}")
        break
    time.sleep(20)
