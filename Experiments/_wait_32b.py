"""Waiter for the 32B time-test. Exits when the test process finishes
(its .out gets the DONE marker) or the runner vanishes. Reads only the .out
marker file and tasklist — never opens the shared data JSON."""
import time, subprocess

OUT = "confirmatory_32btest.out"


def runner_alive():
    r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                       capture_output=True, text=True)
    return r.stdout.count("python.exe") >= 2  # waiter is also python


while True:
    try:
        txt = open(OUT, encoding="utf-8", errors="replace").read()
    except Exception:
        txt = ""
    if "32BTEST DONE" in txt:
        print("DONE 32b time-test finished")
        break
    if not runner_alive():
        print("EXITED runner gone (check .out for DONE or traceback)")
        break
    time.sleep(20)
