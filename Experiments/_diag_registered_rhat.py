"""Pin down per-parameter R-hat at the REGISTERED sampler settings (2000x4,
target_accept=0.95, seed 20260514) -- the settings that produce the official
confirmatory_analysis_results. The regenerated JSON shows rhat_max=1.0128 > 1.01
(genuine, not rounding). This identifies WHICH parameter drives it and confirms
b3 (the H1 coefficient) is well-converged. Pickles idata for reuse.
"""
import json, pickle
import arviz as az
import confirmatory_analysis as ca
import confirmatory_analyze as cz

df = cz.score_cells(json.loads(cz.RESPONSES_PATH.read_text(encoding="utf-8")), cz.load_pool())
d = df.copy(); d["steps_z"] = d["long"].astype("float64")
data = ca.prepare(d[["y", "steps_z", "regime", "model", "item"]])
model = ca.build_model(data, model_effects=(data["n_models"] > 1))

# REGISTERED settings (defaults of ca.fit = PREREG_SAMPLE: 2000x4, ta=0.95, seed fixed)
idata = ca.fit(model)
with open(cz.HERE / "_registered_idata.pkl", "wb") as fh:
    pickle.dump(idata, fh)

post = idata.posterior if hasattr(idata, "posterior") else idata["posterior"]
vars_ = [v for v in ["b0", "b1", "b2", "b3", "sigma_a", "sigma_g", "sigma_u"] if v in post]
rows = {}
for v in vars_:
    rh = float(az.rhat(idata, var_names=[v])[v].values)
    ess = float(az.ess(idata, var_names=[v])[v].values)
    rows[v] = {"rhat": rh, "ess_bulk": ess}
    print(f"  {v:8s} rhat={rh:.6f}  ess_bulk={ess:8.0f}" + ("   <-- H1 coeff" if v == "b3" else ""))
mx = max(rows, key=lambda k: rows[k]["rhat"])
print(f"REGISTERED rhat_max = {rows[mx]['rhat']:.6f} (param {mx});  b3 rhat = {rows['b3']['rhat']:.6f}")
(cz.HERE / "_diag_registered_rhat.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("[done] wrote _diag_registered_rhat.json + _registered_idata.pkl")
