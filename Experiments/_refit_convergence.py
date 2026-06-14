"""Convergence re-fit of the registered PRIMARY eq.6.1' ITT.

The one-shot registered run produced verdict=CONFIRMED but R̂_max=1.010 (>= the
pre-registered 1.01 gate) -> flagged converged=False. This re-fits the IDENTICAL
model + IDENTICAL locked decision rule with MORE samples and higher target_accept
to satisfy the pre-registered convergence criterion. Nothing about the model,
data, regressor, priors, or gate changes -- only draws / adaptation. Completing
the registered analysis to its own standard, not a re-specification.

ROBUSTNESS (after a prior run lost a 54-min sample to a hardcoded-column print
bug): persist idata to netcdf IMMEDIATELY after sampling and write the JSON
summary BEFORE any pretty-printing, so a downstream bug can never cost the fit.
"""
import json
import sys
import arviz as az
import confirmatory_analysis as ca
import confirmatory_analyze as cz

SMOKE = "--smoke" in sys.argv[1:]

df = cz.score_cells(json.loads(cz.RESPONSES_PATH.read_text(encoding="utf-8")), cz.load_pool())

sample_kw = (dict(draws=300, tune=300, chains=2, target_accept=0.9) if SMOKE
             else dict(draws=4000, tune=4000, chains=4, target_accept=0.99))
print(f"[refit] primary ITT, sampler={sample_kw}", flush=True)
amended, registered, idata, data = cz.fit_primary_itt(df, sample_kw)

# 1) PERSIST FIRST -- never lose the sample to a later bug.
# netcdf backends (netCDF4/h5netcdf) are NOT installed in this env, so use pickle
# (always available). Protects the ~54-min sample against any downstream bug.
import pickle
pk = cz.HERE / "_refit_idata.pkl"
try:
    with open(pk, "wb") as fh:
        pickle.dump(idata, fh)
    print(f"[persist] idata -> {pk.name}", flush=True)
except Exception as e:  # persistence must never abort the run
    print(f"[persist] WARNING could not pickle idata: {type(e).__name__}: {e}", flush=True)

# 2) per-parameter diagnostics, accessing only columns that exist.
post = idata.posterior if hasattr(idata, "posterior") else idata["posterior"]
summ = az.summary(idata, var_names=[v for v in ["b0", "b1", "b2", "b3",
                                                "sigma_a", "sigma_g", "sigma_u"]
                                    if v in post])
print("[debug] summary columns:", list(summ.columns), flush=True)
cols = [c for c in ["mean", "sd", "ess_bulk", "ess_tail", "r_hat"] if c in summ.columns]
b3 = summ.loc["b3"]

out = {
    "sampler": sample_kw,
    "verdict": amended["verdict"],
    "p_b3_neg": amended["p_dir"],
    "b3_median": amended["b3_median"],
    "b3_rhat": float(b3["r_hat"]),
    "b3_ess_bulk": float(b3["ess_bulk"]),
    "robust_pp_drop_median": amended["pp_drop_robust_median"],
    "robust_pp_drop_ci": list(amended["pp_drop_robust_ci"]),
    "rhat_max": registered.rhat_max,
    "ess_min": registered.ess_min,
    "converged": registered.converged,
    "per_param_rhat": {k: float(v) for k, v in summ["r_hat"].items()},
    "per_param_ess_bulk": {k: float(v) for k, v in summ["ess_bulk"].items()},
}
# 3) write JSON BEFORE printing.
cz.HERE.joinpath("_refit_convergence_result.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("[per-parameter diagnostics]", flush=True)
print(summ[cols].to_string(), flush=True)
print("[summary]", json.dumps(out, indent=2), flush=True)
print("[done] wrote _refit_convergence_result.json + _refit_idata.pkl", flush=True)
