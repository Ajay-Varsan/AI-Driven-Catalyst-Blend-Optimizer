from flask import Flask, render_template, request, jsonify
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import traceback

app = Flask(__name__, template_folder="templates", static_folder="static")

# ─────────────────────────────────────────────
# 1.  Load dataset and verify columns
# ─────────────────────────────────────────────
CSV_PATH = "blend_dataset(15k).csv"
COST_COL = "Cost"                        # make sure this exists exactly
DF = pd.read_csv(CSV_PATH)
DF.columns = DF.columns.str.strip()

if "Component" not in DF.columns:
    DF.insert(0, "Component", [f"CMPD-{i+1000}" for i in range(len(DF))])

print("📄 CSV columns:", DF.columns.tolist())   # helps diagnose typos

# Front‑end key  →  exact CSV column name
CSV_COL = {
    "attrition_resistance":   "Attrition Resistance",
    "thermal_stability":      "Thermal Stability",
    "avg_particle_size":      "Average Particle Size",          # ← match exactly
    "particle_size_dist":     "Particle Size Distribution (%)",
    "density":                "Density",
    "rare_earth_oxides":      "Rare Earth Oxides (%)",
    "catalyst_surface_area":  "Catalyst Surface Area",
    "micropore_surface_area": "Micropore Surface Area",
    "zeolite_surface_area":   "Zeolite Surface Area",
    "xrf":                    "X-Ray Fluorescence (a.u.)"
}

# ─────────────────────────────────────────────
# 2.  Optimisation helper
# ─────────────────────────────────────────────
def generate_blends(df_work, target, cost_weight,
                    num_candidates=10, num_final=3,
                    min_diff=0.05):
    props = list(target.keys())
    target_vec = np.array([target[p] for p in props])

    data_matrix = df_work[props].to_numpy().T        # shape (P, n)
    cost_arr    = df_work[COST_COL].to_numpy()
    n = len(df_work)

    bounds = [(0, 1)] * n
    constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def loss(w):
        perf = np.sum((data_matrix @ w - target_vec) ** 2)
        cost = np.dot(w, cost_arr)
        return (1 - cost_weight) * perf + cost_weight * cost

    sols, perf_errs, costs, weights = [], [], [], []

    for _ in range(80):
        if len(sols) >= num_candidates:
            break
        x0  = np.random.dirichlet(np.ones(n))
        res = minimize(loss, x0, method="SLSQP",
                       bounds=bounds, constraints=constraint,
                       options={"maxiter": 300})
        if res.success and all(np.linalg.norm(res.x - w) > min_diff for w in weights):
            w = res.x
            sols.append(w)
            weights.append(w)
            perf_errs.append(np.sum((data_matrix @ w - target_vec) ** 2))
            costs.append(np.dot(w, cost_arr))

    if not sols:
        return []

    norm   = lambda x: (x - np.min(x)) / (np.ptp(x) + 1e-12)
    scores = norm(np.array(perf_errs)) + norm(np.array(costs))
    best   = np.argsort(scores)[:num_final]
    return [sols[i] for i in best]

# ─────────────────────────────────────────────
# 3.  Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/blend", methods=["POST"])
def blend():
    try:
        data = request.get_json(force=True)
        print("🔥 Received:", data)

        # Build target dict with exact CSV names
        target = {CSV_COL[k]: float(data.get(k, 0)) for k in CSV_COL}
        cost_weight = float(data.get("cost_weight", 0.3))

        # Sample a manageable subset so weight length matches rows
        df_work = DF.sample(n=min(150, len(DF)), random_state=42).reset_index(drop=True)

        weights = generate_blends(df_work, target, cost_weight)

        if not weights:
            return jsonify(detail="No valid blends found. Relax constraints."), 400

        blends = []
        for rank, w in enumerate(weights, 1):
            props = {k: float((w * df_work[CSV_COL[k]]).sum()) for k in CSV_COL}
            recipe = (
                df_work.assign(Fraction=w)[["Component", "Fraction", COST_COL]]
                .query("Fraction > 0.01")
                .sort_values("Fraction", ascending=False)
                .to_dict(orient="records")
            )
            blends.append({
                "rank": rank,
                "performance_error": float(np.sum(
                    (df_work[[CSV_COL[k] for k in CSV_COL]].to_numpy().T @ w -
                     np.array([target[CSV_COL[k]] for k in CSV_COL])) ** 2)),
                "cost": float(np.dot(w, df_work[COST_COL])),
                "properties": props,
                "recipe": recipe
            })

        return jsonify(blends=blends)

    except Exception as e:
        traceback.print_exc()
        return jsonify(detail=f"Server error: {str(e)}"), 500

# ─────────────────────────────────────────────
# 4.  Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)     # http://localhost:5000
