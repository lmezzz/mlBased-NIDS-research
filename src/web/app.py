import io
import json
from pathlib import Path

import pandas as pd
from flask import (
    Flask, render_template, request, jsonify, send_from_directory, abort,
    redirect, url_for,
)

from config import RESULTS_DIR
from src.facade import DatasetFacade
from src.facade.dataset_facade import EXPERIMENT_DESCRIPTIONS
from src.strategies import REGISTRY, get_strategy
from src.context import (
    IDSContext, load_results_table, lookup_results,
    attack_breakdown, list_figure_files,
)


MODELS = list(REGISTRY.keys())
EXPERIMENTS = DatasetFacade.list_experiments()
DATASETS = DatasetFacade.list_datasets()


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    # ── ROUTES ──

    @app.route("/")
    def index():
        availability = _availability_matrix()
        try:
            results = load_results_table().to_dict(orient="records")
        except FileNotFoundError:
            results = []
        return render_template(
            "index.html",
            models=MODELS,
            experiments=EXPERIMENTS,
            datasets=DATASETS,
            descriptions=EXPERIMENT_DESCRIPTIONS,
            availability=availability,
            results=results,
        )

    @app.route("/results/<model>/<experiment>")
    def view_results(model, experiment):
        row = lookup_results(model, experiment)
        if row is None:
            abort(404, f"No results for {model}/{experiment}")
        breakdown = attack_breakdown(model, experiment).to_dict(orient="records")
        figures = _figures_for(model, experiment)
        return render_template(
            "results.html",
            model=model,
            experiment=experiment,
            description=EXPERIMENT_DESCRIPTIONS.get(experiment.upper(), ""),
            row=row,
            breakdown=breakdown,
            figures=figures,
        )

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        model = request.values.get("model", "lr")
        experiment = request.values.get("experiment", "EXP0").upper()
        ctx = IDSContext.for_(model, experiment)

        if not ctx.strategy.is_available():
            return render_template(
                "predict.html",
                model=model, experiment=experiment,
                schema=None, error=(
                    f"Model artifact missing: {ctx.strategy.pkl_path.name}. "
                    f"Run the legacy training script first to enable live prediction."
                ),
                models=MODELS, experiments=EXPERIMENTS,
            )

        try:
            schema = ctx.feature_schema()
        except Exception as exc:
            return render_template(
                "predict.html",
                model=model, experiment=experiment, schema=None,
                error=f"Failed to load model: {exc}",
                models=MODELS, experiments=EXPERIMENTS,
            )

        result = None
        eval_metrics = None
        if request.method == "POST" and request.form.get("action") == "predict":
            features = {}
            for col in ctx.strategy.feature_cols:
                raw = request.form.get(col, "0")
                try:
                    features[col] = float(raw)
                except (TypeError, ValueError):
                    features[col] = 0.0
            result = ctx.predict_row(features)
        elif request.method == "POST" and request.form.get("action") == "sample":
            kind = request.form.get("kind", "attack")
            dataset = request.form.get("sample_dataset", "kdd_test")
            try:
                sample = DatasetFacade.sample_row(dataset, experiment, kind)
                return render_template(
                    "predict.html",
                    model=model, experiment=experiment, schema=schema,
                    prefill=sample, error=None, result=None, eval_metrics=None,
                    models=MODELS, experiments=EXPERIMENTS, datasets=DATASETS,
                )
            except Exception as exc:
                return render_template(
                    "predict.html",
                    model=model, experiment=experiment, schema=schema,
                    error=f"Could not sample row: {exc}", result=None, eval_metrics=None,
                    models=MODELS, experiments=EXPERIMENTS, datasets=DATASETS,
                )

        return render_template(
            "predict.html",
            model=model, experiment=experiment, schema=schema,
            result=result, eval_metrics=eval_metrics, error=None,
            models=MODELS, experiments=EXPERIMENTS, datasets=DATASETS,
        )

    @app.route("/predict/csv", methods=["POST"])
    def predict_csv():
        model = request.values.get("model", "lr")
        experiment = request.values.get("experiment", "EXP0").upper()
        if "file" not in request.files:
            return jsonify({"error": "no file uploaded"}), 400
        file = request.files["file"]
        try:
            df = pd.read_csv(io.BytesIO(file.read()))
        except Exception as exc:
            return jsonify({"error": f"could not parse CSV: {exc}"}), 400
        ctx = IDSContext.for_(model, experiment)
        if not ctx.strategy.is_available():
            return jsonify({"error": f"model artifact missing: {ctx.strategy.pkl_path.name}"}), 503
        scored = ctx.predict_dataframe(df)
        head = scored.head(50).to_dict(orient="records")
        return jsonify({
            "model": model, "experiment": experiment,
            "n_rows": int(len(scored)),
            "n_attack": int((scored["prediction"] == 1).sum()),
            "n_benign": int((scored["prediction"] == 0).sum()),
            "rows": head,
        })

    @app.route("/evaluate")
    def evaluate():
        model = request.args.get("model", "lr")
        experiment = request.args.get("experiment", "EXP0").upper()
        dataset = request.args.get("dataset", "kdd_test")
        ctx = IDSContext.for_(model, experiment)
        if not ctx.strategy.is_available():
            return jsonify({"error": f"model artifact missing: {ctx.strategy.pkl_path.name}"}), 503
        try:
            return jsonify(ctx.evaluate(dataset))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/compare")
    def compare():
        try:
            df = load_results_table().sort_values(["experiment", "model"])
        except FileNotFoundError:
            return render_template("compare.html", rows=[], pivot=None)
        pivot = df.pivot_table(
            index="experiment", columns="model",
            values=["within_f1", "cross_f1", "performance_drop"],
        ).round(4)
        return render_template(
            "compare.html",
            rows=df.to_dict(orient="records"),
            pivot_html=pivot.to_html(classes="table table-sm table-striped", border=0),
        )

    @app.route("/figures/<path:filename>")
    def figure(filename):
        return send_from_directory(RESULTS_DIR / "figures", filename)

    @app.route("/health")
    def health():
        return jsonify({
            "ok": True,
            "models_available": _availability_matrix(),
            "results_csv": (RESULTS_DIR / "experiment_results.csv").exists(),
        })

    return app


def _availability_matrix() -> dict:
    out = {}
    for m in MODELS:
        out[m] = {e: get_strategy(m, e).is_available() for e in EXPERIMENTS}
    return out


def _figures_for(model: str, experiment: str) -> list[str]:
    """Heuristic mapping from (model, experiment) -> relevant figure filenames."""
    all_figs = list_figure_files()
    keep = []
    model_map = {"lr": ["LogisticRegression", "lr"], "rf": ["RandomForest", "rf"], "svm": ["SVM", "svm"]}
    keys = model_map.get(model.lower(), [model])
    exp = experiment.lower()
    for f in all_figs:
        lower = f.lower()
        if any(k.lower() in lower for k in keys) or exp in lower:
            keep.append(f)
    # always include the cross-cutting summary charts
    for fname in ("chart1_cross_f1_trend.png", "chart2_within_vs_cross_gap.png",
                  "chart3_performance_drop.png"):
        if fname in all_figs and fname not in keep:
            keep.append(fname)
    return keep


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
