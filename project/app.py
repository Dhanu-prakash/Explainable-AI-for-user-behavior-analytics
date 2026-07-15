from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from clustering import add_cluster_labels, add_engagement_score, cluster_users
from explainability import compute_feature_importance
from preprocessing import preprocess_dataframe

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

STATE = {
    "uploaded_file": None,
    "results": None,
    "importance": None,
    "summary": None,
}


def error_response(message: str, status_code: int = 400):
    return jsonify({"success": False, "error": message}), status_code


def is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _build_results_payload(df: pd.DataFrame) -> dict:
    cluster_counts = (
        df["cluster_label"]
        .value_counts()
        .sort_values(ascending=False)
        .rename_axis("cluster")
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    purchases_col = "Items Purchased" if "Items Purchased" in df.columns else None
    scatter_points = []
    if purchases_col:
        for _, row in df.iterrows():
            scatter_points.append(
                {
                    "x": float(row["engagement_score"]),
                    "y": float(row[purchases_col]),
                    "cluster": str(row["cluster_label"]),
                }
            )

    table_rows = df.head(200).to_dict(orient="records")
    for row in table_rows:
        for key, value in list(row.items()):
            if pd.isna(value):
                row[key] = None
            elif isinstance(value, (int, float)):
                row[key] = float(value)

    return {
        "cluster_distribution": cluster_counts,
        "scatter_points": scatter_points,
        "table_columns": list(df.columns),
        "table_rows": table_rows,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/upload")
def upload_data():
    if "file" not in request.files:
        return error_response("No file field found in request.")

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return error_response("Please choose a CSV file before uploading.")
    if not is_allowed_file(uploaded_file.filename):
        return error_response("Invalid file type. Please upload a .csv file.")

    filename = secure_filename(uploaded_file.filename)
    save_path = UPLOAD_DIR / filename
    uploaded_file.save(save_path)

    STATE["uploaded_file"] = str(save_path)
    STATE["results"] = None
    STATE["importance"] = None
    STATE["summary"] = None

    return jsonify({"success": True, "message": "File uploaded successfully.", "filename": filename})


@app.post("/process")
def process_data():
    if not STATE["uploaded_file"]:
        return error_response("No uploaded CSV found. Please upload data first.", 404)

    csv_path = Path(STATE["uploaded_file"])
    if not csv_path.exists():
        return error_response("Uploaded CSV file is missing. Please upload again.", 404)

    try:
        raw_df = pd.read_csv(csv_path)
    except Exception as exc:
        return error_response(f"Could not read CSV file: {exc}")

    if raw_df.empty:
        return error_response("Uploaded CSV is empty.")

    try:
        processed_df = preprocess_dataframe(raw_df)
        clustered_df, silhouette = cluster_users(add_engagement_score(processed_df), k=3)
        clustered_df = add_cluster_labels(clustered_df)
        feature_importance = compute_feature_importance(clustered_df)
    except Exception as exc:
        return error_response(f"Processing failed: {exc}")

    STATE["results"] = _build_results_payload(clustered_df)
    STATE["importance"] = {"features": feature_importance}
    STATE["summary"] = {
        "rows": int(len(clustered_df)),
        "columns": int(clustered_df.shape[1]),
        "silhouette_score": round(float(silhouette), 4),
    }

    return jsonify({"success": True, "message": "Analysis complete.", "summary": STATE["summary"]})


@app.get("/results")
def get_results():
    if not STATE["results"]:
        return error_response("No processed results found. Run analysis first.", 404)
    return jsonify({"success": True, **STATE["results"]})


@app.get("/importance")
def get_importance():
    if not STATE["importance"]:
        return error_response("No feature importance found. Run analysis first.", 404)
    return jsonify({"success": True, **STATE["importance"]})


if __name__ == "__main__":
    app.run(debug=True)
