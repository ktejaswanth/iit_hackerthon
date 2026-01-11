import os
from typing import Optional

from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd

# Flask setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "output", "results.csv")
ALGO_OPTIONS = [
    ("hybrid", "Hybrid (Debate)"),
    ("ml", "ML Classifier"),
]


def load_results() -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(RESULTS_PATH):
            df = pd.read_csv(RESULTS_PATH)
            # Normalize expected columns
            if set(["Story ID", "Prediction", "Rationale"]).issubset(df.columns):
                return df
            # Fallback if columns differ
            return df.rename(columns={
                "id": "Story ID",
                "label": "Prediction",
                "rationale": "Rationale"
            })
        return None
    except Exception as e:
        print(f"Failed to load results.csv: {e}")
        return None


def get_summary(df: Optional[pd.DataFrame]):
    if df is None or df.empty:
        return {"total": 0, "consistent": 0, "contradict": 0, "ratio": 0.0}
    consistent = int((df["Prediction"] == 1).sum())
    contradict = int((df["Prediction"] == 0).sum())
    total = int(len(df))
    ratio = (consistent / total * 100.0) if total > 0 else 0.0
    return {"total": total, "consistent": consistent, "contradict": contradict, "ratio": ratio}


@app.route("/")
def index():
    df = load_results()
    summary = get_summary(df)
    current_algo = (os.getenv("ALGO") or "ml").lower()
    
    # Prepare results data for table
    rows = []
    if df is not None and not df.empty:
        # Optional filter
        q = request.args.get("q", "").strip().lower()
        if q:
            df = df[df.apply(lambda row: q in str(row.values).lower(), axis=1)]
        rows = df.to_dict(orient="records")
    
    return render_template(
        "index.html",
        summary=summary,
        has_results=df is not None and not df.empty,
        algo_options=ALGO_OPTIONS,
        current_algo=current_algo,
        rows=rows,
        total_rows=len(rows),
    )


@app.route("/processing")
def processing():
    status_file = os.path.join(os.path.dirname(__file__), "output", "status.txt")
    status = "RUNNING|unknown"
    
    # Check if status file exists and read it
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                status = f.read().strip()
            print(f"[DEBUG] Status read: {status}")  # Debug output
        except Exception as e:
            print(f"[DEBUG] Failed to read status: {e}")
    else:
        print(f"[DEBUG] Status file not found: {status_file}")
    
    parts = status.split("|")
    status_type = parts[0]
    
    if status_type == "COMPLETE":
        algo = parts[1] if len(parts) > 1 else "unknown"
        api_errors = parts[2] if len(parts) > 2 else "0"
        fallback_used = parts[3] if len(parts) > 3 else "0"
        
        # Clean up status file
        try:
            os.remove(status_file)
        except:
            pass
            
        flash(f"Pipeline completed with '{algo}'. API warnings: {api_errors}, Fallback used: {fallback_used} times.", "success")
        return redirect(url_for("index"))
    elif status_type == "ERROR":
        error_msg = parts[1] if len(parts) > 1 else "Unknown error"
        
        # Clean up status file
        try:
            os.remove(status_file)
        except:
            pass
            
        flash(f"Pipeline failed: {error_msg}", "danger")
        return redirect(url_for("index"))
    else:
        algo = parts[1] if len(parts) > 1 else "unknown"
        return render_template("processing.html", algorithm=algo)


@app.route("/results")
def results():
    df = load_results()
    if df is None or df.empty:
        flash("No results found. Run the pipeline first.", "warning")
        return redirect(url_for("index"))

    # Optional simple filter
    q = request.args.get("q", "").strip().lower()
    if q:
        df = df[df.apply(lambda row: q in str(row.values).lower(), axis=1)]

    # Convert to dict for template rendering
    rows = df.to_dict(orient="records")
    return render_template("results.html", rows=rows, total=len(rows))


@app.route("/train", methods=["POST"]) 
def train_model():
    """Upload a training CSV and retrain the local ML classifier."""
    print("[DEBUG] train_model called")
    import sys
    from io import StringIO
    import threading

    train_file = request.files.get("train_file")
    upload_dir = os.path.join(os.path.dirname(__file__), "data")
    train_path = os.path.join(upload_dir, "train.csv")

    # Save uploaded training file or ensure existing one
    if train_file and train_file.filename:
        try:
            os.makedirs(upload_dir, exist_ok=True)
            train_file.save(train_path)
            print(f"[DEBUG] Uploaded train '{train_file.filename}' saved to {train_path}")
            flash(f"Uploaded training CSV '{train_file.filename}' — replaced train.csv", "info")
        except Exception as e:
            print(f"[DEBUG] Training file upload failed: {e}")
            flash(f"Training upload failed: {e}", "danger")
            return redirect(url_for("index"))
    else:
        if not os.path.exists(train_path):
            flash("No train.csv found. Please upload a training CSV.", "warning")
            return redirect(url_for("index"))

    def run_training():
        from src.ml_model import train_text_classifier
        output_capture = StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = output_capture
            sys.stderr = output_capture
            print(f"[TRAIN] Starting training with {train_path}")
            acc, model_path = train_text_classifier(train_path)
            print(f"[TRAIN] Completed. acc={acc:.3f} model={model_path}")
        except Exception as e:
            print(f"[TRAIN] Error: {e}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            logs = output_capture.getvalue()
            # Show a concise summary via flash
            try:
                acc_line = next((l for l in logs.splitlines() if "Validation accuracy:" in l), None)
            except Exception:
                acc_line = None
            summary = acc_line or "Training complete."
            flash(summary, "success")

    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()
    return redirect(url_for("index"))


@app.route("/evaluate")
def evaluate_page():
    """Run cross-validation evaluation and render results page."""
    try:
        from src.ml_model import crossval_evaluate
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        out_dir = os.path.join(os.path.dirname(__file__), "output")
        report = crossval_evaluate(os.path.join(data_dir, "train.csv"), cv=5, save_dir=out_dir)
        return render_template("evaluate.html", report=report)
    except Exception as e:
        flash(f"Evaluation error: {e}", "danger")
        return redirect(url_for("index"))


@app.route("/run", methods=["POST"]) 
def run_pipeline():
    print("[DEBUG] run_pipeline called")
    import sys
    from io import StringIO
    import threading
    
    algo = request.form.get("algorithm", "ml").lower()
    csv_file = request.files.get("csv_file")
    
    print(f"[DEBUG] Algorithm selected: {algo}")
    print(f"[DEBUG] CSV file received: {csv_file.filename if csv_file and csv_file.filename else 'None'}")
    
    os.environ["ALGO"] = algo
    
    # Always use data/test.csv as the target path
    upload_dir = os.path.join(os.path.dirname(__file__), "data")
    csv_path = os.path.join(upload_dir, "test.csv")
    
    # Handle file upload - save directly as data/test.csv to replace original
    if csv_file and csv_file.filename:
        try:
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            
            # Save uploaded file directly as test.csv (replaces original)
            csv_file.save(csv_path)
            print(f"[DEBUG] Uploaded '{csv_file.filename}' saved to {csv_path}")
            flash(f"Uploaded '{csv_file.filename}' — replaced test.csv", "info")
        except Exception as e:
            print(f"[DEBUG] File upload failed: {e}")
            flash(f"File upload failed: {e}", "danger")
            return redirect(url_for("index"))
    else:
        print(f"[DEBUG] No file uploaded, using existing {csv_path}")
        # Verify the file exists
        if not os.path.exists(csv_path):
            flash("No test.csv found. Please upload a CSV file.", "warning")
            return redirect(url_for("index"))

    # Run pipeline in background thread
    def run_in_background():
        output_capture = StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            sys.stdout = output_capture
            sys.stderr = output_capture
            
            print(f"[PIPELINE] Starting with algorithm={algo}, csv_path={csv_path}")
            from main import main as pipeline_main
            pipeline_main(csv_path=csv_path)
            print(f"[PIPELINE] Completed successfully")
            
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            output = output_capture.getvalue()
            api_errors = output.count("429 RESOURCE_EXHAUSTED")
            fallback_used = output.count("Fallback") + output.count("fallback")
            
            # Store status in a file for the status page to read
            status_file = os.path.join(os.path.dirname(__file__), "output", "status.txt")
            with open(status_file, "w") as f:
                status_text = f"COMPLETE|{algo}|{api_errors}|{fallback_used}"
                f.write(status_text)
                f.flush()
            print(f"[DEBUG] Pipeline complete, wrote status: COMPLETE|{algo}|{api_errors}|{fallback_used}")
                
        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            status_file = os.path.join(os.path.dirname(__file__), "output", "status.txt")
            with open(status_file, "w") as f:
                f.write(f"ERROR|{str(e)}")
    
    # Mark as running
    status_file = os.path.join(os.path.dirname(__file__), "output", "status.txt")
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    with open(status_file, "w") as f:
        f.write(f"RUNNING|{algo}")
        f.flush()
    print(f"[DEBUG] Started pipeline, wrote status: RUNNING|{algo}")
    
    # Start background thread
    thread = threading.Thread(target=run_in_background)
    thread.daemon = True
    thread.start()
    
    return redirect(url_for("processing"))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
