import os
import json
import pandas as pd
import wandb
from wandb.apis.public import Api

def make_serializable(obj):
    required_files = ["config.json", "summary.json"]
    for fname in required_files:
        if not os.path.exists(os.path.join(out_dir, fname)):
            return False
    return True

def export_run(run, out_dir, force=False):
    if os.path.exists(out_dir) and is_run_healthy(out_dir):
        if not force:
            print(f"  ✓ Skipping {run.name} (ID: {run.id}) - already downloaded")
            return

    os.makedirs(out_dir, exist_ok=True)
    print(f"  → Exporting run {run.name} (ID: {run.id})")

    config = run.config
    try:
        config_dict = {k: make_serializable(v) for k, v in config.items()}
    except Exception:
        config_dict = {"config": make_serializable(config)}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    try:
        summary_dict = run.summary._json_dict
    except Exception:
        try:
            summary_dict = {k: make_serializable(v) for k, v in dict(run.summary).items()}
        except Exception:
            summary_dict = {"summary": make_serializable(run.summary)}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary_dict, f, indent=2)

    try:
        hist_iter = run.scan_history(keys=None, page_size=1000)
        hist_rows = list(hist_iter)
        if hist_rows:
            df = pd.DataFrame(hist_rows)
            df.to_csv(os.path.join(out_dir, "history.csv"), index=False)
        else:
            print(f"  → No history rows found for run {run.id}")
    except Exception as e:
        print(f"  → Warning: failed to scan history for run {run.id}: {e}")

    try:
        for fobj in run.files():
            print(f"    • Downloading file {fobj.name}")
            fobj.download(root=out_dir, replace=True)
    except Exception as e:
        print(f"  → Warning: failed to download files for run {run.id}: {e}")

    try:
        for artifact in run.logged_artifacts(per_page=50):
            art_dir = os.path.join(out_dir, "artifact_" + artifact.name.replace("/", "_"))
            print(f"    • Downloading artifact {artifact.name} → {art_dir}")
            artifact_dir = artifact.download(root=art_dir)
            with open(os.path.join(artifact_dir, "artifact_info.json"), "w") as f:
                json.dump({
                    "artifact_name": artifact.name,
                    "artifact_type": artifact.type,
                    "artifact_version": artifact.version,
                    "artifact_path": artifact_dir
                }, f, indent=2)
    except Exception as e:
        print(f"  → Warning: failed to download artifacts for run {run.id}: {e}")

def download_all_runs(entity="ai2-llm", project="OLMo-1B", api_key=None, filters=None, dest_root="olmo-1b_downloads", force=False):
    if api_key is None:
        api_key = os.environ.get("WANDB_API_KEY")

    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()

    api = Api()
    runs = api.runs(f"{entity}/{project}", filters=filters) if filters else api.runs(f"{entity}/{project}")
    print(f"Found {len(runs)} runs for {entity}/{project}")

    downloaded_count = 0
    skipped_count = 0

    for run in runs:
        run_id = run.id
        run_name = run.name or run_id
        out_dir = os.path.join(dest_root, f"{run_name}_{run_id}")

        if os.path.exists(out_dir) and is_run_healthy(out_dir) and not force:
            skipped_count += 1
            print(f"  ✓ Skipping {run_name} (ID: {run_id}) - already downloaded")
        else:
            downloaded_count += 1
            export_run(run, out_dir, force=force)

    print(f"\n{'='*70}")
    print(f"✅ Done! Downloaded: {downloaded_count} | Skipped: {skipped_count} | Total: {len(runs)}")
    print(f"{'='*70}")

if __name__ == "__main__":
    ENTITY = "ai2-llm"
    PROJECT = "OLMo-1B"
    API_KEY = None
    FILTERS = None
    DEST_ROOT = os.path.join(os.path.dirname(__file__), '../../data/olmo/wandb')
    FORCE = os.environ.get("FORCE_DOWNLOAD", "").lower() in ("true", "1", "yes")

    if FORCE:
        print("⚠️  FORCE_DOWNLOAD enabled - re-downloading all runs (including healthy ones)")

    download_all_runs(ENTITY, PROJECT, API_KEY, FILTERS, DEST_ROOT, force=FORCE)



import pandas as pd
import os
from pathlib import Path

history_file = Path("/home/jonas/Code/master-thesis/scripts/wandb_olmo-1b_raw/OLMo-1B-run-001_xz8bu54f/history.csv")
df = pd.read_csv(history_file)

print(f"Shape: {df.shape}")
print(f"\nColumns containing 'train' and 'loss' (case insensitive):")
train_loss_cols = [col for col in df.columns if 'train' in col.lower() and 'loss' in col.lower()]
for col in train_loss_cols:
    print(f"  - {col}")

step_cols = [col for col in df.columns if 'step' in col.lower()]
print(f"\nStep columns:")
for col in step_cols:
    print(f"  - {col}")

if '_step' in df.columns:
    print(f"\nStep range: {df['_step'].min()} to {df['_step'].max()}")
    print(f"Number of rows: {len(df)}")
    
    if 'train/CrossEntropyLoss' in df.columns:
        train_loss = df['train/CrossEntropyLoss'].dropna()
        print(f"\nTraining Loss:")
        print(f"  - Non-null values: {len(train_loss)}")
        print(f"  - Range: {train_loss.min():.4f} to {train_loss.max():.4f}")
        print(f"  - First few values: {train_loss.head(10).tolist()}")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.append('/home/jonas/Code/master-thesis/scripts')
from thesis_figures import THESIS_WIDTH_IN, THESIS_HEIGHT_IN, style_ax, palette

wandb_dir = Path("/home/jonas/Code/master-thesis/scripts/wandb_olmo-1b_raw")
all_runs = []

for run_dir in sorted(wandb_dir.iterdir()):
    if run_dir.is_dir():
        history_file = run_dir / "history.csv"
        if history_file.exists():
            df = pd.read_csv(history_file)
            if 'train/CrossEntropyLoss' in df.columns and '_step' in df.columns:
                run_data = df[['_step', 'train/CrossEntropyLoss']].copy()
                run_data = run_data.dropna()
                run_data['run_name'] = run_dir.name
                all_runs.append(run_data)
                print(f"✓ Loaded {run_dir.name}: {len(run_data)} steps")

print(f"\n{'='*70}")
print(f"SUMMARY: Loaded {len(all_runs)} runs")
print(f"{'='*70}")



















