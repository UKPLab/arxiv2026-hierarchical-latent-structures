import pandas as pd
from pathlib import Path

def load_wandb_training_data(wandb_dir="../data/olmo/wandb"):
    wandb_dir = Path(wandb_dir)
    train_data = []
    
    run_dirs = sorted([d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("OLMo")])
    
    print(f"Loading W&B data from {len(run_dirs)} runs...")
    
    for run_dir in run_dirs:
        history_file = run_dir / "history.csv"
        if not history_file.exists():
            continue
        
        df = pd.read_csv(history_file)
        
        if 'train/CrossEntropyLoss' in df.columns and '_step' in df.columns:
            train_subset = df[['_step', 'train/CrossEntropyLoss']].dropna()
            train_subset = train_subset.rename(columns={
                '_step': 'step',
                'train/CrossEntropyLoss': 'loss',
            })
            train_data.append(train_subset)
    
    if train_data:
        train_df = pd.concat(train_data, ignore_index=True).sort_values('step').drop_duplicates(subset=['step'])
    else:
        train_df = pd.DataFrame()
    
    return train_df

def load_and_print_olmo_data(wandb_dir="../data/olmo/wandb"):
    olmo_df = load_wandb_training_data(wandb_dir)

    print(f"\nallenai/OLMo-1B-hf training data:")
    print(f"  Steps: {olmo_df['step'].min()} - {olmo_df['step'].max()}")
    print(f"  Data points: {len(olmo_df)}")
    print(f"  Loss range: {olmo_df['loss'].min():.4f} - {olmo_df['loss'].max():.4f}")
    print(f"\nFirst few rows:")
    print(olmo_df.head())

    return olmo_df


def load_and_print_ngram_data(ngram_path="../experiments/ngram_large/56a32f27-0a6e-457a-a837-0774f2b99bbb/training_loss.csv"):
    ngram_path = Path(ngram_path)
    ngram_df = pd.read_csv(ngram_path)

    print(f"N-gram language experiment data:")
    print(f"  Steps: {ngram_df['step'].min()} - {ngram_df['step'].max()}")
    print(f"  Data points: {len(ngram_df)}")
    print(f"  Loss range: {ngram_df['loss'].min():.4f} - {ngram_df['loss'].max():.4f}")
    print(f"\nFirst few rows:")
    print(ngram_df.head())

    return ngram_df


def load_and_print_pcfg_data(pcfg_path="../experiments/pcfg_large/9ab00aa2-6728-4d27-a894-c5bb1aad6f5d/training_loss.csv"):
    pcfg_path = Path(pcfg_path)
    pcfg_df = pd.read_csv(pcfg_path)

    print(f"PCFG language experiment data:")
    print(f"  Steps: {pcfg_df['step'].min()} - {pcfg_df['step'].max()}")
    print(f"  Data points: {len(pcfg_df)}")
    print(f"  Loss range: {pcfg_df['loss'].min():.4f} - {pcfg_df['loss'].max():.4f}")
    print(f"\nFirst few rows:")
    print(pcfg_df.head())

    return pcfg_df