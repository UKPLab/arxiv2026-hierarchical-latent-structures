import argparse
import os
from huggingface_hub import HfApi, hf_hub_download, list_repo_refs, snapshot_download

REPO_ID = 'allenai/OLMo-1B-hf'
REQUIRED_FILES = ['model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json']
def get_args():
    parser = argparse.ArgumentParser(description='Download OLMo-1B checkpoints')
    parser.add_argument('target_dir', nargs='?', default=os.path.join(os.path.dirname(__file__), '../../data/olmo/checkpoints'), help='Directory to store models')
    parser.add_argument('--max-checkpoints', type=int, default=None, help='Maximum number of checkpoints to download (evenly spaced)')
    args = parser.parse_args()
    return args
def setup_cache(models_dir):
    cache_dir = os.path.join(models_dir, 'hf_cache')
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir


def format_size(bytes_size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f'{bytes_size:.2f} {unit}'
        bytes_size /= 1024.0
    return f'{bytes_size:.2f} PB'


def calculate_download_size(missing_checkpoints):
    api = HfApi()
    total_size = 0
    sample_count = 0
    sample_limit = min(5, len(missing_checkpoints))
    print(f'\nCalculating download size (sampling {sample_limit} checkpoints)...')
    for branch in missing_checkpoints[:sample_limit]:
        try:
            info = api.repo_info(repo_id=REPO_ID, revision=branch, repo_type='model', files_metadata=True)
            branch_size = sum((f.size for f in info.siblings if f.size is not None))
            if branch_size > 0:
                total_size += branch_size
                sample_count += 1
                print(f'  {branch}: {format_size(branch_size)}')
        except Exception as e:
            print(f'  Failed to sample {branch}: {e}')
    if sample_count == 0:
        print('  Unable to calculate size, using estimate: ~4.4 GB per checkpoint')
        return int(4.4 * 1024 ** 3 * len(missing_checkpoints))
    avg_size = total_size / sample_count
    estimated_total = avg_size * len(missing_checkpoints)
    return int(estimated_total)


def select_evenly_spaced_checkpoints(checkpoints, max_checkpoints):
    if max_checkpoints is None or len(checkpoints) <= max_checkpoints:
        return checkpoints

    def extract_step(checkpoint_name):
        try:
            return int(checkpoint_name.split('-')[0].replace('step', ''))
        except:
            return 0

    sorted_checkpoints = sorted(checkpoints, key=extract_step)
    n = len(sorted_checkpoints)
    indices = [int(i * (n - 1) / (max_checkpoints - 1)) for i in range(max_checkpoints)]
    selected = [sorted_checkpoints[i] for i in indices]
    return selected


def download_checkpoints(models_dir, max_checkpoints=None):
    print('Checking OLMo-1B checkpoints...')
    print('=' * 60)
    refs = list_repo_refs(REPO_ID)
    hf_branches = [b.name for b in refs.branches if b.name != 'main']
    print(f'Found {len(hf_branches)} branches on HuggingFace')

    if max_checkpoints is not None:
        hf_branches = select_evenly_spaced_checkpoints(hf_branches, max_checkpoints)
        print(f'Selected {len(hf_branches)} evenly-spaced checkpoints (max: {max_checkpoints})')

    local_checkpoints = {d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('step') and ('-tokens' in d)}
    missing_checkpoints = [b for b in hf_branches if b not in local_checkpoints]

    if missing_checkpoints:
        total_size = calculate_download_size(missing_checkpoints)
        print(f'\n⚠️  Download required:')
        print(f'  Checkpoints: {len(missing_checkpoints)}')
        print(f'  Estimated size: {format_size(total_size)}')
        print(f'  Destination: {models_dir}')

        response = input('\nProceed with download? [y/N]: ').strip().lower()
        if response not in ['y', 'yes']:
            print('Download cancelled.')
            return local_checkpoints

        print(f'\nDownloading {len(missing_checkpoints)} missing checkpoints...')
        for branch in missing_checkpoints:
            print(f'  Downloading {branch}...')
            try:
                snapshot_download(repo_id=REPO_ID, revision=branch, local_dir=os.path.join(models_dir, branch), local_dir_use_symlinks=False, cache_dir=None)
                print(f'    ✓ Downloaded')
            except Exception as e:
                print(f'    ✗ ERROR: {str(e)}')
    else:
        print('✓ All checkpoints present')

    return local_checkpoints.union(missing_checkpoints)


def verify_files(models_dir):
    print('\n' + '=' * 60)
    print('Verifying checkpoint files...')
    all_checkpoints = sorted([d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('step') and ('-tokens' in d)])

    for checkpoint_dir in all_checkpoints:
        local_path = os.path.join(models_dir, checkpoint_dir)
        missing_files = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(local_path, f))]
        if missing_files:
            print(f'\n{checkpoint_dir}:')
            print(f'  Downloading {len(missing_files)} missing files...')
            for req_file in missing_files:
                try:
                    hf_hub_download(repo_id=REPO_ID, filename=req_file, revision=checkpoint_dir, local_dir=local_path, local_dir_use_symlinks=False, cache_dir=None)
                    print(f'    ✓ {req_file}')
                except Exception as e:
                    print(f'    ✗ {req_file}: {str(e)}')
        else:
            print(f'✓ {checkpoint_dir}')

    print('\n' + '=' * 60)
    print(f'Complete! {len(all_checkpoints)} checkpoints ready')


def main():
    args = get_args()
    target_dir = os.path.abspath(args.target_dir)
    os.makedirs(target_dir, exist_ok=True)

    print(f'Models directory: {target_dir}\n')
    if args.max_checkpoints is not None:
        print(f'Maximum checkpoints: {args.max_checkpoints}\n')

    setup_cache(target_dir)
    download_checkpoints(target_dir, args.max_checkpoints)
    verify_files(target_dir)


if __name__ == '__main__':
    main()