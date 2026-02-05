import argparse
import os
import sys
from datasets import load_dataset

PALOMA_REPO_ID = 'allenai/paloma'
PALOMA_SOURCES = ['4chan_meta_sep', 'c4_100_domains', 'c4_en', 'dolma_100_programing_languages', 'dolma_100_subreddits', 'dolma-v1_5', 'falcon-refinedweb', 'gab', 'm2d2_s2orc_unsplit', 'm2d2_wikipedia_unsplit', 'manosphere_meta_sep', 'mc4', 'ptb', 'redpajama', 'twitterAAE_HELM_fixed', 'wikitext_103']


def get_args():
    parser = argparse.ArgumentParser(description='Download Paloma datasets')
    parser.add_argument('target_dir', nargs='?', default=os.path.join(os.path.dirname(__file__), '../../data/olmo/paloma'), help='Directory to store datasets')
    parser.add_argument('--sources', nargs='+', default=None, help=f"Specific Paloma sources to download (default: all). Available: {', '.join(PALOMA_SOURCES)}")
    args = parser.parse_args()
    return args


def download_paloma_datasets(datasets_dir, sources=None):
    print('Downloading Paloma datasets...')
    print('=' * 60)

    if sources is None:
        sources = PALOMA_SOURCES
    else:
        invalid = [s for s in sources if s not in PALOMA_SOURCES]
        if invalid:
            print(f"⚠️  Invalid sources: {', '.join(invalid)}")
            print(f"Available sources: {', '.join(PALOMA_SOURCES)}")
            sys.exit(1)

    print(f'Sources to download: {len(sources)}')
    for source in sources:
        print(f'  - {source}')

    existing_datasets = set()
    if os.path.exists(datasets_dir):
        existing_datasets = {d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d)) and d.startswith('paloma_')}

    os.makedirs(datasets_dir, exist_ok=True)
    print(f'\nExisting datasets: {len(existing_datasets)}')

    failed_sources = []
    for source in sources:
        dataset_name = f'paloma_{source}'
        dataset_path = os.path.join(datasets_dir, dataset_name)

        if dataset_name in existing_datasets:
            print(f'\n✓ {dataset_name} (already exists)')
            continue

        print(f'\n📥 Downloading {dataset_name}...')
        try:
            dataset = load_dataset(PALOMA_REPO_ID, source, trust_remote_code=True)
            dataset.save_to_disk(dataset_path)
            print(f'  ✓ Saved to {dataset_path}')
        except Exception as e:
            print(f'  ✗ ERROR: {str(e)}')
            failed_sources.append(source)

    print('\n' + '=' * 60)
    print(f'✅ Downloaded {len(sources) - len(failed_sources)}/{len(sources)} datasets')
    if failed_sources:
        print(f"\n⚠️  Failed sources: {', '.join(failed_sources)}")
        return False
    return True


def main():
    args = get_args()
    target_dir = os.path.abspath(args.target_dir)
    os.makedirs(target_dir, exist_ok=True)

    print(f'Datasets directory: {target_dir}\n')
    sources = args.sources if args.sources else None
    success = download_paloma_datasets(target_dir, sources)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
