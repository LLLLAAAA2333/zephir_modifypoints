"""
Generate video only from annotations, worldlines, and data.
This script bypasses all ZephIR tracking/optimization and directly generates annotated.avi (ImageJ compatible)

Usage:
    python generate_zephir_video.py <dataset_path>
    
Example:
    python generate_zephir_video.py ./data/20250730/w3/vol_0_99
"""

import sys
import numpy as np
from pathlib import Path
import argparse

# Add zephir to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'zephir'))

from zephir.models.container import Container
from zephir.methods.save_movie import save_movie
from zephir.utils.io import get_annotation_df, get_annotation


def load_annotations_to_results(container, annotation):
    """
    Convert annotations dataframe to results array format.
    
    :param container: Container with dataset info
    :param annotation: Pandas DataFrame with annotations
    :return: results array (shape_t, shape_n, 3) with xyz coordinates
    """
    shape_t = container.get('shape_t')
    exclude_self = container.get('exclude_self')
    exclusive_prov = container.get('exclusive_prov')
    
    # Get all annotated frames
    t_annot = np.unique(annotation['t_idx']).astype(int)
    
    # Find the frame with most annotations to determine shape_n
    nn_list = []
    worldline_ids_list = []
    
    for t in t_annot:
        u, _, _ = get_annotation(annotation, t, exclusive_prov, exclude_self)
        nn_list.append(len(u))
        worldline_ids_list.append(u)
    
    # Use frame with maximum annotations as reference
    t_max = t_annot[np.argmax(nn_list)]
    worldline_id, _, _ = get_annotation(annotation, t_max, exclusive_prov, exclude_self)
    shape_n = len(worldline_id)
    
    print(f'Found {shape_n} worldlines in frame #{t_max}')
    print(f'Total annotated frames: {len(t_annot)}')
    
    # Store worldline_id in container
    container.props['worldline_id'] = worldline_id
    container.props['t_annot'] = t_annot
    
    # Initialize results array
    results = np.zeros((shape_t, shape_n, 3))
    
    # Fill results with annotations
    for t in t_annot:
        u, annot, prov = get_annotation(annotation, t, exclusive_prov, exclude_self)
        
        # Map annotations to correct worldline indices
        for i, wid in enumerate(worldline_id):
            if wid in u:
                u_idx = np.where(u == wid)[0][0]
                results[t, i, :] = annot[u_idx, :]
    
    return results


def generate_video_only(dataset_path, 
                        channel=None, 
                        gamma=0.5, 
                        include_all=True,
                        exclude_self=False,
                        exclusive_prov=None):
    """
    Generate annotated video from h5 files without running tracking.
    
    :param dataset_path: Path to dataset directory containing annotations.h5 and data.h5
    :param channel: Channel to visualize (None for all channels, -1 for max projection)
    :param gamma: Gamma correction for visualization (default: 0.5)
    :param include_all: Include all frames in video (default: True)
    :param exclude_self: Exclude self-annotations (default: False)
    :param exclusive_prov: Only include annotations from specific provenance (default: None)
    """
    
    dataset = Path(dataset_path)
    
    print(f'\n{"="*60}')
    print(f'Generating video for dataset: {dataset}')
    print(f'{"="*60}\n')
    
    # Check required files exist
    required_files = ['annotations.h5', 'data.h5', 'metadata.json']
    for f in required_files:
        if not (dataset / f).exists():
            raise FileNotFoundError(f'Required file not found: {dataset / f}')
    
    print('✓ All required files found')
    
    # Create container without checkpoint
    print('\nInitializing container...')
    container = Container(
        dataset=dataset,
        channel=channel,
        gamma=gamma,
        include_all=include_all,
        exclude_self=exclude_self,
        exclusive_prov=exclusive_prov,
    )
    
    # Add missing required parameters for save_movie
    container.props['t_list'] = []  # Will be overridden by include_all logic
    
    print('✓ Container initialized')
    
    # Load annotations
    print('\nLoading annotations...')
    annotation = get_annotation_df(dataset)
    print(f'✓ Loaded {len(annotation)} annotation points')
    
    # Convert annotations to results format
    print('\nConverting annotations to results format...')
    results = load_annotations_to_results(container, annotation)
    print(f'✓ Results shape: {results.shape}')
    
    # Generate video
    print('\n' + '='*60)
    print('Starting video generation...')
    print('='*60)
    save_movie(container, results)
    
    print(f'\n{"="*60}')
    print(f'✓ Video saved to: {dataset / "annotated.avi"}')
    print(f'{"="*60}\n')


def main():
    parser = argparse.ArgumentParser(
        description='Generate annotated video from ZephIR h5 files without tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python generate_video_only.py ./data/20250730/w3/vol_0_99
  
  # Specify channel and gamma correction
  python generate_video_only.py ./data/20250730/w3/vol_0_99 --channel 0 --gamma 0.7
  
  # Only show annotated frames
  python generate_video_only.py ./data/20250730/w3/vol_0_99 --no-include-all
        """
    )
    
    parser.add_argument('dataset', type=str, 
                       help='Path to dataset directory')
    parser.add_argument('--channel', type=int, default=None,
                       help='Channel to visualize (default: None for all channels, -1 for max projection)')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='Gamma correction for visualization (default: 0.5)')
    parser.add_argument('--no-include-all', action='store_false', dest='include_all',
                       help='Only include annotated frames in video (default: include all frames)')
    parser.add_argument('--exclude-self', action='store_true',
                       help='Exclude self-annotations (default: False)')
    parser.add_argument('--exclusive-prov', type=str, default=None,
                       help='Only include annotations from specific provenance')
    
    args = parser.parse_args()
    
    # Convert exclusive_prov to bytes if provided
    exclusive_prov = None
    if args.exclusive_prov:
        exclusive_prov = bytes(args.exclusive_prov, 'utf-8')
    
    try:
        generate_video_only(
            dataset_path=args.dataset,
            channel=args.channel,
            gamma=args.gamma,
            include_all=args.include_all,
            exclude_self=args.exclude_self,
            exclusive_prov=exclusive_prov
        )
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
