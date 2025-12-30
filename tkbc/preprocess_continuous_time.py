# Script to normalize timestamps to continuous time values for temporal KG embedding

import os
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np

DATA_PATH = Path(__file__).resolve().parent / "data"

def parse_date(date_str):
    """Parse date string to timestamp."""
    try:
        # Try YYYY-MM-DD format (ICEWS)
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.timestamp()
    except:
        # Try integer format (YAGO)
        try:
            return float(date_str)
        except:
            return None

def normalize_timestamps(dataset_name):
    """
    Read ts_id file, normalize timestamps to [0, 100] range, and save.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'ICEWS14', 'ICEWS05-15', 'yago15k')
    """
    dataset_path = DATA_PATH / dataset_name
    ts_id_path = dataset_path / "ts_id"
    
    if not ts_id_path.exists():
        print(f"Warning: {ts_id_path} does not exist. Skipping {dataset_name}")
        return
    
    print(f"\nProcessing {dataset_name}...")
    
    # Read ts_id file (handle both text and pickle formats)
    timestamp_dict = {}
    try:
        # Try reading as pickle first (for wikidata)
        with open(ts_id_path, 'rb') as f:
            timestamp_dict = pickle.load(f)
        print(f"Loaded {len(timestamp_dict)} timestamps from pickle")
    except (pickle.UnpicklingError, UnicodeDecodeError):
        # Fall back to text format (for ICEWS and YAGO)
        with open(ts_id_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    date_str, ts_id = parts
                    timestamp_dict[date_str] = int(ts_id)
        print(f"Found {len(timestamp_dict)} timestamps from text file")
        # Invert dictionary for consistency
        timestamp_dict = {v: k for k, v in timestamp_dict.items()}
    
    if not timestamp_dict:
        print(f"Warning: No timestamps found in {dataset_name}")
        return
    
    # Convert to real-valued timestamps
    ts_ids = sorted(timestamp_dict.keys())
    date_strings = [timestamp_dict[ts_id] for ts_id in ts_ids]
    real_timestamps = [parse_date(ds) for ds in date_strings]
    
    if None in real_timestamps:
        print(f"Warning: Some timestamps could not be parsed in {dataset_name}")
        return
    
    real_timestamps = np.array(real_timestamps)
    
    # Min-Max normalization to [-1, 1]
    t_min = real_timestamps.min()
    t_max = real_timestamps.max()
    
    if t_max == t_min:
        print(f"Warning: All timestamps are the same in {dataset_name}")
        normalized_timestamps = np.zeros_like(real_timestamps)
    else:
        normalized_timestamps = 2.0 * (real_timestamps - t_min) / (t_max - t_min) - 1.0
    
    print(f"Time range: {date_strings[0]} to {date_strings[-1]}")
    print(f"Normalized range: [{normalized_timestamps.min():.2f}, {normalized_timestamps.max():.2f}]")
    
    # Create mapping: timestamp_id -> normalized_time
    ts_id_to_normalized = {ts_id: normalized_timestamps[i] for i, ts_id in enumerate(ts_ids)}
    
    # Save as pickle
    output_path = dataset_path / "ts_normalized.pickle"
    with open(output_path, 'wb') as f:
        pickle.dump(ts_id_to_normalized, f)
    
    print(f"Saved normalized timestamps to {output_path}")
    
    # Also save as numpy array for easy access
    output_array_path = dataset_path / "ts_normalized_array.npy"
    np.save(output_array_path, normalized_timestamps)
    print(f"Saved normalized timestamp array to {output_array_path}")
    
    return ts_id_to_normalized

if __name__ == "__main__":
    datasets = ['ICEWS14', 'ICEWS05-15', 'yago15k', 'wikidata']
    
    print("="*60)
    print("Continuous Time Normalization Preprocessing")
    print("="*60)
    
    for dataset in datasets:
        try:
            normalize_timestamps(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
