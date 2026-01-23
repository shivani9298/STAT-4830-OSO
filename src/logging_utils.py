"""
Logging utilities - trial logging and artifact management.
Owned by Person D.
"""

import json
from pathlib import Path
from typing import Dict, Any


def log_trial(path: Path, trial_record: Dict[str, Any]) -> None:
    """
    Append a trial record to JSONL file.
    
    Args:
        path: Path to JSONL file
        trial_record: Dictionary to log (will be serialized to JSON)
    """
    path = Path(path)
    
    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Append to JSONL file
    with open(path, 'a') as f:
        json.dump(trial_record, f)
        f.write('\n')
