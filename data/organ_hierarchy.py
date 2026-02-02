"""
Parse organ hierarchy tree to extract class depths.

The tree.json defines anatomical hierarchy:
- human_body (root, depth 0)
  - skeletal_system (depth 1)
    - axial_skeleton (depth 2)
      ...
        - rib_left_1 (leaf, depth 6)
"""
import json
from typing import Dict, List, Optional


def _find_depth_recursive(
    tree: dict,
    target_name: str,
    current_depth: int = 0
) -> Optional[int]:
    """
    Recursively search for target_name in tree and return its depth.

    Args:
        tree: Dictionary representing the hierarchy subtree
        target_name: Class name to find
        current_depth: Current depth in traversal

    Returns:
        Depth if found, None otherwise
    """
    for key, value in tree.items():
        if isinstance(value, str):
            # Leaf node: value is the class name
            if value == target_name:
                return current_depth + 1
        elif isinstance(value, dict):
            # Intermediate node: recurse
            result = _find_depth_recursive(value, target_name, current_depth + 1)
            if result is not None:
                return result
    return None


def load_organ_hierarchy(tree_path: str, class_names: List[str]) -> Dict[int, int]:
    """
    Parse tree.json and compute depth for each class.

    Args:
        tree_path: Path to tree.json
        class_names: List of class names (index = class_idx)

    Returns:
        Dictionary mapping class_idx -> depth
    """
    with open(tree_path, "r") as f:
        tree = json.load(f)

    depths = {}
    for idx, name in enumerate(class_names):
        depth = _find_depth_recursive(tree, name, current_depth=0)
        if depth is None:
            # Default depth for classes not in tree (shouldn't happen)
            depth = 1
        depths[idx] = depth

    return depths


def get_depth_stats(depths: Dict[int, int]) -> Dict[str, int]:
    """
    Get statistics about depth distribution.

    Args:
        depths: Dictionary mapping class_idx -> depth

    Returns:
        Dictionary with min_depth, max_depth, unique_depths
    """
    depth_values = list(depths.values())
    return {
        "min_depth": min(depth_values),
        "max_depth": max(depth_values),
        "unique_depths": len(set(depth_values)),
    }
