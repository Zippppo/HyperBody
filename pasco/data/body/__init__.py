"""
Body semantic segmentation data package.
"""

from pasco.data.body.body_dm import BodyDataModule
from pasco.data.body.params import N_CLASSES, body_class_names

__all__ = ["BodyDataModule", "N_CLASSES", "body_class_names"]
