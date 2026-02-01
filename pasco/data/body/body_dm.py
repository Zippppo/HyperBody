"""
PyTorch Lightning DataModule for body semantic segmentation.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pasco.data.body.body_dataset import BodyDataset, collate_fn


class BodyDataModule(pl.LightningDataModule):
    """
    DataModule for body semantic segmentation.

    Expects dataset structure:
        root/
        ├── train.txt
        ├── val.txt
        ├── test.txt
        └── data/
            ├── sample1.npz
            ├── sample2.npz
            └── ...

    Args:
        root: Dataset root directory
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        target_size: Target grid size (H, W, D)
    """

    def __init__(
        self,
        root,
        batch_size=2,
        num_workers=4,
        target_size=(160, 160, 256),
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        """Setup datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_ds = BodyDataset(
                root=self.root,
                split="train",
                target_size=self.target_size,
            )
            self.val_ds = BodyDataset(
                root=self.root,
                split="val",
                target_size=self.target_size,
            )

        if stage == "test" or stage is None:
            self.test_ds = BodyDataset(
                root=self.root,
                split="test",
                target_size=self.target_size,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )
