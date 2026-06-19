"""LightningModule placeholder for NCF training."""

from __future__ import annotations

import lightning as L


class NCFLightningModule(L.LightningModule):
    """Placeholder training module for step-wise implementation."""

    def __init__(self) -> None:
        super().__init__()

    def training_step(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError("Training step will be implemented in Step 2.")

    def configure_optimizers(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError("Optimizer setup will be implemented in Step 2.")
