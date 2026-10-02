from __future__ import annotations


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        *,
        mode: str = "max",
        minimum_delta: float = 0.0,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be min or max")
        self.patience = patience
        self.mode = mode
        self.minimum_delta = minimum_delta
        self.best: float | None = None
        self.bad_epochs = 0

    def is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.minimum_delta
        return value < self.best - self.minimum_delta

    def step(self, value: float) -> bool:
        if self.is_improvement(value):
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
