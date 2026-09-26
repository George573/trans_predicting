"""tqdm training feedback, automatically adapted to notebooks and terminals."""

import sys
from contextlib import contextmanager

from tqdm.auto import tqdm


class TrainingProgress:
    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stderr
        self.bar = None

    @contextmanager
    def epoch(self, epoch, epochs, total):
        with tqdm(
            total=total,
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            file=self.stream,
            dynamic_ncols=True,
            mininterval=0.5,
        ) as bar:
            self.bar = bar
            try:
                yield
            finally:
                self.bar = None

    def show(self, message, force=False):
        self.message(message)

    def message(self, message):
        tqdm.write(message, file=self.stream)

    def batch(self, completed, mae, steps):
        self.bar.set_postfix(MAE=f"{mae:.5g}", step=steps, refresh=False)
        self.bar.update(completed - self.bar.n)
