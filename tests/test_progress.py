"""Exercise tqdm progress counts, metrics and cleanup without training."""

from io import StringIO

import pytest

from tram_forecast.progress import TrainingProgress


def test_epoch_progress_and_metrics():
    stream = StringIO()
    progress = TrainingProgress(stream)
    with progress.epoch(2, 5, 10):
        bar = progress.bar
        progress.batch(1, 3.0, 4)
        assert bar.n == 1
        progress.batch(10, 1.0, 5)
        assert bar.n == 10
    progress.message('best.pt saved')
    output = stream.getvalue()
    assert 'Epoch 2/5' in output
    assert '100%' in output and 'MAE=1' in output and 'step=5' in output
    assert 'best.pt saved' in output
    assert progress.bar is None and bar.disable


def test_interrupt_closes_bar():
    progress = TrainingProgress(StringIO())
    with pytest.raises(KeyboardInterrupt):
        with progress.epoch(1, 5, 10):
            bar = progress.bar
            progress.batch(1, 3.0, 1)
            raise KeyboardInterrupt
    assert progress.bar is None and bar.disable
