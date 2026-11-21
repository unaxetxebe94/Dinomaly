import torch
from torch.utils.data import TensorDataset

from dinomaly.data import make_loader, release_loader_workers


def test_release_loader_workers_stops_persistent_workers():
    loader = make_loader(TensorDataset(torch.zeros(4, 1)), batch_size=2, shuffle=False, num_workers=1)
    assert sum(1 for _ in loader) == 2
    workers = loader._iterator._workers
    assert workers and all(worker.is_alive() for worker in workers)

    release_loader_workers(loader, None)

    assert loader._iterator is None
    assert not any(worker.is_alive() for worker in workers)
