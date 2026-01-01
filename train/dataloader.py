from torch.utils.data import DataLoader


def build_dataloader(dataset, batch_size=1, shuffle=True):
    """
    Note:
    PerfGraph는 graph-level prediction이라 batch_size=1 권장
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: x[0],  # no batching
    )
