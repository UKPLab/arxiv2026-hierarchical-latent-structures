from typing import Dict, Iterator, Optional, List
import torch
from torch.utils.data import IterableDataset as TorchIterableDataset
from datasets import Dataset


class ScalarUInt16ArrowIterable(TorchIterableDataset):
    def __init__(self, path: str, seq_length: int = 1600, input_col: str = "input_ids"):
        self.path = path
        self.seq_length = seq_length
        self.input_col = input_col
        self._ds: Optional[Dataset] = None

    def _dataset(self) -> Dataset:
        if self._ds is None:
            ds = Dataset.from_file(self.path)
            if self.input_col in ds.column_names and len(ds.column_names) > 1:
                ds = ds.select_columns([self.input_col])
            self._ds = ds
        return self._ds

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        ds = self._dataset()
        info = torch.utils.data.get_worker_info()
        if info is not None and info.num_workers > 1:
            ds = ds.shard(num_shards=info.num_workers, index=info.id, contiguous=True)

        buf: List[int] = []
        for ex in ds:
            buf.append(int(ex[self.input_col]))
            if len(buf) == self.seq_length:
                ids = buf
                yield {"input_ids": ids, "labels": ids, "attention_mask": [1] * self.seq_length}
                buf = []

    def __len__(self) -> int:
        return self._dataset().num_rows // self.seq_length


def load_uint16_as_hf_input_ids(path: str, seq_length: int = 1600, input_col: str = "input_ids") -> ScalarUInt16ArrowIterable:
    return ScalarUInt16ArrowIterable(path=path, seq_length=seq_length, input_col=input_col)
