"""A special benchmark loader for running benchmark during validation."""


class DelegateBenchmarkLoader:
    """This loader only returns information for benchmark dataloader, not traditional dataloader."""

    def __init__(self, benchmarks, **kwargs):
        self._kwargs = kwargs.copy()
        self._benchmarks = benchmarks

    def __len__(self):
        return len(self._benchmarks)

    def __getitem__(self, idx):
        ret = {"benchmark": self._benchmarks[idx]}
        ret.update(self._kwargs)
        return ret


if __name__ == "__main__":
    loader = DelegateBenchmarkLoader(["douyin_recall", "douyin_recall_v2"], vocab_file="xx.vocab")
    for batch in loader:
        print(batch)
