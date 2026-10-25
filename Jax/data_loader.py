from datasets import load_dataset, IterableDataset
import tiktoken
import grain.python as grain
from secret import HF_TOKEN
class Dataset:
    def __init__(self, ds: IterableDataset, maxlen: int, tokenizer):
        self.data_stream = ds
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.end_token = tokenizer.encode('<|endoftext|>',
                                          allowed_special={'<|endoftext|>'})[0]

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        code = self.data_stream[idx]
        tokens = self.tokenizer.encode(code,
                                       allowed_special={'<|endoftext|>'})

        if len(tokens) > self.maxlen:
            tokens = tokens[:self.maxlen]

        tokens.extend([0] * (self.maxlen - len(tokens)))
        return tokens


def create_dataloader(
    ds,
    tokenizer,
    maxlen,
    batch_size,
    shuffle=False,
    num_epochs=1,
    seed=42,
    worker_count=0
):
    dataset = Dataset(ds, maxlen, tokenizer)
    estimated_batches = len(dataset) // batch_size

    sampler = grain.IndexSampler(
        num_records=len(dataset),
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.NoSharding(),
        num_epochs=num_epochs
    )
    dataloader = grain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=True)
        ],
        worker_count=worker_count
    )

    return dataloader, estimated_batches


def get_tokenizer():
    return tiktoken.get_encoding("o200k_base")


def __stream_python_dataset__():
    return load_dataset(
        path="bigcode/the-stack",
        data_dir="data/python",
        streaming=True,
        split="train",
        token=HF_TOKEN
    )
