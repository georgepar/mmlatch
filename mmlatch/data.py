import torch
import gc

from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from cytoolz.functoolz import compose
except ImportError:
    from toolz.functoolz import compose

from torch.nn.utils.rnn import pad_sequence
from mmlatch.util import mktensor


class ToTensor(object):
    def __init__(self, device="cpu", dtype=torch.long):
        self.device = device
        self.dtype = dtype

    def __call__(self, x):
        return mktensor(x, device=self.device, dtype=self.dtype)


class MOSEICollator(object):
    def __init__(
        self,
        modalities=("text", "audio"),
        pad_indx=0,
        device="cpu",
        max_length=-1,
    ):
        self.pad_indx = pad_indx
        self.device = device
        self.modalities = list(modalities)
        self.max_length = max_length
        self.target_dtype = torch.float

    def extract_label(self, l):
        return l[0]

    def extract_sequence(self, s):
        return s[self.max_length :] if self.max_length > 0 else s

    def __call__(self, batch):
        data = {}
        # data["lengths"] = torch.tensor(
        #    [len(self.extract_sequence(b[self.modalities[0]])) for b in batch], device=self.device
        # )

        for m in self.modalities:
            inputs = [self.extract_sequence(b[m]) for b in batch]
            data[m] = pad_sequence(
                inputs, batch_first=True, padding_value=self.pad_indx
            ).to(self.device)

        data["lengths"] = torch.tensor(
            [len(s) for s in data[self.modalities[0]]], device=self.device
        )

        targets = [self.extract_label(b["label"]) for b in batch]
        targets = mktensor(targets, device=self.device, dtype=self.target_dtype)  # type: ignore

        return data, targets.to(self.device)


class MOSEI(Dataset):
    def __init__(
        self,
        data,
        select_label=None,
        modalities={"text", "audio"},
        transforms=None,
    ):
        data1 = {k: [] for k in data[0].keys()}
        self.data = []

        for dat in data:
            for k, v in dat.items():
                data1[k].append(v)
        data = data1
        self.select_label = select_label
        self.labels = data["label"]

        for i in range(len(self.labels)):
            dat = {}

            for k in modalities:
                dat[k] = data[k][i]
            self.data.append(dat)
        gc.collect()
        self.modalities = modalities
        self.transforms = transforms

        if self.transforms is None:
            self.transforms = {m: [] for m in self.modalities}

    def map(self, fn, modality, lazy=True):
        if modality not in self.modalities:
            return self
        self.transforms[modality].append(fn)

        if not lazy:
            self.apply_transforms()

        return self

    def apply_transforms(self):
        for m in self.modalities:
            if len(self.transforms[m]) == 0:
                continue
            fn = compose(*self.transforms[m][::-1])
            # In place transformation to save some mem.

            for i in tqdm(range(len(self.data)), total=len(self.data)):
                self.data[i][m] = fn(self.data[i][m])
        self.transforms = {m: [] for m in self.modalities}

        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dat = self.data[idx]
        dat["label"] = self.labels[idx]

        if self.select_label is not None:
            dat["label"] = dat["label"][self.select_label]

        return dat
