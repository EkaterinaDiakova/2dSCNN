

import logging
import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import pickle
import numpy as np

logger = logging.getLogger(__name__)

class Rusc(Dataset):
    def __init__(
        self,
        data_folder,
        split,
    ):

        if split not in ["train", "test", "valid"]:
            raise ValueError(f"Invalid split {split}")

        # Get paths to all audio files
        self.data_folder = data_folder

        self.get_2dcnn_feats(data_folder, split)


    def __len__(self):
        return len(self.inputs)

    def get_2dcnn_feats(self, data_folder, split):
        if split not in ["train", "test", "valid"]:
            raise ValueError(f"Invalid split {split}")

        # Get paths to all audio files
        self.data_folder = data_folder
        filename_inp = self.data_folder + "/x_" + split + ".pkl"
        filename_labels = self.data_folder + "/y_" + split + ".pkl"

        # Load prep data
        file = open(filename_inp, 'rb')
        self.inputs = torch.tensor(np.array(pickle.load(file)))
        file.close()

        file = open(filename_labels, 'rb')
        self.labels = torch.tensor(pickle.load(file))
        file.close()

    def __getitem__(self, index):

        x = self.inputs[index]
        y = self.labels[index]

        return x, y

    def generateBatch(self, batch):

        xs, ys = zip(*batch)
        xlens = torch.tensor([x.shape[0] for x in xs])
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys)

        return xs, xlens, ys


class HeidelbergDigits(Dataset):

    def __init__(
        self,
        data_folder,
        split
    ):

        if split not in ["train", "test"]:
            raise ValueError(f"Invalid split {split}")

        # Get paths to all audio files
        self.data_folder = data_folder

        self.get_2dcnn_feats(data_folder, split)


    def __len__(self):
        return len(self.inputs)

    def get_2dcnn_feats(self, data_folder, split):
        if split not in ["train", "test"]:
            raise ValueError(f"Invalid split {split}")

            # Get paths to all audio files
        self.data_folder = data_folder
        filename_inp = self.data_folder + "/x_" + split + ".pkl"
        filename_labels = self.data_folder + "/y_" + split + ".pkl"

        # Load prep train / validation data
        file = open(filename_inp, 'rb')
        self.inputs = torch.tensor(np.array(pickle.load(file)))
        file.close()

        file = open(filename_labels, 'rb')
        self.labels = torch.tensor(pickle.load(file))
        file.close()

    def __getitem__(self, index):

        x = self.inputs[index]
        y = self.labels[index]

        return x, y

    def generateBatch(self, batch):

        xs, ys = zip(*batch)
        xlens = torch.tensor([x.shape[0] for x in xs])
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys)

        return xs, xlens, ys


def load_hd(
    dataset_name,
    data_folder,
    split,
    batch_size,
    shuffle=True,
    workers=0,
):

    if dataset_name != "hd":
        raise ValueError(f"Invalid dataset name {dataset_name}")

    if split not in ["train", "test", "valid"]:
        raise ValueError(f"Invalid split name {split}")

    if split in ["valid", "test"]:
        split = "test"
        logging.info("\nHD uses the same split for validation and testing.\n")

    dataset = HeidelbergDigits(
        data_folder, split
    )
    logging.info(f"Number of examples in {dataset_name} {split} set: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.generateBatch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )
    return loader

def load_rusc(
    dataset_name,
    data_folder,
    split,
    batch_size,
    shuffle=True,
    workers=0):

    if dataset_name not in ["rusc"]:
        raise ValueError(f"Invalid dataset name {dataset_name}")

    if split not in ["train", "valid", "test"]:
        raise ValueError(f"Invalid split name {split}")

    dataset = Rusc(data_folder, split)

    logging.info(f"Number of examples in {dataset_name} {split} set: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.generateBatch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )
    return loader

