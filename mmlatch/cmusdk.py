import os
import re

import numpy as np
from tqdm import tqdm

import mmsdk
from mmsdk import mmdatasdk as md


from mmlatch.const import (
    SPECIAL_TOKENS,
    MOSEI_MODALITIES2,
    MOSEI_MODALITIES,
    MOSI_MODALITIES,
)

from mmlatch.util import safe_mkdirs, pickle_dump, pickle_load


def download_mmdata(base_path, dataset):
    safe_mkdirs(base_path)

    try:
        md.mmdataset(dataset.highlevel, base_path)
    except RuntimeError:
        print("High-level features have been downloaded previously.")

    try:
        md.mmdataset(dataset.raw, base_path)
    except RuntimeError:
        print("Raw data have been downloaded previously.")

    try:
        md.mmdataset(dataset.labels, base_path)
    except RuntimeError:
        print("Labels have been downloaded previously.")


def avg_collapse(intervals, features):
    try:
        return np.average(features, axis=0)
    except Exception as e:
        del e

        return features


def deploy(in_dataset, destination):
    deploy_files = {x: x for x in in_dataset.keys()}
    in_dataset.deploy(destination, deploy_files)


def select_dataset(dataset_name):
    if dataset_name == "mosi":
        modality_map = MOSI_MODALITIES
        dataset = md.cmu_mosi
    elif dataset_name == "mosei":
        modality_map = MOSEI_MODALITIES
        dataset = md.cmu_mosei
    elif dataset_name == "mosei2":
        modality_map = MOSEI_MODALITIES2
        dataset = md.cmu_mosei
    else:
        raise ValueError("Unsupported dataset {}".format(dataset_name))

    return dataset, modality_map


def load_modality(base_path, modality_map, modality):
    mfile = modality_map[modality]
    path = os.path.join(base_path, "{}.csd".format(mfile))
    print("Using {} for {} modality".format(path, modality))
    data = md.mmdataset(path)

    return data


def get_vocabulary(text_dataset):
    all_words = []

    for seg in text_dataset.keys():
        words = text_dataset[seg]["features"][0]

        for w in words:
            wi = w.decode("utf-8")
            all_words.append(wi)

    all_words = list(set(all_words))

    return all_words


def create_word2idx(all_words):
    word2idx, idx = {}, 0

    for w in sorted(all_words):
        if w not in word2idx:
            word2idx[w] = idx
            idx += 1

    for t in SPECIAL_TOKENS:
        word2idx[t.value] = idx
        idx += 1

    return word2idx


def load_dataset(
    base_path, dataset="mosi", modalities={"audio", "text"}, collapse=None
):
    dataset, modality_map = select_dataset(dataset)
    download_mmdata(base_path, dataset)
    recipe = {
        f: os.path.join(base_path, "{}.csd".format(f))
        for k, f in modality_map.items()
        if k in modalities
    }
    data = md.mmdataset(recipe)
    if collapse is None:
        collapse = [avg_collapse]
    # first we align to words with averaging
    # collapse_function receives a list of functions

    word_align_path = base_path + "_word_aligned"
    safe_mkdirs(word_align_path)

    data.align(modality_map["text"], collapse_functions=collapse)
    data.impute(modality_map["text"])
    deploy(data, word_align_path)

    all_words = get_vocabulary(data[modality_map["text"]])

    word2idx = create_word2idx(all_words)

    label_recipe = {
        modality_map["labels"]: os.path.join(
            base_path, "{}.csd".format(modality_map["labels"])
        )
    }
    data.add_computational_sequences(label_recipe, destination=None)
    data.align(modality_map["labels"])
    data.hard_unify()
    align_path = base_path + "_final_aligned"
    safe_mkdirs(align_path)
    deploy(data, align_path)

    return data, word2idx


def load_aligned(base_path, dataset="mosi", modalities={"audio", "text"}):
    dataset, modality_map = select_dataset(dataset)
    download_mmdata(base_path, dataset)
    recipe = {
        f: os.path.join(base_path, "{}.csd".format(f))
        for k, f in modality_map.items()
        if k in modalities
    }
    data = md.mmdataset(recipe)

    all_words = get_vocabulary(data[modality_map["text"]])

    word2idx = create_word2idx(all_words)

    label_recipe = {
        modality_map["labels"]: os.path.join(
            base_path, "{}.csd".format(modality_map["labels"])
        )
    }
    data.add_computational_sequences(label_recipe, destination=None)

    return data, word2idx


def clean_split_dataset(
    data,
    dataset="mosi",
    modalities={"audio", "text"},
    remove_pauses=False,
    remove_neutral=False,
    max_length=-1,
    pad_front=False,
    pad_back=False,
):
    dataset, modality_map = select_dataset(dataset)
    pattern = re.compile("(.*)\[.*\]")
    train_split = dataset.standard_folds.standard_train_fold
    dev_split = dataset.standard_folds.standard_valid_fold
    test_split = dataset.standard_folds.standard_test_fold

    train, dev, test = [], [], []

    for segment in tqdm(data[modality_map["labels"]].keys()):
        # get the video ID and the features out of the aligned dataset
        vid = re.search(pattern, segment).group(1)
        label = data[modality_map["labels"]][segment]["features"]
        label = np.nan_to_num(label)  # .item()

        if remove_neutral:
            if np.sign(label) == 0:
                continue
        mods = {k: data[modality_map[k]][segment]["features"] for k in modalities}
        num_drop = 0
        # if the sequences are not same length after alignment,
        # there must be some problem with some modalities
        # we should drop it or inspect the data again
        mod_shapes = {k: v.shape[0] for k, v in mods.items()}

        if not len(set(list(mod_shapes.values()))) <= 1:
            print("Datapoint {} shape mismatch {}".format(vid, mod_shapes))
            num_drop += 1

            continue
        lengths = [len(v) for v in mods.values()]

        for m in modalities:
            if m != "text":
                mods[m] = np.nan_to_num(mods[m])

        if "text" in modalities:
            # Handle speech pause
            mods_nosp = {k: [] for k in modalities}
            sp_idx = []

            for i, w in enumerate(mods["text"]):
                word = w[0].decode("utf-8")

                if word == "sp":
                    sp_idx.append(i)

            if remove_pauses:
                for m in modalities:
                    for i in range(len(mods[m])):
                        if i not in sp_idx:
                            if m == "text":
                                word = mods[m][i][0].decode("utf-8")
                                mods_nosp[m].append(word)
                            else:
                                mods_nosp[m].append(mods[m][i, :])
            else:
                mods_nosp = mods
                mods_nosp["text"] = mods_nosp["text"].tolist()

                for i in range(len(mods["text"])):
                    if i in sp_idx:
                        mods_nosp["text"][i] = SPECIAL_TOKENS.PAUSE.value
                    else:
                        word = mods["text"][i][0].decode("utf-8")
                        mods_nosp["text"].append(word)

            mods = mods_nosp

        if max_length > 0:
            lengths = [len(v) for v in mods.values()]

            for m in modalities:
                t = []
                seglen = len(mods[m])

                if seglen > max_length:
                    for i in range(seglen - max_length, seglen):
                        t.append(mods[m][i])
                    mods[m] = t
                elif seglen < max_length and pad_front:
                    for i in range(max_length - seglen):
                        if m == "text":
                            t.append(SPECIAL_TOKENS.PAD.value)
                        else:
                            vshape = mods[m][0].shape
                            pad = np.zeros(vshape)
                            t.append(pad)
                    t += mods[m]
                    mods[m] = t
                elif seglen < max_length and pad_back:
                    t = mods[m]

                    for i in range(max_length - seglen):
                        if m == "text":
                            t.append(SPECIAL_TOKENS.PAD.value)
                        else:
                            vshape = mods[m][0].shape
                            pad = np.zeros(vshape)
                            t.append(pad)
                    mods[m] = t
                else:
                    continue

        for m in modalities:
            if m != "text":
                mods[m] = np.asarray(mods[m])
        mods["video_id"] = vid
        mods["segment_id"] = segment
        mods["label"] = label

        if vid in train_split:
            train.append(mods)
        elif vid in dev_split:
            dev.append(mods)
        elif vid in test_split:
            test.append(mods)
        else:
            print("{} does not belong to any of the splits".format(vid))
    print("Dropped {} data points".format(num_drop))

    return train, dev, test


def load_splits(
    base_path,
    dataset="mosi",
    modalities={"audio", "text"},
    remove_pauses=False,
    remove_neutral=True,
    max_length=-1,
    pad_front=False,
    pad_back=False,
    aligned=False,
    cache=None,
):
    if cache is not None:
        try:
            return pickle_load(cache)
        except FileNotFoundError:
            pass

    if not aligned:
        data, word2idx = load_dataset(base_path, dataset=dataset, modalities=modalities)
    else:
        data, word2idx = load_aligned(base_path, dataset=dataset, modalities=modalities)

    train, dev, test = clean_split_dataset(
        data,
        dataset=dataset,
        modalities=modalities,
        remove_pauses=remove_pauses,
        remove_neutral=remove_neutral,
        max_length=max_length,
        pad_front=pad_front,
        pad_back=pad_back,
    )

    if cache is not None:
        pickle_dump((train, dev, test, word2idx), cache)

    return train, dev, test, word2idx


def mosi(
    base_path,
    modalities={"audio", "text"},
    remove_pauses=False,
    remove_neutral=False,
    max_length=-1,
    pad_front=False,
    pad_back=False,
    cache=None,
    aligned=False,
):
    return load_splits(
        base_path,
        dataset="mosi",
        modalities=modalities,
        remove_pauses=remove_pauses,
        remove_neutral=remove_neutral,
        max_length=max_length,
        pad_front=pad_front,
        pad_back=pad_back,
        cache=cache,
        aligned=aligned,
    )


def mosei(
    base_path,
    modalities={"audio", "text"},
    remove_pauses=False,
    max_length=-1,
    pad_front=False,
    pad_back=False,
    cache=None,
    aligned=False,
):
    remove_neutral = False

    return load_splits(
        base_path,
        dataset="mosei",
        modalities=modalities,
        remove_pauses=remove_pauses,
        remove_neutral=remove_neutral,
        max_length=max_length,
        pad_front=pad_front,
        pad_back=pad_back,
        cache=cache,
        aligned=aligned,
    )


def mosei2(
    base_path,
    modalities={"audio", "text"},
    remove_pauses=False,
    max_length=-1,
    pad_front=False,
    pad_back=False,
    cache=None,
    aligned=False,
):
    remove_neutral = False

    return load_splits(
        base_path,
        dataset="mosei2",
        modalities=modalities,
        remove_pauses=remove_pauses,
        remove_neutral=remove_neutral,
        max_length=max_length,
        pad_front=pad_front,
        pad_back=pad_back,
        cache=cache,
        aligned=aligned,
    )


def data_pickle(fname):
    data = pickle_load(fname)
    return data["train"], data["valid"], data["test"], None


if __name__ == "__main__":
    import sys

    base_path = sys.argv[1]
    train, dev, test, w2i = mosei2(
        base_path,
        modalities=["audio", "text", "visual", "glove"],
        remove_pauses=True,
    )
