import argparse
import os
import sys
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics import Accuracy, Fbeta, Loss
from torch.utils.data import DataLoader

from mmlatch.cmusdk import mosei
from mmlatch.config import load_config
from mmlatch.data import MOSEI, MOSEICollator, ToTensor
from mmlatch.mm import AudioVisualTextClassifier, AVTClassifier
from mmlatch.trainer import MOSEITrainer
from mmlatch.util import safe_mkdirs


class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()

    def forward(self, out, tgt):
        tgt = tgt.view(-1, 1).float()

        return F.binary_cross_entropy_with_logits(out, tgt)


def get_parser():
    parser = argparse.ArgumentParser(description="CLI parser for experiment")

    parser.add_argument(
        "--dropout",
        dest="common.dropout",
        default=None,
        type=float,
        help="Dropout probabiity",
    )

    parser.add_argument(
        "--proj-size",
        dest="fuse.projection_size",
        default=None,
        type=int,
        help="Modality projection size",
    )

    parser.add_argument(
        "--bidirectional",
        dest="common.bidirectional",
        action="store_true",
        help="Use BiRNNs",
    )

    parser.add_argument(
        "--rnn-type", dest="common.rnn_type", default=None, type=str, help="lstm or gru"
    )

    parser.add_argument(
        "--feedback",
        dest="feedback",
        action="store_true",
        help="Use feedback fusion",
    )

    parser.add_argument(
        "--result-dir",
        dest="results_dir",
        help="Results directory",
    )

    return parser


C = load_config(parser=get_parser())

collate_fn = MOSEICollator(
    device="cpu", modalities=["text", "audio", "visual"], max_length=-1
)


if __name__ == "__main__":
    print("Running with configuration")
    pprint(C)
    train, dev, test, vocab = mosei(
        C["data_dir"],
        modalities=["text", "glove", "audio", "visual"],
        remove_pauses=False,
        max_length=-1,
        pad_front=True,
        pad_back=False,
        aligned=False,
        cache=os.path.join(C["cache_dir"], "mosei_avt.p"),
    )

    # Use GloVe features for text inputs
    for d in train:
        d["text"] = d["glove"]

    for d in dev:
        d["text"] = d["glove"]

    for d in test:
        d["text"] = d["glove"]

    to_tensor = ToTensor(device="cpu")
    to_tensor_float = ToTensor(device="cpu", dtype=torch.float)

    def create_dataloader(data, shuffle=True):
        d = MOSEI(data, modalities=["text", "glove", "audio", "visual"], select_label=0)
        d.map(to_tensor_float, "visual", lazy=True)
        d.map(to_tensor_float, "text", lazy=True)
        d = d.map(to_tensor_float, "audio", lazy=True)
        d.apply_transforms()
        dataloader = DataLoader(
            d,
            batch_size=C["dataloaders"]["batch_size"],
            num_workers=C["dataloaders"]["num_workers"],
            pin_memory=C["dataloaders"]["batch_size"],
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

        return dataloader

    train_loader = create_dataloader(train)
    dev_loader = create_dataloader(dev)
    test_loader = create_dataloader(test)
    print("Running with feedback = {}".format(C["model"]["feedback"]))

    model = AVTClassifier(
        C["model"]["text_input_size"],
        C["model"]["audio_input_size"],
        C["model"]["visual_input_size"],
        C["model"]["projection_size"],
        text_layers=C["model"]["text_layers"],
        audio_layers=C["model"]["audio_layers"],
        visual_layers=C["model"]["visual_layers"],
        bidirectional=C["model"]["bidirectional"],
        dropout=C["model"]["dropout"],
        encoder_type=C["model"]["encoder_type"],
        attention=C["model"]["attention"],
        feedback=C["model"]["feedback"],
        feedback_type=C["model"]["feedback_type"],
        device=C["device"],
        num_classes=C["num_classes"],
    )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("NUMBER OF PARAMETERS: {}".format(count_parameters(model)))

    model = model.to(C["device"])
    optimizer = getattr(torch.optim, C["optimizer"]["name"])(
        [p for p in model.parameters() if p.requires_grad],
        lr=C["optimizer"]["learning_rate"],
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=0.5,
        patience=2,
        cooldown=2,
        min_lr=C["optimizer"]["learning_rate"] / 20.0,
    )

    criterion = nn.L1Loss()

    def bin_acc_transform(output):
        y_pred, y = output
        nz = torch.nonzero(y).squeeze()
        yp, yt = (y_pred[nz] >= 0).long(), (y[nz] >= 0).long()

        return yp, yt

    def acc_transform(output):
        y_pred, y = output
        yp, yt = (y_pred >= 0).long(), (y >= 0).long()

        return yp, yt

    def acc7_transform(output):
        y_pred, y = output
        yp = torch.clamp(torch.round(y_pred) + 3, 0, 6).view(-1).long()
        yt = torch.round(y).view(-1).long() + 3
        yp = F.one_hot(yp, 7)

        return yp, yt

    def acc5_transform(output):
        y_pred, y = output
        yp = torch.clamp(torch.round(y_pred) + 2, 0, 4).view(-1).long()
        yt = torch.round(y).view(-1).long() + 2
        yp = F.one_hot(yp, 5)

        return yp, yt

    metrics = {
        "acc5": Accuracy(output_transform=acc5_transform),
        "acc7": Accuracy(output_transform=acc7_transform),
        "bin_accuracy": Accuracy(output_transform=bin_acc_transform),
        "f1": Fbeta(1, output_transform=bin_acc_transform),
        "accuracy_zeros": Accuracy(output_transform=acc_transform),
        "loss": Loss(criterion),
    }

    if C["overfit_batch"] or C["overfit_batch"] or C["train"]:
        import shutil

        try:
            shutil.rmtree(C["trainer"]["checkpoint_dir"])
        except:
            pass
        if C["trainer"]["accumulation_steps"] is not None:
            acc_steps = C["trainer"]["accumulation_steps"]
        else:
            acc_steps = 1
        trainer = MOSEITrainer(
            model,
            optimizer,
            # score_fn=score_fn,
            experiment_name=C["experiment"]["name"],
            checkpoint_dir=C["trainer"]["checkpoint_dir"],
            metrics=metrics,
            non_blocking=C["trainer"]["non_blocking"],
            patience=C["trainer"]["patience"],
            validate_every=C["trainer"]["validate_every"],
            retain_graph=C["trainer"]["retain_graph"],
            loss_fn=criterion,
            accumulation_steps=acc_steps,
            lr_scheduler=lr_scheduler,
            device=C["device"],
        )

    if C["debug"]:
        if C["overfit_batch"]:
            trainer.overfit_single_batch(train_loader)
        trainer.fit_debug(train_loader, dev_loader)
        sys.exit(0)

    if C["train"]:
        trainer.fit(train_loader, dev_loader, epochs=C["trainer"]["max_epochs"])

    if C["test"]:
        try:
            del trainer
        except:
            pass
        trainer = MOSEITrainer(
            model,
            optimizer,
            experiment_name=C["experiment"]["name"],
            checkpoint_dir=C["trainer"]["checkpoint_dir"],
            metrics=metrics,
            model_checkpoint=C["trainer"]["load_model"],
            non_blocking=C["trainer"]["non_blocking"],
            patience=C["trainer"]["patience"],
            validate_every=C["trainer"]["validate_every"],
            retain_graph=C["trainer"]["retain_graph"],
            loss_fn=criterion,
            device=C["device"],
        )

        predictions, targets = trainer.predict(test_loader)

        pred = torch.cat(predictions)
        y_test = torch.cat(targets)

        import uuid

        from mmlatch.mosei_metrics import (eval_mosei_senti, print_metrics,
                                           save_metrics)

        metrics = eval_mosei_senti(pred, y_test, True)
        print_metrics(metrics)

        results_dir = C["results_dir"]
        safe_mkdirs(results_dir)
        fname = uuid.uuid1().hex
        results_file = os.path.join(results_dir, fname)

        save_metrics(metrics, results_file)
