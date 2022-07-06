import os
from typing import Callable, List, Optional, Tuple, TypeVar, Union, cast

import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, State
from ignite.handlers import EarlyStopping
from ignite.metrics import Loss, RunningAverage
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from mmlatch.handlers import CheckpointHandler, EvaluationHandler
from mmlatch.util import from_checkpoint, to_device, GenericDict

TrainerType = TypeVar("TrainerType", bound="Trainer")


class Trainer(object):
    def __init__(
        self: TrainerType,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler=None,
        newbob_metric="loss",
        checkpoint_dir: str = "../checkpoints",
        experiment_name: str = "experiment",
        score_fn: Optional[Callable] = None,
        model_checkpoint: Optional[str] = None,
        optimizer_checkpoint: Optional[str] = None,
        metrics: GenericDict = None,
        patience: int = 10,
        validate_every: int = 1,
        accumulation_steps: int = 1,
        loss_fn: _Loss = None,
        non_blocking: bool = True,
        retain_graph: bool = False,
        dtype: torch.dtype = torch.float,
        device: str = "cpu",
    ) -> None:
        self.dtype = dtype
        self.retain_graph = retain_graph
        self.non_blocking = non_blocking
        self.device = device
        self.loss_fn = loss_fn
        self.validate_every = validate_every
        self.patience = patience
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = checkpoint_dir

        model_checkpoint = self._check_checkpoint(model_checkpoint)
        optimizer_checkpoint = self._check_checkpoint(optimizer_checkpoint)

        self.model = cast(
            nn.Module,
            from_checkpoint(model_checkpoint, model, map_location=torch.device("cpu")),
        )
        self.model = self.model.type(dtype).to(device)
        self.optimizer = from_checkpoint(optimizer_checkpoint, optimizer)
        self.lr_scheduler = lr_scheduler

        if metrics is None:
            metrics = {}

        if "loss" not in metrics:
            metrics["loss"] = Loss(self.loss_fn)
        self.trainer = Engine(self.train_step)
        self.train_evaluator = Engine(self.eval_step)
        self.valid_evaluator = Engine(self.eval_step)

        for name, metric in metrics.items():
            metric.attach(self.train_evaluator, name)
            metric.attach(self.valid_evaluator, name)

        self.pbar = ProgressBar()
        self.val_pbar = ProgressBar(desc="Validation")

        self.score_fn = score_fn if score_fn is not None else self._score_fn

        if checkpoint_dir is not None:
            self.checkpoint = CheckpointHandler(
                checkpoint_dir,
                experiment_name,
                score_name="validation_loss",
                score_function=self.score_fn,
                n_saved=2,
                require_empty=False,
                save_as_state_dict=True,
            )

        self.early_stop = EarlyStopping(patience, self.score_fn, self.trainer)

        self.val_handler = EvaluationHandler(
            pbar=self.pbar,
            validate_every=1,
            early_stopping=self.early_stop,
            newbob_scheduler=self.lr_scheduler,
            newbob_metric=newbob_metric,
        )
        self.attach()
        print(
            f"Trainer configured to run {experiment_name}\n"
            f"\tpretrained model: {model_checkpoint} {optimizer_checkpoint}\n"
            f"\tcheckpoint directory: {checkpoint_dir}\n"
            f"\tpatience: {patience}\n"
            f"\taccumulation steps: {accumulation_steps}\n"
            f"\tnon blocking: {non_blocking}\n"
            f"\tretain graph: {retain_graph}\n"
            f"\tdevice: {device}\n"
            f"\tmodel dtype: {dtype}\n"
        )

    def _check_checkpoint(self: TrainerType, ckpt: Optional[str]) -> Optional[str]:
        if ckpt is None:
            return ckpt

        ckpt = os.path.join(self.checkpoint_dir, ckpt)

        return ckpt

    @staticmethod
    def _score_fn(engine: Engine) -> float:
        """Returns the scoring metric for checkpointing and early stopping

        Args:
            engine (ignite.engine.Engine): The engine that calculates
            the val loss

        Returns:
            (float): The validation loss
        """
        negloss: float = -engine.state.metrics["loss"]

        return negloss

    def parse_batch(
        self: TrainerType, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0], device=self.device, non_blocking=self.non_blocking)
        targets = to_device(
            batch[1], device=self.device, non_blocking=self.non_blocking
        )

        return inputs, targets

    def get_predictions_and_targets(
        self: TrainerType, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        inputs, targets = self.parse_batch(batch)
        y_pred = self.model(inputs)

        return y_pred, targets

    def train_step(
        self: TrainerType, engine: Engine, batch: List[torch.Tensor]
    ) -> float:
        self.model.train()
        y_pred, targets = self.get_predictions_and_targets(batch)
        loss = self.loss_fn(y_pred, targets)  # type: ignore

        loss = loss / self.accumulation_steps
        loss.backward(retain_graph=self.retain_graph)

        if self.lr_scheduler is not None:
            if (engine.state.iteration - 1) % 128 == 0:
                print("LR = {}".format(self.optimizer.param_groups[0]["lr"]))
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

        if (self.trainer.state.iteration + 1) % self.accumulation_steps == 0:
            self.optimizer.step()  # type: ignore
            self.optimizer.zero_grad()
        loss_value: float = loss.item()

        return loss_value

    def eval_step(
        self: TrainerType, engine: Engine, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        self.model.eval()
        with torch.no_grad():
            y_pred, targets = self.get_predictions_and_targets(batch)

            return y_pred, targets

    def predict(self: TrainerType, dataloader: DataLoader) -> State:
        predictions, targets = [], []

        for batch in dataloader:
            self.model.eval()
            with torch.no_grad():
                pred, targ = self.get_predictions_and_targets(batch)
                predictions.append(pred)
                targets.append(targ)

        return predictions, targets

    def fit(
        self: TrainerType,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
    ) -> State:
        print(
            "Trainer will run for\n"
            f"model: {self.model}\n"
            f"optimizer: {self.optimizer}\n"
            f"loss: {self.loss_fn}"
        )
        self.val_handler.attach(
            self.trainer, self.train_evaluator, train_loader, validation=False
        )
        self.val_handler.attach(
            self.trainer, self.valid_evaluator, val_loader, validation=True
        )
        self.model.zero_grad()
        # self.valid_evaluator.run(val_loader)
        self.trainer.run(train_loader, max_epochs=epochs)

    def overfit_single_batch(self: TrainerType, train_loader: DataLoader) -> State:
        single_batch = [next(iter(train_loader))]

        if self.trainer.has_event_handler(self.val_handler, Events.EPOCH_COMPLETED):
            self.trainer.remove_event_handler(self.val_handler, Events.EPOCH_COMPLETED)

        self.val_handler.attach(  # type: ignore
            self.trainer,
            self.train_evaluator,
            single_batch,
            validation=False,
        )
        out = self.trainer.run(single_batch, max_epochs=100)

        return out

    def fit_debug(
        self: TrainerType, train_loader: DataLoader, val_loader: DataLoader
    ) -> State:
        train_loader = iter(train_loader)  # type: ignore
        train_subset = [next(train_loader), next(train_loader)]  # type: ignore
        val_loader = iter(val_loader)  # type: ignore
        val_subset = [next(val_loader), next(val_loader)]  # type: ignore
        out = self.fit(train_subset, val_subset, epochs=6)  # type: ignore

        return out

    def _attach_checkpoint(self: TrainerType) -> TrainerType:
        ckpt = {"model": self.model, "optimizer": self.optimizer}

        if self.checkpoint_dir is not None:
            self.valid_evaluator.add_event_handler(
                Events.COMPLETED, self.checkpoint, ckpt
            )

        return self

    def attach(self: TrainerType) -> TrainerType:
        ra = RunningAverage(output_transform=lambda x: x)
        ra.attach(self.trainer, "Train Loss")
        self.pbar.attach(self.trainer, ["Train Loss"])
        self.val_pbar.attach(self.train_evaluator)
        self.val_pbar.attach(self.valid_evaluator)
        self.valid_evaluator.add_event_handler(Events.COMPLETED, self.early_stop)
        self = self._attach_checkpoint()

        def graceful_exit(engine, e):
            if isinstance(e, KeyboardInterrupt):
                engine.terminate()
                print("CTRL-C caught. Exiting gracefully...")
            else:
                raise (e)

        self.trainer.add_event_handler(Events.EXCEPTION_RAISED, graceful_exit)
        self.train_evaluator.add_event_handler(Events.EXCEPTION_RAISED, graceful_exit)
        self.valid_evaluator.add_event_handler(Events.EXCEPTION_RAISED, graceful_exit)

        return self


class MOSEITrainer(Trainer):
    def parse_batch(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = {
            k: to_device(v, device=self.device, non_blocking=self.non_blocking)
            for k, v in batch[0].items()
        }
        targets = to_device(
            batch[1], device=self.device, non_blocking=self.non_blocking
        )

        return inputs, targets

    def get_predictions_and_targets(
        self, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        inputs, targets = self.parse_batch(batch)
        y_pred = self.model(inputs)
        y_pred = y_pred.squeeze()
        targets = targets.squeeze()

        return y_pred, targets
