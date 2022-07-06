import shutil

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from typing import Optional

from mmlatch.util import print_separator, GenericDict


class CheckpointHandler(ModelCheckpoint):
    """Augment ignite ModelCheckpoint Handler with copying the best file to a
    {filename_prefix}_{experiment_name}.best.pth.
    This helps for automatic testing etc.
    Args:
        engine (ignite.engine.Engine): The trainer engine
        to_save (dict): The objects to save
    """

    def __call__(self, engine: Engine, to_save: GenericDict) -> None:
        super(CheckpointHandler, self).__call__(engine, to_save)
        # Select model with best loss
        src = self.last_checkpoint
        print(src)
        # for src in paths:
        splitted = src.split("_")
        fname_prefix = splitted[0]
        name = splitted[1]
        dst = f"{fname_prefix}_{name}.best.pth"
        shutil.copy(src, dst)


class EvaluationHandler(object):
    def __init__(
        self,
        pbar: Optional[ProgressBar] = None,
        validate_every: int = 1,
        early_stopping: Optional[EarlyStopping] = None,
        newbob_metric="loss",
        newbob_scheduler=None,
    ):
        self.validate_every = validate_every
        self.print_fn = pbar.log_message if pbar is not None else print
        self.early_stopping = early_stopping
        self.newbob_scheduler = newbob_scheduler
        self.newbob_metric = newbob_metric

    def __call__(
        self,
        engine: Engine,
        evaluator: Engine,
        dataloader: DataLoader,
        validation: bool = True,
    ):
        if engine.state.epoch % self.validate_every != 0:
            return
        evaluator.run(dataloader)

        print_separator(n=35, print_fn=self.print_fn)
        metrics = evaluator.state.metrics
        phase = "Validation" if validation else "Training"
        self.print_fn("Epoch {} {} results".format(engine.state.epoch, phase))
        print_separator(symbol="-", n=35, print_fn=self.print_fn)
        for name, value in metrics.items():
            self.print_fn("{:<15} {:<15}".format(name, value))

        if self.newbob_scheduler is not None:
            if self.newbob_metric == "loss":
                m = metrics[self.newbob_metric]
            else:
                m = -metrics[self.newbob_metric]
            self.newbob_scheduler.step(m, epoch=engine.state.epoch)

        if validation and self.early_stopping:
            loss = self.early_stopping.best_score
            patience = self.early_stopping.patience
            cntr = self.early_stopping.counter
            self.print_fn("{:<15} {:<15}".format("best loss", -loss))
            self.print_fn("{:<15} {:<15}".format("patience left", patience - cntr))
            print_separator(n=35, print_fn=self.print_fn)

    def attach(
        self,
        trainer: Engine,
        evaluator: Engine,
        dataloader: DataLoader,
        validation: bool = True,
    ):
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self, evaluator, dataloader, validation=validation
        )
