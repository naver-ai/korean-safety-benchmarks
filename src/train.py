import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.config.base import load_config
from src.loader import ClassificationDataModule
from src.model import ClassificationModel


def set_experiment_name(cfg):
    # set customized experiment name (useful when tuning hparams)
    name = f"""
    {cfg.task}-ep{cfg.epoch}-lr{cfg.lr}-bsz{cfg.batch_size}-gclip{cfg.gradient_clip_val}
    """.strip()
    return name


def run(cfg):
    pl.seed_everything(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_pretrained)
    datamodule = ClassificationDataModule(config=cfg, tokenizer=tokenizer)
    model = ClassificationModel(config=cfg, tokenizer=tokenizer)

    exp_name = set_experiment_name(cfg)
    wandb_logger = WandbLogger(
        name=exp_name,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
    )
    ckpt_pth = os.path.join(cfg.checkpoint_dirpath, exp_name)
    callbacks = (
        [
            ModelCheckpoint(
                dirpath=ckpt_pth,
                monitor=cfg.checkpoint_monitor,
                save_top_k=cfg.checkpoint_save_top_k,
                filename=cfg.checkpoint_filename,
            ),
            EarlyStopping(
                monitor=cfg.checkpoint_monitor,
                patience=cfg.earlystop_patience,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ]
        if not cfg.do_test_only
        else None
    )

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=os.getcwd(),
        devices=cfg.gpu_devices,
        accelerator=cfg.gpu_accelerator,
        strategy=cfg.gpu_strategy,
        amp_backend=cfg.backend,
        gradient_clip_val=cfg.gradient_clip_val,
        max_epochs=cfg.epoch,
        max_steps=cfg.max_steps,
        precision=cfg.precision,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.validation_interval,
        log_every_n_steps=cfg.log_interval,
    )

    if not cfg.do_test_only:
        trainer.fit(model, datamodule=datamodule)

    if not cfg.do_train_only:
        if cfg.model_load_ckpt_pth is not None:
            model = ClassificationModel.load_from_checkpoint(
                cfg.model_load_ckpt_pth, config=cfg, tokenizer=tokenizer
            )
            trainer.test(model, datamodule=datamodule)

            '''
            import json
            results, golds, preds = [], [], []
            for each in model.testwriter:
                for idx in range(len(each["txt"])):
                    decoded = tokenizer.decode(each["txt"][idx])
                    decoded = decoded.replace("[PAD]", "")
                    results.append({
                        "txt": decoded,
                        "pred": each["pred"][idx].item(),
                        "label": each["label"][idx].item(),
                    })
                    golds.append(each["label"][idx].item())
                    preds.append(each["pred"][idx].item())

            performance = {}
            performance["accuracy"] = accuracy_score(golds, preds)
            performance["macro_f1"] = f1_score(golds, preds, average="macro")
            performance["macro_precision"] = precision_score(golds, preds, average="macro")
            performance["macro_recall"] = recall_score(golds, preds, average="macro")
            performance["ratio_of_1"] = sum(preds) / len(preds)
            results.append(performance)
            with open(f"./{cfg.exp_name}.json", "w", encoding="utf-8") as json_file:
                json.dump(results, json_file, indent=4, ensure_ascii=False)'''
        else:
            trainer.test(datamodule=datamodule)


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
