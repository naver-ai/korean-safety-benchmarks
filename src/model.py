import torch
import evaluate
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

class ClassificationModel(LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer
        num_labels = self.cfg.num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_pretrained, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        self.accuracy = evaluate.load("accuracy")
        self.testwriter = []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ClassificationModel")

        # Dataset and model
        parser.add_argument("--trn_pth", type=str)
        parser.add_argument("--val_pth", type=str)
        parser.add_argument("--tst_pth", type=str)
        parser.add_argument("--num_labels", type=int, default=2)
        parser.add_argument("--model_pretrained", type=str, default="beomi/KcELECTRA-base-v2022")
        parser.add_argument("--model_load_ckpt_pth", help="Default: none")

        # Train
        parser.add_argument("--epoch", type=int, default=10)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--lr", type=float, default=5e-5)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--backend", type=str, default="native")
        parser.add_argument("--precision", type=int, default=32)
        parser.add_argument("--max_steps", type=int, default=-1, help="Default: disabled")
        parser.add_argument(
            "--gradient_clip_val", type=float, default=0.0, help="Default: not clipping"
        )
        parser.add_argument("--accumulate_grad_batches", type=int, default=1)
        parser.add_argument("--warmup_ratio", type=float, default=0.1)

        return parent_parser

    def configure_optimizers(self):
        # optimizer
        optimizer = AdamW(self.parameters(), lr=float(self.cfg.lr))

        # scheduler with warmup
        num_devices = (
            torch.cuda.device_count()
            if self.trainer.num_devices == -1
            else int(self.trainer.num_devices)
        )
        total_steps = (
            len(self.trainer.datamodule.train_dataloader())
            // self.cfg.accumulate_grad_batches
            // num_devices
            * self.cfg.epoch
        )
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            ),
            "interval": "step",  
            "frequency": 1,
            "name": "learning_rate",
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_scores(batch)
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_scores(batch)
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_scores(batch)
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)
        return {"loss": loss, "acc": acc}

    def _get_preds_loss_scores(self, batch):
        output = self.model(
            input_ids=batch["input_ids"].squeeze(1),
            token_type_ids=batch["token_type_ids"].squeeze(1),
            attention_mask=batch["attention_mask"].squeeze(1),
            labels=batch["labels"],
        )
        preds = torch.argmax(output.logits, dim=1)
        loss = output.loss.mean()
        acc = self.accuracy.compute(
            references=batch["labels"].data, predictions=preds.data
        )

        '''# Qualitative analysis
        self.testwriter.append({
            'txt': batch['input_ids'].squeeze(1),
            'pred': preds.data,
            'label': batch['labels'].data,
        })'''
        return preds, loss, acc