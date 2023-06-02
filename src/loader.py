import json
from copy import deepcopy

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

context_dict = {}

class BinaryClassificationDataset(Dataset):
    def __init__(self, task, tok, data):
        self.task = task
        self.tok = tok
        self.data = self.prepare_data(data)
 
    def prepare_data(self, data):
        # exclude "I don't know" annotations
        return [dd for dd in data if dd[self.tgt_name] in [0,1]]

    def __getitem__(self, idx):
        sample = self.data[idx]
        src_text = ' '.join([sample[src] for src in self.src_name])
        src = self.tok(
            src_text, truncation=True, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": src["input_ids"],
            "token_type_ids": src["token_type_ids"],
            "attention_mask": src["attention_mask"],
            "labels": torch.tensor(sample[self.tgt_name]),
        }

    def __len__(self):
        return len(self.data)

class SubjectiveQuestionClassification(BinaryClassificationDataset):
    def __init__(self, task, tok, data):
        self.src_name = ["question"]
        self.tgt_name = "subjective?"
        super().__init__(task, tok, data)

class AcceptableResponseClassification(BinaryClassificationDataset):
    def __init__(self, task, tok, data):
        self.src_name = ["question", "response"]
        self.tgt_name = "acceptable?"
        super().__init__(task, tok, data)

class ContextIncludeGrpDataset(BinaryClassificationDataset):
    def __init__(self, task, tok, data):
        self.src_name = ["demographic_group", "context"]
        self.tgt_name = "include?"
        super().__init__(task, tok, data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        src_text = f' {self.tok.sep_token} '.join([sample[src] for src in self.src_name])
        src = self.tok(
            src_text, truncation=True, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": src["input_ids"],
            "token_type_ids": src["token_type_ids"],
            "attention_mask": src["attention_mask"],
            "labels": torch.tensor(sample[self.tgt_name]),
        }

class SafeSentenceClassification(BinaryClassificationDataset):
    def __init__(self, task, tok, data):
        self.src_name = ["context", "sentence"]
        self.tgt_name = "sentence_label"
        super().__init__(task, tok, data)

    def prepare_data(self, data):
        # actually, do nothing?
        return [dd for dd in data if dd[self.tgt_name] in ['safe','unsafe']]

    def __getitem__(self, idx):
        sample = self.data[idx]
        src_text = ' '.join([sample[src] for src in self.src_name])
        src = self.tok(
            src_text, truncation=True, padding="max_length", return_tensors="pt"
        )
        label = 1 if sample[self.tgt_name] == "safe" else 0
        return {
            "input_ids": src["input_ids"],
            "token_type_ids": src["token_type_ids"],
            "attention_mask": src["attention_mask"],
            "labels": torch.tensor(label),
        }

class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer
        
        self.trn_pth = self.cfg.trn_pth if self.cfg.trn_pth is not None else None
        self.val_pth = self.cfg.val_pth if self.cfg.val_pth is not None else None
        self.tst_pth = self.cfg.tst_pth if self.cfg.tst_pth is not None else None
        
    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            with open(self.trn_pth, "r") as json_file:
                trn_data = json.load(json_file)
            with open(self.val_pth, "r") as json_file:
                val_data = json.load(json_file)
            
            if "square_subj_q_classify" in self.cfg.task:
                self.trn_dset = SubjectiveQuestionClassification(
                    "square_subj_q_classify", self.tokenizer, trn_data)
                self.val_dset = SubjectiveQuestionClassification(
                    "square_subj_q_classify", self.tokenizer, val_data)
            elif "square_accept_r_classify" in self.cfg.task:
                self.trn_dset = AcceptableResponseClassification(
                    "square_accept_r_classify", self.tokenizer, trn_data)
                self.val_dset = AcceptableResponseClassification(
                    "square_accept_r_classify", self.tokenizer, val_data)
            elif "kosbi_incl_grp_classify" in self.cfg.task:
                self.trn_dset = ContextIncludeGrpDataset(
                    "kosbi_incl_grp_classify", self.tokenizer, trn_data)
                self.val_dset = ContextIncludeGrpDataset(
                    "kosbi_incl_grp_classify", self.tokenizer, val_data)
            elif "kosbi_unsafe_sent_classify" in self.cfg.task:
                self.trn_dset = SafeSentenceClassification(
                    "kosbi_unsafe_sent_classify", self.tokenizer, trn_data)
                self.val_dset = SafeSentenceClassification(
                    "kosbi_unsafe_sent_classify", self.tokenizer, val_data)
            else:
                raise NotImplementedError("invalid task")

        if stage == "test":
            with open(self.tst_pth, "r") as json_file:
                tst_data = json.load(json_file)

            if "square_subj_q_classify" in self.cfg.task:
                self.tst_dset = SubjectiveQuestionClassification(
                    "square_subj_q_classify", self.tokenizer, tst_data)
            elif "square_accept_r_classify" in self.cfg.task:
                self.tst_dset = AcceptableResponseClassification(
                    "square_accept_r_classify", self.tokenizer, tst_data)
            elif "kosbi_incl_grp_classify" in self.cfg.task:
                self.tst_dset = ContextIncludeGrpDataset(
                    "kosbi_incl_grp_classify", self.tokenizer, tst_data)
            elif "kosbi_unsafe_sent_classify" in self.cfg.task:
                self.tst_dset = SafeSentenceClassification(
                    "kosbi_unsafe_sent_classify", self.tokenizer, tst_data)
            else:
                raise NotImplementedError("invalid task")

    def train_dataloader(self):
        return DataLoader(
            self.trn_dset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tst_dset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
        )
