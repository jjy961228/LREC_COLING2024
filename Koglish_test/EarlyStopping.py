import numpy as np
import torch
import ipdb
import os
from transformers import AutoModel

class EarlyStopping:

    def __init__(self, task ,ckpt_dir, lang_type ,patience=5, verbose=False, delta=0):
        self.ckpt_dir = ckpt_dir
        self.task = task
        self.lang_type = lang_type
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        model.save_pretrained(self.ckpt_dir +'/' +self.task + '/' +self.lang_type)
        self.val_loss_min = val_loss
