import ipdb
import torch
import argparse
import json
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score, f1_score, recall_score

class TaskSpecificTrainer:

    def __init__(self,model, ckpt_savepath, lang_type,
                train_loader,
                valid_loader,
                test_loader,
                optimizer, device, scheduler, 
                args):
        
        self.model = model
        self.ckpt_savepath = ckpt_savepath
        self.lang_type = lang_type
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.args = args
                
    def train_epoch(self):
        self.model.train()
        train_losses = []
        train_total_examples = 0
        train_correct_pred= 0
        #----------train the model---------- #
        for data in tqdm(self.train_loader):
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                labels = data['labels'].to(self.device)
                if self.args.model in ['mbert_cased', 'mbert_uncased', 'xlmr_base', "xlm","xlmr_large"]:
                    token_type_ids = data['token_type_ids'].to(self.device)
                    loss, logits = self.model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids,
                                            labels=labels)
                elif self.args.model == 'mbart':
                    loss, logits = self.model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            labels=labels) 
                if self.args.task == 'sst2' or self.args.task == 'cola' or self.args.task == 'rte' or self.args.task == 'mrpc' or self.args.task == 'qnli':
                    pred_values,pred_labels = torch.max(logits ,dim=1)
                    train_correct_pred += torch.sum(pred_labels == labels)
                train_total_examples += labels.size(0)
                train_losses.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad() 
        #----------validate the model-----------#
        self.model.eval()
        valid_losses = []
        label1_logits, true_labels = [], []
        valid_correct_pred= 0
        valid_total_examples = 0
        predicted_scores, true_scores = [], []
        with torch.no_grad():
            for data in tqdm(self.valid_loader):
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                labels = data['labels'].to(self.device)
                if self.args.model in ['mbert_cased', 'mbert_uncased', 'xlmr_base', "xlm", "xlmr_large"]:
                    token_type_ids = data['token_type_ids'].to(self.device)
                    loss, logits = self.model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids,
                                            labels=labels)
                elif self.args.model == 'mbart':
                    loss, logits = self.model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            labels=labels) 
                    
                valid_total_examples += labels.size(0)
                valid_losses.append(loss.item())
                
                if self.args.task == 'sst2' or self.args.task == 'cola' or self.args.task == 'rte' or self.args.task == 'mrpc' or self.args.task == 'qnli':
                    pred_values,pred_labels = torch.max(logits ,dim=1)
                    valid_correct_pred += torch.sum(pred_labels == labels)
                    label1_logits.append(logits.detach().cpu().numpy()) 
                    true_labels.append(labels.to('cpu').numpy())
                if self.args.task == 'stsb':
                    predicted_scores.extend(logits.squeeze().cpu().numpy())
                    true_scores.extend(labels.cpu().numpy())
                    
            valid_loss = np.mean(valid_losses) 
            train_loss = np.mean(train_losses)
            print('train_total_examples: ',train_total_examples)
            print('valid_total_examples: ',valid_total_examples)
            
            if self.args.task == 'sst2' or self.args.task == 'cola' or self.args.task == 'rte' or self.args.task == 'mrpc' or self.args.task == 'qnli':
                train_acc =  train_correct_pred.double() / train_total_examples
                valid_acc = valid_correct_pred.double() / valid_total_examples
                
                return train_loss, valid_loss, train_acc, valid_acc, label1_logits, true_labels
                
            if self.args.task == 'stsb' :
                spearman_corr, pvalue = spearmanr(predicted_scores, true_scores)  
                return  train_loss,valid_loss, spearman_corr, pvalue
    
    def test_model(self):
        self.model.load_state_dict(torch.load(self.ckpt_savepath + '/'+self.args.schedular+ '_' + str(self.args.warmup_step) + '_' + str(self.args.lr)+'.pt'))
        print(self.ckpt_savepath + '/'+self.args.schedular+ '_' + str(self.args.warmup_step) + '_' + str(self.args.lr)+'.pt')
        self.model.eval()
        test_losses = []
        label1_logits, true_labels = [], []
        test_correct_pred = 0
        test_total_examples = 0
        predicted_scores, true_scores = [], []
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                labels = data['labels'].to(self.device)
                if self.args.model in ['mbert_cased', 'mbert_uncased', 'xlmr_base', "xlm","xlmr_large"]:
                    token_type_ids = data['token_type_ids'].to(self.device)
                    loss, logits = self.model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids,
                                            labels=labels)
                elif self.args.model == 'mbart':
                    loss, logits = self.model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            labels=labels) 
                test_total_examples += labels.size(0)
                test_losses.append(loss.item())
                
                if self.args.task == 'sst2' or self.args.task == 'cola' or self.args.task == 'rte' or self.args.task == 'mrpc' or self.args.task == 'qnli':
                    pred_values,pred_labels = torch.max(logits ,dim=1)
                    test_correct_pred += torch.sum(pred_labels == labels)
                    label1_logits.append(logits.detach().cpu().numpy()) 
                    true_labels.append(labels.to('cpu').numpy())
                if self.args.task == 'stsb':
                    predicted_scores.extend(logits.squeeze().cpu().numpy())
                    true_scores.extend(labels.cpu().numpy())

            print('test_total_examples : ',test_total_examples)
            test_loss = np.mean(test_losses)
            if self.args.task == 'sst2' or self.args.task == 'cola' or self.args.task == 'rte' or self.args.task == 'mrpc' or self.args.task == 'qnli':    
                test_acc = test_correct_pred.double() / test_total_examples
                return test_loss, test_acc, label1_logits, true_labels
            if self.args.task == 'stsb':
                spearman_corr, pvalue = spearmanr(predicted_scores, true_scores)  
                return test_loss, spearman_corr, pvalue