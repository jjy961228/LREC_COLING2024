import ipdb
import torch
import argparse
import json
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from transformers import BertTokenizerFast
import torch.nn as nn
from scipy.stats import spearmanr

class TaskSpecificTrainer:

    def __init__(self,model, ckpt_savepath,
                train_collator,
                valid_collator,
                test_collator,
                optimizer, device, scheduler,
                args):
        
        self.model = model
        self.ckpt_savepath = ckpt_savepath
        self.train_collator = train_collator
        self.valid_collator = valid_collator
        self.test_collator = test_collator
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.args = args
                
    def train_epoch(self):
        self.train_collator.data_loaders.reset()
        self.model.train()
        train_losses = []
        train_total_examples = 0

        for data in tqdm(self.train_collator, total=len(self.train_collator) ,desc="Training"):
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                token_type_ids = data['token_type_ids'].to(self.device)

                if self.args.method == 'simcse':
                    outputs = self.model(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        token_type_ids = token_type_ids,
                                        simcse = True)
                elif self.args.method == 'ours' :
                    outputs = self.model(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        token_type_ids = token_type_ids,
                                        cross = True) 
                if outputs.logits is not None : 
                    loss, logits = outputs[0] , outputs[1]
                else : # ver1
                    loss = outputs[0]
                bs = input_ids.size(0) 
                train_total_examples += bs
                train_losses.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad() 
                
                
                
        #----------validate the model-----------#
        self.valid_collator.data_loaders.reset()
        self.model.eval()
        valid_total_examples = 0
        predicted_scores, true_scores = [], [] 
        cos_sim_fct = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            for data in tqdm(self.valid_collator, total=len(self.valid_collator) ,desc="Validation"):
                    labels = data['labels']
                    sent1_input_ids = data['input_ids'][:,0,:].to(self.device)
                    sent1_attention_mask = data['attention_mask'][:,0,:].to(self.device)
                    sent1_token_type_ids = data['token_type_ids'][:,0,:].to(self.device)
                    bs = sent1_input_ids.size(0)
                    sent1_outputs = self.model(input_ids = sent1_input_ids,
                                        attention_mask = sent1_attention_mask,
                                        token_type_ids = sent1_token_type_ids,
                                        output_hidden_states = True,
                                        return_dict = True,
                                        sent_emb = True)
                    sent2_input_ids = data['input_ids'][:,1,:].to(self.device)
                    sent2_attention_mask = data['attention_mask'][:,1,:].to(self.device)
                    sent2_token_type_ids = data['token_type_ids'][:,1,:].to(self.device)
                    sent2_outputs = self.model(input_ids = sent2_input_ids,
                                        attention_mask = sent2_attention_mask,
                                        token_type_ids = sent2_token_type_ids,
                                        output_hidden_states = True,
                                        return_dict = True,
                                        sent_emb = True)
                    z1_pooler_output = sent1_outputs['pooler_output'].cpu()
                    z2_pooler_output = sent2_outputs['pooler_output'].cpu()
                    sys_score = cos_sim_fct(z1_pooler_output, z2_pooler_output).detach().numpy()
                    predicted_scores.extend(sys_score)
                    true_scores.extend(labels.numpy())
                    valid_total_examples += bs
            spearman_corr, pvalue = spearmanr(predicted_scores, true_scores)  

            train_loss = np.mean(train_losses) 
            print('train_total_examples: ',train_total_examples)
            print('valid_total_examples: ',valid_total_examples)
            return train_loss, spearman_corr, pvalue
    
    def test_model(self):
        self.test_collator.data_loaders.reset()
        self.model.load_state_dict(torch.load(self.ckpt_savepath+'.pt'))
        self.model.eval()
        test_total_examples = 0
        predicted_scores, true_scores = [], []
        cos_sim_fct = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            for data in tqdm(self.test_collator, total=len(self.test_collator) ,desc="Test"):
                labels = data['labels']
                sent1_input_ids = data['input_ids'][:,0,:].to(self.device)
                sent1_attention_mask = data['attention_mask'][:,0,:].to(self.device)
                sent1_token_type_ids = data['token_type_ids'][:,0,:].to(self.device)
                bs = sent1_input_ids.size(0)
                sent1_outputs = self.model(input_ids = sent1_input_ids,
                                    attention_mask = sent1_attention_mask,
                                    token_type_ids = sent1_token_type_ids,
                                    output_hidden_states = True,
                                    return_dict = True,
                                    sent_emb = True)
                sent2_input_ids = data['input_ids'][:,1,:].to(self.device)
                sent2_attention_mask = data['attention_mask'][:,1,:].to(self.device)
                sent2_token_type_ids = data['token_type_ids'][:,1,:].to(self.device)
                sent2_outputs = self.model(input_ids = sent2_input_ids,
                                    attention_mask = sent2_attention_mask,
                                    token_type_ids = sent2_token_type_ids,
                                    output_hidden_states = True,
                                    return_dict = True,
                                    sent_emb = True)
                z1_pooler_output = sent1_outputs['pooler_output'].cpu()
                z2_pooler_output = sent2_outputs['pooler_output'].cpu()
                sys_score = cos_sim_fct(z1_pooler_output, z2_pooler_output).detach().numpy()
                predicted_scores.extend(sys_score)
                true_scores.extend(labels.numpy())
                test_total_examples += bs
            print('test_total_examples : ',test_total_examples)
            spearman_corr, pvalue = spearmanr(predicted_scores, true_scores) 
            return spearman_corr,pvalue