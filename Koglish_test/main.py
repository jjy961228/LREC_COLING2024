import ipdb
import torch
import argparse
import json
import random
import os
import numpy as np
from torch.utils.data import  DataLoader, RandomSampler,Dataset
from transformers import get_linear_schedule_with_warmup, AdamW, AutoConfig

from transformers import BertTokenizerFast
from transformers import MBart50TokenizerFast
from transformers import XLMRobertaTokenizer
from transformers import  XLMTokenizer

from torch import nn
import pandas as pd
import wandb


from transformers import (get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup)


from load_dataset import LoadCOLA,LoadSST2,LoadMRPC,LoadRTE,LoadSTSB, LoadQNLI
from Loader import Iterator
from model import (MbertForSequenceClassification, MbertForSequenceRegression, 
                   OurMBartForSequenceClassification, MbartForSequenceRegression,
                   OurXLMRobertaForSequenceClassification, OurXLMRobertaSequenceRegression,
                   OurXLMForSequenceClassification, OurXLMSequenceRegression)
from trainer import TaskSpecificTrainer

class Config:

    def __init__(self, config):
        self.epochs = config.get('EPOCHS')
        self.patience = config.get('PATIENCE')
        self.batch_size = config.get('BATCH_SIZE')
        self.max_length = config.get('MAX_SEQ_LEN')

class Arguments:
    def __init__(self, args):
        self.run_wandb = args.run_wandb
        self.random_seed = args.random_seed
        self.model = args.model
        self.task = args.task
        self.lang_type = args.lang_type
        self.ckpt_dir = args.ckpt_dir
        self.schedular = args.schedular
        self.warmup_step = args.warmup_step
        self.lr = args.lr
        self.eps = args.eps

class ScoreEarlyStopping:
    def __init__(self, ckpt_savepath, lang_type, args ,patience=5,verbose=False, delta=0):
        self.ckpt_savepath = ckpt_savepath
        self.lang_type = lang_type
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_max = -np.Inf  
        self.delta = delta
        self.args = args

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        if self.verbose:
            print(f'Score improved ({self.score_max:.6f} --> {score:.6f}). Saving model ...')
        torch.save(model.state_dict(), f"{self.ckpt_savepath}/{self.args.schedular+ '_' + str(self.args.warmup_step) + '_' + str(self.args.lr)}.pt")
        self.score_max = score

def fix_seed(seed):
    random.seed(seed) # 
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(seed)

def main():
    import logging
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_wandb", type=bool)
    parser.add_argument("--random_seed", type = int, default=11111)
    parser.add_argument("--model", type=str, required=True) # mBERT,mBART
    parser.add_argument("--task", type=str, required=True) 
    parser.add_argument("--lang_type", type=str,required=True)
    parser.add_argument("--ckpt_dir", type=str,default='./output')
    parser.add_argument("--schedular", type=str, default='linear') # linear , cosine, restart_cosine 
    parser.add_argument("--warmup_step", type=float, default=0.1) # 5,10,20
    parser.add_argument("--lr", type=float, default=5e-5) # 2e-5, 3e-5, 5e-5    6e-5도 추가하자
    parser.add_argument("--eps", type=float, default=1e-8) # 1e-6 , 1e-8
    args = parser.parse_args()
    if args.task in ['cola', 'sst2']:
        with open('single_sent_config.json', 'r') as f:
            config = json.load(f)
    if args.task in ['mrpc','rte','stsb','qnli']:
        with open('pair_sent_config.json', 'r') as f:
            config = json.load(f)
        
    config = Config(config) #config.json
    args = Arguments(args) #Arguments
    print(vars(config))
    print(vars(args))
    combined_config = {**vars(args), **vars(config)}

    ckpt_savepath = os.path.join(args.ckpt_dir,
                                 'seed' + str(args.random_seed),
                                 args.task,
                                 args.model,
                                 args.lang_type
                                 )
    if not os.path.isdir(ckpt_savepath):
        os.makedirs(ckpt_savepath)
        
    device = torch.device('cuda')
    fix_seed(args.random_seed)
    
    if args.run_wandb == True:
        from dotenv import load_dotenv
        load_dotenv()
        # If you are using WanDB, put your WanDB key here
        WANDB_API_KEY = 'Put in your key'
        wandb.login(key=WANDB_API_KEY)
        if args.model == 'mbert_uncased':
            wandb.init(project="mBERT_uncased",
                       name = str(args.random_seed) +'_' +args.task +'_'+ args.model +'_'+ args.lang_type,
                       entity="Put in your ID", config=combined_config)
        if args.model == 'xlm':
            wandb.init(project="XLM",
                       name = str(args.random_seed) +'_' +args.task +'_'+ args.model +'_'+ args.lang_type,
                       entity="Put in your ID", config=combined_config)
        if args.model == 'xlmr_base':
            wandb.init(project="XLM-R_base",
                       name = str(args.random_seed) +'_' +args.task +'_'+ args.model +'_'+ args.lang_type,
                       entity="Put in your ID", config=combined_config)
        if args.model == 'xlmr_large':
            wandb.init(project="XLM-R_large",
                       name = str(args.random_seed) +'_' +args.task +'_'+ args.model +'_'+ args.lang_type,
                       entity="Put in your ID", config=combined_config)
        if args.model == 'mbart':
            wandb.init(project="mBART",
                       name = str(args.random_seed) +'_' +args.task +'_'+ args.model +'_'+ args.lang_type,
                       entity="Put in your ID", config=combined_config)
        
        
    if args.model == 'mbert_uncased':
        pretrained_model_name_or_path = "bert-base-multilingual-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    if args.model == 'xlmr_base':
        pretrained_model_name_or_path = "xlm-roberta-large"
        tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path)
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    if args.model == 'xlmr_large':
        pretrained_model_name_or_path = "xlm-roberta-large"
        tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path)
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    if args.model == 'mbart':
        pretrained_model_name_or_path = "facebook/mbart-large-50"
        tokenizer = MBart50TokenizerFast.from_pretrained(pretrained_model_name_or_path)
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    if args.model == 'xlm':
        pretrained_model_name_or_path = "xlm-mlm-100-1280"
        tokenizer = XLMTokenizer.from_pretrained(pretrained_model_name_or_path)
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
                        
                        
    if args.task == 'cola':
        if args.lang_type == 'en2en':
            train_data, valid_data, test_data = LoadCOLA.en2en()
        elif args.lang_type == 'en2cross':
            train_data, valid_data, test_data = LoadCOLA.en2cross()
        elif args.lang_type == 'cross2cross':
            train_data, valid_data, test_data = LoadCOLA.cross2cross()        
    
    if args.task == 'sst2':
        if args.lang_type == 'en2en':
            train_data, valid_data, test_data = LoadSST2.en2en()
        elif args.lang_type == 'en2cross':
            train_data, valid_data, test_data = LoadSST2.en2cross()
        elif args.lang_type == 'cross2cross':
            train_data, valid_data, test_data = LoadSST2.cross2cross()
    
    if args.task == 'mrpc':
        if args.lang_type == 'en2en':
            train_data, valid_data, test_data = LoadMRPC.en2en()
        elif args.lang_type == 'en2cross':
            train_data, valid_data, test_data = LoadMRPC.en2cross()
        elif args.lang_type == 'cross2cross':
            train_data, valid_data, test_data = LoadMRPC.cross2cross()

    if args.task == 'rte':
        if args.lang_type == 'en2en':
            train_data, valid_data, test_data = LoadRTE.en2en()
        elif args.lang_type == 'en2cross':
            train_data, valid_data, test_data = LoadRTE.en2cross()
        elif args.lang_type == 'cross2cross':
            train_data, valid_data, test_data = LoadRTE.cross2cross()
            
    if args.task == 'stsb':
        if args.lang_type == 'en2en':
            train_data, valid_data, test_data = LoadSTSB.en2en()
        elif args.lang_type == 'en2cross':
            train_data, valid_data, test_data = LoadSTSB.en2cross()
        elif args.lang_type == 'cross2cross':
            train_data, valid_data, test_data = LoadSTSB.cross2cross()
    
    if args.task == 'qnli':
        if args.lang_type == 'en2en':
            train_data, valid_data, test_data = LoadQNLI.en2en()
        elif args.lang_type == 'en2cross':
            train_data, valid_data, test_data = LoadQNLI.en2cross()
        elif args.lang_type == 'cross2cross':
            train_data, valid_data, test_data = LoadQNLI.cross2cross()

    print('len(train): ',len(train_data), train_data.columns)
    print('len(valid): ',len(valid_data), valid_data.columns)
    print('len(test): ',len(test_data), test_data.columns)
    if args.model in ['mbert_uncased', 'xlmr_base' ,'mbart', 'xlm', 'xlmr_large'] and (args.task == 'cola' or args.task == 'sst2'):
        train_iter = Iterator(train_data, tokenizer, config.max_length, config.batch_size, args)
        valid_iter = Iterator(valid_data, tokenizer, config.max_length, config.batch_size, args)
        test_iter = Iterator(test_data, tokenizer, config.max_length, config.batch_size, args)
        if args.lang_type == 'en2en':
            train_loader = train_iter.single_en_loader()
            valid_loader = valid_iter.single_en_loader()
            test_loader = test_iter.single_en_loader()
        elif args.lang_type == 'en2cross':
            train_loader = train_iter.single_en_loader()
            valid_loader = valid_iter.single_en_loader()
            test_loader = test_iter.single_cross_loader()
        elif args.lang_type == 'cross2cross':
            train_loader = train_iter.single_cross_loader()
            valid_loader = valid_iter.single_cross_loader()
            test_loader = test_iter.single_cross_loader() 
            
    if args.model in ['mbert_uncased', 'xlmr_base' ,'mbart', 'xlm', 'xlmr_large'] and (args.task == 'mrpc' or args.task =='rte' or args.task == 'stsb' or args.task == 'qnli'):
        train_iter = Iterator(train_data, tokenizer, config.max_length, config.batch_size, args)
        valid_iter = Iterator(valid_data, tokenizer, config.max_length, config.batch_size, args)
        test_iter = Iterator(test_data, tokenizer, config.max_length, config.batch_size, args)
        if args.lang_type == 'en2en':
            train_loader = train_iter.pair_en_loader()
            valid_loader = valid_iter.pair_en_loader()
            test_loader = test_iter.pair_en_loader()
        elif args.lang_type == 'en2cross':
            train_loader = train_iter.pair_en_loader()
            valid_loader = valid_iter.pair_en_loader()
            test_loader = test_iter.pair_cross_loader()
        elif args.lang_type == 'cross2cross':
            train_loader = train_iter.pair_cross_loader()
            valid_loader = valid_iter.pair_cross_loader()
            test_loader = test_iter.pair_cross_loader()
    print('===========Data_Loader_Examples==========')
    print(f'task : {args.task} / lang_type : {args.lang_type}')
    print('=====Train_Loader=====')
    print(tokenizer.batch_decode(next(iter(train_loader))['input_ids'][0]))
    print('=====Valid_Loader=====')
    print(tokenizer.batch_decode(next(iter(valid_loader))['input_ids'][0]))
    print('=====Test_Loader=====')
    print(tokenizer.batch_decode(next(iter(test_loader))['input_ids'][0]))
    ## mBERT
    if args.model in ['mbert_cased', 'mbert_uncased'] and (args.task == 'cola' or args.task == 'sst2' or args.task == 'mrpc' or args.task == 'rte' or args.task == 'qnli') :
        num_labels = 2
        model = MbertForSequenceClassification(args, 
                                               num_labels=num_labels)
    if args.model in ['mbert_cased', 'mbert_uncased'] and (args.task == 'stsb'):
        num_labels = 1
        model = MbertForSequenceRegression(args,
                                           num_labels = num_labels)
    ## XLM-R
    if (args.model in 'xlmr_base' or args.model in 'xlmr_large') and (args.task == 'cola' or args.task == 'sst2'or args.task == 'mrpc' or args.task == 'rte' or args.task == 'qnli'):
        num_labels = 2
        model = OurXLMRobertaForSequenceClassification(args = args,
                                                num_labels = num_labels)
    if (args.model in 'xlmr_base' or args.model in 'xlmr_large') and (args.task == 'stsb'):
        num_labels = 1
        model = OurXLMRobertaSequenceRegression(args = args,
                                                num_labels = num_labels)
    ## mBART
    if args.model in 'mbart' and (args.task == 'cola' or args.task == 'sst2'or args.task == 'mrpc' or args.task == 'rte' or args.task == 'qnli'):
        num_labels = 2
        model = OurMBartForSequenceClassification(args = args,
                                                num_labels = num_labels)
    if args.model in 'mbart' and (args.task == 'stsb'):
        num_labels = 1
        model = MbartForSequenceRegression(args = args,
                                                num_labels = num_labels)
    
    ## XLM
    if args.model in 'xlm' and (args.task == 'cola' or args.task == 'sst2'or args.task == 'mrpc' or args.task == 'rte' or args.task == 'qnli'):
        num_labels = 2
        model = OurXLMForSequenceClassification(args = args,
                                                num_labels = num_labels)
    if args.model in 'xlm' and (args.task == 'stsb'):
        num_labels = 1
        model = OurXLMSequenceRegression(args = args,
                                                num_labels = num_labels)
    

    
    model.to(device)
    optimizer = AdamW(model.parameters(),
                    lr = args.lr,
                    eps = args.eps
                    )
    total_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(args.warmup_step  * total_steps)

    if args.schedular == 'linear' : 
        schedular_type = get_linear_schedule_with_warmup
    elif args.schedular == 'cosine' :
        schedular_type = get_cosine_schedule_with_warmup
    elif args.schedular == 'restart_cosine' :
        schedular_type = get_cosine_with_hard_restarts_schedule_with_warmup
    scheduler = schedular_type(optimizer, 
                                num_warmup_steps = num_warmup_steps,
                                num_training_steps = total_steps)
    taskSpecificTrainer = TaskSpecificTrainer(model= model,
                                              ckpt_savepath=ckpt_savepath,
                                              lang_type = args.lang_type,
                                              train_loader=train_loader,
                                              valid_loader=valid_loader,
                                              test_loader=test_loader,
                                              optimizer=optimizer,
                                              device=device,
                                              scheduler=scheduler,
                                              args = args)
    early_stopping = ScoreEarlyStopping(ckpt_savepath = ckpt_savepath,
                                        lang_type = args.lang_type,
                                        patience= config.patience,
                                        args = args,
                                        verbose=True)
 
    ##Train
    prt_task = str(args.random_seed) +'_' + args.model+ '_' + args.task + '_' + args.lang_type + '_' + args.schedular + '_step' + str(args.warmup_step) + '_lr' + str(args.lr)
    
    for epoch in range(config.epochs):
            print(f'Epoch:{epoch+1}/{config.epochs}')
            print(f'----------Train roop: {prt_task}-----------')
            
            if args.task == 'sst2' or args.task == 'cola' or args.task == 'rte' or args.task == 'mrpc' or args.task == 'qnli':
                from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score, f1_score, recall_score
                train_loss ,valid_loss, train_acc, valid_acc, label1_logits, true_labels = taskSpecificTrainer.train_epoch()
                predictions = np.concatenate(label1_logits, axis=0) 
                true_labels = np.concatenate(true_labels, axis=0)
                pred_probas = torch.softmax(torch.tensor(predictions), dim=-1)[:,1] 
                threshold = 0.5
                pred_labels = np.where(pred_probas >= threshold , 1, 0)
                auc_score = roc_auc_score(true_labels, pred_probas)
                auprc_score = average_precision_score(true_labels, pred_probas)
                accuracy = accuracy_score(true_labels, pred_labels)
                recall = recall_score(true_labels, pred_labels)
                f1 = f1_score(true_labels, pred_labels)
                confunsion = confusion_matrix(true_labels, pred_labels)

                if args.task == 'sst2' or args.task == 'cola' or args.task == 'rte' or args.task == 'qnli':
                    early_stopping(valid_acc, model)
                elif args.task == 'mrpc':
                    early_stopping(f1, model)    
                    
            elif args.task == 'stsb':
                train_loss ,valid_loss,spearman_corr,pvalue = taskSpecificTrainer.train_epoch()
                early_stopping(spearman_corr, model)
            
            if early_stopping.early_stop :
                print("Early stopping")
                break

            if args.task == 'sst2' or args.task == 'cola' or args.task == 'rte' or args.task == 'mrpc' or args.task == 'qnli':
                print(f'Train_loss: {train_loss}/ Train_acc : {train_acc}')
                print(f'Valid_loss : {valid_loss} / Valid_acc : {valid_acc}')
                if args.run_wandb == True:
                    wandb.log({'Train_loss': train_loss,'Train_acc': train_acc, 
                               'Valid_loss': valid_loss,'Valid_acc': valid_acc})
            if args.task == 'stsb':
                print(f'Train_loss: {train_loss}')
                print(f'Vrain_loss: {train_loss}')
                print(f'Spearman_corr: {spearman_corr}')
                print(f'P-Value: {pvalue}')
                if args.run_wandb == True:
                    wandb.log({'Train_loss': train_loss, 
                               'Valid_loss': valid_loss,
                               'Spearman_corr': spearman_corr,
                               'P-Value': pvalue})
    
    ##Test
    print(f'----------Test: {prt_task}-----------')
    if args.task == 'sst2' or args.task == 'cola' or args.task == 'rte' or args.task == 'mrpc' or args.task == 'qnli':
        test_loss, test_acc, label1_logits, true_labels = taskSpecificTrainer.test_model()
        print(f'Test_loss : {test_loss} / Test_acc: {test_acc} ')
 
        from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score, f1_score, recall_score
        predictions = np.concatenate(label1_logits, axis=0) 
        true_labels = np.concatenate(true_labels, axis=0)
        pred_probas = torch.softmax(torch.tensor(predictions), dim=-1)[:,1] 
        threshold = 0.5
        pred_labels = np.where(pred_probas >= threshold , 1, 0)
        auc_score = roc_auc_score(true_labels, pred_probas)
        auprc_score = average_precision_score(true_labels, pred_probas)
        accuracy = accuracy_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        confunsion = confusion_matrix(true_labels, pred_labels)
        print('=====AUC score:===== \n', auc_score,'\n')
        print('=====AUPRC score:===== \n', auprc_score,'\n')
        print('=====Accuracy score:===== \n', accuracy,'\n')
        print('=====recall score:===== \n', recall,'\n')
        print('=====f1 score:===== \n', f1,'\n')
        print('=====confusion matrix:===== \n', confunsion,'\n')

        if args.run_wandb == True:
            wandb.log({'test_acc': test_acc,'test_loss': test_loss})
            wandb.log({
                        'AUROC': auc_score,
                        'AUPRC': auprc_score,
                        'ACC': accuracy,
                        'Recall': recall,
                        'F1': f1,})

    elif args.task == 'stsb':
        test_loss, spearman_corr, pvalue = taskSpecificTrainer.test_model()
        print(f'test_loss: {test_loss}')
        print(f'Spearman_corr: {spearman_corr}')
        print(f'P-Value: {pvalue}')
        if args.run_wandb == True:
            wandb.log({'Test_loss': test_loss, 
                        'Spearman_corr': spearman_corr,
                        'P-Value': pvalue})
if __name__ == "__main__":
    main()



    