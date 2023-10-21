import ipdb
import torch
import argparse
import json
import random
import os
import numpy as np
from torch.utils.data import  DataLoader, RandomSampler,Dataset
from transformers import BertTokenizerFast,get_linear_schedule_with_warmup, AdamW, AutoConfig
from transformers import (get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup)
from torch import nn
import pandas as pd
from transformers import XLMRobertaTokenizer, RobertaTokenizer
import logging
from typing import Optional


## Customized packages
from load_dataset import LoadNLI,LoadSTSB, TestLoadSTS_12, TestLoadSTS_13, TestLoadSTS_14, TestLoadSTS_15, TestLoadSTS_16,TestLoadSICK
from Loader import Iterator,CustomCollator,CustomDataLoader
from model import BertForCL,RobertaForCL
from trainer import TaskSpecificTrainer

class TrainingConfig:

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
        self.lang_type = args.lang_type
        self.ckpt_dir = args.ckpt_dir
        self.pooler_type = args.pooler_type      
        self.hard_negative_weight = args.hard_negative_weight    
        self.method = args.method  
        self.ours_version = args.ours_version
        self.task = args.task
        self.eval_type = args.eval_type
        self.temp = args.temp
        self.schedular = args.schedular
        self.warmup_step = args.warmup_step
        self.lr = args.lr
        self.eps = args.eps
        self.margin = args.margin
        self.triplet = args.triplet

class EarlyStopping:
    def __init__(self, ckpt_savepath, lang_type, patience=5, verbose=False, delta=0):
        self.ckpt_savepath = ckpt_savepath
        self.lang_type = lang_type
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.spearman_corr_max = -np.Inf
        self.delta = delta

    def __call__(self, spearman_corr, model):
        score = spearman_corr

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(spearman_corr, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(spearman_corr, model)
            self.counter = 0

    def save_checkpoint(self, spearman_corr, model):
        if self.verbose:
            print(f'Spearman correlation increased ({self.spearman_corr_max:.6f} --> {spearman_corr:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.ckpt_savepath +'.pt')
        self.spearman_corr_max = spearman_corr

def fix_seed(seed):
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_wandb", 
                        type=bool)
    parser.add_argument("--random_seed", 
                        type=int, 
                        default=11111) 
    parser.add_argument("--model", # Encoder_base_model
                        type=str, 
                        required=True)  
    parser.add_argument("--eval_type", 
                        type=str,
                        required= True)
    parser.add_argument("--method",
                        type=str,
                        required=True) 
    parser.add_argument("--task",
                        type=str,
                        required=True) 
    parser.add_argument("--ours_version", # if ours_version == 999 : ConCSE
                        type = str)
    parser.add_argument("--lang_type", 
                        type=str,
                        required=True)
    parser.add_argument("--ckpt_dir",
                        type=str,default='./output')
    parser.add_argument("--pooler_type",
                        type=str,
                        default='cls')
    parser.add_argument("--hard_negative_weight",
                        type=int,
                        default=0)
    parser.add_argument("--temp",
                        type=float,
                        default=0.05)
    parser.add_argument("--schedular",
                        type=str,
                        default='linear')  
    parser.add_argument("--warmup_step",
                        type=float,
                        default=0.1) 
    parser.add_argument("--lr",
                        type=float,
                        required=True) 
    parser.add_argument("--eps",
                        type=float,
                        default=1e-8) 
    parser.add_argument("--triplet" ,
                        type = float)
    parser.add_argument("--margin",
                        type=float) 
    
    args = parser.parse_args()
    args = Arguments(args) 
    
    ## simcse and base 
    if args.method == 'simcse' and (args.model == 'mbert_uncased' or args.model == 'xlmr_base'):
        with open('config_simcse_base.json', 'r') as f:
            train_config = json.load(f)
    ## ours and base
    elif args.method == 'ours' and (args.model == 'mbert_uncased' or args.model == 'xlmr_base'):
        with open('config_ours_base.json', 'r') as f:
            train_config = json.load(f)
            
    ## simcse and large
    elif args.method == 'simcse' and (args.model == 'xlmr_large'):
        with open('config_simcse_large.json', 'r') as f:
            train_config = json.load(f)
    ## ours and large
    elif args.method == 'ours' and (args.model == 'xlmr_large'):
        with open('config_ours_large.json', 'r') as f:
            train_config = json.load(f)    
            
    train_config = TrainingConfig(train_config) 
    combined_config = {**vars(args), **vars(train_config)}
    print('======random_seed====== :  ', args.random_seed)
    print(vars(train_config))
    print(vars(args))
    if ((args.method == 'simcse' and args.ours_version is not None) or 
        (args.method == 'ours' and args.ours_version is None)):
        print("Exiting.")
        import sys
        sys.exit()
    
    # ConCSE
    if args.ours_version is not None :
        save_dir = os.path.join(args.ckpt_dir,'seed' + str(args.random_seed),
                                args.eval_type
                                ,args.method,args.model, args.task , args.lang_type, 'temp' + str(args.temp))
        save_filename = 'ver' + str(args.ours_version) + '_' + args.schedular + '_warm_' + str(args.warmup_step) + '_lr_' + str(args.lr) + '_margin_' + str(args.margin) + '_triplet_' + str(args.triplet)
        namer = ('seed' + str(args.random_seed) + '_'+args.eval_type + '_'+args.method + '_'
            + args.model +'_'+args.task +'_' +args.lang_type
            + '_temp' + str(args.temp) + '_ver' + str(args.ours_version) + '_' + args.schedular 
            + '_warm_' + str(args.warmup_step) + '_lr_' + str(args.lr) 
            + '_margin_' + str(args.margin) + '_triplet_' + str(args.triplet))

    
    # SimCSE 
    else:
        save_dir = os.path.join(args.ckpt_dir,'seed'+str(args.random_seed),
                                args.eval_type
                                ,args.method, args.model, args.task ,args.lang_type)
        save_filename = '_E' + str(train_config.epochs) + '_BS' + str(train_config.batch_size) 
        namer = 'seed' + str(args.random_seed) +'_'+args.eval_type + '_'+args.method + '_' + args.model + '_' +args.task +'_' +args.lang_type + '_E' + str(train_config.epochs) + '_BS' + str(train_config.batch_size)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    ckpt_savepath = os.path.join(save_dir, save_filename)
    device = torch.device('cuda')
    fix_seed(args.random_seed)
    
    # WanDB setting
    if args.run_wandb == True:
        try:
            import wandb
            from dotenv import load_dotenv
            load_dotenv()
            # If you are using WanDB, put your WanDB key here
            WANDB_API_KEY = 'Put in your key'
            wandb.login(key=WANDB_API_KEY)
        
            ## mbert
            if args.model == 'mbert_uncased' and args.eval_type == 'transfer' and (args.method == 'simcse' or args.method == 'ours'):
                wandb.init(project="mBERT_uncased",
                        name = namer,
                        entity="Put in your ID", config=combined_config)
            
            ## roberta
            if 'xlmr_base' in args.model and args.eval_type == 'transfer' and (args.method == 'simcse' or args.method == 'ours'):
                wandb.init(project="XLM-R_base",
                        name = namer,
                        entity="Put in your ID", config=combined_config)
            
            if 'xlmr_large' in args.model and args.eval_type == 'transfer' and (args.method == 'simcse' or args.method == 'ours'):
                wandb.init(project="XLM-R_large",
                        name = namer,
                        entity="Put in your ID", config=combined_config)

        except Exception as e:
            print(f'wandb error:{e}')
            pass
 
    ## mBERT
    if args.model == 'mbert_cased':
        pretrained_model_name_or_path = "bert-base-multilingual-cased"
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    if args.model == 'mbert_uncased':
        pretrained_model_name_or_path = "bert-base-multilingual-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        
    ## XLM-R
    if args.model == 'xlmr_base':
        pretrained_model_name_or_path = "xlm-roberta-base"
        tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path)
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    if args.model == 'xlmr_large':
        pretrained_model_name_or_path = "xlm-roberta-large"
        tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path)
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

    if args.eval_type == 'transfer':
            if args.lang_type == 'en2en':
                train_data = LoadNLI.en_train()
                valid_data = LoadSTSB.en_valid()
                if args.task == 'stsb':
                    test_data = LoadSTSB.en_test()
                elif args.task == 'sts12':
                    testloadSTS_12 = TestLoadSTS_12(args)
                    test_data = testloadSTS_12.en_test()
                elif args.task == 'sts13':
                    testloadSTS_13 = TestLoadSTS_13(args)
                    test_data = testloadSTS_13.en_test()
                elif args.task == 'sts14':
                    testloadSTS_14 = TestLoadSTS_14(args)
                    test_data = testloadSTS_14.en_test()
                elif args.task == 'sts15':
                    testloadSTS_15 = TestLoadSTS_15(args)
                    test_data = testloadSTS_15.en_test()
                elif args.task == 'sts16':
                    testloadSTS_16 = TestLoadSTS_16(args)
                    test_data = testloadSTS_16.en_test()
                elif args.task == 'sick':
                    testLoadSICK = TestLoadSICK(args)
                    test_data = testLoadSICK.en_test()
    
            if args.lang_type == 'en2cross':
                train_data = LoadNLI.en_train()
                valid_data = LoadSTSB.en_valid()
                if args.task == 'stsb':
                    test_data = LoadSTSB.cross_test()
                elif args.task == 'sts12':
                    testLoadSTS_12 = TestLoadSTS_12(args)
                    test_data = testLoadSTS_12.cross_test()
                elif args.task == 'sts13':
                    testLoadSTS_13 = TestLoadSTS_13(args)
                    test_data = testLoadSTS_13.cross_test()
                elif args.task == 'sts14':
                    testLoadSTS_14 = TestLoadSTS_14(args)
                    test_data = testLoadSTS_14.cross_test()
                elif args.task == 'sts15':
                    testLoadSTS_15 = TestLoadSTS_15(args)
                    test_data = testLoadSTS_15.cross_test()
                elif args.task == 'sts16':
                    testLoadSTS_16 = TestLoadSTS_16(args)
                    test_data = testLoadSTS_16.cross_test()
                elif args.task == 'sick':
                    testLoadSICK = TestLoadSICK(args)
                    test_data = testLoadSICK.cross_test()
        
            if args.lang_type == 'cross2cross':
                train_data = LoadNLI.cross_train()
                valid_data = LoadSTSB.cross_valid()
                if args.task == 'stsb':
                    test_data = LoadSTSB.cross_test()
                elif args.task == 'sts12':
                    testLoadSTS_12 = TestLoadSTS_12(args)
                    test_data = testLoadSTS_12.cross_test()
                elif args.task == 'sts13':
                    testLoadSTS_13 = TestLoadSTS_13(args)
                    test_data = testLoadSTS_13.cross_test()
                elif args.task == 'sts14':
                    testLoadSTS_14 = TestLoadSTS_14(args)
                    test_data = testLoadSTS_14.cross_test()
                elif args.task == 'sts15':
                    testLoadSTS_15 = TestLoadSTS_15(args)
                    test_data = testLoadSTS_15.cross_test()
                elif args.task == 'sts16':
                    testLoadSTS_16 = TestLoadSTS_16(args)
                    test_data = testLoadSTS_16.cross_test()
                elif args.task == 'sick':
                    testLoadSICK = TestLoadSICK(args)
                    test_data = testLoadSICK.cross_test()
    
    print('len(train): ',len(train_data), train_data.columns)
    print('len(valid): ',len(valid_data), valid_data.columns)
    print('len(test): ',len(test_data), test_data.columns)
    
    train_iter = Iterator(train_data, tokenizer, train_config, args)
    train_data_loaders = CustomDataLoader(train_iter, train_config, args, iter_type='train')
    train_collator = CustomCollator(train_data_loaders, args, iter_type='train')
    
    valid_iter = Iterator(valid_data, tokenizer, train_config, args)
    valid_data_loaders = CustomDataLoader(valid_iter, train_config, args, iter_type='valid')
    valid_collator = CustomCollator(valid_data_loaders, args, iter_type='valid')
    
    test_iter = Iterator(test_data, tokenizer, train_config, args)
    test_data_loaders = CustomDataLoader(test_iter, train_config, args, iter_type='test')
    test_collator = CustomCollator(test_data_loaders, args, iter_type='test')
    print('===========Collator&DataLoader_Examples==========')
    print(f'lang_type : {args.lang_type}_{args.method}')
    print('=====Train_Collator=====')
    print(tokenizer.batch_decode(train_collator.__next__()['input_ids'][0])) 
    print('=====Valid_collator=====')
    print(tokenizer.batch_decode(valid_collator.__next__()['input_ids'][0])) 
    print('=====Test_collator=====')
    print(tokenizer.batch_decode(test_collator.__next__()['input_ids'][0])) 
    if args.model == 'mbert_uncased' or args.model == 'mbert_cased':
        model = BertForCL.from_pretrained(pretrained_model_name_or_path = pretrained_model_name_or_path,
                                        args = args ,
                                        model_config = model_config)
    if 'xlmr' in args.model:
        model = RobertaForCL.from_pretrained(pretrained_model_name_or_path = pretrained_model_name_or_path,
                                            args = args ,
                                            model_config = model_config)
    model.to(device)
    optimizer = AdamW(model.parameters(),
                    lr = args.lr,
                    eps = args.eps
                    )
    total_steps = len(train_data_loaders) * train_config.epochs
    num_warmup_steps = int(args.warmup_step * total_steps)
    
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
                                              train_collator=train_collator,
                                              valid_collator=valid_collator,
                                              test_collator=test_collator,
                                              optimizer=optimizer,
                                              device=device,
                                              scheduler=scheduler,
                                              args = args)
    early_stopping = EarlyStopping(ckpt_savepath=ckpt_savepath,
                                lang_type=args.lang_type,
                                patience=train_config.patience,
                                verbose=True)
    ##Train
    prt_task = namer
    for epoch in range(train_config.epochs):
        print(f'Epoch:{epoch+1}/{train_config.epochs}')
        print(f'----------Train loop: {prt_task}-----------')
        train_loss, spearman_corr, pvalue = taskSpecificTrainer.train_epoch()

        early_stopping(spearman_corr, model) # Validate and check early stopping
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print(f'Train_loss: {train_loss}')
        print(f'spearman_corr: {spearman_corr}')
        print(f'P-Value: {pvalue}')
        if args.run_wandb == True:
            try:
                wandb.log({'Train_loss': train_loss, 
                            'Spearman_corr': spearman_corr,
                            'P-Value': pvalue})
            except Exception as e:
                print(f'wandb error:{e}')
                pass
    
    spearman_corr, pvalue = taskSpecificTrainer.test_model()
    print(f'Spearman_corr: {spearman_corr}')
    print(f'P-Value: {pvalue}')
    if args.run_wandb == True:
        try :
            wandb.log({'Spearman_corr': spearman_corr,
                        'P-Value': pvalue})
        except Exception as e:
            print(f'wandb error:{e}')
            pass
    
if __name__ == "__main__":
    main()



    