import ipdb
import numpy
import torch
from torch.utils.data import  DataLoader, Dataset, RandomSampler

class Single_Tokenization(Dataset):
  def __init__(self, data, labels, tokenizer, max_length,args):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.args = args

  def __len__(self):
        return len(self.data)
  
  def __getitem__(self, item):
    data = self.data[item]
    labels = self.labels[item]
    if self.args.model in ['mbert_cased', 'mbert_uncased', 'xlmr_base', 'xlm' , 'xlmr_large'] :
        encoding = self.tokenizer(
        str(data),
        truncation=True,
        padding='max_length',
        max_length=self.max_length,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt'
        )
        return {
        'input_ids': encoding['input_ids'].flatten(), #flatten() = flatten()
        'attention_mask': encoding['attention_mask'].flatten(),
        'token_type_ids': encoding['token_type_ids'].flatten(),
        'labels': torch.tensor(labels, dtype=torch.long)
        }
    elif self.args.model == 'mbart':
        encoding = self.tokenizer(
        str(data),
        add_special_tokens=True,
        max_length=self.max_length,
        return_attention_mask=True,
        pad_to_max_length=True,
        truncation=True,
        return_tensors='pt',
        )
        ids = encoding['input_ids']
        masks = encoding['attention_mask']
        
        return {
            'input_ids': ids.flatten(),
            'attention_mask': masks.flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
  

class Pair_Tokenization(Dataset):
  def __init__(self, sentence1, sentence2, labels, tokenizer, max_length, args):
      self.sentence1 = sentence1
      self.sentence2 = sentence2
      self.labels = labels
      self.tokenizer = tokenizer
      self.max_length = max_length
      self.args = args
        
  def __len__(self):
        return len(self.sentence1)  
  
  def __getitem__(self, item):
        sentence1 = self.sentence1[item]
        sentence2 = self.sentence2[item]
        labels = self.labels[item]
        if self.args.task == 'mrpc' or self.args.task == 'rte' or self.args.task == 'qnli':
            labels = torch.tensor(labels, dtype=torch.long)
        elif self.args.task == 'stsb' :
            labels = torch.tensor(labels, dtype = torch.float)
        if self.args.model in ['mbert_cased', 'mbert_uncased','xlmr_base','xlm', 'xlmr_large'] :
            encoding = self.tokenizer(
                text = str(sentence1),
                text_pair = str(sentence2),
                truncation=True, 
                padding='max_length',
                max_length=self.max_length,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors='pt',
                )
            return {
                'input_ids': encoding['input_ids'].flatten(), 
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
                'labels': labels
            }
        elif self.args.model == 'mbart':
            encoding = self.tokenizer(
                text = str(sentence1),
                text_pair = str(sentence2),
                add_special_tokens=True,
                max_length=self.max_length,
                return_attention_mask=True,
                pad_to_max_length=True,
                truncation=True,
                return_tensors='pt',
                )
            ids = encoding['input_ids']
            masks = encoding['attention_mask']
            
            return {
                'input_ids': ids.flatten(),
                'attention_mask': masks.flatten(),
                'labels': labels
            }
        
class Iterator:
    
    def __init__(self, df, tokenizer, max_length, batch_size, args):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.args = args

    def single_en_loader(self):
        ds = Single_Tokenization(  
                        data = self.df['sentence'].to_numpy(),
                        labels= self.df['label'].to_numpy(),
                        tokenizer=self.tokenizer,
                        max_length=self.max_length,
                        args = self.args
                        )
        return DataLoader(ds,batch_size=self.batch_size, sampler = RandomSampler(ds)) 
    
    def single_cross_loader(self):
        ds = Single_Tokenization(  
                        data = self.df['cross'].to_numpy(),
                        labels= self.df['label'].to_numpy(),
                        tokenizer=self.tokenizer,
                        max_length=self.max_length,
                        args = self.args
                        )
        return DataLoader(ds,batch_size=self.batch_size, sampler = RandomSampler(ds)) 

    ###### mrpc(s1,s2), qqp(q1,q2), rte(s1,s2), wnli(s1,s2),stsb
    def pair_en_loader(self):
        ds = Pair_Tokenization(
                        sentence1 = self.df['sentence1'].to_numpy(),
                        sentence2 = self.df['sentence2'].to_numpy(),
                        labels= self.df['label'].to_numpy(),
                        tokenizer=self.tokenizer,
                        max_length=self.max_length,
                        args = self.args                  
                            )
        return DataLoader(ds,batch_size=self.batch_size, sampler = RandomSampler(ds)) 

    def pair_cross_loader(self):
        ds = Pair_Tokenization(
                        sentence1 = self.df['cross1'].to_numpy(),
                        sentence2 = self.df['cross2'].to_numpy(),
                        labels= self.df['label'].to_numpy(),
                        tokenizer=self.tokenizer,
                        max_length=self.max_length,
                        args = self.args                    
                            )
        return DataLoader(ds,batch_size=self.batch_size, sampler = RandomSampler(ds)) 