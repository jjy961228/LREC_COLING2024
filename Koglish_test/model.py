import ipdb
import torch
import numpy as np
from transformers import BertForSequenceClassification, MBartForSequenceClassification, XLMRobertaForSequenceClassification, XLMForSequenceClassification
from torch import nn


## mBERT
class MbertForSequenceClassification(nn.Module):
    def __init__(self,args ,num_labels):
        super(MbertForSequenceClassification, self).__init__()
        if args.model == 'mbert_cased':
            self.mbert = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)
        if args.model == 'mbert_uncased':
            self.mbert = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=num_labels)

    def forward(self, input_ids, attention_mask,token_type_ids ,labels): 
        loss,logits = self.mbert(
          input_ids = input_ids,
          attention_mask = attention_mask,
          token_type_ids=token_type_ids,
          labels = labels
          )[:2]      
        return loss,logits

class MbertForSequenceRegression(nn.Module):
     def __init__(self,args,num_labels):
        super(MbertForSequenceRegression, self).__init__()
        if args.model == 'mbert_cased':
            self.mbert = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",num_labels=num_labels)
        if args.model == 'mbert_uncased':
            self.mbert = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased",num_labels=num_labels)
        self.loss_fct = nn.MSELoss()
     def forward(self, input_ids, attention_mask,token_type_ids, labels): 
        outputs = self.mbert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids=token_type_ids,
            )
        logits = outputs['logits']
        loss = self.loss_fct(logits.squeeze(),labels.squeeze())    
        return loss,logits

## mBART
class OurMBartForSequenceClassification(nn.Module):
    def __init__(self,args ,num_labels):
        super(OurMBartForSequenceClassification, self).__init__()
        if args.model == 'mbart':
            self.mbart = MBartForSequenceClassification.from_pretrained("facebook/mbart-large-50", num_labels=num_labels)

    def forward(self, input_ids, attention_mask,labels): 
        loss,logits = self.mbart(
          input_ids = input_ids,
          attention_mask = attention_mask,
          labels = labels
          )[:2]      
        return loss,logits

class MbartForSequenceRegression(nn.Module):
    def __init__(self,args ,num_labels):
        super(MbartForSequenceRegression, self).__init__()
        if args.model == 'mbart':
            self.mbart = MBartForSequenceClassification.from_pretrained("facebook/mbart-large-50", num_labels=num_labels)

    def forward(self, input_ids, attention_mask,labels): 
        loss,logits = self.mbart(
          input_ids = input_ids,
          attention_mask = attention_mask,
          labels = labels
          )[:2]      
        return loss,logits

## XLM-R
class OurXLMRobertaForSequenceClassification(nn.Module):
    def __init__(self,args ,num_labels):
        super(OurXLMRobertaForSequenceClassification, self).__init__()
        if args.model == 'xlmr_base':
            self.roberta = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)
        if args.model == 'xlmr_large':
            self.roberta = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-large", num_labels=num_labels)
    def forward(self, input_ids, attention_mask,token_type_ids ,labels): 
        loss,logits = self.roberta(
          input_ids = input_ids,
          attention_mask = attention_mask,
          token_type_ids=token_type_ids,
          labels = labels
          )[:2]    
        return loss,logits

class OurXLMRobertaSequenceRegression(nn.Module):
     def __init__(self,args,num_labels):
        super(OurXLMRobertaSequenceRegression, self).__init__()
        if args.model == 'xlmr_base':
            self.roberta = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base",num_labels=num_labels)
        if args.model == 'xlmr_large':
            self.roberta = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-large",num_labels=num_labels)
        self.loss_fct = nn.MSELoss()
     def forward(self, input_ids, attention_mask,token_type_ids, labels): 
        outputs = self.roberta(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids=token_type_ids,
            )
        logits = outputs['logits']
        loss = self.loss_fct(logits.squeeze(),labels.squeeze())    
        return loss,logits


## XLM
class OurXLMForSequenceClassification(nn.Module):
    def __init__(self,args ,num_labels):
        super(OurXLMForSequenceClassification, self).__init__()
        if args.model == 'xlm':
            self.model = XLMForSequenceClassification.from_pretrained("xlm-mlm-100-1280", num_labels=num_labels)

    def forward(self, input_ids, attention_mask,token_type_ids ,labels): 
        loss,logits = self.model(
          input_ids = input_ids,
          attention_mask = attention_mask,
          token_type_ids=token_type_ids,
          labels = labels
          )[:2]    
        return loss,logits

class OurXLMSequenceRegression(nn.Module):
     def __init__(self,args,num_labels):
        super(OurXLMSequenceRegression, self).__init__()
        if args.model == 'xlm':
            self.model = XLMForSequenceClassification.from_pretrained("xlm-mlm-100-1280",num_labels=num_labels)
        self.loss_fct = nn.MSELoss()
     def forward(self, input_ids, attention_mask,token_type_ids, labels): 
        outputs = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids=token_type_ids,
            )
        logits = outputs['logits']
        loss = self.loss_fct(logits.squeeze(),labels.squeeze())    
        return loss,logits