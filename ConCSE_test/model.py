import torch
import numpy as np
from transformers import BertModel
import torch.nn as nn

from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers import BertTokenizerFast

from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
import ipdb

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, model_config):
        super().__init__()
        self.dense = nn.Linear(model_config.hidden_size,model_config.hidden_size)
        self.activation = nn.Tanh()
    def forward(self, features):
        x= self.dense(features)
        x= self.activation(x)

        return x 
    
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    

class TripletLoss(nn.Module):
    
    def __init__(self, margin=1.0):
        super(TripletLoss,self).__init__()
        self.margin = margin
    
    def calc_l2_norm(self,x1,x2):
        return torch.norm(x1-x2,p=2,dim=1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor): 
	    # anchor : sen1[64,768] CLS임 , positive: sent2 , neg : hard_neg
        distance_positive = self.calc_l2_norm(anchor, positive)
        distance_negative = self.calc_l2_norm(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
    

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state #encoder의 last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states
        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

def cl_init(cls, args ,model_config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = args.pooler_type 
    cls.pooler = Pooler(cls.pooler_type)
    if cls.pooler_type == "cls":
        cls.mlp = MLPLayer(model_config)
    cls.sim = Similarity(temp = args.temp)
    if cls.args.ours_version == '999' : # TripletLoss 
        cls.triplet = TripletLoss(cls.args.margin)
    
    cls.init_weights()  #BertModel,Pooler, MLPLayer, Similarity 모두 random initialize

def cl_forward(cls, 
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):  

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None

    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len) 
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len) 
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len) 

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.args.pooler_type in ['avg_top2', 'avg_first_last'] else False,

        return_dict=True,
    )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)  #encoder(bs*num_sent, max_len, 768) 

    # (bs*num_ent, hidden) => (bs, num_sent, hidden)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) 

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.args.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)
    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1] #positive pair(sentence1,sentence1) 

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0)) 
    
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0)) #z1_z3_cos = [64,64]
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1) # cos_sim = [64,128]
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.args.hard_negative_weight 
                                                
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device) 
        cos_sim = cos_sim + weights 

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

## cross_forward is ConCSE
def cross_forward(cls, 
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):  
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 6: ours
    num_sent = input_ids.size(1)

    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )
    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)  #encoder(bs*num_sent, max_len, 768) 

    # (bs*num_ent, hidden) => (bs, num_sent, hidden)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) 

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.args.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output) 

    e0, e1 = pooler_output[:,0], pooler_output[:,1] # en_positive pair
    e2 = pooler_output[:,2]                         # en_hard_negative
    c0, c1 = pooler_output[:,3], pooler_output[:,4] # cs_positive pair   
    c2 = pooler_output[:,5]                         # cs_hard_negative

    if cls.args.ours_version == '999': 

        e0_e1_cos = cls.sim(e0.unsqueeze(1), e1.unsqueeze(0))
        e0_e2_cos = cls.sim(e0.unsqueeze(1), e2.unsqueeze(0))
        c0_c1_cos = cls.sim(c0.unsqueeze(1), c1.unsqueeze(0))
        c0_c2_cos = cls.sim(c0.unsqueeze(1), c2.unsqueeze(0))
        e0_c0_cos = cls.sim(e0.unsqueeze(1), c0.unsqueeze(0))
        e1_c1_cos = cls.sim(e1.unsqueeze(1), c1.unsqueeze(0))
        e2_c2_cos = cls.sim(e2.unsqueeze(1), c2.unsqueeze(0))
        
        e0_c2_cos = cls.sim(e0.unsqueeze(1), c2.unsqueeze(0))
        c0_e2_cos = cls.sim(c0.unsqueeze(1), e2.unsqueeze(0))
        e1_c2_cos = cls.sim(e1.unsqueeze(1), c2.unsqueeze(0))
        
        sim1 = torch.cat([e0_e1_cos, e0_e2_cos], 1)  
        sim2 = torch.cat([c0_c1_cos, c0_c2_cos], 1)  
        sim3 = torch.cat([e0_e1_cos, e0_c2_cos],1)  
        sim4 = torch.cat([c0_c1_cos, c0_e2_cos],1)  
        sim5 = torch.cat([e0_c0_cos,c0_e2_cos],1)
        sim6 = torch.cat([e1_c1_cos,e1_c2_cos],1) 

        cat_labels = torch.arange(sim1.size(0)).long().to(cls.device)
        e0_c0_labels = torch.arange(e0_c0_cos.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss()
        e2_weight = cls.args.hard_negative_weight 
        weights = torch.tensor(
            [[0.0] * (sim1.size(-1) - e0_e2_cos.size(-1)) + [0.0] * i + [e2_weight] + [0.0] * (e0_e2_cos.size(-1) - i - 1) for i in range(e0_e2_cos.size(-1))]
        ).to(cls.device) 
        
        sim1 = sim1 + weights 
        sim2 =  sim2 + weights
        sim3 =  sim3 + weights 
        sim4 = sim4 + weights 
        sim5 = sim5 + weights 
        sim6 =  sim6 + weights
         
        ## Cross Contrastive loss 
        sim1_loss = loss_fct(sim1, cat_labels) 
        sim2_loss = loss_fct(sim2, cat_labels) 
        sim3_loss = loss_fct(sim3, cat_labels) 
        sim4_loss = loss_fct(sim4, cat_labels) 
        sim5_loss = loss_fct(sim5, cat_labels) 
        sim6_loss = loss_fct(sim6, cat_labels)  
        
        ## Neg_align_loss
        cs_pos_loss = loss_fct(e2_c2_cos, e0_c0_labels) # cs_pos_loss
        
        ce_loss = sim1_loss + sim2_loss + sim3_loss + sim4_loss + sim5_loss + sim6_loss + cs_pos_loss
       
        ## Cross Triplet loss
        sim1_tri_loss = cls.triplet(e0,e1,e2)
        sim2_tri_loss = cls.triplet(c0,c1,c2)
        sim3_tri_loss = cls.triplet(e0,e1,c2)
        sim4_tri_loss =  cls.triplet(c0,c1,e2)
        sim5_tri_loss = cls.triplet(e0,c0,e2)
        sim6_tri_loss = cls.triplet(e1,c1,c2)
        
        tri_loss = sim1_tri_loss + sim2_tri_loss + sim3_tri_loss+ sim4_tri_loss + sim5_tri_loss + sim6_tri_loss
        
        total_loss = ce_loss + cls.args.triplet * tri_loss
        total_logits = None

    
    return SequenceClassifierOutput(
        loss=total_loss,
        logits=total_logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.args.pooler_type in ['avg_top2', 'avg_first_last'] else False, 
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs) 
    '''
    if cls.args.pooler_type == "cls" and not cls.args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)
    '''
    if cls.args.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)
    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    def __init__(self, pretrained_model_name_or_path ,args, model_config):
        super().__init__(model_config)
        self.args = args
        self.model_config = model_config
        self.model_name = self.args.model
        self.pooler_type = self.args.pooler_type
        self.hard_negative_weight = self.args.hard_negative_weight
            
        self.bert = BertModel(self.model_config, add_pooling_layer=False)
        cl_init(self, self.args, self.model_config)
    def forward(self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        simcse = False,
        cross = False
    ):  
        if sent_emb:
            # Encoder
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif simcse:
            # SimCSE
            return cl_forward(self, 
                                self.bert, 
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds,
                                labels=labels,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict,
                                mlm_input_ids=mlm_input_ids,
                                mlm_labels=mlm_labels,
                            )
        elif cross: 
            # ConCSE
            return  cross_forward(self, 
                                self.bert, 
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds,
                                labels=labels,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict,
                                mlm_input_ids=mlm_input_ids,
                                mlm_labels=mlm_labels)
             

class RobertaForCL(RobertaPreTrainedModel):
    def __init__(self, pretrained_model_name_or_path ,args, model_config):
        super().__init__(model_config)
        self.args = args
        self.model_config = model_config
        self.model_name = self.args.model
        self.pooler_type = self.args.pooler_type
        self.hard_negative_weight = self.args.hard_negative_weight      

        self.roberta = RobertaModel(self.model_config, add_pooling_layer=False)
        cl_init(self, self.args, self.model_config)
    def forward(self, #여기서 print(self)하면 model의 전체구조가 출력되는 이유는  self자체가 자기자신 인스턴스니까 print(model)과 같이 출력되는것
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        simcse = False,
        cross = False
    ):  
        if sent_emb:
            # Encoder
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif simcse:
            # SimCSE
            return cl_forward(self, 
                                self.roberta, 
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds,
                                labels=labels,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict,
                                mlm_input_ids=mlm_input_ids,
                                mlm_labels=mlm_labels,
                            )
        elif cross:
            # ConCSE
            return  cross_forward(self, 
                                self.roberta,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds,
                                labels=labels,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict,
                                mlm_input_ids=mlm_input_ids,
                                mlm_labels=mlm_labels)