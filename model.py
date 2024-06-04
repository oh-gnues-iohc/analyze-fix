import copy
from transformers import T5ForConditionalGeneration
import torch
import torch.nn as nn
from torch.nn import L1Loss, CrossEntropyLoss
import copy
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from transformers.utils import ModelOutput


@dataclass
class AnalystOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    regression_logits: torch.FloatTensor = None
    classification_logits: torch.FloatTensor = None
    tagging_logits: torch.FloatTensor = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None

class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class Analyst(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        regression_config = copy.deepcopy(config)
        regression_config.num_labels = 1
        self.regression_head = ClassificationHead(regression_config)
        
        tagging_config = copy.deepcopy(config)
        tagging_config.num_labels = 2
        self.tagging_head = ClassificationHead(tagging_config)
        
        self.classification_head = ClassificationHead(config)
        
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        labels_regression: Optional[torch.FloatTensor] = None,
        labels_tagging: Optional[torch.LongTensor] = None,
        labels_classification: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], AnalystOutput]:
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            labels=labels,
            return_dict=return_dict
        )
        encoder_hidden_state = output.encoder_last_hidden_state
        lm_logits = output.logits
        
        loss = output.loss
        regression_logits = None
        classification_logits = None
        tagging_logits = None
        
        if input_ids is not None:
            eos_mask = input_ids.eq(self.config.eos_token_id).to(encoder_hidden_state.device)
            batch_size, _, hidden_size = encoder_hidden_state.shape
            sentence_representation = encoder_hidden_state[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
            
            regression_logits = self.regression_head(sentence_representation)
            classification_logits = self.classification_head(sentence_representation)
            tagging_logits = self.tagging_head(encoder_hidden_state)
        
        if labels_regression is not None:
            labels_regression = labels_regression.to(lm_logits.device)
            loss_fct = L1Loss()
            regression_loss = loss_fct(regression_logits.squeeze(), labels_regression.squeeze())
            loss += regression_loss
        else:
            regression_loss = None

        if labels_classification is not None:
            labels_classification = labels_classification.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            classification_loss = loss_fct(classification_logits.view(-1, self.config.num_labels), labels_classification.squeeze())
            loss += classification_loss
        else:
            classification_loss = None

        if labels_tagging is not None:
            labels_tagging = labels_tagging.to(lm_logits.device)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            tagging_loss = loss_fct(tagging_logits.view(-1, tagging_logits.size(-1)), labels_tagging.view(-1))
            loss += tagging_loss
        else:
            tagging_loss = None
        
        if not return_dict:
            output = (loss, lm_logits, regression_logits, classification_logits, tagging_logits)
            return output

        return AnalystOutput(
            loss=loss,
            logits=lm_logits, 
            regression_logits=regression_logits, 
            classification_logits=classification_logits, 
            tagging_logits=tagging_logits,
            encoder_last_hidden_state=encoder_hidden_state
        )