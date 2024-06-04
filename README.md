# analyze-fix

### Demo

https://huggingface.co/spaces/ohgnues/ANALYZE-FIX

## **프로젝트 목표**

아래 네가지의 태스크를 수행하는 단일 모델을 학습하는것이 목표

1. Seq2Seq 오타 수정 텍스트 생성
2. 회귀 분석을 통한 데이터 품질 평가
3. 토큰 분류를 통한 오타 위치 태깅
4. 데이터 도메인 분류

## **프로젝트 구현**

- **모델 선택**
    - KETI-AIR/ke-t5-base
- **데이터 수집 및 전처리**
    - Aihub에서 제공하는 [**문서요약 텍스트](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)** 데이터 사용 위에 나열한 태스크를 수행하기 위해 원천 데이터를 기반으로 `'IT', '기업', '종교', '일반행정', '인물', '특허', '스포츠', '사회', '문화', '정책', '경제', '정치’` 도메인 분류, 직접 제작한 [오타 생성 모듈](https://github.com/oh-gnues-iohc/korean-noise-augmentation)을 이용하여 오염된 데이터 생성 후, LCS를 이용한 데이터 품질, 오타 위치 라벨링 진행
    
    https://huggingface.co/datasets/ohgnues/unified_text_task
    
    ![image](https://github.com/oh-gnues-iohc/analyze-fix/assets/79557937/e80ff0bf-6898-4f24-a8ad-c80e1a53ea5e)

    
- **모델 구현**
    1. Seq2Seq 오타 수정 텍스트 생성 : 기존의 ***T5ForConditionalGeneration*** 방식을 그대로 사용
    2. 회귀 분석을 통한 데이터 품질 평가 : Encoder의 `last_hidden_state`에서 eos 토큰에 대한 representation을 이용하여 회귀 분석 진행
    3. 토큰 분류를 통한 오타 위치 태깅 : Encoder의 `last_hidden_state`에서 각 위치에 대한 이진 분류를 통해 오타 태깅
    4. 데이터 도메인 분류 : Encoder의 `last_hidden_state`에서 패딩을 제외한 토큰들의 임베딩 평균을 이용하여 도메인 분류
    
    ```python
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
    ```
    

## **프로젝트 결과**

기본적으로 오타 위치 태깅과 도메인 분류를 제외한 태스크는 준수한 성능을 보임

![image](https://github.com/oh-gnues-iohc/analyze-fix/assets/79557937/40d0b550-3236-465a-9502-ddd75d82b45d)

### 결과

1. **오타 위치 태깅 성능 부족**
    - 오타 위치를 정확하게 태깅하는 성능이 기대에 미치지 못함
    - **해결방안**
        - 더 많은 데이터를 사용하여 학습하거나 데이터 증강 기법을 통해 다양한 오타 케이스를 추가적으로 학습시키는 방법을 고려
2. **오타 태깅과 오타 수정의 연계 문제**
    - 현재 오타 태깅과 오타 재생성이 별도로 수행되어 두 과정이 어울리지 않는 문제가 있음
    - **해결방안**
        - 오타 태깅을 먼저 수행한 뒤, 태깅된 토큰 부분만 재생성하는 방법을 고려.
        - 마스크드 언어 모델링(MLM) 기법을 적용하여, 오타로 태깅된 부분만 마스크하고 해당 부분을 재생성하는 방식으로 모델을 개선할 수 있어보임
3. **도메인 분류 정확도 향상 필요**
    - 도메인 분류 작업의 성능이 예상보다 훨씬 낮게 나옴
    - **해결방안**
        - 모델의 크기에 비해 너무 많은 태스크를 학습한게 주 원인으로 생각됨
        - Encoder의 `last_hidden_state`를 통해 여러 태스크를 학습하다 보니 예상보다 낮게 나온 듯 함

![image](https://github.com/oh-gnues-iohc/analyze-fix/assets/79557937/08c04791-aec8-41b6-9c0d-43903041dd6d)
