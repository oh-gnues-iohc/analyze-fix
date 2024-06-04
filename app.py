import streamlit as st
import numpy as np
import pandas as pd
from model import Analyst
from transformers import AutoTokenizer
import torch
import plotly.express as px

DEVICE = "cpu"

def calculate_text_quality(score):
    return score[0][0]

def generate_corrected_text(inputs, model, tokenizer):
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_return_sequences=3, 
            num_beams=3,
            max_length=512,
            early_stopping=True
        )
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return "\n".join([f"{i+1}. {output}" for i, output in enumerate(decoded_outputs)])

def classify_text_domain(logits):
    softmax = torch.nn.functional.softmax(logits, dim=1)
    probabilities = softmax.squeeze().tolist()
    labels = ['IT', '기업', '종교', '일반행정', '인물', '특허', '스포츠', '사회', '문화', '정책', '경제', '정치']
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_probabilities = [probabilities[i] for i in sorted_indices]
    return sorted_labels, sorted_probabilities

def identify_typos(logits, text, tokenizer):
    logits = logits.squeeze(0)  # Remove batch dimension
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    typo_indices = torch.argmax(probabilities, dim=-1).tolist()

    offset_mapping = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
    ).offset_mapping[0]
    
    typo_positions = [index for index, value in enumerate(typo_indices) if value == 1]
    typo_positions = list(set([o.item() for idx in typo_positions for o in offset_mapping[idx]]))
    
    highlighted_indices = []
    for start, end in offset_mapping:
        if start in typo_positions or end in typo_positions:
            highlighted_indices.append(start)
            highlighted_indices.append(end)
    
    tagged_text = []
    for i, char in enumerate(text):
        if i in highlighted_indices:
            tagged_text.append((char, "TYPO"))
        else:
            tagged_text.append((char, "O"))
    return tagged_text

@st.cache_resource()
def load_model_and_tokenizer(path):
    model = Analyst.from_pretrained(path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

st.title("ANALYZE & FIX")

input_text = st.text_area("Input:")

if st.button("Submit"):
    model, tokenizer = load_model_and_tokenizer("./temp")
    with st.spinner("Analyzing..."):
        tokenized_inputs = tokenizer(
            input_text,
            return_tensors="pt",
        ).to(model.device)
        
        with torch.no_grad():
            model_outputs = model(**tokenized_inputs, decoder_input_ids=tokenized_inputs.input_ids, return_dict=True)
        
        text_quality = calculate_text_quality(model_outputs.regression_logits)
        corrected_text = generate_corrected_text(tokenized_inputs, model, tokenizer)
        domain_labels, domain_probabilities = classify_text_domain(model_outputs.classification_logits)
        typo_tags = identify_typos(model_outputs.tagging_logits, input_text, tokenizer)

    st.markdown("## Results")

    st.markdown("### Text Quality")
    st.metric("Score", f"{text_quality:.5f}")

    st.markdown("### Corrected Text")
    st.markdown(corrected_text)

    st.markdown("### Domain Classification Probabilities")
    df = pd.DataFrame({"Label": domain_labels, "Probability": domain_probabilities})
    fig = px.bar(df, x="Label", y="Probability", title="Domain Classification Probabilities", labels={"Label": "Domain", "Probability": "Probability"})
    st.plotly_chart(fig)

    st.markdown("### Typo Location Tagging")
    formatted_text = ""
    for char, tag in typo_tags:
        if tag == "TYPO" and char:
            formatted_text += f"<b><font color='red'>{char}</font></b>"
        else:
            formatted_text += char
    st.markdown(formatted_text, unsafe_allow_html=True)
