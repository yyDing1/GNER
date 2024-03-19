from typing import Dict, Union
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import sys
import torch
import gradio as gr
import re

from evaluate import extract_predictions, parser


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name_or_path = "dyyyyyyyy/GNER-LLaMA-7B"
config = AutoConfig.from_pretrained(model_name_or_path)
is_encoder_decoder = config.is_encoder_decoder
MODEL_CLASS = AutoModelForSeq2SeqLM if is_encoder_decoder else AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = MODEL_CLASS.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16).to(device)

examples = [
    [
        "Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.",
        "person, company, location, airplane",
    ],
    [
        "On 25 July 1948, on the 39th anniversary of Bleriot's crossing of the English Channel, the Type 618 Nene-Viking flew Heathrow to Paris (Villacoublay) in the morning carrying letters to Bleriot's widow and son (secretary of the FAI), who met it at the airport.",
        "date, location, person, organization",
    ],
    [
        "Leo & Ian won the 1962 Bathurst Six Hour Classic at Mount Panorama driving a Daimler SP250 sports car, (that year the 500 mile race for touring cars were held at Phillip Island)",
        "person, date, location, organization, competition",
    ],
    [
        "The Shore Line route of the CNS & M until 1955 served, from south to north, the Illinois communities of Chicago, Evanston, Wilmette, Kenilworth, Winnetka, Glencoe, Highland Park, Highwood, Fort Sheridan, Lake Forest, Lake Bluff, North Chicago, Waukegan, Zion, and Winthrop Harbor as well as Kenosha, Racine, and Milwaukee (the ``KRM'') in Wisconsin.",
        "location, organization, date",
    ],
    [
        "Comet C/2006 M4 (SWAN) is a non-periodic comet discovered in late June 2006 by Robert D. Matson of Irvine, California and Michael Mattiazzo of Adelaide, South Australia in publicly available images of the Solar and Heliospheric Observatory (SOHO).",
        "person, organization, date, location",
    ],
    [
        "From November 29, 2011 to March 31, 2012, Karimloo returned to ``Les Misérables`` to play the lead role of Jean Valjean at The Queen's Theatre, London, for which he won the 2013 Theatregoers' Choice Award for Best Takeover in a Role.",
        "person, award, date, location",
    ],
    [
        "A Mexicali health clinic supported by former Baja California gubernatorial candidate Enrique Acosta Fregoso (PRI) was closed on June 15 after selling a supposed COVID-19 ``cure'' for between MXN $10,000 and $50,000.",
        "location, organization, person, date, currency",
    ],
    [
        "Built in 1793, it was the home of Mary Young Pickersgill when she moved to Baltimore in 1806 and the location where she later sewed the ``Star Spangled Banner'', in 1813, the huge out-sized garrison flag that flew over Fort McHenry at Whetstone Point in Baltimore Harbor in the summer of 1814 during the British Royal Navy attack in the Battle of Baltimore during the War of 1812.",
        "date, person, location, organization, event, flag",
    ],
]

def extract_span_info(words, predictions):
    span_l, span_r, span_type = -1, -1, None
    span_list = []
    for idx, (word, label) in enumerate(zip(words, predictions)):
        if label == "O" or label[:2] == "B-":
            if span_l != -1 and span_r != -1 and span_type is not None:
                span_list.append((span_l, span_r, span_type))
                span_l = span_r = -1
                span_type = None
        if label[:2] == "B-":
            span_l = span_r = idx
            span_type = label[2:]
        else:
            span_r = idx
            span_type = label[2:]
    if span_l != -1 and span_r != -1 and span_type is not None:
        span_list.append((span_l, span_r, span_type))
    return span_list

def predict_entities(text, labels):
    words = []
    start_token_idx_to_text_idx = []
    end_token_idx_to_text_idx = []
    for word_match in re.finditer(r'\w+(?:[-_]\w+)*|\S', text):
        words.append(word_match.group())
        start_token_idx_to_text_idx.append(word_match.start())
        end_token_idx_to_text_idx.append(word_match.end())

    # fit in the instruction template
    instruction_template = "Please analyze the sentence provided, identifying the type of entity for each word on a token-by-token basis.\nOutput format is: word_1(label_1), word_2(label_2), ...\nWe'll use the BIO-format to label the entities, where:\n1. B- (Begin) indicates the start of a named entity.\n2. I- (Inside) is used for words within a named entity but are not the first word.\n3. O (Outside) denotes words that are not part of a named entity.\n"
    instruction = f"{instruction_template}\nUse the specific entity tags: {', '.join(labels)} and O.\nSentence: {' '.join(words)}"
    if not is_encoder_decoder:
        instruction = f"[INST] {instruction} [/INST]"

    # tokenize -> generate -> detokenize
    inputs = tokenizer(instruction, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=640)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if not is_encoder_decoder:
        response = response[response.find("[/INST]") + len("[/INST]"):].strip()

    # structure
    example = {
        "label_list": labels,
        "instance": {"words": words},
        "prediction": response,
    }
    bio_predictions = extract_predictions(example)
    span_predictions_info = extract_span_info(words, bio_predictions)
    entities = []
    for start_token_idx, end_token_idx, ent_type in span_predictions_info:
        start_text_idx = start_token_idx_to_text_idx[start_token_idx]
        end_text_idx = end_token_idx_to_text_idx[end_token_idx]
        entities.append({
            "start": start_token_idx_to_text_idx[start_token_idx],
            "end": end_token_idx_to_text_idx[end_token_idx],
            "text": text[start_text_idx:end_text_idx],
            "label": ent_type,
        })
    return entities

def ner(text, labels: str) -> Dict[str, Union[str, int, float]]:
    labels = labels.split(",")
    for i in range(len(labels)):
        labels[i] = labels[i].strip()
    return {
        "text": text,
        "entities": [
            {
                "entity": entity["label"],
                "word": entity["text"],
                "start": entity["start"],
                "end": entity["end"],
                "score": 0,
            }
            for entity in predict_entities(text, labels)
        ],
    }

with gr.Blocks(title="GNER-LLaMA") as demo:
    gr.Markdown(
        """
<p align="center"><h2 align="center">Rethinking Negative Instances for Generative Named Entity Recognition</h2></p>

We introduce GNER, a **G**enerative **N**amed **E**ntity **R**ecognition framework, which demonstrates enhanced zero-shot capabilities across unseen entity domains. Experiments on two representative generative models, i.e., LLaMA and Flan-T5, show that the integration of negative instances into the training process yields substantial performance enhancements. The resulting models, GNER-LLaMA and GNER-T5, outperform state-of-the-art (SoTA) approaches by a large margin, achieving improvements of 8 and 11 points in $F_1$ score, respectively. Code and models are publicly available.

* 💻 Code: [https://github.com/yyDing1/GNER/](https://github.com/yyDing1/GNER/)
* 📖 Paper: [Rethinking Negative Instances for Generative Named Entity Recognition](https://arxiv.org/abs/2402.16602)
* 💾 Models in the 🤗 HuggingFace Hub: [GNER-Models](https://huggingface.co/collections/dyyyyyyyy/gner-65dda2cb96c6e35c814dea56)
* 🧪 Reproduction Materials: [Materials](https://drive.google.com/drive/folders/1m2FqDgItEbSoeUVo-i18AwMvBcNkZD46?usp=drive_link)
* 🎨 Example Jupyter Notebooks: [GNER Notebooks](notebook.ipynb)
        """
    )

    input_text = gr.Textbox(
        value=examples[0][0], label="Text input", placeholder="Enter your text here"
    )
    with gr.Row() as row:
        labels = gr.Textbox(
            value=examples[0][1],
            label="Labels",
            placeholder="Enter your labels here (comma separated)",
            scale=1,
        )

    output = gr.HighlightedText(label="Predicted Entities")
    submit_btn = gr.Button("Submit")
    examples = gr.Examples(
        examples,
        fn=ner,
        inputs=[input_text, labels],
        outputs=output,
        cache_examples=True,
    )

    # Submitting
    input_text.submit(fn=ner, inputs=[input_text, labels], outputs=output)
    submit_btn.click(fn=ner, inputs=[input_text, labels], outputs=output)

demo.queue()
demo.launch(share=True)
