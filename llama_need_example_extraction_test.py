def extract_entities_from_spacy_file(spacy_file_path):
    nlp = spacy.blank("en")  # 使用空模型载入 DocBin
    doc_bin = DocBin().from_disk(spacy_file_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    all_data = []
    for doc in docs:
        text = doc.text
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })
        all_data.append({
            "text": text,
            "entities": entities
        })
    return all_data


import spacy
from spacy.tokens import DocBin


# 使用示例
spacy_file_path = "train_TAC2017.spacy"  # 替换成你的文件路径  
spacy_file_path = "LLMdata_FAERS_R3.spacy"
data = extract_entities_from_spacy_file(spacy_file_path)

# 打印前两个样本预览
import json
print(json.dumps(data[:2], indent=2))

##extract first entity from first sentence
data[0]['text'][data[0]['entities'][0]['start']:data[0]['entities'][0]['end']]
#check first entity text
data[0]['entities'][0]

####check number of entities
from collections import Counter

labels = []
for item in data:
    labels.extend(ent['label'] for ent in item['entities'])

label_counts = Counter(labels)

for label, count in label_counts.items():
    print(f"标签: '{label}', 出现次数: {count}")


########change the output look of data

def insert_tags(text, entities):
    # 按照 start 倒序排序，避免位置冲突
    entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)
    tagged_text = text
    for ent in entities_sorted:
        start, end, label = ent['start'], ent['end'], ent['label']
        tagged_text = (
            tagged_text[:start] +
            f"<{label}>{tagged_text[start:end]}</{label}>" +
            tagged_text[end:]
        )
    return tagged_text

def process_ner_data(data):
    result = []
    for item in data:
        original = item['text']
        tagged = insert_tags(original, item['entities'])
        result.append({
            "original": original,
            "tagged": tagged
        })
    return result

# 示例使用
converted_data = process_ner_data(data)

# 打印结果查看
for item in converted_data:
    print("Original:\n", item['original'])
    print("\nTagged:\n", item['tagged'])
    print("\n" + "-"*50 + "\n")


####try 
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="meta-llama/Llama-3.3-70B-Instruct",
    token="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",
    provider="hf-inference"
)

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token="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",
    provider="hf-inference"
)

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token="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",
    provider="hf-inference"
)

sentence = ("5 WARNINGS AND PRECAUTIONS\n\n\n\n   EXCERPT:    *  "
            "Tendon rupture or serious injury to the injected finger/hand: "
            "Avoid injecting XIAFLEX into tendons, nerves, blood vessels, or other collagen-containing structure of the hand.")

# 构造 prompt
prompt = f"""<s>[INST] You are an expert biomedical annotator. 
Tag named entities in the sentence using XML-style tags.
Only use <AdverseReaction> and <Severity> tags. 
Keep the rest of the text unchanged. Do NOT add extra explanation.

Sentence:
{sentence}

Tagged: [/INST]
"""

# 调用 LLaMA 70B 推理
response = client.text_generation(
    prompt,
    max_new_tokens=300,
    temperature=0.2,
    do_sample=False
)

print(response)

######################
################################
##############data training using LoRA
import json

def format_lora_prompt(sample):
    return f"""<s>[INST] You are an expert biomedical annotator. 
Tag named entities in the sentence using XML-style tags.
Only use <age>, <sex>, <drug>, <tempo>, <dose>, <hx>, <lab>, <ae>, <treatment>, <status>, <dx>, <bsym>, <ro> and <cod> tags. 
Keep the rest of the text unchanged. Do NOT add extra explanation.

Sentence:
{sample["original"]}

Tagged: [/INST]
{sample["tagged"]}</s>"""

# 保存成 jsonl，用于 LLaMA/LoRA 训练
with open("llama_training_data.jsonl", "w", encoding="utf-8") as f:
    for item in converted_data[0:10]:
        formatted = format_lora_prompt(item)
        json.dump({"text": formatted}, f)
        f.write("\n")


####train
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7" 

dataset = load_dataset("json", data_files="llama_training_data.jsonl", split="train")


# 1. 加载你的数据
# dataset = load_dataset("json", data_files="llama_training_data.jsonl", split="train")

# 2. 加载模型 & tokenizer（比如使用 LLaMA2-7B）
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # 确保你有访问权限
###hunggingface login
# huggingface-cli login

###
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",  # ✅ GPU 支持 fp4；如出错可换成 "nf4"
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16  # 如果你有 A100/H100 可换成 torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # 兼容性

# 3. 加载量化模型 + LoRA 设置
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # 使用 QLoRA
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 4. Tokenize 数据
def tokenize(sample):
    return tokenizer(sample["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

# 5. 训练参数
args = TrainingArguments(
    output_dir="./lora_llama_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="epoch",
    bf16=True,
    save_total_limit=1,
)

# 6. 开始训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)


trainer.train()
import gc
import torch

del trainer
del model
del tokenized
gc.collect()
torch.cuda.empty_cache()


######evaluation
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# 微调后的模型路径
output_dir = "./lora_llama_output/checkpoint-3"
peft_config = PeftConfig.from_pretrained(output_dir)
# base_model = AutoModelForCausalLM.from_pretrained(
#     peft_config.base_model_name_or_path,
#     load_in_4bit=True,
#     device_map="auto"
# )
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",  # nf4通常效果好
#     bnb_4bit_compute_dtype="float16"
# )
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,  # 如果是4bit就改为 load_in_4bit=True
#     llm_int8_enable_fp32_cpu_offload=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",  # nf4通常效果好
#     bnb_4bit_compute_dtype="float16"
# )

# # 这里device_map可以用"auto"，如果显存不足，可以指定分配或启用offload
# base_model = AutoModelForCausalLM.from_pretrained(
#     peft_config.base_model_name_or_path,
#     quantization_config=bnb_config,
#     device_map="auto",
#     # 如果显存不够，可以加下面这行开启fp32的cpu offload
#     llm_int8_enable_fp32_cpu_offload=True,
# )
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,  # 这个只给 BitsAndBytesConfig
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载 LoRA adapter
model = PeftModel.from_pretrained(base_model, output_dir)
model.eval()

# 确认是否加载成功
print(model.peft_config)

# Step 3: Prepare prompt (LLaMA3 Instruct格式)
prompt = """[INST] You are an expert biomedical annotator.
Tag only named entities in the sentence using these tags: <age>, <sex>, <drug>, <tempo>, <dose>, <hx>, <lab>, <ae>, <treatment>, <status>, <dx>, <bsym>, <ro>, <cod>.
Keep the sentence exactly the same.
Do NOT add any explanation or extra text.
Output only the annotated sentence.
Sentence: This spontaneous report was received from a consumer or other non health professional (patient's family via a call center) and concerns a male patient in his eighties from Japan: Local Case ID: 201412714.
[/INST]"""

# Step 4: Generate output
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0,
        top_p=0.9,
        do_sample=True
    )

# Step 5: Decode and print
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# # 加载 base model（你训练用的是 Llama-3.1-8B-Instruct）
# base_model = "meta-llama/Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
# tokenizer.pad_token = tokenizer.eos_token

# # 加载 LoRA 微调后的模型
# model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     load_in_4bit=True,
#     device_map="auto"
# )

# model = PeftModel.from_pretrained(model, output_dir)
# model.eval()


def extract_entities_from_spacy_file(spacy_file_path):
    nlp = spacy.blank("en")  # 使用空模型载入 DocBin
    doc_bin = DocBin().from_disk(spacy_file_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    all_data = []
    for doc in docs:
        text = doc.text
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })
        all_data.append({
            "text": text,
            "entities": entities
        })
    return all_data


import spacy
from spacy.tokens import DocBin


# 使用示例
spacy_file_path = "valid_FAVERS_R1_LLMLLM_revision_v1.spacy"
data = extract_entities_from_spacy_file(spacy_file_path)
# def lora_annotate(text):
#     prompt = f"请用标签标注以下句子中的时间表达式：{text}"
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=100,
#             do_sample=False,
#             top_k=50,
#             top_p=0.95,
#             temperature=0.7,
#         )
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return decoded

# def lora_annotate(text):
#     prompt = f"""<s>[INST] You are an expert biomedical annotator. 
# Tag named entities in the sentence using XML-style tags.
# Only use <age>, <sex>, <drug>, <tempo>, <dose>, <hx>, <lab>, <ae>, <treatment>, <status>, <dx>, <bsym>, <ro> and <cod> tags. 
# Keep the rest of the text unchanged. Do NOT add extra explanation.

# Sentence: {text}
# [/INST]"""
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=200,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#         )
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
#     return decoded

# def lora_annotate(text):
#     prompt = f"""<s>[INST] You are an expert biomedical annotator. 
# Tag named entities in the sentence using XML-style tags.
# Only use <age>, <sex>, <drug>, <tempo>, <dose>, <hx>, <lab>, <ae>, <treatment>, <status>, <dx>, <bsym>, <ro> and <cod> tags. 
# Keep the rest of the text unchanged. Do NOT add extra explanation.
# Sentence: {text} [/INST]"""
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=512,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#         )
#     full_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # 👇 只保留回答部分（去掉 prompt）
#     response = full_decoded.split("[/INST]")[-1].strip()
#     return response
import re

def clean_output(text):
    allowed_tags = ['age','sex','drug','tempo','dose','hx','lab','ae','treatment','status','dx','bsym','ro','cod']
    pattern = re.compile(r'</?([a-z]+)[^>]*>')
    def repl(m):
        tag = m.group(1)
        if tag in allowed_tags:
            return m.group(0)
        else:
            return ''
    return pattern.sub(repl, text)

def lora_annotate(text):
#     prompt = f"""<s> You are an expert biomedical annotator.
# Tag only named entities in the sentence using these tags: <age>, <sex>, <drug>, <tempo>, <dose>, <hx>, <lab>, <ae>, <treatment>, <status>, <dx>, <bsym>, <ro>, <cod>.
# Keep the sentence exactly the same.
# Do NOT add any explanation or extra text.
# Output only the annotated sentence.
# Sentence: {text}
# [/INST]"""
    prompt = f"""<s>[INST] You are an expert biomedical annotator. 
    Tag named entities in the sentence using XML-style tags.
    Only use <age>, <sex>, <drug>, <tempo>, <dose>, <hx>, <lab>, <ae>, <treatment>, <status>, <dx>, <bsym>, <ro> and <cod> tags. 
    Keep the rest of the text unchanged. Do NOT add extra explanation.
    Sentence:
    {text}
    Tagged: [/INST]
    </s>"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()
    # 清理不允许标签
    decoded = clean_output(decoded)
    return decoded




# ==== 后处理，提取 <tempo>...</tempo> 等标注 ====
import re

def postprocess_prediction(original_text, annotated_text):
    annotated_output = ""
    all_annotations = []
    offset = 0
    entity_pattern = re.finditer(r"<(?P<label>\w+)>(?P<text>.*?)</\1>", annotated_text)
    last_end = 0
    for match in entity_pattern:
        label = match.group("label")
        ent_text = match.group("text")
        start = annotated_text.find(ent_text, last_end)
        end = start + len(ent_text)
        # 累积 offset 以匹配原文
        entity_global_start = start
        entity_global_end = end + len(label) * 2 + 5
        all_annotations.append({
            "label": label,
            "note": "LoRA",
            "textContext": {
                "start": entity_global_start,
                "end": entity_global_end,
                "text": ent_text,
                "page": 0
            }
        })
        last_end = end
    annotated_output = f"<p>{annotated_text}</p>"
    return {
        "pages": [original_text],
        "annotated_pages": [annotated_output],
        "annotations": all_annotations
    }

# ==== 总体处理 ====
import torch
output = []
for item in data[0:6]:
    text = item["text"]
    predicted = lora_annotate(text)
    example = postprocess_prediction(text, predicted)
    output.append(example)

# ==== 写入文件 ====
with open("lora_annotation_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("✅ 写入完成：lora_annotation_output.json")


############test 6/11/2025
##this is result one in text file
def lora_annotate(text):
#     system_prompt = """You are going to annotate these important attributes that were reported in a FAERS case report narrative. Annotate with the following annotation types.
# The annotation should be in-text, which means you should not alter the original text but only adding tags surround the term-of-interest.
# For example, the original sentence may be: "Concomitant medications included on an unknown <Date>, atenolol tablet at a dose of 25 milligrams twice a day via unknown route for unknown indication"
# The annotated sentence should be "Concomitant medications included on an unknown Date, <Drug>atenolol</Drug> tablet at a dose of <Dose>25 milligrams</Dose> twice a day via unknown route for unknown indication"
# Annotation types:
#     'Drug': 'Drug Products',
#     'AE': 'Adverse Event',
#     'Treatment': 'Drug used for Treatment',
#     'RO': 'Rule-out diagnosis',
#     'Dx': 'Diagnostic test results',
#     'CoD': 'Cause of Death',
#     'Age': 'Age',
#     'Sex': 'Sex',
#     'Dose': 'Dose Administered',
#     'bSYM': 'Baseline Symptom before AE Occurred',
#     'Hx': 'Medical History',
#     'Status': 'Patient Status',
#     'Lab': 'Laboratory Finding',
#     'TEMPO': 'Temporal Information, such as Date, Time, etc.'
# Output Guidance:
#     Be cautious. Do NOT alter the original text. Only add tags of Annotation_type such as <AE>...</AE> to the identified entity.
#     The tag must be complete. Do not miss any right > or left <.
#     The output should start with and only contain:
#     The annotated text is shown as below:"""
    system_prompt = """You are an expert biomedical annotator.
    Your task is to highlight important clinical attributes in the FAERS case report narrative using XML-style tags.
    NEVER alter the original sentence content. Just wrap the entity using exact tags.
    Available annotation types:
    <Drug>, <AE>, <Treatment>, <RO>, <Dx>, <CoD>, <Age>, <Sex>, <Dose>, <bSYM>, <Hx>, <Status>, <Lab>, <TEMPO>
    Example:
    Input: "Patient was given ibuprofen 200mg daily on 5th July for pain"
    Output: "Patient was given <Drug>ibuprofen</Drug> <Dose>200mg</Dose> daily on <TEMPO>5th July</TEMPO> for <Treatment>pain</Treatment>"
    The annotated text is shown as below:"""
    user_prompt = f"""Sentence:
{text}"""
    # 拼接成 Chat 模式的 prompt
    # prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
    prompt = f"{system_prompt}\n\nSentence:\n{text}\n\nThe annotated text is shown as below:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # 去除前面的 prompt（只保留模型生成的回答）
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]", 1)[-1].strip()
    return clean_output(decoded)


def clean_output(output_text):
    # Step 1: 找到包含至少一个开始标签的第一段
    pattern = r"(?:<Drug>|<AE>|<Treatment>|<RO>|<Dx>|<CoD>|<Age>|<Sex>|<Dose>|<bSYM>|<Hx>|<Status>|<Lab>|<TEMPO>).*?"
    match = re.search(pattern, output_text, re.DOTALL)
    if match:
        start_index = match.start()
        # 回溯找到最靠近它的句子开头（比如上一段的 \n\n 或 Sentence: 后面）
        prefix = output_text[:start_index]
        if "Sentence:" in prefix:
            start_index = prefix.rfind("Sentence:") + len("Sentence:")
        return output_text[start_index:].strip()
    # fallback
    return output_text.strip()

output = []
for item in data[0:7]:
    text = item["text"]
    print("text: ", text)
    predicted = lora_annotate(text)
    print(predicted)
    example = postprocess_prediction(text, predicted)
    output.append(example)
#########6/11/2025  night
import re
import torch

def lora_annotate(text):
    system_prompt = """You are an expert biomedical annotator.
Your task is to highlight important clinical attributes in the FAERS case report narrative using XML-style tags.
NEVER alter the original sentence content. Just wrap the entity using exact tags.
Available annotation types:
<Drug>, <AE>, <Treatment>, <RO>, <Dx>, <CoD>, <Age>, <Sex>, <Dose>, <bSYM>, <Hx>, <Status>, <Lab>, <TEMPO>. 
The annotated text is shown as below:"""
    prompt = f"{system_prompt}\n\n{text}\n\nThe annotated text is shown as below:"
    # prompt = f"{system_prompt}\n\nSentence:\n{text}\n\nThe annotated text is shown as below:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return decoded
    # 清理输出文本
    return clean_output(decoded)


def clean_output(output_text):
    """
    从模型输出中提取以“The annotated text is shown as below:”开头的、真正带标签的原文段落。
    """
    pattern = r"The annotated text is shown as below:(.*)"
    match = re.search(pattern, output_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return output_text.strip()



import json

with open("annotated_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)


##############6/12/2015 

from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()
openai_api_key = os.getenv('LLM_KEY', '')
openai_api_base = os.getenv('LLM_URL', '')
llm_model_name = os.getenv('LLM_MODEL', "")
 
# Validate required variables
if not openai_api_key:
    raise ValueError("Environment variable OPENAI_API_KEY is not set")
 
# Initialize OpenAI client
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,  # this can be None if using OpenAI's default endpoint
)

# def get_70b_hints(text: str) -> str:
#     response = client.chat.completions.create(
#         model=llm_model_name,
#         messages=[
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a prompt engineering expert.\n"
#                     "Given a request to annotate biomedical text, suggest the most effective concise system prompt."
#                     "Focus on clarity, minimalism, and correctness."
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": f"Provide an improved system_prompt for annotating:\n{text}"
#             }
#         ],
#         temperature=0.0,
#         max_tokens=200
#     )
#     # 在新版接口里，choices 在 data 字段下
#     return response.choices[0].message.content.strip()

def get_70b_reasoning(text: str) -> str:
    """
    使用 70B 大模型阐述如何一步步得出标注结果的思路。
    只输出思路，不要包含任何 XML 标签或直接标注结果。
    """
    # 用户原始 annotation prompt（示例，仅用于上下文，不会要求模型输出标签）
    user_system_prompt = (
        "Summarize the following annotation reasoning into **one short sentence** "
        "You are going to annotate these important attributes from a FAERS case report narrative.\n"
        "Your task is NOT to output the annotated text, but to explain step by step the reasoning process you get the result of identifying and tag entities.\n"
        "Refer to the following guidelines when crafting your reasoning:\n"
        "- Explain clearly how you identify the entities related to clinical concepts\n"
        "- Focus on how you decide tag boundaries, select tag types, and ensure no overlap."
    )
    # 调用 70B 模型获取思路
    response = client.chat.completions.create(
        model=llm_model_name,
        messages=[
            {"role": "system", "content": user_system_prompt},
            {"role": "user", "content": f"Explain your reasoning for annotating this sentence: {text}"}
        ],
        temperature=0.0,
        max_tokens=300
    )
    # 返回完整的思路文本
    reasoning = response.choices[0].message.content.strip()
    return reasoning




def lora_annotate(text):
    # 获取 reasoning
    reasoning_full = get_70b_reasoning(text)
    # 提取首句或精简 reasoning（例如只保留前三句话）
    reasoning_hint = "\n".join(reasoning_full.split("\n")[:3])
    # 构建 system prompt，插入精简 reasoning 作为额外提示
    system_prompt = (
        f"[Annotation hint from reasoning]: {reasoning_hint}\n\n"
        "You are a biomedical annotation expert. Your task is to identify and tag only clinical entities using XML-style tags.\n\n"
        "Strict rules:\n"
        "1. Only wrap recognized entities with the following tags: <Drug>, <AE>, <Treatment>, <RO>, <Dx>, <CoD>, <Age>, <Sex>, <Dose>, <bSYM>, <Hx>, <Status>, <Lab>, <TEMPO>.\n"
        "2. Absolutely do NOT create or use tags outside this list. Tags like <Company>, <Prof>, <CorrectedOutput> are forbidden.\n"
        "3. Do not explain, summarize, or say 'Corrected output'. Just return the one annotated sentence and nothing else.\n"
        "4. Do not change or omit any part of the original sentence. If no tags are needed, return the sentence unchanged.\n\n"
        "For example, the original sentence may be:\n"
        "\"Concomitant medications included on an unknown Date, atenolol tablet at a dose of 25 milligrams twice a day via unknown route for unknown indication\"\n"
        "The annotated sentence should be:\n"
        "\"Concomitant medications included on an unknown Date, <Drug>atenolol</Drug> tablet at a dose of <Dose>25 milligrams</Dose> twice a day via unknown route for unknown indication\"\n\n"
        "Annotation types:\n"
        "    'Drug': Drug Products\n"
        "    'AE': Adverse Event\n"
        "    'Treatment': Drug used for Treatment\n"
        "    'RO': Rule-out diagnosis\n"
        "    'Dx': Diagnostic test results\n"
        "    'CoD': Cause of Death\n"
        "    'Age': Age\n"
        "    'Sex': Sex\n"
        "    'Dose': Dose Administered\n"
        "    'bSYM': Baseline Symptom before AE Occurred\n"
        "    'Hx': Medical History\n"
        "    'Status': Patient Status\n"
        "    'Lab': Laboratory Finding\n"
        "    'TEMPO': Temporal Information, such as Date, Time, etc.\n\n"
        f"Here is the sentence to annotate:\n\n\"{text}\"\n\nPlease output the annotated sentence now:"
    )
    # 接下来跟你现有生成逻辑一致
    prompt = f"{system_prompt}\n\"{text}\"\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return clean_output(decoded)

import re


def clean_output(output_text):
    """
    提取第一个以 Output: 开头，后接英文双引号括起的标注结果句子，包括引号本身。
    如果没有符合的格式，则返回原始文本。
    """
    match = re.search(r'Output:\s*"[^"]*"\s*', output_text)
    if match:
        return match.group(0).strip()
    return output_text.strip()



# def lora_annotate(text):
#     # 第一步：从 70B 模型获取 prompt hints
#     hint_prompt = get_70b_hints(text)
#     print("hint_prompt: ",hint_prompt)
#     # 第二步：构建最终 system_prompt，将 hint 与固定规则结合
#     system_prompt = (
#         hint_prompt + "\n"
#         "You are a biomedical annotation expert. Your task is to identify and tag only clinical entities using XML-style tags.\n\n"
#         "Strict rules:\n"
#         "1. Only wrap recognized entities with the following tags: <Drug>, <AE>, <Treatment>, <RO>, <Dx>, <CoD>, <Age>, <Sex>, <Dose>, <bSYM>, <Hx>, <Status>, <Lab>, <TEMPO>.\n"
#         "2. Absolutely do NOT create or use tags outside this list. Tags like <Company>, <Prof>, <CorrectedOutput> are forbidden.\n"
#         "3. Do not explain, summarize, or say 'Corrected output'. Just return the one annotated sentence and nothing else.\n"
#         "4. Do not change or omit any part of the original sentence. If no tags are needed, return the sentence unchanged.\n\n"
#         "For example, the original sentence may be:\n"
#         "\"Concomitant medications included on an unknown Date, atenolol tablet at a dose of 25 milligrams twice a day via unknown route for unknown indication\"\n"
#         "The annotated sentence should be:\n"
#         "\"Concomitant medications included on an unknown Date, <Drug>atenolol</Drug> tablet at a dose of <Dose>25 milligrams</Dose> twice a day via unknown route for unknown indication\"\n\n"
#         "Annotation types:\n"
#         "    'Drug': Drug Products\n"
#         "    'AE': Adverse Event\n"
#         "    'Treatment': Drug used for Treatment\n"
#         "    'RO': Rule-out diagnosis\n"
#         "    'Dx': Diagnostic test results\n"
#         "    'CoD': Cause of Death\n"
#         "    'Age': Age\n"
#         "    'Sex': Sex\n"
#         "    'Dose': Dose Administered\n"
#         "    'bSYM': Baseline Symptom before AE Occurred\n"
#         "    'Hx': Medical History\n"
#         "    'Status': Patient Status\n"
#         "    'Lab': Laboratory Finding\n"
#         "    'TEMPO': Temporal Information, such as Date, Time, etc.\n\n"
#         f"Here is the sentence to annotate:\n\n\"{text}\"\n\nPlease output the annotated sentence now:"
#     )
#     # 准备调用 LoRA 模型
#     prompt = f"{system_prompt}\n\"{text}\"\nOutput:"  
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=300,
#             do_sample=False,
#             temperature=0.0,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#         )
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#     return clean_output(decoded)





# def lora_annotate(text):
#     # system_prompt = (
#     #     "You are an expert biomedical annotator. Your task is to annotate important clinical entities in the given text using XML-style tags.\n\n"
#     #     "Rules:\n"
#     #     "1. Do NOT change or add any words, only add XML tags around the exact entity words.\n"
#     #     "2. Use ONLY the following tags: <Drug>, <AE>, <Treatment>, <RO>, <Dx>, <CoD>, <Age>, <Sex>, <Dose>, <bSYM>, <Hx>, <Status>, <Lab>, <TEMPO>.\n"
#     #     "3. Output ONLY the annotated sentence once, without any explanation or extra text.\n"
#     #     "4. Do NOT repeat or duplicate the sentence.\n"
#     #     "5. If no entities are found, output the original sentence exactly as is.\n\n"
#     #     f"Here is the sentence to annotate:\n\n\"{text}\"\n\nPlease output the annotated sentence now:"
#     # )
# #     system_prompt = (
# #     "You are an expert biomedical annotator. Your task is to annotate important clinical entities in the given text using XML-style tags.\n\n"
# #     "Rules:\n"
# #     "0. Output the entire original sentence exactly, only adding XML tags around recognized entities, do NOT omit or shorten anything.\n"
# #     "1. Do NOT change or add any words, only add XML tags around the exact entity words.\n"
# #     "2. Use ONLY the following tags: <Drug>, <AE>, <Treatment>, <RO>, <Dx>, <CoD>, <Age>, <Sex>, <Dose>, <bSYM>, <Hx>, <Status>, <Lab>, <TEMPO>.\n"
# #     "3. Output ONLY the annotated sentence once, without any explanation or extra text.\n"
# #     "4. Do NOT repeat or duplicate the sentence.\n"
# #     "5. If no entities are found, output the original sentence exactly as is.\n\n"
# #     f"Here is the sentence to annotate:\n\n\"{text}\"\n\nPlease output the annotated sentence now:"
# # )   ###this is the result 2 in text file
# #     system_prompt = (
# #     "You are a biomedical annotation expert. Your task is to identify and tag only clinical entities using XML-style tags.\n\n"
# #     "Strict rules:\n"
# #     "1. Only wrap recognized entities with the following tags: <Drug>, <AE>, <Treatment>, <RO>, <Dx>, <CoD>, <Age>, <Sex>, <Dose>, <bSYM>, <Hx>, <Status>, <Lab>, <TEMPO>.\n"
# #     "2. Absolutely do NOT create or use tags outside this list. Tags like <Company>, <Prof>, <CorrectedOutput> are forbidden.\n"
# #     "3. Do not explain, summarize, or say 'Corrected output'. Just return the one annotated sentence and nothing else.\n"
# #     "4. Do not change or omit any part of the original sentence. If no tags are needed, return the sentence unchanged.\n\n"
# #     "For example, the original sentence may be:\n"
# #     "\"Concomitant medications included on an unknown Date, atenolol tablet at a dose of 25 milligrams twice a day via unknown route for unknown indication\"\n"
# #     "The annotated sentence should be:\n"
# #     "\"Concomitant medications included on an unknown Date, <Drug>atenolol</Drug> tablet at a dose of <Dose>25 milligrams</Dose> twice a day via unknown route for unknown indication\"\n\n"
# #     "Annotation types:\n"
# #     "    'Drug': Drug Products\n"
# #     "    'AE': Adverse Event\n"
# #     "    'Treatment': Drug used for Treatment\n"
# #     "    'RO': Rule-out diagnosis\n"
# #     "    'Dx': Diagnostic test results\n"
# #     "    'CoD': Cause of Death\n"
# #     "    'Age': Age\n"
# #     "    'Sex': Sex\n"
# #     "    'Dose': Dose Administered\n"
# #     "    'bSYM': Baseline Symptom before AE Occurred\n"
# #     "    'Hx': Medical History\n"
# #     "    'Status': Patient Status\n"
# #     "    'Lab': Laboratory Finding\n"
# #     "    'TEMPO': Temporal Information, such as Date, Time, etc.\n\n"
# #     f"Here is the sentence to annotate:\n\n\"{text}\"\n\nPlease output the annotated sentence now:"
# # )    ###this is the result 3 in text file
#     # system_prompt = (
#     #     "You are a biomedical annotation expert. Your task is to identify and tag only clinical entities using XML-style tags.\n\n"
#     #     "Strict rules:\n"
#     #     "1. Only wrap recognized entities with the following tags: <Drug>, <AE>, <Treatment>, <RO>, <Dx>, <CoD>, <Age>, <Sex>, <Dose>, <bSYM>, <Hx>, <Status>, <Lab>, <TEMPO>.\n"
#     #     "2. Absolutely do NOT create or use tags outside this list. Tags like <Company>, <Prof>, <CorrectedOutput> are forbidden.\n"
#     #     "3. Do not explain, summarize, or say 'Corrected output'. Just return the one annotated sentence and nothing else.\n"
#     #     "4. Do not change or omit any part of the original sentence. If no tags are needed, return the sentence unchanged.\n\n"
#     #     "5. Please be sure to annotate **all temporal expressions** such as dates, times, or durations with the <TEMPO> tag."
#     #     "For example, the original sentence may be:\n"
#     #     "\"Concomitant medications included on an unknown Date, atenolol tablet at a dose of 25 milligrams twice a day via unknown route for unknown indication\"\n"
#     #     "The annotated sentence should be:\n"
#     #     "\"Concomitant medications included on an unknown <TEMPO>Date</TEMPO>, <Drug>atenolol</Drug> tablet at a dose of <Dose>25 milligrams</Dose> twice a day via unknown route for unknown indication\"\n\n"
#     #     "Second example, the original sentence may be:\n"
#     #     "\"Patient was given ibuprofen 200mg daily on 5th July for pain\"\n"
#     #     "The annotated sentence should be:\n"
#     #     "\"Patient was given <Drug>ibuprofen</Drug> <Dose>200mg</Dose> daily on <TEMPO>5th July</TEMPO> for <Treatment>pain</Treatment>\"\n\n"
#     #     "Annotation types:\n"
#     #     "    'Drug': Drug Products\n"
#     #     "    'AE': Adverse Event\n"
#     #     "    'Treatment': Drug used for Treatment\n"
#     #     "    'RO': Rule-out diagnosis\n"
#     #     "    'Dx': Diagnostic test results\n"
#     #     "    'CoD': Cause of Death\n"
#     #     "    'Age': Age\n"
#     #     "    'Sex': Sex\n"
#     #     "    'Dose': Dose Administered\n"
#     #     "    'bSYM': Baseline Symptom before AE Occurred\n"
#     #     "    'Hx': Medical History\n"
#     #     "    'Status': Patient Status\n"
#     #     "    'Lab': Laboratory Finding\n"
#     #     "    'TEMPO': Temporal Information, such as Date, Time, etc.\n\n"
#     #     f"Here is the sentence to annotate:\n\n\"{text}\"\n\nPlease output the annotated sentence now:"
#     # )  
# #     system_prompt = (
# #     "You are a biomedical annotation expert. Annotate only clinical entities with these tags: "
# #     "<Drug>, <AE>, <Treatment>, <RO>, <Dx>, <CoD>, <Age>, <Sex>, <Dose>, <bSYM>, <Hx>, <Status>, <Lab>, <TEMPO>.\n"
# #     "Do NOT use any other tags or add extra text.\n"
# #     "Do NOT change or omit any text.\n"
# #     "Output ONLY the single annotated sentence, nothing else.\n"
# #     "\n"
# #     "Examples:\n"
# #     "Original: Concomitant medications included on an unknown Date, atenolol tablet at a dose of 25 milligrams twice a day.\n"
# #     "Annotated: Concomitant medications included on an unknown <TEMPO>Date</TEMPO>, <Drug>atenolol</Drug> tablet at a dose of <Dose>25 milligrams</Dose> twice a day.\n"
# #     "\n"
# #     f"Now annotate:\n\"{text}\"\n"
# #     "Output ONLY the annotated sentence:"
# # )
#     prompt = f"{system_prompt}\n\n{text}\n\nThe annotated text is shown as below:"
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=300,  # 合理限制长度，避免重复生成
#             do_sample=False,
#             temperature=0.0,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#         )
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#     return clean_output(decoded)

def clean_output(output_text):
    """
    提取“The annotated text is shown as below:”后第一次标注的句子，并去除多余重复内容。
    """
    pattern = r"The annotated text is shown as below:\s*(.*?)(?=(?:\n{2,}|The annotated text is shown as below:|$))"
    match = re.search(pattern, output_text, re.DOTALL)
    if match:
        result = match.group(1).strip()
        # optional: 去除句子末尾多余的句号或空格
        return result.rstrip(". \n") + "."
    return output_text.strip()



import gc
import torch

# 如果你已经不需要继续用这部分模型
del model
del tokenizer
del base_model
del data
del output

gc.collect()
torch.cuda.empty_cache()


###look at which process occupy the GPU
# nvidia-smi --query-compute-apps=pid,used_memory --format=csv





#####most accurate so far
def lora_annotate(text):
    system_prompt = """You are going to annotate these important attributes that were reported in a FAERS case report narrative. Annotate with the following annotation types.
The annotation should be in-text, which means you should not alter the original text but only adding tags surround the term-of-interest.
For example, the original sentence may be: "Concomitant medications included on an unknown <Date>, atenolol tablet at a dose of 25 milligrams twice a day via unknown route for unknown indication"
The annotated sentence should be "Concomitant medications included on an unknown Date, <Drug>atenolol</Drug> tablet at a dose of <Dose>25 milligrams</Dose> twice a day via unknown route for unknown indication"
Annotation types:
    'Drug': 'Drug Products',
    'AE': 'Adverse Event',
    'Treatment': 'Drug used for Treatment',
    'RO': 'Rule-out diagnosis',
    'Dx': 'Diagnostic test results',
    'CoD': 'Cause of Death',
    'Age': 'Age',
    'Sex': 'Sex',
    'Dose': 'Dose Administered',
    'bSYM': 'Baseline Symptom before AE Occurred',
    'Hx': 'Medical History',
    'Status': 'Patient Status',
    'Lab': 'Laboratory Finding',
    'TEMPO': 'Temporal Information, such as Date, Time, etc.'
Output Guidance:
    Be cautious. Do NOT alter the original text. Only add tags of Annotation_type such as <AE>...</AE> to the identified entity.
    The tag must be complete. Do not miss any right > or left <.
    The output should start with: The annotated text is shown as below:"""
    
    user_prompt = f"""Sentence:
{text}"""
    # 拼接成 Chat 模式的 prompt
    prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # 去除前面的 prompt（只保留模型生成的回答）
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]", 1)[-1].strip()
    return clean_output(decoded)


import re

def clean_output(output_text):
    """
    提取"The annotated text is shown as below:"后面紧跟的一行内容（去除前后空格）。
    如果找不到该行，则返回原始文本。
    """
    lines = output_text.splitlines()
    for i, line in enumerate(lines):
        if "The annotated text is shown as below:" in line:
            # 返回紧接其后的一行（如果存在）
            if i + 1 < len(lines):
                return lines[i + 1].strip()
            else:
                return ""
    return output_text.strip()

output = []
for item in data[0:7]:
    text = item["text"]
    print("text: ", text)
    predicted = lora_annotate(text)
    print(predicted)
    example = postprocess_prediction(text, predicted)
    output.append(example)



nohup python llama_test.py > llama_test.log &