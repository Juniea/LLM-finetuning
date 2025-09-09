# LLM_URL=http://ncslphpcgpu02.fda.gov:11434/v1
# LLM_KEY=dbbllmforresearchonly
# LLM_MODEL=llama-3.3-70b-instruct


from tqdm.notebook import tqdm
import os, json
import pandas as pd
### using HPC LLM API
from openai import OpenAI

LLM_URL="http://ncslphpcgpu02.fda.gov:11434/v1"
LLM_KEY="dbbllmforresearchonly"
LLM_MODEL="llama-3.3-70b-instruct" 

# LLM_URL="http://ncslp30300:8814/v1"
# LLM_KEY="dbbllmforresearchonly"
# LLM_MODEL="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
# LLM_MODEL="meta-llama/Llama-3.3-70B-Instruct"

openai_api_key = LLM_KEY
openai_api_base = LLM_URL 

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def call_llm(message, prompt='Help answer the following requests.', max_token=1024):
    if not message:
        return 'No input!'
    chat_response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        max_tokens=max_token,
        temperature = 0
    )
    
    answer = chat_response.choices[0].message.content
    return answer

response = call_llm("列出三种常见的副作用。")
print(response)

def annotate_text(text):
    # prompt_ner= '''**You are an expert medical annotator tasked with identifying and tagging key clinical and contextual attributes in a FAERS (FDA Adverse Event Reporting System) case report narrative. Follow these instructions meticulously:**

    # ### ✳️ Annotation Format and Rules
    # 1. Perform **in-text annotation** by enclosing the relevant text with appropriate XML-style tags.
    # 2. **Do not alter the original text in any way**. Your task is solely to insert annotation tags.
    # 3. Use capitalized tags in this format: `<TAG>relevant text</TAG>`.
    # 4. Ensure all tags are properly opened and closed. For example: `<SDRUG>atenolol</SDRUG>`.
    # 5. Be precise in your tag placement. Include only the specific words that correspond to the entity being tagged.

    # ### 🏷️ Tag Types
    # Include a list of all possible tags here, such as SDRUG, CDRUG, AE, DATE, DOSE, etc., with a brief description of each.

    # ### ✅ Annotation Example  
    # Original text:  
    # Concomitant medications included on an unknown date, atenolol tablet at a dose of 25 milligrams twice a day via unknown route for unknown indication.

    # Correctly annotated text:  
    # Concomitant medications included on an unknown date, atenolol tablet at a dose of 25 milligramstwice a day via unknown route for unknown indication.

    # ### 🚫 Common Mistakes to Avoid
    # 1. Incomplete tags (e.g., `<SDRUG>atenolol` without a closing tag)
    # 2. Altering the original text content
    # 3. Inconsistent capitalization of tags
    # 4. Overlapping or nested tags

    # Your task is to apply these annotation principles to the provided FAERS case report narrative. Maintain accuracy and consistency throughout your annotations.
    # ---

    # ### 🔖 Annotation Types and Their Descriptions  
    # Use the following table as your reference to annotate the provided text:

    # | Clinical Concept/Temporal Feature | Description | Annotation Rule | Trigger Words |
    # |-----------------------------------|-------------|-----------------|---------------|
    # | SDrug | A drug or biological product believed to have caused, contributed to, or been associated with the adverse event. | The drug is explicitly linked to the AE through temporal or causal language. A clinical action was taken on the drug (e.g., discontinuation, dose adjustment). | Suspected, implicated, caused, linked to, attributed to, associated with, following administration of, resolved after stopping, symptoms worsened after increasing dose. |
    # | CDrug | Drugs that were concurrently administered with other drugs (e.g., suspect drugs), as part of the patient's ongoing or routine regimen. | The drug is mentioned as part of the patient's medication list and taken concurrently. The drug is mentioned as part of the patient's medication regimen but lacks a causal link to AE. | Background therapy, chronic therapy, maintained on, patient's usual medications, taken concurrently, other medications included, concomitant drug. |
    # | ODrug | Include drugs mentioned but not clearly linked to current adverse event, concomitant drugs or therapy, or explicit treatment purpose. Illicit substances, whether explicitly relief, supportive care, or as a response to an adverse event. | Annotate references to specific drugs (both suspect and concomitant) as "drug" when they are listed without symptoms, adverse events, or causal indicators directly linked to them. Include any references to the disposition of drugs (e.g. "Drug discontinued," "dose adjusted"). | Illicit substances, past medications, drug history, drug class, general drug references without clinical context, drug under investigation, dosage, frequency, route, supportive care, absence of treatment. |
    # | Dose | Explicitly stated dosage information of any drugs. | Annotate dosage quantity, frequency, adjustments, or changes. | Mg, dose, once daily, twice daily, adjusted, increased, decreased. |
    # | Treatment| Capture only drug products/drug administration explicitly described as treatments addressing disease, adverse events or symptoms. | Annotate drug names when explicitly stated as treatments or therapeutic measures in the narrative. | Treatment, therapy, managed by, treatment with, intervention (when drug-based). |
    # | AE| Any negative health outcome, condition, or symptom that could represent an adverse event, regardless of their relation to the drug or baseline status. | Annotate as Adverse Event: Diagnosed conditions and symptoms that are explicitly described as occurring during or immediately after adverse event. | Caused by, due to, result of, associated with, linked to, following, induced by, after starting, led to, Feels, reports, experiences, complains of, sensation of, unwell, discomfort, suffers from, dx, diagnosed with, developed. |
    # | mAE| mAE-Manifestations: immediate signs, symptoms that appear as part of the adverse event. mAE-Sequelae: consequences or complications resulting from the adverse event. | Annotate signs and symptoms that are explicitly described as occurring during or immediately after adverse event. Annotate consequences or complications resulting from the adverse event. | After taking, developed, symptoms included, resulted in, caused by, following resolution of, long-term effects included. |
    # | bSYM| Symptoms, conditions, or findings that existed prior to the adverse event and are unrelated to it. | Describe pre-existing conditions unrelated to the adverse event. May be chronic conditions or ongoing symptoms before AE onset. | Prior to, pre-existing, before the onset of. |
    # | RO | Conditions considered but ultimately ruled out as causes of symptoms or adverse events. | Annotate as Rule-Out Diagnosis if the language indicates that a condition was ruled out as a cause. | Ruled out, R/O, no evidence of, excluded. |
    # | Dx | Captured only the names of diagnostic procedures performed to evaluate or confirm a medical condition or diagnosis. | Annotate the procedure name (e.g. Biopsy, CT, MRI, endoscopy, ultrasound). | Imaging test: CT, MRI, ultrasound, X-ray, radiograph. Diagnostic procedures: biopsy, histopathology, colonoscopy, endoscopy. |
    # | CoD | Specific cause or reason for patient death that may be related to the drug. | Annotate as Cause of Death if causative language specifically attributes the death to a condition or event potentially related to the drug. | |
    # | Lab | Results (test named and specific results) specifically from laboratory tests indicating quantitative measurements or biochemical markers. | Annotate as Laboratory Finding if: Test and result indicators: results, level, measurement, value, findings. Test name: include test results not included/reported. | Normal and negative indicators: normal, within range, negative, unremarkable. Abnormal and positive indicators: elevated, increased, decreased, positive for, abnormal, high, low. Specific tests: names of lab tests, hemoglobin, blood test, urine analysis, serum measurement, height, weight, BMI. |
    # | FHx | Patient's family medical history, often related to genetic predispositions. | Annotate as Family History if the condition is described in the context of family members and not linked to drug-related events. | Family history, FHx, inherited, genetic predisposition. |
    # | MHx | Symptoms, conditions, or medical findings that pre-existed before the adverse event and are not caused by it. | Annotate any medical condition or symptom clearly described as present prior to the AE. | Past medical history, history of, baseline, chronic, prior diagnosis, pre-existing, known condition, underlying, previously diagnosed, maintained on. |
    # | IND | The reason or intended medical purpose for which a drug, treatment, or procedure is used. | Annotate as Indication if the narrative states that a drug or treatment is used for a specific condition or symptom. | Used for, given for, prescribed for, indicated for, to treat, for the management of, for prevention of, due to, because of, unknown indication, reason for use unclear. |
    # | Status | Captures statements about the progression or outcome of either the patient's overall clinical condition or the adverse event status after a treatment, or intervention. | Annotate as Patient Status if: Resolution or recovery, deterioration or worsening, stability or no change, long-term effects or complications. | Admitted, discharged, recovered, deteriorated, outcome, improvement, unknown, resolved, unchanged, persistent symptoms, stable. |
    # | Age | The age of the patient when the AE occurred or during the described event. | The age of the patient explicitly mentioned in the text. | Exact or approximate age is mentioned. Adult, older, infant, newborn. |
    # | Sex | The biological sex of the patient, as described in the clinical narrative. | The biological sex of the patient explicitly mentioned in the text. | Biological sex is explicitly mentioned. Boy, girl, female, male. |
    # | Date | Specific or partial dates provided in the text. | Annotate as Date if the text specifies an exact or partial date. | On, in, exact dates (e.g., 15-MAR-2007, November 1991, 2005). |
    # | Time | Specific times of day mentioned in the report. | Annotate as Time if the text specifies a clock time. | AM, PM, morning, evening, specific times (e.g., 9pm). Minutes, hours, days, weeks, months, years. |
    # | Duration | Length of time over which a clinical event occurs or persists. | Annotate as Duration if the text describes a time span. | |
    # | Relative | Time expressions relative to other events. | Annotate as Relative if the text indicates time relative to another event. | Before, after, following, within, later. |
    # | Latency | Refers to the time interval that occurs between an initial event and a subsequent adverse event. | Mark expressions that specify the amount of time passing between an intervention or exposure and the occurrence of an adverse event as latency. | After, within, following. |
    # | Temporal | Involves annotating explicit temporal markers that show the timing of adverse events. | Annotate only those parts of the text that use explicit time-related expressions indicating when an adverse event occurs in relation to others. | Hours later, days before, next morning. |

    # > 🔔 *Note:* Adverse Events should be tagged as `<AE>...</AE>` only when clearly distinguishable from general diagnoses.
    # > 🔔 *Note:* The tag can only be chosen from the first column, which is from: [SDrug, CDrug, ODrug, Dose, Treatment, AE, mAE, bSYM, RO, Dx, CoD, Lab, FHx, MHx, IND, Status, Age, Sex, Date, Time, Duration, Relative, Latency, Temporal].
    #     DO NOT add other tags.
    # ---

    # ### 📝 Output Instructions  
    # - Begin your output with:  
    # The annotated text is shown as below:  
    # - Follow this with the full narrative, inserting only the annotation tags where appropriate.
    # - Ensure **the original text structure and punctuation remain unchanged** aside from the added tags.

    # '''
    prompt_ner= '''**You are an expert medical annotator tasked with identifying and tagging key clinical and contextual attributes in a FAERS (FDA Adverse Event Reporting System) case report narrative. Follow these instructions meticulously:**

    ### ✳️ Annotation Format and Rules
    1. Perform **in-text annotation** by enclosing the relevant text with appropriate XML-style tags.
    2. **Do not alter the original text in any way**. Your task is solely to insert annotation tags.
    3. Use capitalized tags in this format: `<TAG>relevant text</TAG>`.
    4. Ensure all tags are properly opened and closed. For example: `<SDRUG>atenolol</SDRUG>`.
    5. Be precise in your tag placement. Include only the specific words that correspond to the entity being tagged.

    ### 🏷️ Tag Types
    Include a list of all possible tags here, such as SDRUG, CDRUG, AE, DATE, DOSE, etc., with a brief description of each.

    ### ✅ Annotation Example  
    Original text:  
    Concomitant medications included on an unknown date, atenolol tablet at a dose of 25 milligrams twice a day via unknown route for unknown indication.

    Correctly annotated text:  
    Concomitant medications included on an unknown date, atenolol tablet at a dose of 25 milligramstwice a day via unknown route for unknown indication.

    ### 🚫 Common Mistakes to Avoid
    1. Incomplete tags (e.g., `<SDRUG>atenolol` without a closing tag)
    2. Altering the original text content
    3. Inconsistent capitalization of tags
    4. Overlapping or nested tags

    Your task is to apply these annotation principles to the provided FAERS case report narrative. Maintain accuracy and consistency throughout your annotations.
    ---

    ### 🔖 Annotation Types and Their Descriptions  
    Use the following table as your reference to annotate the provided text:

    Tag	Description	Annotation Rule	Triggers
    SDrug	Drug suspected to cause AE.	Linked to AE with causal or timing cues; action taken.	Suspected, associated with, stopped, worsened after.
    CDrug	Drug used at same time, not linked to AE.	Part of ongoing meds; no AE link.	Concomitant, background therapy, usual meds.
    Drug	Mentioned drug with unclear AE/treatment link.	No clear AE/treatment link or used for history/support.	Drug history, illicit, discontinued, dose adjusted.
    Dose	Describes amount/frequency/changes of drug.	Include dose, frequency, adjustments.	Mg, daily, increased, reduced.
    Treatment	Drug used to treat disease or AE.	Clear mention of drug as therapy.	Treated with, managed by.
    AE	Adverse condition/symptom after drug.	Diagnosed AE or symptom clearly linked to time/event.	Developed, diagnosed with, complained of.
    mAE	Symptom during AE or its consequence.	Immediate or resulting symptoms from AE.	Symptoms included, led to, long-term effect.
    bSYM	Pre-existing symptom/condition before AE.	Mentioned before AE; unrelated.	Prior to, pre-existing.
    RO	Ruled-out diagnoses.	Explicitly excluded conditions.	Ruled out, no evidence.
    Dx	Diagnostic test/procedure name.	Name the test only.	CT, biopsy, MRI, colonoscopy.
    CoD	Cause of death related to drug.	Specific death cause tied to drug/event.	—
    Lab	Test results with values/findings.	Result + test name if possible.	Elevated, decreased, positive, hemoglobin.
    FHx	Family medical history.	Describes family health, not drug-linked.	Family history, inherited.
    MHx	Medical history before AE.	Chronic or baseline conditions.	History of, chronic, baseline.
    IND	Reason for drug/treatment use.	Condition that prompted use.	Prescribed for, indicated for.
    Status	Outcome or progress after AE/treatment.	Resolution, worsening, stable.	Recovered, deteriorated, stable.
    Age	Patient's age at time of event.	Exact or approximate age.	60-year-old, adult, infant.
    Sex	Biological sex of patient.	Explicit mention only.	Male, female, boy, girl.
    Date	Exact or partial date.	Include if calendar-like.	2005, Nov 1991, on Jan 5.
    Time	Specific clock time or moment.	Time of day or event point.	9pm, morning, hours later.
    Duration	Timespan of event/symptom.	Time length expressions.	3 days, two weeks.
    Relative	Time relative to another event.	Temporal relation only.	Before, after, following.
    Latency	Delay between drug and AE.	Time gap from drug to AE.	Within 3 days, 1 week after.
    Temporal	Time markers for AE order.	Use if timing order is stated.	Hours later, next morning.
    > 🔔 *Note:* Adverse Events should be tagged as `<AE>...</AE>` only when clearly distinguishable from general diagnoses.
    > 🔔 *Note:* The tag can only be chosen from the first column, which is from: [SDrug, CDrug, ODrug, Dose, Treatment, AE, mAE, bSYM, RO, Dx, CoD, Lab, FHx, MHx, IND, Status, Age, Sex, Date, Time, Duration, Relative, Latency, Temporal].
        DO NOT add other tags.
    ---

    ### 📝 Output Instructions  
    - Begin your output with:  
    The annotated text is shown as below:  
    - Follow this with the full narrative, inserting only the annotation tags where appropriate.
    - Ensure **the original text structure and punctuation remain unchanged** aside from the added tags.

    '''

    # 使用 call_llm 函数
    result = call_llm(text, prompt=prompt_ner)
    # 提取真正的 annotated text（移除前缀）
    # if "The annotated text is shown as below:" in result:
    #     return result.split("The annotated text is shown as below:")[-1].strip()
    # else:
    #     return result.strip()
    match = re.sub(r'The annotated text is shown as below:\s*', '', result.strip(), flags=re.IGNORECASE)
    if match:
        answer = match
    return answer




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
# spacy_file_path = "valid_FAVERS_R1_LLMLLM_revision_v1.spacy"
spacy_file_path = "valid_FAVERS_R1_v1_SME1_new_revision.spacy"
data = extract_entities_from_spacy_file(spacy_file_path)

import re

for item in data:
    item['text'] = re.sub(r'[\r\n\u21b5]+', ' ', item['text'])

import re
def postprocess_prediction(original_text, annotated_text):
    annotated_output = ""
    all_annotations = []
    offset = 0
    entity_pattern = re.finditer(r"<(?P<label>\w+)>(?P<text>.*?)</\1>", annotated_text)
    for match in entity_pattern:
        last_end = 0
        label = match.group("label")
        ent_text = match.group("text")
        start = original_text.find(ent_text, last_end)
        end = start + len(ent_text)
        # 累积 offset 以匹配原文
        entity_global_start = start
        entity_global_end = end
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


import json

output_file = "annotated_output_test1000_llama70B_R1_v2.json"


# output = []
# for item in data:
#     text = item["text"]
#     print("text: ", text)
#     predicted = annotate_text(text)
#     print(predicted)
#     example = postprocess_prediction(text, predicted)
#     output.append(example)

# 读取已完成的条目数（如存在）
try:
    with open(output_file, "r") as f:
        completed = [json.loads(line) for line in f]
except FileNotFoundError:
    completed = []

# processed_texts = {tuple(item["pages"]) for item in completed}
processed_texts = {tuple(item["pages"]) for item in completed}
 # 用 text 去重

with open(output_file, "a") as f:
    for item in data:
        text = item["text"]
        if text in processed_texts:
            continue  # 跳过已处理的
        try:
            predicted = annotate_text(text)
            example = postprocess_prediction(text, predicted)
            f.write(json.dumps(example) + "\n")
            f.flush()  # 写入磁盘
        except Exception as e:
            print(f"Failed on: {text}")
            print(e)
            break



del data
del output


nohup python llama_test1000_llama70B_v1.py > llama_test1000_llama70B_v1.log &




#########################################################
##############################################################
############################################################
#####################################################################
###correct entites start and end position

import re
import json
from bs4 import BeautifulSoup

def strip_tags(html_text):
    """去除HTML标签，返回纯文本"""
    return BeautifulSoup(html_text, "html.parser").get_text()



def extract_entities(html_text):
    """
    从带有标签的 HTML 文本中提取实体，返回 [(label, text), ...]
    忽略结构标签（如 <p>）
    """
    soup = BeautifulSoup(html_text, "html.parser")
    entities = []
    blacklist = {"p", "div", "span", "br"}
    for tag in soup.find_all():
        label = tag.name.lower()
        if label in blacklist:
            continue  # 跳过结构标签
        text = tag.get_text(strip=True)
        entities.append((label, text))
    return entities

# def generate_annotations_from_annotated_and_pages(pages, annotated_pages):
#     annotations = []
#     for page_idx, (page_text, annotated_text) in enumerate(zip(pages, annotated_pages)):
#         plain_text = strip_tags(annotated_text)
#         # 截取第一个以句号（中英文）结尾的句子
#         first_sent_match = re.search(r'^.*?[。.]', plain_text)
#         if not first_sent_match:
#             print(f"⚠️ Page {page_idx} 找不到以句号结尾的句子")
#             continue
#         first_sentence = first_sent_match.group()
#         # 提取所有实体
#         entities = extract_entities(annotated_text)
#         print(entities,plain_text)
#         # 不过滤实体，全部处理
#         used_spans = []
#         can_locate_all = True
#         for label, ent_text in entities:
#             ent_text_clean = ent_text.strip()
#             candidate_spans = []
#             # 忽略大小写匹配实体在纯文本中的位置
#             for match in re.finditer(re.escape(ent_text_clean), page_text, re.IGNORECASE):
#                 span = (match.start(), match.end())
#                 if all(not (max(span[0], u[0]) < min(span[1], u[1])) for u in used_spans):
#                     candidate_spans.append(span)
#             if not candidate_spans:
#                 print(f"⚠️ Page {page_idx} 找不到实体文本: '{ent_text_clean}'")
#                 can_locate_all = False
#                 break
#             used_spans.append(candidate_spans[0])
#         if not can_locate_all:
#             # 如果找不到实体位置，依然跳过这个页面（你也可以选择继续）
#             continue
#         for label, ent_text in entities:
#             ent_text_clean = ent_text.strip()
#             start, end = None, None
#             for match in re.finditer(re.escape(ent_text_clean), page_text, re.IGNORECASE):
#                 span = (match.start(), match.end())
#                 if span in used_spans:
#                     start, end = span
#                     break
#             annotation = {
#                 "label": label,
#                 "note": "LoRA",
#                 "relationships": {
#                     "date": {"page": page_idx, "text": ""},
#                     "frequency": {"page": page_idx, "text": ""},
#                     "relatives": {"page": page_idx, "text": ""},
#                     "span": {"page": page_idx, "text": ""},
#                     "time": {"page": page_idx, "text": ""}
#                 },
#                 "textContext": {
#                     "start": start,
#                     "end": end,
#                     "page": page_idx,
#                     "text": ent_text_clean,
#                     "text_raw": ""
#                 }
#             }
#             annotations.append(annotation)
#     return annotations

annotations = []
def generate_annotations_from_existing(annotations_raw):
    for ann in annotations_raw:
        label = ann.get("label")
        text_ctx = ann.get("textContext", {})
        start = text_ctx.get("start")
        end = text_ctx.get("end")
        page = text_ctx.get("page")
        text = text_ctx.get("text")
        # 如果 start 是 -1，说明位置有问题，可以选择跳过
        if start is None or end is None or start == -1:
            print(f"⚠️ 跳过无效实体: {text} (label: {label})")
            continue
        annotation = {
            "label": label,
            "note": "LoRA",
            "relationships": {
                "date": {"page": page, "text": ""},
                "frequency": {"page": page, "text": ""},
                "relatives": {"page": page, "text": ""},
                "span": {"page": page, "text": ""},
                "time": {"page": page, "text": ""}
            },
            "textContext": {
                "start": start,
                "end": end,
                "page": page,
                "text": text,
                "text_raw": ""
            }
        }
        annotations.append(annotation)
    return annotations




# 主流程：读取、生成标注、写文件

input_file = "annotated_output_test1000_llama70B_R1_v1.json"
output_file = "annotated_output_test1000_llama70B_filtered_R1_v1.json"

pages_all = []
annotations_all = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        pages_all.extend(record.get("pages", []))
        annotations_all.extend(record.get("annotations", []))



new_annotations = generate_annotations_from_existing(annotations_all)
print(f"生成 {len(new_annotations)} 个标注实体")

# new_records = []

# with open(input_file, "r", encoding="utf-8") as f:
#     for record_idx, line in enumerate(f):
#         line = line.strip()
#         if not line:
#             continue
#         record = json.loads(line)
#         pages = record.get("pages", [])
#         annotated_pages = record.get("annotated_pages", [])
#         page_indices = set(range(len(pages)))
#         # 找出所有 new_annotations 中属于当前这条记录的 annotation（按 page 编号）
#         anns = [ann for ann in new_annotations if ann["textContext"]["page"] in page_indices]
#         # ⚠️ 可选检查 page number 是否偏移
#         min_page = min(ann["textContext"]["page"] for ann in anns) if anns else 0
#         if min_page > 0:
#             # 如果原始数据是拼接过的，你需要把 page 编号偏移回来
#             for ann in anns:
#                 ann["textContext"]["page"] -= min_page
#         new_record = {
#             "pages": pages,
#             "annotated_pages": annotated_pages,
#             "annotations": anns
#         }
#         new_records.append(new_record)

new_records = []
ann_index = 0

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        pages = record.get("pages", [])
        annotated_pages = record.get("annotated_pages", [])
        num_pages = len(pages)
        anns = new_annotations[ann_index: ann_index + num_pages]
        ann_index += num_pages
        new_record = {
            "pages": pages,
            "annotated_pages": annotated_pages,
            "annotations": anns
        }
        new_records.append(new_record)

# 写出
with open(output_file, "w", encoding="utf-8") as f_out:
    for rec in new_records:
        json.dump(rec, f_out, ensure_ascii=False)
        f_out.write("\n")

print(f"✅ 已保存新文件到：{output_file}")

###################################
##############################################
################################################
########################################################
#convert json to spacy
import json
import spacy
from spacy.tokens import DocBin

def json_to_spacy_bin(input_json_path, output_spacy_path):
    import spacy
    from spacy.tokens import DocBin
    import json
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    with open(input_json_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            record = json.loads(line)
            pages = record.get("pages", [])
            print(f"Line {line_num} Pages: {pages}")
            annotations = record.get("annotations", [])
            for page_idx, text in enumerate(pages):
                entities = []
                seen_spans = set()  # 用于去重 (start, end, label)
                for ann in annotations:
                    ctx = ann.get("textContext", {})
                    start = ctx.get("start")
                    end = ctx.get("end")
                    label = ann.get("label")
                    # 基础有效性检查
                    if start is None or end is None or label is None:
                        continue
                    if not (0 <= start < end <= len(text)):
                        print(f"⚠️ 索引越界或无效: start={start}, end={end}, 文本长度={len(text)}")
                        continue
                    # 去重判断，避免重复实体
                    key = (start, end, label.upper())
                    if key in seen_spans:
                        continue
                    seen_spans.add(key)
                    ent_text = text[start:end]
                    print(f"实体: '{ent_text}', 标签: '{label}', 范围: ({start}, {end})")
                    entities.append((start, end, label.upper()))
                # 构造 spaCy Doc 和实体
                if entities or True:  # 你想加判断页是否含实体可调这里
                    doc = nlp(text)
                    ents = []
                    for start, end, label in entities:
                        span = doc.char_span(start, end, label=label, alignment_mode="contract")
                        if span is None:
                            print(f"⚠️ 创建span失败: 文本[{start}:{end}] '{text[start:end]}', 标签 '{label}'")
                            # 尝试 expand 模式
                            span = doc.char_span(start, end, label=label, alignment_mode="expand")
                            if span is None:
                                print(f"⚠️ 扩展模式也失败了")
                                continue
                            else:
                                print(f"✅ 扩展模式成功创建span")
                        ents.append(span)
                    # 设置实体，注意 ents 不应有重叠
                    try:
                        doc.ents = ents
                    except Exception as e:
                        print(f"❌ 设置实体失败: {e}")
                        # 失败时可以考虑只用不重叠的实体或跳过该页
                        continue
                    doc_bin.add(doc)
                    print(f"📄 添加了页面 {page_idx + 1}，包含 {len(ents)} 个实体")
    doc_bin.to_disk(output_spacy_path)
    print(f"✅ spaCy数据保存至: {output_spacy_path}")


# 用法示例
input_json_file = "annotated_output_test1000_llama70B_R1_v1.json"
output_spacy_file = "output_data_1000llama70B_R1_v1.spacy"

json_to_spacy_bin(input_json_file, output_spacy_file)


nohup python test_llama_accuracy_1000_llama70B.py > test_llama_accuracy_1000_llama70B.log &