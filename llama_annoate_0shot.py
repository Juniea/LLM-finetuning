# LLM_URL=http://ncslphpcgpu02.fda.gov:11434/v1
# LLM_KEY=dbbllmforresearchonly
# LLM_MODEL=llama-3.3-70b-instruct


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
 
def call_llm(message, prompt='Help answer the following requests.', max_token=100000):
    if not message:
        return 'No input!'
    chat_response = client.chat.completions.create(
        # model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        model=llm_model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        max_tokens=max_token
    )
   
    answer = chat_response.choices[0].message.content
    return answer


response = call_llm("列出三种常见的副作用。")
print(response)

def annotate_text(text):
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
    The output should start with:
    The annotated text is shown as below:"""
    # 使用 call_llm 函数
    result = call_llm(text, prompt=system_prompt)
    # 提取真正的 annotated text（移除前缀）
    if "The annotated text is shown as below:" in result:
        return result.split("The annotated text is shown as below:")[-1].strip()
    else:
        return result.strip()



annotated_data = []

for item in data:
    original_text = item["text"]
    annotated_text = annotate_text(original_text)
    print(annotated_text)
    annotated_data.append({
        "original": original_text,
        "annotated": annotated_text
    })

# 打印结果
for entry in annotated_data:
    print("Original:", entry["original"])
    print("Annotated:", entry["annotated"])
    print("---")
