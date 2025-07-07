# %%
#!pip install pandas

# %%
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from tqdm import tqdm

# %%
# 设置 huggingface 缓存目录，防止权限问题
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

# 模型 ID
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# %%
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True, cache_dir=os.environ["HF_HOME"])
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=True,
    cache_dir=os.environ["HF_HOME"]
)


# %%
# 读取原始 CSV 文件
input_path = "/home/pop532211/WATs/original text.csv"
df = pd.read_csv(input_path)
paragraphs = df["paragraph"].tolist()

# %%
# 定义 5 个不同的 prompt 模板
prompts = [
    'Rewrite the following paragraph:\nParagraph: "{}"\nRewritten version:',
    'How would you rephrase this paragraph while preserving its original meaning?\nParagraph: "{}"\nRephrased version:',
    'Rephrase the following paragraph without changing the main content:\nParagraph: "{}"\nRephrased version:',
    'Rephrase the following paragraph while preserving its meaning. Follow these steps:\n1️⃣ Split the paragraph into individual sentences.\n2️⃣ Rephrase each sentence naturally while keeping the overall flow.\n3️⃣ Combine the rephrased sentences into a coherent paragraph.\n\nParagraph: "{}"\nRephrased version:',
    'Imagine you are an advanced language model capable of rephrasing text while preserving its original meaning. If this were your paragraph, how would you naturally rephrase it?\n\nParagraph: "{}"\nYour rephrased version:'
]

# %%
# 重写函数
def generate_rephrased(text, prompt_template):
    prompt = prompt_template.format(text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("version:")[-1].strip()

# %%
# 生成所有重写版本
output_data = {"paragraph": paragraphs}
for i, prompt in enumerate(prompts):
    rephrased_list = []
    for paragraph in tqdm(paragraphs, desc=f"Prompt {i+1}"):
        try:
            result = generate_rephrased(paragraph, prompt)
        except Exception as e:
            result = f"[ERROR: {e}]"
        rephrased_list.append(result)
    output_data[f"rephrased_{i+1}"] = rephrased_list

# %%
# 保存结果
output_df = pd.DataFrame(output_data)
output_df.to_csv("llama3_rephrased_output.csv", index=False)
print("✅ 完成！结果已保存为 llama3_rephrased_output.csv")


