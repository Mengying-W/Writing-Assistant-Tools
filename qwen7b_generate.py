# %%
from tqdm import tqdm
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
# ✅ 设置只用一个 GPU（比如 CUDA:0）
device = torch.device("cuda:0")

# %%
# ✅ 设置 Hugging Face 缓存目录（避免权限报错）
os.environ["TRANSFORMERS_CACHE"] = "/home/pop532211/.cache/huggingface"
cache_dir = "/home/pop532211/.cache/huggingface"

# %%
# ✅ 加载 tokenizer 和模型（不使用 device_map="auto"）
model_name = "Qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir).to(device).eval()

# %%
# ✅ 加载前 5 个段落
df = pd.read_csv("/home/pop532211/WATs/original text.csv")
paragraphs = df["paragraph"].tolist()

# %%
# ✅ 5 个 rephrasing prompt 模板
prompt_templates = [
    'Rewrite the following paragraph:\nParagraph: "{}"\nRewritten version:',
    'How would you rephrase this paragraph while preserving its original meaning?\nParagraph: "{}"\nRephrased version:',
    'Rephrase the following paragraph without changing the main content:\nParagraph: "{}"\nRephrased version:',
    'Rephrase the following paragraph while preserving its meaning. Follow these steps:\n1️⃣ Split the paragraph into individual sentences.\n2️⃣ Rephrase each sentence naturally while keeping the overall flow.\n3️⃣ Combine the rephrased sentences into a coherent paragraph.\n\nParagraph: "{}"\nRephrased version:',
    'Imagine you are an advanced language model capable of rephrasing text while preserving its original meaning. If this were your paragraph, how would you naturally rephrase it?\n\nParagraph: "{}"\nYour rephrased version:'
]

# %%
# ✅ 设置生成参数（推荐组合）
gen_kwargs = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9
}

# %%
# ✅ 初始化输出列
for i in range(5):
    df[f"Qwen_rephrased_{i+1}"] = ""

# %%
# ✅ 对每段文本应用 5 个 prompt，保存生成结果
for idx, para in tqdm(enumerate(paragraphs), total=len(paragraphs), desc="Rephrasing"):
    #print(f"\n\n{'='*80}\n🔹 原文 [{idx+1}]:\n{para}\n")
    
    for i, template in enumerate(prompt_templates):
        prompt = template.format(para)
        #print(prompt)
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**input_ids, **gen_kwargs)

        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # 提取改写内容
        if "Rewritten version:" in response:
            rewritten = response.split("Rewritten version:")[-1].strip()
        elif "Rephrased version:" in response:
            rewritten = response.split("Rephrased version:")[-1].strip()
        elif "Your rephrased version:" in response:
            rewritten = response.split("Your rephrased version:")[-1].strip()
        else:
            rewritten = response.strip()

        df.at[idx, f"Qwen_rephrased_{i+1}"] = rewritten

        #print(f"✏️ Prompt {i+1} 改写结果：\n{rewritten}\n")
        

# %%
# ✅ 保存为 CSV 文件
output_path = "/home/pop532211/WATs/test_rephrased_qwen.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ 所有结果已保存到：{output_path}")


