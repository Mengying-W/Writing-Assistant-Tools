# %%
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
# 设置模型名称和设备
model_id = "CohereLabs/aya-23-8B"
device = torch.device("cuda:0")

# %%
cache_dir = "/home/pop532211/hf_cache"


# %%
# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    device_map={"": 0}  # 强制只用第0张GPU
)

# %%
# 读取前5条段落
df = pd.read_csv("/home/pop532211/WATs/original text.csv")
paragraphs = df["paragraph"].tolist()

# %%
# 定义5个 prompt 模板
prompts = [
    'Rewrite the following paragraph:\nParagraph: "{}"\nRewritten version:',
    'How would you rephrase this paragraph while preserving its original meaning? \nParagraph: "{}"\nRephrased version:',
    'Rephrase the following paragraph without changing the main content: \nParagraph: "{}"\nRephrased version:',
    'Rephrase the following paragraph while preserving its meaning. Follow these steps:\n1️⃣ Split the paragraph into individual sentences.\n2️⃣ Rephrase each sentence naturally while keeping the overall flow.\n3️⃣ Combine the rephrased sentences into a coherent paragraph.\n\nParagraph: "{}"\nRephrased version:',
    'Imagine you are an advanced language model capable of rephrasing text while preserving its original meaning. If this were your paragraph, how would you naturally rephrase it?\n\nParagraph: "{}"\nYour rephrased version:'
]

# %%
# 定义生成函数
def rephrase(paragraph, prompt_template):
    prompt = prompt_template.format(paragraph.strip())
    #print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取回答部分
    if "Rewritten version:" in prompt:
        return decoded.split("Rewritten version:")[-1].strip()
    elif "Rephrased version:" in prompt:
        return decoded.split("Rephrased version:")[-1].strip()
    elif "Your rephrased version:" in prompt:
        return decoded.split("Your rephrased version:")[-1].strip()
    else:
        return decoded.strip()

# %%
# 对每个 prompt 执行生成
for i, prompt_template in enumerate(prompts):
    column_name = f"Aya23_Prompt{i+1}"
    rephrased_list = []
    for para in tqdm(paragraphs, desc=f"Prompt {i+1}"):
        rephrased = rephrase(para, prompt_template)
        #print(rephrased)
        rephrased_list.append(rephrased)
    df[column_name] = rephrased_list

# %%
# 保存输出
output_path = "/home/pop532211/WATs/aya23_test.csv"
df.to_csv(output_path, index=False)
print(f"✅ 已完成 Aya-23-8B 对前5条段落的 rephrase，结果保存在：{output_path}")


