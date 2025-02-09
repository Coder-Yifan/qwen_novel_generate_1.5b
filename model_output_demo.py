
# 测试本地预训练模型文件
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "huihui-ai/Qwen2.5-1.5B-Instruct-abliterated"
model_path = './pre_model_obi/models--huihui-ai--Qwen2.5-1.5B-Instruct-abliterated/snapshots/60801bee812f39cde0fe5bafc7afca56e3c87eef'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(prompt, max_length=100, temperature=0.7, top_k=30):
    """ 使用训练好的模型生成文本 """
    
    # 将输入转换为张量
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 生成文本
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        do_sample=True  # 采用采样策略，增加生成多样性
    )
    
    # 解码输出
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 测试生成
prompt = "故事开始了"
generated_text = generate_text(prompt,max_length=200,temperature=0.8)
print(f"Prompt: {prompt}")
print(f"Generated Text: {generated_text}")


