from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

import torch

# 加载微调后的模型文件
model_path = "/code/model_pre/qwen_obti/qwen_lora_model_all"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(prompt, max_length=100, temperature=0.7, top_k=30):
    """ 使用训练好的模型生成文本 """
    
    # 将输入转换为张量
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 生成文本
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=temperature,
        top_k=top_k,
        do_sample=True  # 采用采样策略，增加生成多样性
    )
    
    # 解码输出
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_story_segment(prompt, previous_text=None):
    if previous_text:
        prompt = previous_text + "\n\n" + prompt  # 将上一段的结尾与新段落的开头合并
    
    generation_config = {
    "max_new_tokens": 700,
    "temperature": 0.7,
    "top_k": 30,
    "do_sample": True,
}

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **generation_config)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text



# 测试生成
prompt = "你想要的故事开始了！"
generated_text = generate_text(prompt,max_length=800,temperature=0.7)
print(f"Prompt: {prompt}")
print(f"Generated Text: {generated_text}")



