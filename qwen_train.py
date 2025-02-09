from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch
import joblib
from peft import LoraConfig, get_peft_model

model_save_path = '/code/model_pre/qwen_obti'
# 确保显存足够（至少需要4GB显存）
torch.cuda.empty_cache()

# ----------------------------------
# 自定义数据（仅3条样本）
# ----------------------------------
paragraphs = joblib.load('/code/model_pre/data/novels_dataset29.pkl')

# 转换为 Hugging Face Dataset 格式
dataset = Dataset.from_dict({"text": paragraphs})

# ----------------------------------
# 模型与分词器初始化
# ----------------------------------
# model_name = "huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated"  # 替换成实际的Qwen模型名称
model_path = './pre_model_obi/models--huihui-ai--Qwen2.5-1.5B-Instruct-abliterated/snapshots/60801bee812f39cde0fe5bafc7afca56e3c87eef'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# 设置填充token
tokenizer.pad_token = tokenizer.eos_token

# ----------------------------------
# LoRA配置
# ----------------------------------
lora_config = LoraConfig(
    r=16,                # 提高秩，提高 LoRA 适配能力
    lora_alpha=32,       # 2*r 的标准设置
    target_modules=["q_proj", "v_proj",'k_proj'],  # 适配更多注意力层
    lora_dropout=0.1,    # 适合文本任务
    bias="none",         # 不调整偏置，节省显存
    task_type="CAUSAL_LM"
)

# 应用LoRA到模型
model = get_peft_model(model, lora_config)

# ----------------------------------
# 数据预处理（适应短文本）
# ----------------------------------
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,  # 根据样本最大长度设置
        padding="max_length",
        add_special_tokens=True
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

split_result = tokenized_dataset.train_test_split(test_size=0.2, shuffle=True)
train_dataset = split_result["train"] 
eval_dataset = split_result["test"]

# ----------------------------------
# 训练参数配置（防止过拟合）
# ----------------------------------
training_args = TrainingArguments(
    output_dir="/code/model_pre/qwen_obti/qwen_lora_trained",
    num_train_epochs=30,
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=2,  
    learning_rate=3e-5,              
    warmup_steps=40,   
    lr_scheduler_type="cosine",
    logging_dir='/code/model_pre/qwen_obti/logs_all',               
    logging_steps=5,
    save_strategy="no",
    evaluation_strategy="no",
    fp16=torch.cuda.is_available(),
)

# ----------------------------------
# 数据整理器
# ----------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 使用CLM（因果语言建模）
)

# ----------------------------------
# 早停回调
# ----------------------------------
# early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)  # 如果连续3次评估都没有提升，则停止训练

# ----------------------------------
# 训练执行
# ----------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

# ----------------------------------
# 模型保存
# ----------------------------------
model.save_pretrained("/code/model_pre/qwen_obti/qwen_lora_model_all")
tokenizer.save_pretrained("/code/model_pre/qwen_obti/qwen_lora_model_all")

# ----------------------------------
# 生成测试
# ----------------------------------
generation_config = {
    "max_length": 500,
    "temperature": 0.7,
    "top_k": 30,
    "do_sample": True,
}

test_prompts = ["故事开始了，"]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **generation_config)
    print(f"Prompt: {prompt}")
    print(f"Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")