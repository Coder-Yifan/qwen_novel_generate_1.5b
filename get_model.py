# 从hugging face获取模型
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import transformers
print(transformers.__version__)

# 模型名称
model_name = "huihui-ai/Qwen2.5-1.5B-Instruct-abliterated"
# 模型文件地址
local_dir = "./pre_model_obi"

# 执行前终端添加镜像源
# export HF_ENDPOINT=https://hf-mirror.com

# 获取模型文件 ，timeout后重新执行resume
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir,timeout=300,resume_download=True)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_dir,resume_download=True) 

# export HTTP_PROXY=http://your-proxy-server:port
# export HTTPS_PROXY=http://your-proxy-server:port

# 加载模型文件验证
# model_path = local_dir
# config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype="auto",
#     device_map="auto",
#     trust_remote_code=True,
#     config=config
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)


# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)