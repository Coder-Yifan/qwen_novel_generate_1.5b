## 执行顺序
* data_process.py 加工预处理训练数据
* get_model.py 获取预训练基础模型
* model_output_demo.py 验证预训练模型文件是否完整，也可执行varify_model.py
* qwen_train.py 微调训练
* novels_generate.py 使用微调模型文件生成文本

## 环境
torch                      2.3.0
transformers               4.43.1
