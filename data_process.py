# 读取文件夹内所有txt文件，按段落拆解成可训练数据集合，数据集为列表形式，保存成pkl文件方便加载

import os
import re

# 拆解时按样本token长度设置max_length ，过大会导致很多填充token浪费缓存，过小会导致训练效果不佳
def split_long_paragraphs(paragraphs, max_length=500):
    new_paragraphs = []
    for para in paragraphs:
        while len(para) > max_length:
            # 找到最接近但不超过max_length的最大结束位置，优先在换行符处分割
            split_pos = para.rfind('\n', 0, max_length)
            if split_pos == -1:  # 如果没有找到合适的换行符，则直接按照max_length切割
                split_pos = max_length
            new_paragraphs.append(para[:split_pos].strip())
            para = para[split_pos:]
        new_paragraphs.append(para.strip())  # 添加剩余部分
    return new_paragraphs

def read_and_merge_txt_files(folder_path):
    paragraphs = []
    
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='GBK', errors='ignore') as f:
                        content = f.read()
                    
                    # 清理文本
                    content = content.replace("\r", "").strip()

                    # 按段落拆分（假设段落之间有空行）
                    temp_paragraphs = re.split(r'\n+', content)  # 使用一个或多个换行符进行分割

                    # 去掉空的段落
                    temp_paragraphs = [p.strip() for p in temp_paragraphs if p.strip()]

                    paragraphs.extend(temp_paragraphs)
                except Exception as e:
                    print(f"读取文件 {file_path} 时出错: {e}")
    
    # 处理长段落
    paragraphs = split_long_paragraphs(paragraphs)
    
    return paragraphs

# 使用示例
folder_path = '/code/model_pre/data/'  # 替换为你的文件夹路径
paragraphs = read_and_merge_txt_files(folder_path)

# 输出结果检查（可选）
for idx, paragraph in enumerate(paragraphs):
    print(f"Paragraph {idx + 1}: Length={len(paragraph)}")

import joblib
joblib.dump(paragraphs,'/code/model_pre/data/novels_dataset29.pkl')

example = max(paragraphs,key=len)
print(example)
