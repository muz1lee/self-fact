import json
from datasets import Dataset
import os
import random
import string


def generate_random_prefix():
    """生成随机的5个字母"""
    return ''.join(random.choices(string.ascii_uppercase, k=5))


def process_predictions(file_path):
    processed_data = []
    prefix = generate_random_prefix()  # 为整个数据集生成一个统一的前缀
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # 确保有两个回答
            if len(item['response']) >= 2:
                # 创建新的索引格式：XXXXX_数字
                new_index = f"{prefix}_{item['index']}"
                processed_data.append({
                    'index': new_index,
                    'instruction': item['prompt'],
                    'input': "",  # 添加空的 input 字段
                    'chosen': item['response'][0],
                    'rejected': item['response'][1]
                })

    # 转换为Dataset格式
    dataset = Dataset.from_dict({
        'index': [item['index'] for item in processed_data],
        'instruction': [item['instruction'] for item in processed_data],
        'input': [item['input'] for item in processed_data],  # 添加 input 到 Dataset
        'chosen': [item['chosen'] for item in processed_data],
        'rejected': [item['rejected'] for item in processed_data]
    })
    
    return dataset

def save_data_to_json(data, output_path):
    """保存数据为 JSON 格式"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 写入 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    file_path = "/mnt/sda/llmfact/LLaMA-Factory/generated_predictions.jsonl"
    dataset = process_predictions(file_path)
    
    # 将数据集转换为列表格式
    data_list = [entry for entry in dataset]
    
    # 保存为 JSON 文件
    output_path = "/mnt/sda/llmfact/LLaMA-Factory/data/biography_preference.json"
    save_data_to_json(data_list, output_path)
    
    print(f"Total processed pairs: {len(data_list)}")
    print("\nFirst example:")
    print(json.dumps(data_list[0], indent=2, ensure_ascii=False))
    print(f"\nData saved to: {output_path}")

if __name__ == "__main__":
    main() 