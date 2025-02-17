import json
from datasets import Dataset

def process_predictions(file_path):
    processed_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # 确保有两个回答
            if len(item['response']) >= 2:
                processed_data.append({
                    'index': str(item['index']),
                    'prompt': item['prompt'],
                    'chosen': [
                        {"content": item['prompt'], "role": "user"},
                        {"content": item['response'][0], "role": "assistant"}
                    ],
                    'rejected': [
                        {"content": item['prompt'], "role": "user"},
                        {"content": item['response'][1], "role": "assistant"}
                    ]
                })

    # 转换为Dataset格式
    dataset = Dataset.from_dict({
        'index': [item['index'] for item in processed_data],
        'prompt': [item['prompt'] for item in processed_data],
        'chosen': [item['chosen'] for item in processed_data],
        'rejected': [item['rejected'] for item in processed_data]
    })
    
    return dataset

# 使用示例
if __name__ == "__main__":
    file_path="/mnt/sda/llmfact/LLaMA-Factory/generated_predictions.jsonl"
    dataset = process_predictions()
    # 保存为新的数据集
    dataset.save_to_disk("processed_spa_data")
    print(f"Total processed pairs: {len(dataset)}")
    print("\nFirst example:")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False)) 