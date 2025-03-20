import json
import random
def filter_and_save_jsonl(input_file: str, output_file: str, level_filter: str = "Level 5"):
    # 读取 jsonl 文件
    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    # 筛选出 level 为 "Level 5" 的问题
    filtered_data = [entry for entry in data if entry.get("level") == level_filter]
    random.shuffle(filtered_data)
    filtered_data=filtered_data[:10]
    # 将筛选后的数据保存为新的 json 文件
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)

    print(f"Filtered data has been saved to {output_file}")

# 使用示例
input_file = "./data/test.json"  # 替换为你的文件路径
output_file = "./data/test_10.json"  # 输出路径
filter_and_save_jsonl(input_file, output_file)
