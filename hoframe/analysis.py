import os
from collections import defaultdict
import numpy as np
import pandas as pd

def process_file(file_path):
    """
    读取单个文件，返回：
    {
        function_id: (mean, variance)
    }
    """
    data = defaultdict(list)

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                function_id, seed, value = line.split(':')
                value = float(value)
                data[function_id].append(value)
            except ValueError:
                print(f"格式错误，跳过: {line}")

    stats = {}
    for func, values in data.items():
        mean = np.mean(values)
        var = np.std(values)
        stats[func] = (mean, var)

    return stats


def process_directory(dir_path):
    """
    遍历目录，生成最终表格
    """
    all_results = {}
    all_functions = set()

    # 遍历文件
    for filename in sorted(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, filename)
        if not os.path.isfile(file_path):
            continue

        stats = process_file(file_path)
        all_results[filename] = stats
        all_functions.update(stats.keys())

    # 构造表格
    table = {}

    for func in sorted(all_functions):
        row = {}
        for filename in all_results:
            if func in all_results[filename]:
                mean, var = all_results[filename][func]
                row[filename] = f"{mean:.4e} ({var:.4e})"
            else:
                row[filename] = ""
        table[func] = row

    df = pd.DataFrame.from_dict(table, orient='index')
    return df


if __name__ == "__main__":
    dir_path = "record"  # 修改为你的目录路径
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', None)  # 自动换行宽度
    pd.set_option('display.max_colwidth', None)  # 列内容不截断
    df = process_directory(dir_path)

    print(df)
