import os
import numpy as np

from scipy.io import loadmat


def load_groups(base_dir="datafiles/rdg3"):

    result = []
    for func_id in range(1,21):
        filename = f"F{func_id:02d}.mat"
        filepath = os.path.join(base_dir, filename)

        if not os.path.exists(filepath):
            print(f"[跳过] 文件不存在: {filepath}")
            continue

        try:
            mat_data = loadmat(filepath)
            if 'nonseps' not in mat_data or 'seps' not in mat_data:
                print(f"[警告] {filename} 中缺少 'nonseps' 或 'seps' 变量。可用变量: {list(mat_data.keys())}")
                continue

            raw_nonseps = mat_data['nonseps']
            raw_seps = mat_data['seps']

            group_list = []

            for group in raw_nonseps[0]:
                group_list.append(np.asarray(group[0]) -1)

            seps = np.asarray(raw_seps).flatten()
            if seps.size:
                group_list.append(seps -1)


            result.append(group_list)
        except Exception as e:
            print(f"[错误] 处理 F{func_id}.mat 时发生异常: {e}")

    return result