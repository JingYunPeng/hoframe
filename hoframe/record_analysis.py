"""
算法性能分析脚本
功能：
1. 读取record目录下所有*.txt文件，计算每个算法的均值和标准差（保留四位小数）
2. 使用Wilcoxon秩和检验（Mann-Whitney U检验）判断每个算法在显著性水平0.05下
   是否优于vst算法（单侧检验：算法结果 > vst结果）

数据格式示例（每行）：
    f3:1:-19811035.213572297
表示指标f3在第1次独立运行时的结果。

使用方法：
    将本脚本放在与record目录同级的位置，然后运行：
        python record_analysis.py
"""

import os
import glob
import numpy as np
from scipy import stats


# ============================================================
# 配置区 —— 按需修改
# ============================================================
RECORD_DIR = "record"          # 数据文件所在目录（相对于本脚本）
BASELINE_ALGO = "vst"          # 基线算法名（对应 record/vst.txt）
ALPHA = 0.05                   # 显著性水平


# ============================================================
# 数据读取
# ============================================================
def load_algorithm_data(record_dir):
    """
    读取record目录下所有*.txt文件。
    返回：
        algo_data: dict[algo_name] -> dict[metric_name] -> list of float
    """
    algo_data = {}

    txt_files = glob.glob(os.path.join(record_dir, "*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"在目录 '{record_dir}' 中未找到任何 *.txt 文件")

    for filepath in sorted(txt_files):
        algo_name = os.path.splitext(os.path.basename(filepath))[0]
        metric_runs = {}

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(":")
                if len(parts) < 3:
                    continue
                metric = parts[0]                          # 如 "f3"
                value = float(parts[2])                    # 如 -19811035.213572297
                metric_runs.setdefault(metric, []).append(value)

        algo_data[algo_name] = metric_runs
        print(f"  ✓ 已加载 [{algo_name}] {len(metric_runs)} 个指标")

    return algo_data


# ============================================================
# 任务1：均值 & 标准差
# ============================================================
def format_sci(val):
    """将数值格式化为科学计数法字符串，保留4位有效数字。
    示例：-19811035.21357 → -1.9811e+07
    """
    return f"{val:.4e}"


def print_mean_std(algo_data):
    """打印每个算法在每个指标上的均值和标准差（科学计数法，4位有效数字）"""
    print("\n" + "=" * 72)
    print("【任务1】各算法各指标的均值与标准差（科学计数法，4位有效数字）")
    print("=" * 72)

    # 收集并排序所有指标名
    all_metrics = sorted(set().union(*(d.keys() for d in algo_data.values())))

    for metric in all_metrics:
        print(f"\n■ 指标: {metric}")
        header = f"  {'算法':<24s} {'均值':>18s} {'标准差':>18s} {'N':>6s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for algo in sorted(algo_data.keys()):
            values = algo_data[algo].get(metric, [])
            if not values:
                print(f"  {algo:<24s} {'N/A':>18s} {'N/A':>18s} {'0':>6s}")
                continue
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)   # 样本标准差
            print(f"  {algo:<24s} {format_sci(mean_val):>18s} {format_sci(std_val):>18s} {len(values):>6d}")


# ============================================================
# 任务2：Mann-Whitney U 秩和检验
# ============================================================
def print_wilcoxon_test(algo_data):
    """
    以 BASELINE_ALGO 为基线，对每个指标做单侧 Mann-Whitney U 检验：
        备择假设 H1: 算法结果 > 基线结果（即算法更优）

    说明：
        对于最小化问题（如误差、损失），"优于"意味着数值更小。
        脚本默认按"数值更大 = 更优"来判断（适用于最大化指标）。
        如果你的指标是最小化指标（如 f3 是负数误差），请修改下面的
        BETTER_DIRECTION 变量。
    """
    print("\n" + "=" * 72)
    print(f"【任务2】Mann-Whitney U 秩和检验（基线: {BASELINE_ALGO}，α = {ALPHA}）")
    print("         单侧检验 H1: 算法结果 > 基线结果（数值更大 = 更优）")
    print("=" * 72)

    if BASELINE_ALGO not in algo_data:
        print(f"\n⚠️  基线算法 '{BASELINE_ALGO}' 的数据文件不存在，跳过检验。")
        return

    all_metrics = sorted(set().union(*(d.keys() for d in algo_data.values())))

    for metric in all_metrics:
        baseline_vals = np.array(algo_data[BASELINE_ALGO].get(metric, []))
        if len(baseline_vals) == 0:
            print(f"\n■ 指标: {metric}\n  ⚠️  基线算法无此指标数据，跳过。")
            continue

        print(f"\n■ 指标: {metric}")
        header = f"  {'算法':<24s} {'N':>4s} {'基线N':>6s} {'U统计量':>10s} {'p值(单尾)':>12s} {'结论':>20s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for algo in sorted(algo_data.keys()):
            if algo == BASELINE_ALGO:
                continue
            algo_vals = np.array(algo_data[algo].get(metric, []))
            if len(algo_vals) == 0:
                print(f"  {algo:<24s} {'-':>4s} {'-':>6s} {'-':>10s} {'-':>12s} {'无数据':>20s}")
                continue

            # 双尾检验先算出来
            u_stat, p_two_tailed = stats.mannwhitneyu(
                algo_vals, baseline_vals, alternative="two-sided"
            )

            # 判断方向：算法中位数是否大于基线中位数
            med_diff = np.median(algo_vals) - np.median(baseline_vals)

            if med_diff > 0:
                # 算法值更大 → 单侧 p = 双尾 p / 2
                p_one = p_two_tailed / 2.0
                if p_one < ALPHA:
                    conclusion = "✅ 显著优于基线"
                else:
                    conclusion = "➖ 优于但不显著"
            else:
                # 算法值更小或相等 → 单侧 p = 1 - 双尾p/2（方向不对）
                p_one = 1.0 - p_two_tailed / 2.0
                conclusion = "⚠️ 未优于基线"

            print(f"  {algo:<24s} {len(algo_vals):>4d} {len(baseline_vals):>6d} "
                  f"{u_stat:>10.1f} {p_one:>12.4e} {conclusion:>20s}")


# ============================================================
# 主程序
# ============================================================
def main():
    # 自动定位 record 目录（与脚本同级的 record/）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    record_dir = os.path.join(script_dir, RECORD_DIR)

    print("=" * 72)
    print("算法性能分析工具")
    print(f"数据目录: {record_dir}")
    print("=" * 72)

    # 加载数据
    algo_data = load_algorithm_data(record_dir)
    print(f"\n共发现 {len(algo_data)} 个算法: {', '.join(sorted(algo_data.keys()))}")

    # 任务1
    print_mean_std(algo_data)

    # 任务2
    print_wilcoxon_test(algo_data)

    print("\n" + "=" * 72)
    print("✅ 分析完成。")
    print("=" * 72)


if __name__ == "__main__":
    main()
