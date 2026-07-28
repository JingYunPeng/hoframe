import glob
import os

import numpy as np
from scipy.stats import wilcoxon

RECORD_DIR = "record"
BASELINE = "vst"
ALPHA = 0.05
MIN_COMMON = 5      # 至少多少个共同seed才进行Wilcoxon


def read_record(filename):
    """
    data[fid][seed] = value
    """
    data = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                func, seed, value = line.split(":")
            except ValueError:
                continue

            fid = int(func[1:])
            seed = int(seed)
            value = float(value)

            data.setdefault(fid, {})
            data[fid][seed] = value

    return data


def get_values(data, fid):
    if fid not in data:
        return np.array([])

    return np.array(list(data[fid].values()))


def get_common_values(data1, data2, fid):

    if fid not in data1:
        return np.array([]), np.array([])

    if fid not in data2:
        return np.array([]), np.array([])

    common = sorted(
        set(data1[fid].keys()) &
        set(data2[fid].keys())
    )

    x = np.array([data1[fid][i] for i in common])
    y = np.array([data2[fid][i] for i in common])

    return x, y


def latex_number(x):

    if np.isnan(x):
        return "-"

    if abs(x) < 1e-300:
        return "0.00e+00"

    return "{:.2e}".format(x)


def mean_std(values):

    values = np.asarray(values)

    if len(values) == 0:
        return None, None

    mean = np.mean(values)

    if len(values) == 1:
        std = 0
    else:
        std = np.std(values, ddof=1)

    return mean, std


def mean_std_str(values):

    mean, std = mean_std(values)

    if mean is None:
        return "-"

    return f"${latex_number(mean)}\\pm{latex_number(std)}$"


#######################################################
# Read all algorithms
#######################################################

algorithms = {}

for file in sorted(glob.glob(os.path.join(RECORD_DIR, "*.txt"))):
    name = os.path.splitext(os.path.basename(file))[0]
    algorithms[name] = read_record(file)

alg_names = sorted(algorithms.keys())

if BASELINE not in algorithms:
    raise RuntimeError(f"Cannot find baseline {BASELINE}")

wins = {alg: 0 for alg in alg_names}
ties = {alg: 0 for alg in alg_names}
losses = {alg: 0 for alg in alg_names}

#######################################################
# Latex table
#######################################################

print(r"\begin{tabular}{l" + "c" * len(alg_names) + "}")
print(r"\toprule")

print("Function & " + " & ".join(alg_names) + r"\\")
print(r"\midrule")

for fid in range(1, 11):

    means = {}

    for alg in alg_names:

        values = get_values(algorithms[alg], fid)

        if len(values):
            means[alg] = np.mean(values)

    best_alg = None

    if len(means):
        best_alg = min(means, key=means.get)

    row = [f"$f_{{{fid}}}$"]

    baseline = algorithms[BASELINE]

    for alg in alg_names:

        values = get_values(algorithms[alg], fid)

        txt = mean_std_str(values)

        if txt == "-":
            row.append("-")
            continue

        if alg == best_alg:
            txt = r"\mathbf{" + txt + "}"

        if alg == BASELINE:
            row.append(txt)
            continue

        x, y = get_common_values(
            algorithms[alg],
            baseline,
            fid,
        )

        if len(x) < MIN_COMMON:
            symbol = "-"
        else:

            try:

                _, p = wilcoxon(
                    x,
                    y,
                    alternative="two-sided"
                )

                if p >= ALPHA:

                    symbol = "="
                    ties[alg] += 1

                else:

                    # vst 更好
                    if np.median(y) < np.median(x):

                        symbol = "+"
                        wins[alg] += 1

                    # 当前算法更好
                    else:

                        symbol = "-"
                        losses[alg] += 1

            except Exception:

                symbol = "-"

        row.append(txt + f"$^{{({symbol})}}$")

    print(" & ".join(row) + r"\\")

print(r"\midrule")

last = ["+ / = / -"]

for alg in alg_names:

    if alg == BASELINE:
        last.append("--")
    else:
        last.append(
            f"{wins[alg]}/{ties[alg]}/{losses[alg]}"
        )

print(" & ".join(last) + r"\\")

print(r"\bottomrule")
print(r"\end{tabular}")