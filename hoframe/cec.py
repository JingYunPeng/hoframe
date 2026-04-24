import numpy as np
import os
import warnings
from scipy.io import loadmat


class BenchmarkSuite:
    def __init__(self, data_dir='datafiles/cec'):
        self.data_dir = data_dir
        self.cache = {}
        self.func_map = {i: getattr(self, f'f{i}') for i in range(1, 21)}

    def load_data(self, func_num):
        if func_num in self.cache:
            return self.cache[func_num]

        filename = f"f{func_num:02d}.mat"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"错误: 找不到数据文件 {filepath}。")

        try:
            raw_data = loadmat(filepath)
        except Exception as e:
            raise IOError(f"读取文件 {filepath} 失败: {e}")

        try:
            s = raw_data['s'].flatten().astype(int)
            xopt = raw_data['xopt'].flatten().astype(np.float64)
            p = raw_data['p'].flatten().astype(int) - 1  # 转 0-based
            m = int(np.squeeze(raw_data['m']))
            lb = raw_data['lb'].flatten().astype(np.float64)
            ub = raw_data['ub'].flatten().astype(np.float64)
            w = raw_data['w'].flatten().astype(np.float64)
            R25 = raw_data['R25'].astype(np.float64)
            R50 = raw_data['R50'].astype(np.float64)
            R100 = raw_data['R100'].astype(np.float64)
        except KeyError as e:
            raise KeyError(f"文件 {filename} 中缺少必要的变量：{e}。请检查文件内容。")

        c = np.cumsum(s).astype(int)
        data = {
            'xopt': xopt,
            'p': p,
            's': s,
            'R25': R25,
            'R50': R50,
            'R100': R100,
            'm': m,
            'c': c,  # 使用计算出的 c
            'lb': lb,
            'ub': ub,
            'w': w
        }

        self.cache[func_num] = data
        return data

    def evaluate(self, x, func_num):
        if func_num not in self.func_map:
            raise ValueError(f"不支持的函数编号: {func_num}")
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return self.func_map[func_num](x)

    def T_asy(self, f, beta):
        D, popsize = f.shape
        temp = beta * np.linspace(0, 1, D)[:, None]

        exponent = 1.0 + temp * np.sqrt(np.maximum(f, 0))
        g = np.where(f > 0, np.power(f, exponent), f)

        return g

    def T_irreg(self, f):
        """Transformation to create smooth local irregularities"""
        a = 0.1
        g = f.copy()

        idx_pos = f > 0
        if np.any(idx_pos):
            val = np.log(f[idx_pos]) / a
            g[idx_pos] = np.exp(val + 0.49 * (np.sin(val) + np.sin(0.79 * val))) ** a

        idx_neg = f < 0
        if np.any(idx_neg):
            val = np.log(-f[idx_neg]) / a
            g[idx_neg] = - (np.exp(val + 0.49 * (np.sin(0.55 * val) + np.sin(0.31 * val))) ** a)
        return g

    def checkBounds(self, x, lb, ub):
        """检查哪些列（解）越界"""
        violations = (x > ub.reshape(-1, 1)) | (x < lb.reshape(-1, 1))
        indices = np.where(np.sum(violations, axis=0) > 0)[0]
        return indices

    def schwefel(self, x):
        x_trans = self.T_asy(self.T_irreg(x), 0.2)
        cum_x = np.cumsum(x_trans, axis=0)  # shape (D, ps)
        fit = np.sum(cum_x ** 2, axis=0)  # shape (ps,)
        return fit.reshape(1, -1)  # 返回 (1, ps)

    def _base_func_logic(self, x, func_num, conflicting=False):
        data = self.load_data(func_num)

        xopt = data['xopt']
        p = data['p']  # 0-based permutation
        s = data['s']
        R25, R50, R100 = data['R25'], data['R50'], data['R100']
        m = data['m']
        c = data['c']  # cumulative sum of s
        lb, ub = data['lb'], data['ub']
        w = data['w']

        D_input, ps = x.shape
        idx_violate = self.checkBounds(x, lb, ub)

        fit = np.zeros((1, ps))

        # 2. 预处理 Shift
        # f1-f10: 全局 shift: x = x - xopt
        # f11-f20: 局部 shift: 在循环内部对每个子分量 shift
        if not conflicting:
            # 确保 xopt 形状为 (D, 1) 以便广播
            x_shifted = x - xopt.reshape(-1, 1)
        else:
            x_shifted = x.copy()

        # 3. 循环处理子分量
        n_sub = len(s)
        for i in range(n_sub):
            if i == 0:
                ldim_idx = 0
                ldim_shift_idx = 0
            else:
                ldim_idx = int(c[i - 1] - i * m)
                if conflicting:
                    ldim_shift_idx = int(c[i - 1])
            udim_idx = int(c[i] - i * m)
            udim_shift_idx = int(c[i])

            if udim_idx > len(p):
                raise IndexError(f"Function {func_num}: Calculated index {udim_idx} exceeds permutation size {len(p)}. "
                                 f"Input D={D_input} might mismatch the function's designed dimension.")

            sub_p_indices = p[ldim_idx:udim_idx]
            sub_x = x_shifted[sub_p_indices, :]

            if conflicting:
                if udim_shift_idx > len(xopt):
                    raise IndexError(f"Function {func_num}: xopt index out of bounds.")

                xopt_seg = xopt[ldim_shift_idx:udim_shift_idx]
                if len(xopt_seg) != sub_x.shape[0]:
                    min_len = min(len(xopt_seg), sub_x.shape[0])
                    xopt_seg = xopt_seg[:min_len]
                    sub_x = sub_x[:min_len, :]
                z = sub_x - xopt_seg.reshape(-1, 1)
            else:
                z = sub_x

            # 选择旋转矩阵
            si = s[i]
            if si == 25:
                R = R25
            elif si == 50:
                R = R50
            elif si == 100:
                R = R100
            else:
                raise ValueError(f"Function {func_num}: Subcomponent size {si} is not 25, 50, or 100.")

            # 维度对齐检查
            if R.shape[0] != z.shape[0]:
                raise ValueError(
                    f"Function {func_num}: Rotation matrix size {R.shape[0]} mismatch with sub-vector size {z.shape[0]} at component {i + 1}. "
                    f"This usually means input D ({D_input}) does not match the function's expected dimension.")

            f_val = self.schwefel(R @ z)
            fit += w[i] * f_val

        # 处理越界结果
        if len(idx_violate) > 0:
            fit[:, idx_violate] = np.nan
            warnings.warn(f"Function {func_num}: Some solutions violate boundary constraints.")

        return fit

    # 定义 f1 - f10 (Conforming)
    def f1(self, x):
        return self._base_func_logic(x, 1, conflicting=False)

    def f2(self, x):
        return self._base_func_logic(x, 2, conflicting=False)

    def f3(self, x):
        return self._base_func_logic(x, 3, conflicting=False)

    def f4(self, x):
        return self._base_func_logic(x, 4, conflicting=False)

    def f5(self, x):
        return self._base_func_logic(x, 5, conflicting=False)

    def f6(self, x):
        return self._base_func_logic(x, 6, conflicting=False)

    def f7(self, x):
        return self._base_func_logic(x, 7, conflicting=False)

    def f8(self, x):
        return self._base_func_logic(x, 8, conflicting=False)

    def f9(self, x):
        return self._base_func_logic(x, 9, conflicting=False)

    def f10(self, x):
        return self._base_func_logic(x, 10, conflicting=False)

    # 定义 f11 - f20 (Conflicting)
    def f11(self, x):
        return self._base_func_logic(x, 11, conflicting=True)

    def f12(self, x):
        return self._base_func_logic(x, 12, conflicting=True)

    def f13(self, x):
        return self._base_func_logic(x, 13, conflicting=True)

    def f14(self, x):
        return self._base_func_logic(x, 14, conflicting=True)

    def f15(self, x):
        return self._base_func_logic(x, 15, conflicting=True)

    def f16(self, x):
        return self._base_func_logic(x, 16, conflicting=True)

    def f17(self, x):
        return self._base_func_logic(x, 17, conflicting=True)

    def f18(self, x):
        return self._base_func_logic(x, 18, conflicting=True)

    def f19(self, x):
        return self._base_func_logic(x, 19, conflicting=True)

    def f20(self, x):
        return self._base_func_logic(x, 20, conflicting=True)


class F:
    def __init__(self, benchmark, overlap):
        self.benchmark = benchmark
        self.overlap = overlap
        self.D = 1000 - 19 * overlap
        func_num = self.add_inx() + overlap
        benchmark.load_data(func_num)
        self.p = benchmark.cache[func_num]['p']

    def compute(self, x):
        idx = self.add_inx()
        return self.benchmark.evaluate(x, self.overlap + idx)[0][0]

    def add_inx(self):
        raise NotImplementedError


class F13(F):
    def add_inx(self):
        return 0


class F14(F):
    def add_inx(self):
        return 10


def ori_test():
    import time

    print("正在初始化基准测试套件...")
    try:
        suite = BenchmarkSuite()
    except FileNotFoundError as e:
        print(e)
        print("请确保 'datafiles' 文件夹存在且包含 f01.mat 到 f20.mat。")
        exit(1)

    # 构造 k 序列: 1->10, 然后 1->10
    k_sequence = list(range(1, 11)) + list(range(1, 11))

    print(f"开始测试，当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试逻辑: x = ones(D), D = 1000 - k")
    print("-" * 75)
    print(f"{'Func':<5} | {'k':<3} | {'D':<5} | {'Status':<12} | {'Result (Val)':<20}")
    print("-" * 75)

    error_count = 0

    for func_num in range(1, 21):
        k = k_sequence[func_num - 1]

        D = 1000 - 19 * k
        # 创建输入向量 x = ones(D)，形状 (D, 1)
        x = np.ones((D, 1), dtype=np.float64)

        status = "OK"
        res_val = ""

        try:
            val = suite.evaluate(x, func_num)
            # val 形状 (1, 1)
            res_val = f"{val[0, 0]:.6e}"

            if np.isnan(val[0, 0]):
                status = "NaN (Bound)"

        except Exception as e:
            status = "Error"
            res_val = str(e)[:50]  # 截断长错误信息
            error_count += 1

        print(f"{func_num:<5} | {k:<3} | {D:<5} | {status:<12} | {res_val}")

    print("-" * 75)
    if error_count > 0:
        print(f"测试完成。发现 {error_count} 个错误。")
        print("提示：如果错误是 'dimension mismatch'，说明 1000-k 与该函数固定的设计维度不符。")
        print("例如：f1 可能设计为 D=981，当 k!=19 时 (D!=981) 会报错。")
    else:
        print("测试全部完成，无运行时错误。")


# ================= 测试执行部分 =================

if __name__ == "__main__":
    suite = BenchmarkSuite()
    for overlap in range(1, 11):
        f13 = F13(suite, overlap=overlap)
        x = np.linspace(1,50,f13.D)
        t = f13.compute(x)
        print(t)
    for overlap in range(1, 11):
        f13 = F14(suite, overlap=overlap)
        x = np.linspace(1, 50, f13.D)
        t = f13.compute(x)
        print(t)
