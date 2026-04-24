import numpy as np


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        return f"Node({self.val})"


def build_optimal_tree(A):
    n = len(A)
    if n == 0:
        return None, 0

    # dp[i][j] 存储区间 A[i...j] 的最小最大路径和
    # 初始化为无穷大
    dp = [[float('inf')] * n for _ in range(n)]

    # root_idx[i][j] 存储区间 A[i...j] 最优解对应的根节点下标
    root_idx = [[-1] * n for _ in range(n)]

    # 初始化长度为 1 的区间
    for i in range(n):
        dp[i][i] = A[i]
        root_idx[i][i] = i

    # 枚举区间长度 len，从 2 到 n
    for length in range(2, n + 1):
        # 枚举起点 i
        for i in range(n - length + 1):
            j = i + length - 1

            # 枚举根节点位置 k
            best_k = -1
            min_max_path = float('inf')

            for k in range(i, j + 1):
                # 计算左子树的贡献
                left_val = 0
                if k > i:
                    left_val = dp[i][k - 1]

                # 计算右子树的贡献
                right_val = 0
                if k < j:
                    right_val = dp[k + 1][j]

                # 当前根节点的最大路径和
                current_max_path = A[k] + max(left_val, right_val)

                # 寻找最小值
                if current_max_path < min_max_path:
                    min_max_path = current_max_path
                    best_k = k

            dp[i][j] = min_max_path
            root_idx[i][j] = best_k

    # --- 重建树结构 ---
    def reconstruct(i, j):
        if i > j:
            return None
        k = root_idx[i][j]
        node = TreeNode(A[k])
        node.left = reconstruct(i, k - 1)
        node.right = reconstruct(k + 1, j)
        return node

    root = reconstruct(0, n - 1)
    min_score = dp[0][n - 1]

    return root, min_score


def transform_tree(A):
    # 1. 中序遍历收集节点
    nodes = []

    def inorder(node):
        if not node:
            return
        inorder(node.left)
        nodes.append(node)
        inorder(node.right)

    inorder(A)

    # 2. 生成分块数组
    blocks = []
    total = 0
    for node in nodes:
        block = list(range(total, total + node.val))
        blocks.append(block)
        total += node.val

    # 4. 重新映射回树结构
    idx = 0

    def build(node):
        nonlocal idx
        children = []
        if node.left:
            left = build(node.left)
            children.append(left)

        root_block = blocks[idx]
        idx += 1

        if node.right:
            right = build(node.right)
            children.append(right)

        return [root_block, children]

    return build(A)


# --- 辅助函数：打印树结构 (缩进式) ---
def print_tree(node, level=0, prefix="Root: "):
    if node is not None:
        print(" " * (level * 4) + prefix + str(node.val))
        if node.left is None and node.right is None:
            # 叶子节点标记
            pass
        else:
            if node.left:
                print_tree(node.left, level + 1, "L--- ")
            else:
                print(" " * ((level + 1) * 4) + "L--- None")

            if node.right:
                print_tree(node.right, level + 1, "R--- ")
            else:
                print(" " * ((level + 1) * 4) + "R--- None")


# --- 辅助函数：验证并打印所有根到叶子的路径 ---
def get_all_paths(node, current_path, all_paths):
    if node is None:
        return
    current_path.append(node.val)
    if node.left is None and node.right is None:
        all_paths.append(list(current_path))
    else:
        get_all_paths(node.left, current_path, all_paths)
        get_all_paths(node.right, current_path, all_paths)
    current_path.pop()


def compress(vst):
    post_order_arrays = []

    def post_order(node):
        if not node:
            return

        root_data, children = node[0], node[1]

        # 先递归处理子节点
        for child in children:
            post_order(child)

        # 再处理当前节点 (root_data 是一个 int 数组)
        post_order_arrays.append(root_data)

    if vst:
        post_order(vst)

    print(len(post_order_arrays))
    print([len(_) for _ in post_order_arrays])

    final_result = []
    buffer = []

    for arr in post_order_arrays:
        current_len = len(arr)
        if current_len >= 50:
            if buffer:
                final_result.append(buffer)
                buffer = []
            final_result.append(arr)
        else:
            buffer.extend(arr)
            if len(buffer) >= 50:
                final_result.append(buffer)
                buffer = []
    if buffer:
        final_result.append(buffer)

    print(len(final_result))
    print([len(_) for _ in final_result])
    return final_result


def vst_list():
    vst_list = []
    for overlap in range(1, 11):
        vst_list.append(build_vst(overlap))
    return vst_list


def build_vst(overlap):
    dims = np.array([50, 50, 25, 25, 100, 100, 25, 25, 50, 25, 100, 25, 100, 50, 25, 25, 25, 100, 50, 25])
    A = []
    for i in range(20):
        if i == 0:
            A.append(dims[i] - overlap)
        elif i == 19:
            A.append(dims[i] - overlap)
            continue
        else:
            A.append(dims[i] - 2 * overlap)
        A.append(overlap)
    root, score = build_optimal_tree(A)

    return transform_tree(root)


def test():
    global s
    dims = np.array([50, 50, 25, 25, 100, 100, 25, 25, 50, 25, 100, 25, 100, 50, 25, 25, 25, 100, 50, 25])
    A = []
    overlap = 1
    for i in range(20):
        if i == 0:
            A.append(dims[i] - overlap)
        elif i == 19:
            A.append(dims[i] - overlap)
            continue
        else:
            A.append(dims[i] - 2 * overlap)
        A.append(overlap)

    print(sum(A))
    root, score = build_optimal_tree(A)

    # 3. 输出结果
    print("\n" + "=" * 30)
    print(f"输入数组: {A}")
    print(f"最低评分 (最小化的最大路径和): {score}")
    print("=" * 30)
    print("\n构造出的树结构:")
    print_tree(root)

    # 4. 验证路径
    print("\n所有根到叶子的路径及其和:")
    paths = []
    get_all_paths(root, [], paths)
    max_check = 0
    for p in paths:
        s = sum(p)
        print(f"路径: {p} -> 和 = {s}")
        if s > max_check:
            max_check = s

    print("-" * 20)
    print(f"验证计算出的最大路径和: {max_check}")
    if max_check == score:
        print("验证成功！评分一致。")
    else:
        print("验证失败！逻辑有误。")


# ================= 主程序入口 =================
if __name__ == "__main__":
    vst = build_vst(5)
    compress(vst)

