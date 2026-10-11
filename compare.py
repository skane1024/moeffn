
import numpy as np




import numpy as np

def compare_arrays(arr1, arr2, print_results=True):

    if arr1.shape != arr2.shape:
        raise ValueError("数据shape不同")
    
    a1 = arr1.astype(np.float64)
    a2 = arr2.astype(np.float64)
    
    flat1 = a1.ravel()
    flat2 = a2.ravel()
    
    mean1 = np.mean(flat1)
    mean2 = np.mean(flat2)
    var1 = np.var(flat1)
    var2 = np.var(flat2)
    
    error = flat1 - flat2
    mae = np.mean(np.abs(error))                # 平均绝对误差
    max_abs_error = np.max(np.abs(error))       # 最大绝对误差
    
    tol = 1e-8
    nonzero_ratio = np.count_nonzero(np.abs(error) > tol) / error.size
    
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    if norm1 == 0 or norm2 == 0:
        cosine_sim = 0.0
    else:
        cosine_sim = np.dot(flat1, flat2) / (norm1 * norm2)
    
    results = {
        "mean_arr1": mean1,
        "mean_arr2": mean2,
        "variance_arr1": var1,
        "variance_arr2": var2,
        "mean_absolute_error": mae,
        "max_absolute_error": max_abs_error,
        "nonzero_error_ratio": nonzero_ratio,
        "cosine_similarity": cosine_sim,
    }
    
    if print_results:
        print("========== 数组对比结果 ==========")
        print(f"数组1 均值: {mean1:.6f}")
        print(f"数组2 均值: {mean2:.6f}")
        print(f"数组1 方差: {var1:.6f}")
        print(f"数组2 方差: {var2:.6f}")
        print(f"平均绝对误差 (MAE): {mae:.6f}")
        print(f"最大绝对误差: {max_abs_error:.6f}")
        print(f"误差非零比例: {nonzero_ratio:.2%}")
        print(f"余弦相似度: {cosine_sim:.6f}")
        print("===================================")
    
    return results


# ================== 示例用法 ==================
if __name__ == "__main__":
    data1 = np.fromfile("data/fm0_0001_int8.bin", dtype = np.int8)
    data2 = np.fromfile("data/fm0_0001_int8.bin", dtype = np.int8)

    compare_arrays(data1,data2)