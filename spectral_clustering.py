# %%
import numpy as np
import datetime as dt
from sklearn.cluster import SpectralClustering
from tslearn.metrics import cdist_dtw  # 用于计算两两之间的 DTW 距离矩阵
from tslearn.utils import to_time_series_dataset
from cleaner import AppActive, GlobalParams
# %%
gp = GlobalParams(hospital_uuid=['1001'])
app_data = AppActive(gp,end_time=dt.date.today())
pts = app_data.pts
# cgm = app_data.t_cgm(columns=['patient_uuid', 'date', 'daily_data'])
# cgm
pts
# %%
# 转换为 tslearn 要求的格式 (n_samples, n_timestamps, n_features)。
# windows_data：由上游切窗得到的序列列表/数组，每条约为一日的血糖等时序；需在此处接入真实数据。
formatted_windows = to_time_series_dataset(windows_data)
n_segments = len(formatted_windows)
if n_segments < 2:
    raise ValueError("谱聚类至少需要 2 条序列样本（sklearn 要求）。")
# %%
# 2. 计算 DTW 距离矩阵 (这一步计算量最大，O(N^2))
# 注意：论文中提到了 CID-DTW，tslearn 原生支持标准 DTW。
# 如果需要完全还原 CID，需要自定义 metric 或对原始数据进行复杂度预处理。
dist_matrix = cdist_dtw(formatted_windows, n_jobs=-1)

# 3. 将“距离矩阵”转换为“相似度矩阵（Affinity Matrix）”
# 谱聚类需要的是相似度（值越大越相似），而 DTW 返回的是距离（值越小越相似）。
# 我们通常使用高斯核（RBF Kernel）进行转换：A = exp(-d^2 / (2*sigma^2))
# 中位数启发式应排除对角线（自距为 0），否则会低估 sigma、affinity 过于尖锐。
triu = np.triu_indices_from(dist_matrix, k=1)
sigma = np.median(dist_matrix[triu])
sigma = max(sigma, np.finfo(float).eps)
affinity_matrix = np.exp(-(dist_matrix**2) / (2 * (sigma**2)))

# 4. 调用 sklearn 的 SpectralClustering，设置 affinity='precomputed'
n_clusters = min(3, n_segments)  # k-means 阶段要求 n_clusters <= n_samples
model = SpectralClustering(
    n_clusters=n_clusters,
    affinity="precomputed",
    random_state=42,
)
labels = model.fit_predict(affinity_matrix)

print(f"成功将 {n_segments} 个血糖片段聚类为 {n_clusters} 类。")