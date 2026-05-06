# %%
import datetime as dt
import pandas as pd
from connections import engine
from cleaner import AppActive
import seaborn as sns
from engineer import CGM_cluster_non_diabetic, CGM_cal_metrics, CGM_AGP_plot, yutang_palette, yutang_theme
sns.set_theme(rc=yutang_theme, style='white', palette=yutang_palette)
# %%
tbl = pd.read_excel('体重管理糖尿病与非糖尿病肥胖对照-脱敏.xlsx')
tbl = tbl.drop('序号', axis=1)

rename_cols = {
    '与糖系统ID': 'uuid',
    '性别': 'sex',
    '年龄': 'age',
    '是否糖尿病': 'is_diabetic',
    '入组时间': 'enroll_date',
    '身高': 'height',
    '体重': 'weight',
    'BMI': 'bmi',
    'FMI指数': 'fmi',
    'A/G': 'a_g',
    '骨密度值': 'bone_density',
    'RMR': 'rmr',
    'ASMI': 'asmi',
    '体脂率': 'body_fat_ratio',
    '腰围': 'waist_circumference',
    '臀围': 'hip_circumference',
    '腰臀比': 'waist_hip_ratio',
    '全部肌肉': 'total_muscle',
    '全部脂肪': 'total_fat',
    '内脏脂肪（g）': 'visceral_fat_g',
    '内脏脂肪（㎝3）': 'visceral_fat_cm3',
    '收缩压': 'systolic_bp',
    '舒张压': 'diastolic_bp',
    '心率': 'heart_rate',
    '饮食': 'diet',
    '运动': 'exercise',
    '药物': 'medication',
    'WBC': 'wbc',
    'RBC': 'rbc',
    'Hb': 'hb',
    'PLT': 'plt',
    'CRP': 'crp',
    'γ-GGT': 'gamma_ggt',
    'ALT': 'alt',
    'AST': 'ast',
    'BUN': 'bun',
    'Cr': 'cr',
    'UA': 'ua',
    'TG': 'tg',
    'TC': 'tc',
    'LDL': 'ldl',
    'Na': 'na',
    'K': 'k',
    'Cl': 'cl',
    '血脂肪酶': 'lipase',
    '血淀粉酶': 'amylase',
    'FFA': 'ffa',
    'VitD': 'vit_d',
    'ACTH': 'acth',
    'Cor': 'cor',
    'TSH': 'tsh',
    'FT3': 'ft3',
    'FT4': 'ft4',
    'TPOAb': 'tpo_ab',
    '空腹葡萄糖': 'fasting_glucose',
    '空腹胰岛素': 'fasting_insulin',
    'HOMA-IR': 'homa_ir',
    'C肽': 'c_peptide',
    '胰高血糖素': 'glucagon',
    '糖化血红蛋白': 'hba1c',
    '尿ACR': 'urine_acr',
    '脂肪肝': 'fatty_liver',
    '甲状腺结节': 'thyroid_nodule',
    '颈动脉斑块': 'carotid_plaque',
    '糖尿病前期': 'prediabetes',
    '高血压': 'hypertension',
    '甲减': 'hypothyroidism',
    '脂蛋白代谢紊乱': 'lipoprotein_disorder',
    '高尿酸血症': 'hyperuricemia'
}

tbl = tbl.rename(columns=rename_cols)
tbl.columns
# %%
app_data = AppActive(pts = tbl['uuid'], start_time = dt.date(1, 1, 1), end_time = dt.date.today())
cgm = app_data.t_cgm()
# %%
cgm['patient_uuid'].nunique()
# %%
tbl['enroll_date'] = pd.to_datetime(tbl['enroll_date'])
# %%
cgm_new = cgm.join(tbl.set_index('uuid')[['enroll_date', 'is_diabetic']], on='patient_uuid', how='left')
cgm_new['date'] = pd.to_datetime(cgm_new['date'])
cgm_new['enroll_days'] = cgm_new.eval('date - enroll_date').dt.days
cgm_new = cgm_new.query('enroll_days >= 0 and enroll_days <= 6')
# %%
# 1. 计算各个患者的血糖特征指标
# 这里传入我们刚刚获取到的 cgm 数据表，指定需要计算的患者 UUID 列表
cgm_nondm =cgm_new.query('is_diabetic == "否"')
cgm_dm = cgm_new.query('is_diabetic == "是"')
cal = CGM_cal_metrics(df=cgm_new)
# 调用 cal_all_metrics() 触发所有核心指标的计算
# 这会把指标计算结果保存在 cal.metrics 中，并返回用于展示的 metrics_display 表
metrics = cal.cal_all_metrics()
# %%
# 2. 对患者进行聚类划分波动类型
cluster = CGM_cluster_non_diabetic()
cluster.load_model(
    model_file=r'd:\Jie\Git\source\cgm_kmeans_model.joblib',
    scaler_file=r'd:\Jie\Git\source\cgm_scaler.joblib'
)
# %%
# 使用 predict 方法基于已有的模型对新数据进行预测
# 在预测前，先剔除那些因为缺失数据导致指标为 NaN 的患者
invalid_uuids = cal.metrics[cal.metrics[cluster.features].isna().any(axis=1)].index.tolist()
if invalid_uuids:
    print(f"以下 {len(invalid_uuids)} 位患者因为有效数据少于 2 个点（导致 MAG/MAGE 等计算为 NaN）将被过滤：")
    print(invalid_uuids)

# 过滤掉这些无效行再送入模型预测
valid_metrics = cal.metrics.dropna(subset=cluster.features).copy()

# 为了能借用 cluster.predict，我们将 cal 中的数据临时替换为有效的
cal_temp = cal
cal_temp.metrics = valid_metrics
labels = cluster.predict(cal_temp)

# 为了能借用原来类中的画图方法，我们手动把包含预测结果的表赋给 cluster.result
cluster.result = valid_metrics.copy()
cluster.result['cluster'] = labels
# %%
# —— 画图看看在该模型下的预测聚类效果 ——
# 绘制 PCA 降维散点图，查看簇与簇之间的分离情况
cluster.plot_pca()
# 绘制 雷达图，观察每类人群在不同生理指标上的表现（比如哪类是高波动，哪类是黎明现象）
cluster.plot_radar()
# %%
tbl = tbl.join(cluster.result['cluster'], on='uuid', how='left')
tbl
# %%
# 3. 绘制单个重点患者（或者所有人）的 AGP 动态图谱
sample_uuid = tbl.query('cluster == "黎明型" and is_diabetic == "是"')['uuid'].iloc[0] # 取第一位患者看一下AGP图
lable = tbl.set_index('uuid').loc[sample_uuid, 'cluster']
is_diabetic = tbl.set_index('uuid').loc[sample_uuid, 'is_diabetic']
agp = CGM_AGP_plot(
    cal=cal,                       # 必须传入一开始计算的 cal 对象
    title=f"患者 {sample_uuid} {lable} - 糖尿病：{is_diabetic}",
    patient_uuid=sample_uuid,
    target_range=(3.9, 10.0),       # 绿色安全范围判定线
    is_save=True
)
# %%
tbl.to_excel('体重管理糖尿病与非糖尿病肥胖对照-脱敏-分类.xlsx', index=False)
# %%

# %%
tbl.query('uuid == "PAT_2fvicb7jx8"')