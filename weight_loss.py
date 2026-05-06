# %%
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from connections import engine
from engineer import yutang_palette, yutang_theme, get_bmi
import pickle
sns.set_theme(rc=yutang_theme, style='white', palette=yutang_palette)
# %%
# 选取样本，取购买体重管理服务且有体脂称记录的患者，检查是否含CGM设备
sql = '''select h.city, h.name as hospital_name, yz.*
from sanyi_care.d_yz_order as yz
left join sanyi_care.u_hospital as h
on yz.user_hospital_uuid = h.uuid
where pkg_type = "体重管理"
and h.name in ('天津医科大学朱宪彝纪念医院(代谢病医院)', '前海人寿广州总医院')
'''
with engine.connect() as con:
    orders = pd.read_sql(sql, con)
orders['user_uuid'].nunique()
# %%
sql = f'''select pbd.have_glycuresis, pbd.glycuresis_type, sw.*
from sanyi_care.t_self_weight as sw
left join sanyi_care.u_patient as p
on sw.patient_uuid = p.uuid
left join sanyi_care.u_hospital as h
on p.hospital_uuid = h.uuid
left join sanyi_care.u_team as t
on p.team_uuid = t.uuid
left join sanyi_care.u_patient_base_disease as pbd
on p.uuid = pbd.patient_uuid
where p.status in ('join', 'wait_join', 'take_over')
and h.status = "on"
and t.status = "on"
and p.uuid in ('{"','".join(orders['user_uuid'])}')
and sw.deleted_at is null
and pbd.have_glycuresis = 0
'''
with engine.connect() as con:
    tbl = pd.read_sql(sql, con)
print(tbl.shape)
print(tbl['patient_uuid'].nunique())
tbl.head()
# %%
orders = orders.query('user_uuid in @tbl["patient_uuid"].unique()')
orders
# %%
orders['cgm'] = orders['sku_no'].str.contains("微泰").combine_first(orders['sku_name'].str.contains("微泰"))
display(orders[['sku_no', 'cgm']].value_counts(),
orders['paid_at'].describe())
# %%
chk = orders.groupby(['user_uuid'])['cgm'].mean().to_frame()
chk.query('cgm > 0 and cgm < 1')
# %%
# 为简化叙事，有多个订单的患者，只取所有订单均有或均无CGM设备的样本
orders = orders.query('user_uuid not in @chk.query("(cgm > 0 and cgm < 1) or cgm != cgm").index')
tbl = tbl.query('patient_uuid not in @chk.query("(cgm > 0 and cgm < 1) or cgm != cgm").index')
# %%
orders['user_uuid'].nunique()
# %%
tbl['patient_uuid'].nunique()
# %%
orders_agg = orders.groupby(['user_uuid', 'city', 'hospital_name']).agg(
    cgm = ('cgm', 'mean'),
    paid_at = ('paid_at', 'min')
)
orders_agg['cgm'] = orders_agg['cgm'].replace({0: '无CGM', 1: '有CGM'})
orders_agg = orders_agg.reset_index(['city', 'hospital_name'])
# %%
tbl = tbl.join(orders_agg, on='patient_uuid', how='inner')
tbl = tbl.query('measured_at >= paid_at')
print(tbl['patient_uuid'].nunique())
print(tbl.shape)
# %%
# 观察各组样本量
print(tbl.drop_duplicates(subset = ['patient_uuid', 'have_glycuresis', 'cgm'])[['cgm']].value_counts(),
tbl.drop_duplicates(subset = ['patient_uuid', 'have_glycuresis', 'cgm'])[['cgm', 'city', 'hospital_name']].value_counts()
)
# %%
start_date = tbl.groupby(['patient_uuid'])['paid_at'].min().to_frame('start_date')
tbl = tbl.join(start_date, on='patient_uuid', how='left')
tbl['start_date'] = pd.to_datetime(tbl['start_date'].dt.date)
tbl['measure_days'] = tbl.eval('measured_at - start_date').dt.days
tbl['measure_week'] = tbl['measure_days'].map(lambda x: np.ceil((x+1)/7) - 1)
# %%
# 每周体重变化
tbl_agg = (tbl.groupby(['patient_uuid', 'have_glycuresis', 'cgm', 'measure_week', 'start_date'])
           [['weight', 'bmi']].mean().reset_index())
# %%
# 观察每周的样本量，选取足量的时间节点
df = tbl_agg.query('have_glycuresis == 0').copy() # 是否过滤糖尿病患者，加 .query('have_glycuresis == 0')
samples = df.groupby(['measure_week', 'cgm'])['patient_uuid'].nunique().unstack('cgm')
samples_chk = (samples >= 10).sum(axis = 1)
cut_week = min(samples_chk[samples_chk < 2].index) -1
df_model = df.query('measure_week <= @cut_week').copy()
sns.lineplot(data=df_model, x='measure_week', y='bmi', hue='cgm')
# %%
# 构建混合效应模型
# bmi ~ measure_week * cgm 意思是包含两者各自的效应，以及它们的交互项
# groups='patient_uuid' 表示考虑同一个患者的多次测量具有相关性（设定个体随机截距）
# Gemini给的重要tips
# 在处理混合模型时，统计学界有一个标准流程（黄金准则）：
# 比较哪个模型包含的协变量更好的时候（对比固定效应）：一定要用 ML 拟合模型，然后对比不同模型的 AIC / BIC 或做似然比检验（Likelihood Ratio Test）。
# 最终选定了包含哪些协变量之后，用来汇报系数和 P 值时：再换回系统默认的 REML 重新 fit() 一次，因为 REML 给出的方差估计是无偏的，系数结论更严谨。

model = smf.mixedlm("bmi ~ measure_week * cgm", data=df_model, groups=df_model["patient_uuid"])
result = model.fit(reml=False)
# result = model.fit()
# 打印结果
print(result.summary()) # 看交互项的p值`
print("AIC:", result.aic) # 看AIC值
# %%
# 交互项 measure_week:cgm 的 P>|z| 为 0.001，远小于 0.05 的显著性水平。这意味着在统计学上，有无 CGM 设备对患者 BMI 随时间变化的斜率（减重速度）有显著的调节作用。

# measure_week (系数 -0.210)：这代表对照组（无设备 cgm=0）的平均减重速度，也就是没有设备的人每周平均下降 0.210 的 BMI。
# measure_week:cgm (系数 -0.028)：这就是交互效应大小。它说明配备了设备（cgm=1）的患者，在每周下降 0.210 的基础上，会显著地额外多下降 0.028的 BMI（即有设备组的平均减重速度是 -0.238/周）。
# %%
# 探索混杂因素
# 1.基线BMI
base_bmi = (df_model
            .query('measure_week == 0')
            .sort_values(['patient_uuid', 'measure_week'])
            .groupby(['patient_uuid'])['bmi'].first().to_frame('base_bmi'))
df_model_cov = df_model.join(base_bmi, on='patient_uuid', how='left').copy()
# %%
sql = f'''select patient_uuid, weight, height, created_at
from sanyi_care.t_exam_body_bp
where patient_uuid in ('{"','".join(df_model['patient_uuid'].unique())}')
'''
with engine.connect() as con:
    exam = pd.read_sql(sql, con)
exam = exam.join(orders_agg[['paid_at']], on='patient_uuid', how='left')
exam['created_at'] = pd.to_datetime(exam['created_at'])
# exam['start_date'] = pd.to_datetime(exam['start_date'])
exam['measure_days'] = exam.eval('created_at - paid_at').dt.days
exam = exam.query('measure_days >= 0 and measure_days <= 6')
exam['measure_date'] = exam['created_at'].dt.date
exam = exam.sort_values(['created_at']).drop_duplicates(subset=['patient_uuid', 'measure_date'], keep='last')
exam = exam.sort_values(['measure_date']).drop_duplicates(subset=['patient_uuid'], keep='first')
exam['base_bmi_sup'] = get_bmi(data=exam, height='height', weight='weight')
df_model_cov = df_model_cov.join(exam.set_index('patient_uuid')[['base_bmi_sup']], on='patient_uuid', how='left')
df_model_cov['base_bmi'] = df_model_cov['base_bmi'].fillna(df_model_cov['base_bmi_sup'])
# df_model_cov
# %%
model_cov = smf.mixedlm("bmi ~ measure_week * cgm + base_bmi", data=df_model_cov.dropna(subset=['base_bmi']),
                        groups=df_model_cov.dropna(subset=['base_bmi'])["patient_uuid"])
result_cov = model_cov.fit(reml=False)
# result_cov = model_cov.fit()
print(result_cov.summary())
print("AIC:", result_cov.aic)
# %% 
sql = f'''select pres.* from sanyi_care.c_prescription_tag as pres
where patient_uuid in ('{"','".join(df_model['patient_uuid'].unique())}')
-- and medicine_name like "%%肽%%"
'''
with engine.connect() as con:
    pres = pd.read_sql(sql, con)
print(pres['medicine_name'].unique())
print(pres['patient_uuid'].nunique())
print(pres.shape)
pres = pres.join(tbl_agg[['patient_uuid', 'start_date']].drop_duplicates()
.set_index('patient_uuid')[['start_date']], on='patient_uuid', how='left')
# %%
pres['date'] = pd.to_datetime(pres['date'])
pres['pres_days'] = pres.eval('date - start_date').dt.days
pres = pres.query('pres_days >= -30 and pres_days <= 30')
pres.shape
# %%
print("GLP1药物：", pres.query('medicine_name.str.contains("肽")')['medicine_name'].unique(),
      "\n非GLP1药物：", pres.query('not medicine_name.str.contains("肽")')['medicine_name'].unique())
# %%
pres = pres.query('medicine_name.str.contains("肽")')
# pres = pres.query("medicine_name in ('德谷胰岛素利拉鲁肽注射液', '司美格鲁肽注射液', '利拉鲁肽注射液', '替尔泊肽注射液', '玛仕度肽', '甘精胰岛素利司那肽注射液（Ⅰ）', '聚乙二醇洛塞那肽注射液', '甘精胰岛素利司那肽', '贝那鲁肽注射液')")
pres.shape
# %%
pres.groupby(['patient_uuid'])['tag_id'].nunique()
# %%
df_model_cov['is_glp1'] = df_model_cov['patient_uuid'].isin(pres['patient_uuid']).replace({False: '无GLP1', True: '有GLP1'})
# %%
df_model_cov.groupby(['is_glp1', 'cgm'])['patient_uuid'].nunique()
# %%
model_cov = smf.mixedlm("bmi ~ measure_week * cgm + base_bmi + is_glp1", data=df_model_cov.dropna(subset=['is_glp1', 'base_bmi']),
                        groups=df_model_cov.dropna(subset=['is_glp1', 'base_bmi'])["patient_uuid"])
result_cov = model_cov.fit(reml=False)
# result_cov = model_cov.fit()
print(result_cov.summary())
print("AIC:", result_cov.aic)
# %%
model_cov = smf.mixedlm("bmi ~ measure_week * cgm + base_bmi + measure_week * is_glp1", data=df_model_cov.dropna(subset=['is_glp1', 'base_bmi']),
                        groups=df_model_cov.dropna(subset=['is_glp1', 'base_bmi'])["patient_uuid"])
result_cov = model_cov.fit(reml=False)
# result_cov = model_cov.fit()
print(result_cov.summary())
print("AIC:", result_cov.aic)
# %%
model_cov = smf.mixedlm("bmi ~ measure_week * cgm * is_glp1 + base_bmi", data=df_model_cov.dropna(subset=['is_glp1', 'base_bmi']),
                        groups=df_model_cov.dropna(subset=['is_glp1', 'base_bmi'])["patient_uuid"])
result_cov = model_cov.fit(reml=False)
# result_cov = model_cov.fit()
print(result_cov.summary())
print("AIC:", result_cov.aic)
# %%
# 作图
# 1. 提取我们刚才建好的全变量且无缺失的模型数据
variables = [model_cov.endog_names] + model_cov.exog_names
plot_df = df_model_cov.dropna(subset=['bmi', 'measure_week', 'cgm', 'base_bmi', 'is_glp1']).copy()
# 2. 预测值！生成模型算出来的“绝对公平的校正 BMI”
plot_df['predicted_bmi'] = result_cov.predict(plot_df)
# 3. 开始用 seaborn 画图 (注意这里的 y 是 predicted_bmi)
plt.figure(figsize=(8, 4))
# 你可以根据需要决定给不给 GLP1 分组，如果你只想看设备的作用：
sns.lineplot(
    data=plot_df,
    x='measure_week',
    y='predicted_bmi',
    hue='cgm',
    # 可以用 style='is_glp1' 同时看看用药不用药的区别
    style='is_glp1',
    linewidth=2.5,
    errorbar=None # 预测拟合线一般不带置信带，或者你可以用 errorbar=None
)
plt.title('模型校正后的 BMI 每周下降轨迹 (Predicted Trajectories)', fontsize=14)
plt.xlabel('减重服务的周次', fontsize=12)
plt.ylabel('校正预测 BMI', fontsize=12)
# 完善图例 (获取当前图例标签，将其中的英文变量名替换为中文)
handles, labels = plt.gca().get_legend_handles_labels()
labels = [label.replace('cgm', '佩戴设备').replace('is_glp1', '药物干预') for label in labels]
plt.legend(handles=handles, labels=labels, title='干预分组', bbox_to_anchor=(1.05, 1), loc='upper left', handlelength=3.0)

# 提取关键统计学信息用于标注（这里可以手动写死，也可以从 result_cov 里取，手写最简单）
stat_text = (
    f"$N_{{total}} = {result_cov.model.n_groups}$\n"
    f"$P_{{interaction}} = {result_cov.pvalues['measure_week:cgm[T.有CGM]']:.3f}$" # 交互项P值填写
)
# 使用相对坐标 (transform=plt.gca().transAxes) 把文字钉在图上
# x=0.03, y=0.05 表示距离左侧3%，距离底部5%的位置（通常左下角比较空）
plt.text(
    x=0.05, y=0.05,
    s=stat_text, 
    transform=plt.gca().transAxes, 
    fontsize=12,
    verticalalignment='bottom',
    # 加一个半透明的白色背景框，防止被折线网格线挡住
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
    alpha=0, edgecolor='lightgray')
)
plt.xticks(range(0, int(cut_week)+1, 2))
sns.despine()
plt.tight_layout()
plt.savefig('predicted_bmi.png', dpi=300, transparent=True)
plt.show()
# %%
model_glp1 = smf.mixedlm("bmi ~ measure_week * cgm + base_bmi", data=df_model_cov.query('is_glp1 == "有GLP1"').dropna(subset=['is_glp1', 'base_bmi']),
                        groups=df_model_cov.query('is_glp1 == "有GLP1"').dropna(subset=['is_glp1', 'base_bmi'])["patient_uuid"])
result_glp1 = model_glp1.fit(reml=False)
# result_cov = model_cov.fit()
print(result_glp1.summary())
print("AIC:", result_glp1.aic)
# %%
model_no_glp1 = smf.mixedlm("bmi ~ measure_week * cgm + base_bmi", data=df_model_cov.query('is_glp1 == "无GLP1"').dropna(subset=['is_glp1', 'base_bmi']),
                        groups=df_model_cov.query('is_glp1 == "无GLP1"').dropna(subset=['is_glp1', 'base_bmi'])["patient_uuid"])
result_no_glp1 = model_no_glp1.fit(reml=False)
# result_cov = model_cov.fit()
print(result_no_glp1.summary())
print("AIC:", result_no_glp1.aic)

# %%
# 使用 sns.relplot 画出分层子图
g = sns.relplot(
    data=df_model_cov,
    x='measure_week',
    y='bmi',
    hue='cgm',       # 用颜色区分是否佩戴 CGM
    col='medicine_name',   # 核心：根据有没有用 GLP-1 拆成两张并排的图
    kind='line',
    height=5, 
    aspect=1.2,
    marker='o',      # 在折线上把每周的实际点加上
    err_style='bars' # 或者默认带阴影区
)
g.set_axis_labels('减重服务的周次', '真实平均 BMI') #"随访周次"
g.set_titles("是否开具 GLP1={col_name}")
g.set(xticks=range(0, 21, 2))
sns.move_legend(g, "center right", bbox_to_anchor=(1, 0.5), title="有无CGM")
g.figure.subplots_adjust(top=0.88)
g.figure.suptitle('不同用药人群下 CGM 对减重速度的真实作用趋势', fontsize=16)
# glp1_text = (
#     f"$N_{{total}} = {result_glp1.model.n_groups}$\n"
#     f"$P_{{interaction}} = {result_glp1.pvalues['measure_week:cgm[T.有CGM]']:.3f}$" # 交互项P值填写
# )
# no_glp1_text = (
#     f"$N_{{total}} = {result_no_glp1.model.n_groups}$\n"
#     f"$P_{{interaction}} = {result_no_glp1.pvalues['measure_week:cgm[T.有CGM]']:.3f}$" # 交互项P值填写
# )
# g.fig.text(0.1, 0.2, glp1_text, fontsize=10)
# g.fig.text(0.6, 0.2, no_glp1_text, fontsize=10)
# plt.savefig('real_bmi_medicine.png', dpi=300, transparent=True)
# %%
save = False # 手动决定是否保存
if save:
    with open('CGM对减重作用的研究.pkl', 'wb') as f:
        pickle.dump((tbl, orders, pres, tbl_agg, df_model, df_model_cov, result_cov, plot_df, g), f)
# %%
read = False # 手动决定是否读取
if read:
    with open('CGM对减重作用的研究.pkl', 'rb') as f:
        tbl, orders, pres, tbl_agg, df_model, df_model_cov, result_cov, plot_df, g = pickle.load(f)
# %%
# 各组样本量
(df_model_cov.dropna(subset=['patient_uuid', 'cgm', 'is_glp1', 'bmi', 'base_bmi'])
.drop_duplicates(['patient_uuid', 'cgm', 'is_glp1'])
[['cgm', 'is_glp1']].value_counts().sort_index().to_frame())
# %%
# %%
# 每周有多少样本
df_model_cov.dropna(subset=['patient_uuid', 'cgm', 'is_glp1', 'bmi', 'base_bmi']).groupby(['measure_week'])['patient_uuid'].count().to_frame('patient_count')
# %%
# 区分不同药物种类的亚组
# 检查是否一个患者可能有多种药物
pres.groupby(['patient_uuid'])['medicine_name'].nunique().sort_values()
# %%
# 选取最新的一个处方
pres_latest = pres.sort_values('date').drop_duplicates(['patient_uuid'], keep='last')
# %%
df_model_cov = df_model_cov.join(pres_latest.set_index('patient_uuid')[['medicine_name']],
on='patient_uuid', how = 'left')
df_model_cov
# %%
model_cov = smf.mixedlm("bmi ~ measure_week * cgm + base_bmi + measure_week * medicine_name",
data=df_model_cov.query('medicine_name != "玛仕度肽"').dropna(subset=['medicine_name', 'base_bmi']),
                        groups=df_model_cov.query('medicine_name != "玛仕度肽"').dropna(subset=['medicine_name', 'base_bmi'])["patient_uuid"])
result_cov = model_cov.fit(reml=False)
# result_cov = model_cov.fit()
print(result_cov.summary())
print("AIC:", result_cov.aic)
# %%
res_df = pd.DataFrame(result_cov.summary().tables[1])
res_df
res_title = result_cov.summary().tables[0]
res_title = pd.DataFrame(res_title)
res_title.columns = [''] * res_title.shape[1]
display(
res_title,
res_df)
# %%

# %%
df_model_cov.dropna(subset=['medicine_name', 'base_bmi']).groupby(['cgm', 'medicine_name'])['patient_uuid'].nunique().sort_index().to_frame('patient_counts')
# %%
model_cov = smf.mixedlm("bmi ~ base_bmi + measure_week * medicine_name",
data=df_model_cov.dropna(subset=['medicine_name', 'base_bmi']),
                        groups=df_model_cov.dropna(subset=['medicine_name', 'base_bmi'])["patient_uuid"])
result_cov = model_cov.fit(reml=False)
# result_cov = model_cov.fit()
print(result_cov.summary())
print("AIC:", result_cov.aic)
# %%
res_df = pd.DataFrame(result_cov.summary().tables[1])
res_df
res_title = result_cov.summary().tables[0]
res_title = pd.DataFrame(res_title)
res_title.columns = [''] * res_title.shape[1]
display(
res_title,
res_df)

# %%
# 时间能不能对齐
df_model_cov.groupby(['measure_week'])['patient_uuid'].nunique().to_frame('patient_counts')
# %%
df_model_cov['start_date'] = pd.to_datetime(df_model_cov['start_date']).dt.date
df_model_cov['end_days'] = df_model_cov['start_date'].map(lambda x: dt.date(2026, 3, 24) - x).dt.days
df_model_cov['end_weeks'] = df_model_cov['end_days'].map(lambda x: np.ceil((x+1)/7) - 1)
df_model_cov[['end_days', 'end_weeks']]
# df_model_cov.groupby(['end_weeks'])['patient_uuid'].nunique().to_frame('patient_counts')
# %%
model_cov = smf.mixedlm("bmi ~ measure_week * cgm + base_bmi + measure_week * medicine_name",
data=df_model_cov.query('medicine_name != "玛仕度肽" and end_weeks >= 11').dropna(subset=['medicine_name', 'base_bmi']),
                        groups=df_model_cov.query('medicine_name != "玛仕度肽" and end_weeks >= 11').dropna(subset=['medicine_name', 'base_bmi'])["patient_uuid"])
result_cov = model_cov.fit(reml=False)
# result_cov = model_cov.fit()
print(result_cov.summary())
print("AIC:", result_cov.aic)
# %%
# %%
model_cov = smf.mixedlm("bmi ~ measure_week * cgm + base_bmi + measure_week * is_glp1",
data=df_model_cov.query('end_weeks >= 19').dropna(subset=['is_glp1', 'base_bmi', 'cgm']),
                        groups=df_model_cov.query('end_weeks >= 19').dropna(subset=['is_glp1', 'base_bmi', 'cgm'])["patient_uuid"])
result_cov = model_cov.fit(reml=False)
# result_cov = model_cov.fit()
print(result_cov.summary())
print("AIC:", result_cov.aic)
# %%
# 核心人群
week_goal = 19
core_sample = (df_model_cov.query('end_weeks >= @week_goal and measure_week <= @week_goal and bmi > 0')
                .groupby('patient_uuid')['measure_week'].nunique().to_frame('measure_week_counts')
                .query('measure_week_counts >= @week_goal+1').index)
core_sample
# %%
data = (df_model_cov
        .query('patient_uuid in @core_sample and measure_week <= @week_goal')
        .dropna(subset=['is_glp1', 'base_bmi', 'cgm']))
data.groupby(['cgm', 'is_glp1'])['patient_uuid'].nunique().to_frame('patient_counts')
# %%
model_cov_core = smf.mixedlm("bmi ~ measure_week * cgm + base_bmi + measure_week * is_glp1",
                            data=data, groups=data["patient_uuid"])
result_cov_core = model_cov_core.fit(reml=False)
# result_cov = model_cov.fit()
print(result_cov_core.summary())
print("AIC:", result_cov_core.aic)
# %%
# plot_df['predicted_bmi'] = result_cov.predict(plot_df)
data['predicted_bmi'] = result_cov_core.predict(data.drop(columns=['medicine_name']))
# 3. 开始用 seaborn 画图 (注意这里的 y 是 predicted_bmi)
plt.figure(figsize=(8, 4))
# 你可以根据需要决定给不给 GLP1 分组，如果你只想看设备的作用：
sns.lineplot(
    data=data,
    x='measure_week',
    y='bmi',
    hue='cgm',
    # 可以用 style='is_glp1' 同时看看用药不用药的区别
    style='is_glp1',
    linewidth=2.5,
    errorbar=None # 预测拟合线一般不带置信带，或者你可以用 errorbar=None
)
plt.title('模型校正后的 BMI 每周下降轨迹 (Predicted Trajectories)', fontsize=14)
plt.xlabel('减重服务的周次', fontsize=12)
plt.ylabel('校正预测 BMI', fontsize=12)
# 完善图例 (获取当前图例标签，将其中的英文变量名替换为中文)
handles, labels = plt.gca().get_legend_handles_labels()
labels = [label.replace('cgm', '佩戴设备').replace('is_glp1', '药物干预') for label in labels]
plt.legend(handles=handles, labels=labels, title='干预分组', bbox_to_anchor=(1.05, 1), loc='upper left', handlelength=3.0)

# 提取关键统计学信息用于标注（这里可以手动写死，也可以从 result_cov 里取，手写最简单）
stat_text = (
    f"$N_{{total}} = {result_cov_core.model.n_groups}$\n"
    f"$P_{{interaction}} = {result_cov_core.pvalues['measure_week:cgm[T.有CGM]']:.3f}$" # 交互项P值填写
)
# 使用相对坐标 (transform=plt.gca().transAxes) 把文字钉在图上
# x=0.03, y=0.05 表示距离左侧3%，距离底部5%的位置（通常左下角比较空）
plt.text(
    x=0.05, y=0.05,
    s=stat_text, 
    transform=plt.gca().transAxes, 
    fontsize=12,
    verticalalignment='bottom',
    # 加一个半透明的白色背景框，防止被折线网格线挡住
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
    alpha=0, edgecolor='lightgray')
)
plt.xticks(range(0, week_goal+2, 2))
sns.despine()
plt.tight_layout()
# plt.savefig('predicted_bmi.png', dpi=300, transparent=True)
plt.show()
# %%
# 生存曲线
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
# %%
# ----------------- 步骤1：严谨定义“反弹事件” -----------------
# 针对每个患者，我们先找到他随访中达到的“最瘦（最低 BMI）”的那一周
survival_data = []

for uuid, group in (df_model_cov
                    .dropna(subset=['bmi', 'base_bmi', 'cgm', 'is_glp1', 'measure_week'])
                    .groupby('patient_uuid')):
    # 找到该患者历史最低的 BMI 及其对应的周次
    min_bmi = group['bmi'].min()
    base_bmi = group['base_bmi'].min()
    min_week = group.loc[group['bmi'].idxmin(), 'measure_week']

    # 获取他最低点之后的所有随访记录
    after_min = group[group['measure_week'] > min_week]

    # 【核心！】定义什么是“反弹”？这里定义为：相比最低点回升了 1.0 个 BMI
    # todo:查文献，反弹的定义
    # REBOUND_THRESHOLD = 0.5
    REBOUND_RATIO = 0.05
    is_rebound = 0
    time_to_event = 0

    if len(after_min) == 0:
        # 如果达到了最低点之后就再也没测过了（人跑了），记作随访 0 周删失
        time_to_event = 0
        is_rebound = 0
    else:
        # 看看最低点之后，有没有哪一周的 BMI 超过了反弹阈值
        rebound_records = (after_min[after_min['bmi'] >= (min_bmi + (base_bmi-min_bmi)*REBOUND_RATIO)]
                            .sort_values(by='measure_week', ascending=True)
                            .reset_index(drop=True))

        if len(rebound_records) > 0:
            # 发生反弹了！记录从最低点到反弹，熬了多少周
            is_rebound = 1
            rebound_week = rebound_records.iloc[0]['measure_week']
            time_to_event = rebound_week - min_week
        else:
            # 没发生反弹，直到他最后一次称重！记作删失（Censored）
            is_rebound = 0
            last_week = after_min['measure_week'].max()
            time_to_event = last_week - min_week

    # 其他协变量，只取一个值
    cgm_status = group['cgm'].iloc[0]
    is_glp1 = group['is_glp1'].iloc[0]
    base_bmi = group['base_bmi'].iloc[0]

    survival_data.append({
        'patient_uuid': uuid,
        'T': time_to_event, # 维持了多少周没反弹
        'E': is_rebound,    # 是否死了（是否反弹了）
        'cgm': cgm_status,
        'is_glp1': is_glp1,
        'base_bmi': base_bmi
    })
# %%
# 构建生存分析 DataFrame
df_survival = pd.DataFrame(survival_data)
print(df_survival.shape)
# 剔除那些最低点就是最后一天，根本没有后续随访时间的人
df_survival = df_survival[df_survival['T'] > 0]
print(df_survival.shape)
df_survival
# %%
# ----------------- 步骤2：画出 Kaplan-Meier -----------------
plt.figure(figsize=(8, 6))
kmf_cgm = KaplanMeierFitter()
kmf_no_cgm = KaplanMeierFitter()

# 画有 CGM 组的线
mask_cgm = df_survival['cgm'] == '有CGM'
if sum(mask_cgm) > 0:
    kmf_cgm.fit(df_survival[mask_cgm]['T'], df_survival[mask_cgm]['E'], label='有CGM 患者')
    kmf_cgm.plot_survival_function(ci_show=True, linewidth=2.5, color='#2ca02c') # 留存率曲线

# 画无 CGM 组的线
mask_no_cgm = df_survival['cgm'] == '无CGM'
if sum(mask_no_cgm) > 0:
    kmf_no_cgm.fit(df_survival[mask_no_cgm]['T'], df_survival[mask_no_cgm]['E'], label='无CGM 患者')
    kmf_no_cgm.plot_survival_function(ci_show=True, linewidth=2.5, linestyle='--', color='#ff7f0e')

# 进行 Log-Rank 检验，算出神圣的 P 值
results = logrank_test(df_survival[mask_cgm]['T'], df_survival[mask_no_cgm]['T'], 
                       event_observed_A=df_survival[mask_cgm]['E'], event_observed_B=df_survival[mask_no_cgm]['E'])
p_value = results.p_value

# 装扮图表
plt.title('患者“无反弹”生存曲线 (Kaplan-Meier Relapse-Free Survival)', fontsize=14)
plt.xlabel('达到个人最低体重后的维持周数', fontsize=12)
plt.ylabel('尚未发生反弹的生存概率 (未反弹率)', fontsize=12)
plt.ylim(0, 1.05)
plt.text(0.05, 0.1, f"$P_{{Log-Rank}} = {p_value:.4f}$", transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

sns.despine()
plt.tight_layout()
plt.show()
# %%
# ----------------- 步骤3：多因素 Cox 比例风险回归模型 -----------------
# %%
# 准备 Cox 模型的数据，需要把分类变量转换为数值虚拟变量 (Dummy variables)
# 设定参照组为0：'无CGM' -> 0, '有CGM' -> 1；'无GLP1' -> 0, '有GLP1' -> 1
df_cox = df_survival.copy()

df_cox['is_cgm'] = df_cox['cgm'].apply(lambda x: 1 if x == '有CGM' else 0)
df_cox['is_glp1'] = df_cox['is_glp1'].apply(lambda x: 1 if x == '有GLP1' else 0)

# 只保留用于建模的列：持续时间(T)、结局事件(E)、需要纳入的自变量
df_cox = df_cox[['T', 'E', 'is_cgm', 'is_glp1', 'base_bmi']]

# 拟合 Cox 比例风险模型
cph = CoxPHFitter()
cph.fit(df_cox, duration_col='T', event_col='E')

# 打印回归结果详细报告
cph.print_summary()

# 画出各个变量的风险比森林图（Hazard Ratio Forest Plot）
plt.figure(figsize=(8, 4))
cph.plot()
plt.title('多因素 Cox 回归森林图 (Hazard Ratios)', fontsize=14)
plt.tight_layout()
plt.show()
# %%
