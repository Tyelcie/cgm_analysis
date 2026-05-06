# %%
import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from cleaner import AppActive
from engineer import yutang_palette, yutang_theme, CGM_AGP_plot,CGM_cal_metrics
sns.set_theme(style='white', palette=yutang_palette, rc=yutang_theme)
plt.rcParams.update(yutang_theme)
from connections import engine
end = dt.date.today()
# %%
hospital_name = '天津医科大学朱宪彝纪念医院（代谢病医院）'
# hospital_name = "山东省立医院"
# %%
sql = f'''select cgm.*
from stats.t_cgm_params as cgm
join sanyi_care.u_patient as p
on cgm.patient_uuid = p.uuid
where p.hospital_name = "{hospital_name}";'''

with engine.connect() as con:
    cgm = pd.read_sql(sql, con)
for i in ['date', 'first_date']:
    cgm[i] = cgm[i].astype('datetime64[ns]')
cgm = cgm.query('valid_data > 0')
# %%
print(f"有多少患者传入了CGM数据: {cgm['patient_uuid'].nunique()}")
# %%
cgm_pts = cgm.groupby(['patient_uuid']).agg(days = ('date', 'nunique'),
data_valid = ('valid_data', 'sum'))
cgm_pts['data_valid_rate'] = cgm_pts.eval('data_valid/(days*1440)')
print(f"人均有效数据比例: {cgm_pts['data_valid_rate'].mean()}")
# %%
cgm['days'] = cgm.eval('(date - first_date).dt.days')
cgm['weeks'] = cgm.eval('days/7').map(np.floor)
# 计算每周的总有效数据量，过滤不完整周（阈值：5天 * 1440 = 7200
week_validity = cgm.groupby(['patient_uuid', 'weeks'])['valid_data'].sum().to_frame('week_valid_sum').reset_index()
valid_weeks = week_validity.query('week_valid_sum >= 1440*5')[['patient_uuid', 'weeks']]

# 仅保留数据量充足的周进行对比
cgm_valid = cgm.merge(valid_weeks, on=['patient_uuid', 'weeks'])
# %%
# 按有效数据周数进行患者分组打标签
cgm_weeks = cgm_valid.groupby(['patient_uuid'])['weeks'].nunique().to_frame('week_count').reset_index()
cgm_valid = cgm_valid.merge(cgm_weeks, on=['patient_uuid'])
# %%
cgm_valid['care_period'] = pd.cut(cgm_valid['week_count'], bins=[0, 2, 4, 8, 12, np.inf],
    labels=['≤2周', '3-4周', '5-8周', '9-12周', '≥13周'], include_lowest=True).astype(str)
cgm_valid['care_period'].value_counts().sort_index()
# %%
# 按周计算聚合指标
cgm_valid['mean_sum'] = cgm_valid.eval('mean*valid_data')
cgm_valid['sum_sq'] = cgm_valid.eval('(std**2 + mean**2) * valid_data')
cgm_valid['lbgi_sum'] = cgm_valid.eval('lbgi * valid_data')
cgm_valid['hbgi_sum'] = cgm_valid.eval('hbgi * valid_data')

cgm_week_agg = cgm_valid.groupby(['patient_uuid', 'weeks', 'week_count'], observed=False).agg(
    valid_data = ('valid_data', 'sum'),
    vlow = ('vlow', 'sum'),
    low = ('low', 'sum'),
    high = ('high', 'sum'),
    vhigh = ('vhigh', 'sum'),
    in_range_tight = ('in_range_tight', 'sum'),
    mean_sum = ('mean_sum', 'sum'),
    sum_sq = ('sum_sq', 'sum'),
    lbgi_sum = ('lbgi_sum', 'sum'),
    hbgi_sum = ('hbgi_sum', 'sum')
)

cgm_week_agg['mean'] = cgm_week_agg.eval('mean_sum/valid_data')
cgm_week_agg['std'] = np.sqrt(cgm_week_agg.eval('sum_sq/valid_data - mean**2'))
cgm_week_agg['cv'] = cgm_week_agg.eval('std/mean')
cgm_week_agg['lbgi'] = cgm_week_agg.eval('lbgi_sum/valid_data')
cgm_week_agg['hbgi'] = cgm_week_agg.eval('hbgi_sum/valid_data')
cgm_week_agg['TIR'] = cgm_week_agg.eval('(valid_data - vlow - low - high - vhigh)/valid_data')
cgm_week_agg['TBR'] = cgm_week_agg.eval('(low+vlow)/valid_data')
cgm_week_agg['TBR_II'] = cgm_week_agg.eval('vlow/valid_data')
cgm_week_agg['TAR'] = cgm_week_agg.eval('(high+vhigh)/valid_data')
cgm_week_agg['TAR_II'] = cgm_week_agg.eval('vhigh/valid_data')
cgm_week_agg['GRI'] = cgm_week_agg.eval('(3.0 * vlow + 2.4 * low + 1.6 * vhigh + 0.8 * high)/valid_data')
cgm_week_agg['valid_rate'] = cgm_week_agg.eval('valid_data/(1440*7)')
cgm_week_agg = cgm_week_agg.reset_index()
# %%
# 对有效周进行正向和反向排名，确定基线和末周
cgm_week_agg['seq_asc'] = cgm_week_agg.groupby(['patient_uuid'])['weeks'].rank(ascending=True, method='dense')
cgm_week_agg['seq_desc'] = cgm_week_agg.groupby(['patient_uuid'])['weeks'].rank(ascending=False, method='dense')

# 提取基线和末周，并排除只有一周数据的患者
cgm_pair = cgm_week_agg.query('seq_asc == 1 or seq_desc == 1')
cgm_pair = cgm_pair.query('~(seq_asc == 1 and seq_desc == 1)')

print(f"数据量充足的有2周以上的人数: {cgm_pair['patient_uuid'].nunique()}")
cgm_pair['seq'] = cgm_pair.apply(lambda x: '首周' if x['seq_asc'] == 1 else "末周", axis=1)
# %%
metrics_name = {'TBR': 'TBR（<3.9mmol/L）',
'TBR_II': '二级TBR（<3mmol/L）',
'TAR': 'TAR（>10mmol/L）',
'TAR_II': '二级TAR（>13.9mmol/L）',
'TIR': 'TIR（3.9-10mmol/L）',
'GRI': '血糖风险指数（GRI）',
'lbgi': '低血糖风险指数（LBGI）',
'hbgi': '高血糖风险指数（HBGI）',
'mean': '平均值', 'std': '标准差', 'cv': '变异系数（CV）'}
cgm_pair_melt = cgm_pair.melt(id_vars=['patient_uuid', 'seq', 'weeks', 'week_count'], value_vars=list(metrics_name.keys()))
cgm_pair_melt
# %%
order_seq = ['首周', '末周']
order_variable = ['TBR', 'TBR_II', 'TAR', 'TAR_II', 'TIR', 'GRI', 'lbgi','hbgi', 'mean', 'std', 'cv']
# %%
# cgm_pair_melt['seq'] = pd.Categorical(cgm_pair_melt['seq'], categories=order_seq, ordered=True)
cgm_pair_melt['variable'] = pd.Categorical(cgm_pair_melt['variable'], categories=order_variable, ordered=True)
cgm_pair_melt = cgm_pair_melt.sort_values(['variable', 'seq'])
# %%
obs = cgm_pair_melt.copy()
week_cond = '>=7'
save_folder = f'./{hospital_name}/有效佩戴{week_cond.replace(">=", "大于等于")}周的患者'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
obs = obs.query(f'week_count {week_cond}')
print(f'样本量(人数): {obs["patient_uuid"].nunique()}')
obs['care_period'] = pd.cut(obs['week_count'], bins=[0, 2, 4, 8, 12, np.inf],
    labels=['≤2周', '3-4周', '5-8周', '9-12周', '≥13周'], include_lowest=True).astype(str)
g = sns.FacetGrid(obs, col='variable', col_wrap=4, sharey=False)
g.map(sns.boxplot, 'seq', 'value', order = order_seq, showfliers=False)
for ax in g.axes.flat:
    # 获取指标名称
    metrics = ax.get_title().split('=')[1].strip()
    
    # 统计检验逻辑
    base_vals = obs.query(f'variable == "{metrics}" and seq == "首周"').sort_values('patient_uuid')['value']
    last_vals = obs.query(f'variable == "{metrics}" and seq == "末周"').sort_values('patient_uuid')['value']
    
    if len(base_vals) == len(last_vals) and len(base_vals) > 0:
        try:
            stat, p_val = stats.wilcoxon(base_vals, last_vals)
            p_text = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
            f_color = 'darkred' if p_val < 0.05 else 'black'
        except ValueError:
            p_text = "p = 1.000"
            f_color = 'black'
        
        ax.text(0.95, 0.95, p_text, transform=ax.transAxes,
                va='top', ha='right',
                weight='bold', color=f_color, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        ax.text(0.25, 0, f"N = {len(base_vals)}", transform=ax.transAxes, 
                va='bottom', ha='center',
                weight='bold', color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        ax.text(0.75, 0, f"N = {len(base_vals)}", transform=ax.transAxes, 
                va='bottom', ha='center',
                weight='bold', color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        ax.text(0, base_vals.median(), f"{base_vals.median():.2f}", 
                va='bottom', ha='center',
                weight='bold', color='black', bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))
        ax.text(1, last_vals.median(), f"{last_vals.median():.2f}", 
                va='bottom', ha='center',
                weight='bold', color='black', bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))
    ax.set_title(metrics_name[metrics])
    ax.set_xlabel('')
    ax.set_ylabel('')
plt.suptitle(f'各指标首周和末周的变化(取有效佩戴{week_cond}周的患者)', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(save_folder, '00_首周和末周指标对比.png'), dpi=300, transparent = False)
# %%
for i,metrics in enumerate(order_variable):
    plot_data = cgm_week_agg.query(f'weeks <= 12 and week_count {week_cond}').copy()
    plot_data['weeks'] = plot_data['weeks'].map(lambda x: format(x, '.0f'))

    # 绘图
    ax = sns.boxplot(data = plot_data, x = 'weeks', y = metrics, showfliers=False, order=[str(i) for i in range(13)])

    # 计算统计量并标注
    stats_df = plot_data.groupby('weeks')[metrics].agg(['median', 'count'])
    for j, label in enumerate(ax.get_xticklabels()):
        week_str = label.get_text()
        if week_str in stats_df.index:
            median = stats_df.loc[week_str, 'median']
            n = stats_df.loc[week_str, 'count']
            # 标注中位数
            ax.text(j, median, f'{median:.2f}', ha='center', va='bottom',
                    color='black', fontsize=9, weight='bold')
            # 标注样本量
            ax.text(j, ax.get_ylim()[0], f'N={int(n)}', ha='center', va='bottom',
                    color='black', fontsize=8, weight='bold')

    plt.xlabel('佩戴周次')
    plt.ylabel(metrics_name[metrics])
    plt.title(f'{metrics_name[metrics]}的佩戴周次变化(取有效佩戴{week_cond}周的患者)')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'{i+1:02d}_{metrics}_趋势变化.png'), dpi=300, transparent = False)
    plt.close()
# %%
# 检查一些特别的个体
pt = 'PAT_hxcuva6wdf'
start_date, end_date = cgm_valid.query('patient_uuid == @pt and weeks == 6')['date'].agg(['min','max'])
print(start_date, end_date)
# %%
# pt = 'PAT_hxcuva6wdf'
app_data = AppActive(pts = [pt], start_time=start_date, 
end_time=end_date)
cgm_origin = app_data.t_cgm()
df = CGM_cal_metrics(df=cgm_origin)
CGM_AGP_plot(df, title=f'AGP图谱: {pt}')
# %%
