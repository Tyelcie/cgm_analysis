# %%
from narwhals import col
from matplotlib.pyplot import axis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from engineer import get_bmi, yutang_palette, yutang_theme
from connections import engine
from cleaner import AppActive
from exporter import data_wide
import data_dict as dd
from recoding import uuid_decoding
sns.set_theme(rc=yutang_theme, style='white', palette=yutang_palette)
# %%
sql = '''select p.uuid, p.team_name, p.age, p.sex, p.collect_date, pbd.glycuresis_year
from sanyi_care.u_patient as p
left join sanyi_care.u_patient_base_disease as pbd
on p.uuid = pbd.patient_uuid
where team_name in ("张景云主任内分泌科—体重管理VIP照护团队", "张景云主任内分泌科—体重管理照护团队", 
"李金金主任内分泌科—体重管理VIP照护团队")
'''
with engine.connect() as con:
    tbl = pd.read_sql(sql, con)
tbl['collect_date'] = pd.to_datetime(tbl['collect_date'])
tbl
# %%
sql = f'''select patient_uuid, weight, height, created_at
from sanyi_care.t_exam_body_bp
where patient_uuid in ('{"','".join(tbl['uuid'].unique())}')
'''
with engine.connect() as con:
    exam = pd.read_sql(sql, con)
exam = exam.join(tbl.set_index('uuid')[['collect_date']], on='patient_uuid', how='left')
exam['created_at'] = pd.to_datetime(exam['created_at'])
exam = exam.query('created_at >= collect_date')
exam['measure_date'] = exam['created_at'].dt.date
exam = exam.sort_values(['created_at']).drop_duplicates(subset=['patient_uuid', 'measure_date'], keep='last')
exam['bmi'] = get_bmi(data=exam, height='height', weight='weight')
exam['bmi分级'] = get_bmi(data=exam, height='height', weight='weight', cut = True)
# %%
# %%
sql = f'''select patient_uuid, weight, bmi, measured_at
from sanyi_care.t_self_weight
where patient_uuid in {tuple(tbl['uuid'].unique())}'''
with engine.connect() as con:
    weight = pd.read_sql(sql, con)
weight = weight.join(tbl.set_index('uuid')['collect_date'], on='patient_uuid', how='left')
weight = weight.query('measured_at >= collect_date')
weight['bmi分级'] = get_bmi(data=weight, bmi='bmi', cut=True)
# %%
exam_weight = pd.concat([exam[['patient_uuid', 'bmi', 'bmi分级', 'created_at']].rename(columns={'created_at': 'measured_at'}),
weight[['patient_uuid', 'bmi', 'bmi分级', 'measured_at']]])
exam_weight
# %%
exam['measure_days'] = exam.eval('created_at - collect_date').dt.days
exam['seq'] = exam['measure_days'].map(lambda x: '基线' if x <= 30 else '最新')
exam = exam.sort_values(['measure_date'])
exam_base = exam.query('seq == "基线"').drop_duplicates(subset=['patient_uuid'], keep='first')
exam_last = exam.query('seq == "最新"').drop_duplicates(subset=['patient_uuid'], keep='last')
exam_widt = pd.concat([exam_base, exam_last])
exam
# %%
exam_pivot = exam.pivot_table(index=['patient_uuid'], columns='seq', values=['bmi', 'bmi分级'],
aggfunc='first')
exam_pivot.columns = [f'{j}{i}' for i, j in exam_pivot.columns]
exam_pivot
# %%
tbl = tbl.join(exam_pivot, on='uuid', how='left')
tbl
# %%
sql = f'''select *
from sanyi_care.c_prescription_tag
where patient_uuid in {tuple(tbl['uuid'])}
'''
with engine.connect() as con:
    pres = pd.read_sql(sql, con)
pres['is_glp1'] = pres['medicine_name'].str.contains('肽')
pres['is_glp1'] = pres.apply(lambda x: False if x['medicine_name'] in ['胰激肽原酶肠溶片'] else x['is_glp1'], axis=1)
display('GLP1:', pres.query('is_glp1 == True')['medicine_name'].unique().tolist(),
'非GLP1:', pres.query('is_glp1 == False')['medicine_name'].unique().tolist())
# %%
tbl['is_glp1'] = tbl['uuid'].isin(pres.query('is_glp1')['patient_uuid'])
tbl
# %%
weight['rank'] = weight.groupby(['patient_uuid'])['measured_at'].rank(method = 'dense')
weight['rev_rank'] = weight.groupby(['patient_uuid'])['measured_at'].rank(method = 'dense', ascending=False)
weight['seq'] = weight.apply(lambda x: '体脂称第一条' if x['rank'] == 1 else '体脂称最后一条' if x['rank'] > 1 and x['rev_rank'] == 1 else pd.NA, axis=1)
weight = weight.dropna()
weight
# %%
weight_pivot = weight.pivot_table(index='patient_uuid', columns='seq', values=['bmi', 'bmi分级'],
aggfunc='first')
weight_pivot.columns = [f'{j}{i}' for i, j in weight_pivot.columns]
weight_pivot
# %%
tbl = tbl.join(weight_pivot, on='uuid', how='left')
tbl
# %%
exam_weight['rank'] = exam_weight.groupby(['patient_uuid'])['measured_at'].rank(method = 'dense')
exam_weight['rev_rank'] = exam_weight.groupby(['patient_uuid'])['measured_at'].rank(method = 'dense', ascending=False)
exam_weight['seq'] = exam_weight.apply(lambda x: '两个来源第一条' if x['rank'] == 1 else '两个来源最后一条' if x['rank'] > 1 and x['rev_rank'] == 1 else pd.NA, axis=1)
exam_weight =exam_weight.dropna()
exam_weight
# %%
exam_weight_pivot = exam_weight.pivot_table(index='patient_uuid', columns='seq', values=['bmi', 'bmi分级'],
aggfunc='first')
exam_weight_pivot.columns = [f'{j}{i}' for i, j in exam_weight_pivot.columns]
exam_weight_pivot
# %%
tbl = tbl.join(exam_weight_pivot, on='uuid', how='left')
tbl
# %%
# tbl.to_excel('张景云基础数据.xlsx', index=False)
# %%
plt.figure(figsize=[8, 5])
sns.histplot(data = tbl, x = 'age', bins=range(0, max(tbl['age']+1), 10))
plt.xlabel('年龄')
sns.despine()
plt.ylabel('人数')
plt.title('年龄分布')
plt.savefig('年龄分布.png', dpi=300)
# %%
plt.figure(figsize=[8, 5])
tbl['sex'] = tbl['sex'].map({'male': '男', 'female': '女'})
sns.countplot(data = tbl, x = 'sex')
# plt.xlabel('年龄')
sns.despine()
# plt.ylabel('人数')
# plt.title('年龄分布')
# plt.savefig('年龄分布.png', dpi=300)
# %%
data = data_wide(approve_code='00000000', mail=False, base_date='collect_date',
other_cond='''and vpc.team_name in ("张景云主任内分泌科—体重管理VIP照护团队", "张景云主任内分泌科—体重管理照护团队", 
"李金金主任内分泌科—体重管理VIP照护团队")''', return_tbls=True, return_meta=False,
sequance='time', limit_times=93*4+1, seq_days=93,
basic_cols=['collect_date', 'vpc.team_name', 'vpc.age', 'vpc.sex'],
t_exam_body_bp_arg={'columns': ['patient_uuid', 'height', 'weight', 'systolic', 'diastolic', 'created_at']},
t_lab_data_arg={'code_list': ['1591', '205', '204', '202', '184','183','1469',
'1470', '1471', '1472','1473','9003']},
t_self_weight_arg={'columns': ['patient_uuid', 'weight', 'bmi', 'measured_at']},
aggs = {'化验': ['last'], '自测血糖': ['mean'], '体格检查': ['last'], '体脂称': ['last']}
                                                                        )
data['患者ID'] = data['患者ID'].map(uuid_decoding)
data.loc[:, data.columns.str.contains('体重_值')]
# %%
data.to_excel('张景云基础数据-横版.xlsx', index = False)
# %%
data_miss = pd.read_csv('weight_followup_missing_v1v4.csv')
# %%
data_miss['系统ID'] = data_miss['患者ID'].map(uuid_decoding)
data_miss
# %%
data_miss.to_csv('weight_followup_missing_v1v4_decoding.csv', index=False)
# %%
data_all = pd.read_excel('张景云基础数据-横版.xlsx')
data_all_rank = pd.read_excel('张景云基础数据-横版_按次序.xlsx')
# %%
data.loc[data['患者ID'].isin(data_miss['患者ID']), data.columns.str.contains('体重_值')]
# %%
data_all_rank.loc[data_all_rank['患者ID'].isin(data_miss['患者ID']),
                  data_all_rank.columns.str.contains('体重_值')]
# %%
pts = pd.Series([
    'Y3745404233125650021321170758', 
                 'Y3745404253620631433343074346', 
                 'Y3745404207575347534301140001', 'Y3745404259570117583628001217'
                 ]).map(uuid_decoding).tolist()
pts
# %%
data_supp = data_wide(approve_code='00000000', mail=False, base_date='vip_start_date',
                      package_type='体重管理',
other_cond=f'''and vpc.uuid in {tuple(pts)}''', 
return_tbls=True, return_meta=False,
sequance='time', limit_times=93*4+1, seq_days=93,
basic_cols=['collect_date', 'vpc.team_name', 'vpc.age', 'vpc.sex', 'vpc.vip_start_date'],
t_exam_body_bp_arg={'columns': ['patient_uuid', 'height', 'weight', 'systolic', 'diastolic', 'created_at']},
t_lab_data_arg={'code_list': ['1591', '205', '204', '202', '184','183','1469',
'1470', '1471', '1472','1473','9003']},
t_self_weight_arg={'columns': ['patient_uuid', 'weight', 'bmi', 'measured_at']},
aggs = {'化验': ['last'], '自测血糖': ['mean'], '体格检查': ['last'], '体脂称': ['last']}
                                                                        )
# %%
data = pd.concat([data.query('患者ID not in @pts'), data_supp], ignore_index=True)
data.to_excel('张景云基础数据-横版_补充4人数据.xlsx', index=False)
# %%
data_supp.loc[:, ['患者ID', 'VIP开始日期'] + data_supp.columns[data_supp.columns.str.contains('体重_值')].tolist()]
# %%
data_supp.insert(0, '系统ID', data_supp['患者ID'].map(uuid_decoding))
# %%
data_supp.to_excel('张景云补充4人数据-横版_以减重包为基线日期.xlsx', index=False)
# %%
