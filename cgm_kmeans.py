# %%
import datetime as dt
import pandas as pd
import seaborn as sns
from connections import engine
from cleaner import AppActive
from engineer import CGM_cluster_non_diabetic, CGM_cal_metrics, CGM_AGP_plot, yutang_palette, yutang_theme
sns.set_theme(rc=yutang_theme, style='white', palette=yutang_palette)
# %%
sql = '''select p.uuid, p.collect_date, p.group_name, p.join_date, p.vip_start_date, p.vip_end_date
from sanyi_care.u_patient as p
left join sanyi_care.u_hospital as h
on p.hospital_uuid = h.uuid
left join sanyi_care.u_team as t
on p.team_uuid = t.uuid
where h.status = "on"
and t.status = "on"
and p.status in ("join", "take_over", "wait_join")
and exists (
    select 1 from sanyi_care.t_cgm as cgm 
    where cgm.patient_uuid = p.uuid
)
'''
with engine.connect() as con:
    tbl = pd.read_sql(sql, con)
tbl
# %%
app_data = AppActive(pts=tbl['uuid'], start_time=dt.date(1, 1, 1),
end_time=dt.date.today())
# %%
cgm = app_data.t_cgm()
cgm
# %%
with open('cgm_all_patient.json', 'w') as f:
    cgm.to_json(f, orient='records', force_ascii=False)
# %%
sql = '''
select uuid, collect_date, join_date, vip_start_date, vip_end_date, group_name
from sanyi_care.u_patient as p
left join sanyi_care.d_yz_order as yz
on p.uuid = yz.user_uuid
where p.hospital_uuid = "1001"
and yz.pkg_type = "体重管理"
'''
with engine.connect() as con:
    tbl_weight = pd.read_sql(sql, con)
# %%
cgm_weight = cgm[cgm['patient_uuid'].isin(tbl_weight['uuid'])]
cgm_weight
# %%
with open('cgm_weight_patient.json', 'w') as f:
    cgm_weight.to_json(f, orient='records', force_ascii=False)
# %%
exam = app_data.t_exam_body_bp()
exam_weight = exam[exam['patient_uuid'].isin(tbl_weight['uuid'])]
exam_weight
with open('exam_weight_patient.json', 'w') as f:
    exam_weight.to_json(f, orient='records', force_ascii=False)
# %%
weighter = app_data.t_self_weight()
weighter_weight = weighter[weighter['patient_uuid'].isin(tbl_weight['uuid'])]
weighter_weight
with open('weighter_weight_patient.json', 'w') as f:
    weighter_weight.to_json(f, orient='records', force_ascii=False)
# %%
cgm_new = cgm.join(tbl.set_index('uuid')[['collect_date']], on='patient_uuid', how='inner')
cgm_new['collect_date'] = pd.to_datetime(cgm_new['collect_date'])
cgm_new['date'] = pd.to_datetime(cgm_new['date'])
cgm_new['enroll_days'] = cgm_new.eval('date - collect_date').dt.days
cgm_new = cgm_new.query('enroll_days >= 0 and enroll_days <= 6')
cgm_new
# %%
# %%
cal = CGM_cal_metrics(df=cgm_new)
# 调用 cal_all_metrics() 触发所有核心指标的计算
# 这会把指标计算结果保存在 cal.metrics 中，并返回用于展示的 metrics_display 表
metrics = cal.cal_all_metrics()
metrics
# %%
cluster = CGM_cluster_non_diabetic()
cluster_res = cluster.fit(cal)
cluster_res
# %%
# 可视化本次新的聚类结果
cluster.plot_pca()
cluster.plot_radar()
# %%
