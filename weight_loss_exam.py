# %%
from narwhals import col
from matplotlib.pyplot import axis
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from engineer import get_bmi, yutang_palette, yutang_theme
from cleaner import AppActive
from connections import engine
sns.set_theme(rc=yutang_theme, style='white', palette=yutang_palette)
# %%
sql = '''select p.uuid, p.team_name, p.age, p.sex, p.collect_date, pbd.glycuresis_year, vip_status
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
# 他们是否购买过含CGM的产品
sql = f'''select h.city, h.name as hospital_name, yz.*, yzr.refund_amount
from sanyi_care.d_yz_order as yz
left join sanyi_care.u_hospital as h
on yz.user_hospital_uuid = h.uuid
left join 
(select oid, sum(total) as refund_amount from sanyi_care.d_yz_return_order group by oid) as yzr
on yz.oid = yzr.oid
where yz.user_uuid in {tuple(tbl['uuid'].unique())}
and yz.paid_at > 0
and (yz.payment_fee > yzr.refund_amount or yzr.refund_amount is null)
and (yz.pkg_type = "体重管理" or yz.product_name = "【天津代谢专属】血糖试纸福袋（自提）")
'''
with engine.connect() as con:
    orders = pd.read_sql(sql, con)
orders['user_uuid'].nunique()
# 找出真正的开始日期，取所有目标商品的订单，找出最早日期
orders['rank'] = orders.groupby('user_uuid')['paid_at'].rank(method='dense')
orders['cgm'] = orders['sku_no'].str.contains('微泰') | orders['sku_name'].str.contains('微泰')
orders
# %%
chk = orders.groupby(['user_uuid'])['cgm'].mean().to_frame()
chk.query('cgm != 0 and cgm != 1')
# orders.query('')
# %%
tbl = tbl.join(orders.query('rank == 1').set_index(['user_uuid'])[['paid_at', 'cgm']], 
               on = 'uuid', how='left')
tbl['is_paid'] = tbl['paid_at'].notnull()
tbl['cgm'] = tbl['cgm'].fillna(False)
tbl
# %%
tbl['start_date'] = tbl.apply(lambda x: x['paid_at'].date() if x['is_paid']
                              else x['collect_date'].date(), axis=1)
tbl
# %% 
app_data = AppActive(pts=tbl['uuid'], start_time=dt.date(1, 1, 1),
end_time=dt.date.today())
pres = app_data.c_prescription_tag()
# pres = pres.query('sub_category == "glp1"')

print("GLP1药物：", pres.query('sub_category == "glp1"')['medicine_name'].unique(),
      "\n非GLP1药物：", pres.query('sub_category != "glp1"')['medicine_name'].unique())
# %%
sql = f'''select pres.*, med.category
from sanyi_care.c_prescription_tag as pres
left join sanyi_care.c_medicine_info_v2 as med
on pres.medicine_id = med.id
where patient_uuid in ('{"','".join(tbl['uuid'].unique())}')
-- and medicine_name like "%%肽%%"
'''
with engine.connect() as con:
    pres = pd.read_sql(sql, con)
# print(pres['medicine_name'].unique())
# print(pres['patient_uuid'].nunique())
# print(pres.shape)
pres = pres.join(tbl.set_index('uuid')[['start_date']], on='patient_uuid', how='left')
pres['date'] = pd.to_datetime(pres['date'])
pres['start_date'] = pd.to_datetime(pres['start_date'])
pres['pres_days'] = pres.eval('date - start_date').dt.days
# pres = pres.query('pres_days >= -30 and pres_days <= 30')
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
