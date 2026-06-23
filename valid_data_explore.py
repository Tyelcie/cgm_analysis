# %%
import ast
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import bindparam, text
from connections import engine
# %%
# 所有含异常数据的记录，剔除测试设备
sql = '''select cgm.id, cgm.date, cgm.device_id, cp.valid_data, cgm.data_count, cgm.updated_at, cgm.created_at
from stats.t_cgm_params as cp
join sanyi_cgm.t_cgm as cgm
on cp.id = cgm.id
where exists (
select 1 from sanyi_care.u_device_usage as du
where cgm.device_id = du.device_id
)
-- and cp.valid_data < cgm.data_count
and cp.device_id not in ("WTSDK00080", "WTSDK00081", "WTSDK00082", "WTSDK00083")
'''
with engine.connect() as conn:
    df = pd.read_sql(sql, conn)
# %%
# 按设备号聚合
df_new = df.query('updated_at >= @dt.date.today()').copy()
df_old = df.query('updated_at < @dt.date.today()').copy()
df_dev_old = df_old.groupby(['device_id']).agg(
    valid_data = ('valid_data', 'sum'),
    data_count = ('data_count', 'sum'),
    last_updated_at = ('updated_at', 'max'),
    days = ('date', 'count')).reset_index()
df_dev_old['abnormal_rate'] = df_dev_old.eval('1 - valid_data/data_count')
df_dev_old = df_dev_old.sort_values('abnormal_rate', ascending = False)#.query('days >= 3')

df_dev_new = df_new.groupby(['device_id']).agg(
    valid_data = ('valid_data', 'sum'),
    data_count = ('data_count', 'sum'),
    last_updated_at = ('updated_at', 'max'),
    days = ('date', 'count')).reset_index()
df_dev_new['abnormal_rate'] = df_dev_new.eval('1 - valid_data/data_count')
df_dev_new = df_dev_new.sort_values('abnormal_rate', ascending = False)#.query('days >= 3')
# %%
# 总数据异常量
print('非当天更新的总数据量异常：', format(1 - np.divide(*df_dev_old[['valid_data', 'data_count']].sum()), '.2%'))
print('当天更新的总数据量异常', format(1 - np.divide(*df_dev_new[['valid_data', 'data_count']].sum()), '.2%'))
# %%
# (df_dev_new.rename({'device_id': '设备号',
#                         'valid_data': '>=1.1的数据量',
#                         'data_count': '回传的数据量',
#                         'last_updated_at': '最后更新时间',
#                         'days': f'佩戴天数（截至{dt.date.today()})',
#                         'abnormal_rate': '异常比例'},
#                                     axis = 1)
#     #   .to_excel('CGM疑似异常数据量探查.xlsx', index = False)
#       )
# %%
def data_to_plot(data, data_col = 'daily_data'):
    data_plot = (
        data.set_index('date')[data_col]
        .apply(pd.Series)
        .reset_index()
        .melt(id_vars='date', var_name='time', value_name='glucose')
        .dropna()
    )
    data_plot['date'] = data_plot['date'].astype(str)
    data_plot['time'] = pd.to_datetime(data_plot['time'], format='%H:%M:%S')
    return data_plot
# %%
def cgm_origin_line(data, save = False, ax = None, title = None, xlim = None, vline_time = None):
    if ax is None:
        plt.figure(figsize =[10, 5])
    ax = sns.lineplot(data=data, x='time', y='glucose', hue='date', ax=ax)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.axhline(y = 1.1, linestyle = '--', color = 'gray', label = "1.1")
    ax.axhline(y = 0, linestyle = '--', color = 'red', label = "0")
    if vline_time is not None:
        ax.axvline(x = vline_time, linestyle = '--', color = 'black', alpha = 0.7)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)  # 把图例放到图形外侧喵~
    sns.despine()
    # plt.title(f'device id: {dev_id}, valid_data: {valid_data: .0f}, data_count: {data_count: .0f}')
    ax.set_title(title or f'device id: {dev_id}')
    plt.tight_layout()
    if save:
        plt.savefig(f'CGM疑似异常数据量探查-{dev_id}.png', dpi = 300)
    return ax
# %%
def valid_data_compare_facet(data, id = None, device_id = None, save = False):
    if id is not None:
        data_chk = data.query('id == @id').copy()
    elif device_id is not None:
        data_chk = data.query('device_id == @device_id').copy()
    else:
        data_chk = data.copy()

    if data_chk.empty:
        print('没有找到可对比的数据，喵~')
        return None

    local_plot = data_to_plot(data_chk, 'daily_data')
    latest_plot = data_to_plot(data_chk, 'daily_data_new')
    # 两边用时间点并集确定横轴范围，并用本地最后时间点画竖向虚线喵~
    time_union = pd.Index(local_plot['time']).union(pd.Index(latest_plot['time'])).sort_values()
    xlim = (time_union.min(), time_union.max()) if len(time_union) else None
    local_last_time = local_plot['time'].max() if not local_plot.empty else None

    fig, axes = plt.subplots(1, 2, figsize=[16, 5], sharey=True)
    cgm_origin_line(local_plot, ax=axes[0], title='local JSON', xlim=xlim, vline_time=local_last_time)
    cgm_origin_line(latest_plot, ax=axes[1], title='remote latest', xlim=xlim, vline_time=local_last_time)
    title = f'data id: {id}' if id is not None else f'device id: {device_id}' if device_id is not None else 'valid_data变化记录'
    fig.suptitle(title)
    fig.tight_layout()
    if save:
        save_key = id or device_id or 'all'
        fig.savefig(f'CGM疑似异常数据量对比-{save_key}.png', dpi=300)
    # return fig, axes

def compare_daily_data_detail(old, new):
    if not isinstance(old, dict) or not isinstance(new, dict) or not old:
        return pd.Series({
            'local_last_key': None,
            'new_last_key': None,
            'key_added_before_local_end': [],
            'key_missing_before_local_end': [],
            'key_delay_after_local_end': 0,
            'value_changed_detail': [],
            'key_changed_cnt': 0,
            'value_changed_cnt': 0,
            'daily_data_changed_detail': old != new,
        })

    def round_bg(x):
        return round(float(x), 1) if pd.notna(x) else x

    local_last_key = max(old)
    new_last_key = max(new)
    old_before_end = {k: round_bg(v) for k, v in old.items() if k <= local_last_key}
    new_before_end = {k: round_bg(v) for k, v in new.items() if k <= local_last_key}
    new_after_end = {k: round_bg(v) for k, v in new.items() if k > local_last_key}
    old_keys = set(old_before_end)
    new_keys = set(new_before_end)
    new_after_key = set(new_after_end)
    key_added = sorted(new_keys - old_keys)
    key_missing = sorted(old_keys - new_keys)
    # key_delay = len(new_after_key)
    value_changed = [
        {
            'time': k,
            'local_value': old_before_end[k],
            'new_value': new_before_end[k],
        }
        for k in sorted(old_keys & new_keys)
        if old_before_end[k] != new_before_end[k]
    ]

    return pd.Series({
        'local_last_key': local_last_key,
        'new_last_key': new_last_key,
        'key_added_before_local_end': key_added,
        'key_missing_before_local_end': key_missing,
        'key_delay_after_local_end': len(new_after_key) if len(new_after_key) > 3 else 0,
        'value_changed_detail': value_changed,
        'key_changed_cnt': len(key_added) + len(key_missing) + len(new_after_key) if len(new_after_key) > 3 else 0,
        'value_changed_cnt': len(value_changed),
        'daily_data_changed_detail': bool(key_added or key_missing or value_changed or len(new_after_key) > 3),
    })
# %%
dev_id = "22222CBTGZ"
dev_id = "22222B6V7C"
dev_id = "22222BRDNE"
dev_id = "22222DZGDE"
dev_id = '2222286M4K' # 更新中,后续再观察valid_data变化
did = '6a32cadfec2fbcac7c7652c0' # 6月22日12:07观察cp表：data_count:1207, valid_data:803, updated_at: 2026-06-22 11:13:54
sql = f'''select device_id, date, daily_data, updated_at
from sanyi_cgm.t_cgm
-- where device_id = "{dev_id}"
where id = '{did}'
-- and updated_at < "{dt.date.today()}"
'''
with engine.connect() as con:
    df_cgm = pd.read_sql(sql, con)
df_cgm['daily_data'] = df_cgm['daily_data'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df_plt = data_to_plot(data = df_cgm)
cgm_origin_line(data = df_plt)
display(df.query('id==@did'))
# %%
# 查今天有异常的
# sql = '''select cgm.id, cgm.date, cgm.device_id, cgm.data_count, cgm.daily_data,
# cgm.created_at, cgm.updated_at, current_timestamp
# from sanyi_cgm.t_cgm as cgm
# where exists (
# select 1 from sanyi_care.u_device_usage as du
# where cgm.device_id = du.device_id
# )
# and date(cgm.updated_at) = current_date
# '''
# with engine.connect() as conn:
#     df_today = pd.read_sql(sql, conn)
# df_today['daily_data'] = df_today['daily_data'].apply(lambda x: ast.literal_eval(x)
#                                    if isinstance(x, str) else x)
# df_today['invalid_data'] = df_today['daily_data'].map(lambda x: sum(a < 1.1 for a in x.values())
#                                    if isinstance(x, dict) else 0)
# print(df_today.shape)
# df_today = df_today.query('invalid_data > 0')
# print(df_today.shape)
# # %%
# # 保存当天异常回传明细为本地 JSON，方便后续查看喵~
# json_path = f'CGM疑似异常数据量探查_原始数据_{dt.date.today()}.json'
# df_today.to_json(json_path, orient='records', force_ascii=False, indent=2, date_format='iso')
# print(f'当天异常回传明细已保存到：{json_path}，喵~')
# %%
# 后续检查
chk_date = dt.date(2026, 6, 22)
df_local = pd.read_json(f'CGM疑似异常数据量探查_原始数据_{chk_date}.json', orient='records')
chk_date_2 = dt.date(2026, 6, 21)
df_remote = pd.read_json(f'CGM疑似异常数据量探查_原始数据_{chk_date_2}.json', orient='records')
df_remote.rename({'invalid_data': 'invalid_data_new'}, axis=1,inplace=True)
# %%
ids = df_local['id'].dropna().drop_duplicates().tolist()
sql = text('''
select id, data_count, daily_data, updated_at
from sanyi_cgm.t_cgm
where id in :ids
''').bindparams(bindparam('ids', expanding=True))
with engine.connect() as conn:
    df_remote = pd.read_sql(sql, conn, params={'ids': ids})
df_remote['daily_data'] = df_remote['daily_data'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df_remote['invalid_data_new'] = df_remote['daily_data'].map(
    lambda x: sum(a < 1.1 for a in x.values()) if isinstance(x, dict) else 0
)
# %%
df_valid_compare = df_local[['id', 'date', 'device_id', 'data_count',
    'invalid_data', 'daily_data', 'updated_at']].merge(
    (df_remote[['id', 'data_count', 'invalid_data_new', 'daily_data', 'updated_at']]
      .rename(columns={'updated_at': 'updated_at_new',
                       'daily_data': 'daily_data_new',
                       'data_count': 'data_count_new'})),
    on='id',
    how='left',
)
df_valid_compare['invalid_data_delta'] = df_valid_compare['invalid_data_new'] - df_valid_compare['invalid_data']
df_valid_changed = df_valid_compare.query('invalid_data_delta != 0')
print(f'本地记录 {len(df_local)} 条，回查 {len(df_remote)} 条，valid_data 变化 {len(df_valid_changed)} 条，喵~')
df_valid_changed_sorted = df_valid_changed.sort_values('invalid_data_delta', ascending = False)
# did = df_valid_changed_sorted['id'].iloc[0] if not df_valid_changed_sorted.empty else None
# if did is not None:
#     valid_data_compare_facet(data = df_valid_compare, id = did, save=False)
# else:
#     print('没有 invalid_data 变化的目标 id，喵~')
# %
df_valid_compare = pd.concat(
    [
        df_valid_compare,
        df_valid_compare.apply(
            lambda x: compare_daily_data_detail(x['daily_data'], x['daily_data_new']),
            axis=1,
        ),
    ],
    axis=1,
)
df_daily_data_changed = df_valid_compare.query('daily_data_changed_detail')
print(
    f'daily_data 截止本地最后时间点前不一致 {len(df_daily_data_changed)} 条，'
    f'key 不一致 {sum(df_valid_compare["key_changed_cnt"] > 0)} 条，'
    f'value 不一致 {sum(df_valid_compare["value_changed_cnt"] > 0)} 条，喵~'
)
# %%
display(df_valid_compare.sort_values('invalid_data_delta', ascending = False).query('key_changed_cnt>0')[['id', 
# 'key_added_before_local_end', 'key_missing_before_local_end',
'data_count', 'data_count_new',
'invalid_data', 'invalid_data_new',
'key_delay_after_local_end',
'updated_at','updated_at_new']])
# %%
# 18日只有一半数据，19日传完，但20日仍有更新，值不变，不知道什么变了
did = "6a3346d3342ace99516bc6c9"
did = '6a380a94ec2fbcac7cc42fcb'
# 以下三个valid_data有变化的，都是在尾部新增数据上的变化
did = "6a380aab342ace99518f8e25"
did = "6a38252c342ace99519030b0"
did = "6a380aa6ec2fbcac7cc4319c"
valid_data_compare_facet(data = df_valid_compare, id = did)
# %%
# 理论上一天当中最后一段时间（平滑算法的窗口期内）的值会在次日观察到值的变化
display((df_valid_compare
.query('local_last_key > "23:45:00" and date == @dt.date(2026, 6, 22)')
[['id', 'value_changed_cnt', 'key_changed_cnt', 'local_last_key',
'new_last_key', 'data_count', 'data_count_new']])
)
# valid_data_compare_facet(data = df_valid_compare, id = "6a36e891342ace9951872fc9")
# %%
def daily_data_by_key(data, id, before_local_end = True, only_changed = False):
    data_chk = data.query('id == @id')
    if data_chk.empty:
        print('没有找到目标 id 的数据，喵~')
        return pd.DataFrame()

    old = data_chk['daily_data'].iloc[0]
    new = data_chk['daily_data_new'].iloc[0]
    if not isinstance(old, dict) or not isinstance(new, dict):
        print('目标 id 的 daily_data 不是可展开的字典，喵~')
        return pd.DataFrame()

    local_last_key = max(old) if old else None
    keys = sorted(set(old) | set(new))
    if before_local_end and local_last_key is not None:
        keys = [k for k in keys if k <= local_last_key]

    def round_bg(x):
        return round(float(x), 1) if pd.notna(x) else x

    data_detail = pd.DataFrame({
        'time': keys,
        'local_value': [round_bg(old.get(k)) for k in keys],
        'new_value': [round_bg(new.get(k)) for k in keys],
    })
    data_detail['key_status'] = np.select(
        [
            data_detail['local_value'].isna(),
            data_detail['new_value'].isna(),
        ],
        [
            '仅回查版有key',
            '仅本地版有key',
        ],
        default='两边都有key',
    )
    data_detail['value_changed'] = (
        data_detail['local_value'].notna()
        & data_detail['new_value'].notna()
        & (data_detail['local_value'] != data_detail['new_value'])
    )
    data_detail['value_delta'] = data_detail['new_value'] - data_detail['local_value']
    if only_changed:
        data_detail = data_detail.query('key_status != "两边都有key" or value_changed')
    return data_detail

df_daily_data_by_key = (
    daily_data_by_key(df_valid_compare, id=did, only_changed=False)
    if did is not None else pd.DataFrame()
)
df_daily_data_by_key
#%%
# %%
if did is not None:
    print(
        max(list(df_valid_compare.set_index('id').loc[did, 'daily_data'].keys())),
        max(list(df_valid_compare.set_index('id').loc[did, 'daily_data_new'].keys())),
    )
else:
    print('没有可查看最后 key 的目标 id，喵~')
# %%
