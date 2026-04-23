import pandas as pd
from pathlib import Path

ROOT = Path.cwd()
DATA_DIR = ROOT / 'data'
START_DATE = pd.Timestamp('2014-01-31')
END_DATE = pd.Timestamp.today().normalize()

start_str = START_DATE.strftime('%d.%m.%Y')
end_str = END_DATE.strftime('%d.%m.%Y')
url = (
    'https://www.cbr.ru/hd_base/KeyRate/'
    f'?UniDbQuery.Posted=True&UniDbQuery.From={start_str}&UniDbQuery.To={end_str}'
)
tables = pd.read_html(url, decimal=',', thousands=' ')
table = tables[0]
date_col = table.columns[0]
rate_col = table.columns[1]
out = table[[date_col, rate_col]].copy()
out.columns = ['date', 'key_rate']
out['date'] = pd.to_datetime(out['date'], dayfirst=True, errors='coerce')
out['key_rate'] = pd.to_numeric(
    out['key_rate'].astype(str).str.replace(',', '.', regex=False),
    errors='coerce',
)
out = out.dropna(subset=['date', 'key_rate']).drop_duplicates('date').sort_values('date')
daily = out.set_index('date')['key_rate']
daily.to_csv('data/key_rate_daily.csv')