import pandas as pd
import numpy as np
from pathlib import Path

# Directories
RAW_DIR = Path(__file__).parent.parent / "CREMP_CSV_files"
PROC_DIR = Path(__file__).parent / "processed_data"
PROC_DIR.mkdir(exist_ok=True, parents=True)

# Load CSV or Excel and standardize column names
def load_csv_or_excel(name):
    csv_path = RAW_DIR / f"{name}.csv"
    xls_path = RAW_DIR / f"{name}.xlsx"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif xls_path.exists():
        df = pd.read_excel(xls_path)
    else:
        raise FileNotFoundError(f"{name} not found in {RAW_DIR}")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

# Save DataFrame to CSV
def save_df(df, fname):
    df.to_csv(PROC_DIR / fname, index=False)

# Filter outliers (cleaned_final)
def filter_outliers(df, cols, low_q=0.01, high_q=0.99):
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            lo, hi = df2[c].quantile([low_q, high_q])
            df2 = df2[df2[c].between(lo, hi)]
    return df2

# Enhance data types and parse dates
def enhance_data_types_and_dates(df_st, df_temp, df_scor, df_octo, df_cover):
    # Stations
    cat_cols = ['region', 'site_code', 'habitat', 'subregion', 'site_name']
    for col in cat_cols:
        if col in df_st.columns:
            df_st[col] = df_st[col].astype('category')
    num_cols = ['siteid', 'stationid', 'first_year_surveyed', 'length_m', 'depth_ft']
    for col in num_cols:
        if col in df_st.columns:
            df_st[col] = pd.to_numeric(df_st[col], errors='coerce')
    # Temperature
    for col in ['siteid', 'year', 'month', 'day']:
        if col in df_temp.columns:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
    if 'time' in df_temp.columns:
        df_temp['time'] = pd.to_numeric(df_temp['time'], errors='coerce')
    if set(['year', 'month', 'day']).issubset(df_temp.columns):
        df_temp['date'] = pd.to_datetime(df_temp[['year','month','day']], errors='coerce')
    # SCOR, OCTO, Cover
    for df_dict in [df_scor, df_octo, df_cover]:
        for key, df in df_dict.items():
            for dt in ['date', 'firstofdate']:
                if dt in df.columns:
                    df[dt] = pd.to_datetime(df[dt], errors='coerce')
    print('Data types and date parsing complete.')

# Normalize text columns
def normalize_text_columns(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

def normalize_all_text_columns(df_st, df_temp, df_scor, df_octo, df_cover):
    text_cols = ['site_name', 'region', 'habitat', 'subregion', 'site_code']
    normalize_text_columns(df_st, text_cols)
    normalize_text_columns(df_temp, ['site_name'])
    for df_dict in [df_scor, df_octo, df_cover]:
        for key, df in df_dict.items():
            normalize_text_columns(df, text_cols)
    print('Text normalization complete.')

if __name__ == '__main__':
    # 1. Stations
    st_csv = RAW_DIR / 'CREMP_Stations_2023.csv'
    st_xls = RAW_DIR / 'CREMP_Stations_2023.xlsx'
    df_st_csv = pd.read_csv(st_csv)
    try:
        df_st_xls = pd.read_excel(st_xls)
    except FileNotFoundError:
        df_st_xls = None
    if df_st_xls is not None:
        df_stations = pd.concat([df_st_csv, df_st_xls]).drop_duplicates().reset_index(drop=True)
    else:
        df_stations = df_st_csv
    df_stations.columns = df_stations.columns.str.lower().str.replace(' ', '_')
    save_df(df_stations, 'stations_with_stations.csv')
    save_df(df_stations, 'stations_cleaned_final.csv')

    # 2. Temperature
    df_temp = load_csv_or_excel('CREMP_Temperatures_2023')
    save_df(df_temp, 'temperature_with_stations.csv')
    df_temp_clean = df_temp.groupby('siteid').ffill()
    save_df(df_temp_clean, 'temperature_cleaned_final.csv')

    # 3. SCOR datasets
    scor_files = {
        'raw_data': 'CREMP_SCOR_RawData_2023',
        'lta_summary': 'CREMP_SCOR_Summaries_2023_LTA',
        'density_summary': 'CREMP_SCOR_Summaries_2023_Density',
        'counts_summary': 'CREMP_SCOR_Summaries_2023_Counts',
        'condition_counts': 'CREMP_SCOR_Summaries_2023_ConditionCounts'
    }
    df_scor = {}
    for key, name in scor_files.items():
        df = load_csv_or_excel(name)
        df_scor[key] = df
        df_merge = df.merge(df_stations, on='stationid', how='left') if 'stationid' in df.columns else df
        save_df(df_merge, f'scor_{key}_with_stations.csv')
        if key in ['lta_summary', 'density_summary']:
            metric = 'scleractinia' if key == 'lta_summary' else 'density'
            df_clean = filter_outliers(df_merge, [metric])
        else:
            df_clean = df_merge
        save_df(df_clean, f'scor_{key}_cleaned_final.csv')

    # 4. OCTO datasets
    octo_files = {
        'raw_data': 'CREMP_OCTO_RawData_2023',
        'density_summary': 'CREMP_OCTO_Summaries_2023_Density',
        'height_summary': 'CREMP_OCTO_Summaries_2023_MeanHeight'
    }
    df_octo = {}
    for key, name in octo_files.items():
        df = load_csv_or_excel(name)
        df_octo[key] = df
        df_merge = df.merge(df_stations, on='stationid', how='left') if 'stationid' in df.columns else df
        save_df(df_merge, f'octo_{key}_with_stations.csv')
        if key == 'density_summary':
            df_clean = filter_outliers(df_merge, ['total_octocorals'])
        else:
            df_clean = df_merge
        save_df(df_clean, f'octo_{key}_cleaned_final.csv')

    # 5. Percent cover datasets
    cover_files = {
        'stony_species': 'CREMP_Pcover_2023_StonyCoralSpecies',
        'taxa_groups': 'CREMP_Pcover_2023_TaxaGroups'
    }
    df_cover = {}
    for key, name in cover_files.items():
        df = load_csv_or_excel(name)
        df_cover[key] = df
        df_merge = df.merge(df_stations, on='stationid', how='left') if 'stationid' in df.columns else df
        save_df(df_merge, f'pcover_{key}_with_stations.csv')
        numeric_cols = df_merge.select_dtypes(include=[np.number]).columns.tolist()
        df_clean = filter_outliers(df_merge, numeric_cols)
        save_df(df_clean, f'pcover_{key}_cleaned_final.csv')

    # Enhance and normalize
    enhance_data_types_and_dates(df_stations, df_temp_clean, df_scor, df_octo, df_cover)
    normalize_all_text_columns(df_stations, df_temp_clean, df_scor, df_octo, df_cover)
    print('Preprocessing complete. Files in:', PROC_DIR)

# Convenience loader for EDA
def load_processed(name, version='with_stations'):
    return pd.read_csv(PROC_DIR / f"{name}_{version}.csv")
