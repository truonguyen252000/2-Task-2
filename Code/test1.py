import streamlit as st
st.set_page_config(page_title="Power Quality Analysis", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>

/* Global styles */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: "Sitka Display Semibold", "Sitka Display", "Segoe UI", Roboto, sans-serif;
    text-align: center;
    font-size: 18px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    padding: 2rem 1rem;
}
section[data-testid="stSidebar"] * {
    color: #ecf0f1 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #3498db !important;
    font-weight: 700;
    margin-bottom: 1rem;
}

/* Main content area */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 100%;
}

/* Title styling */
.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

/* Card styling */
.stat-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.stat-number {
    font-size: 2.5rem;
    font-weight: 700;
    color: #3498db;
}
.stat-label {
    font-size: 1rem;
    color: #7f8c8d;
    margin-top: 0.5rem;
}

/* Button styling */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-size: 1rem;
    box-shadow: 0px 4px 10px rgba(52, 152, 219, 0.3);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #2980b9, #3498db);
    transform: translateY(-2px);
    box-shadow: 0px 6px 14px rgba(52, 152, 219, 0.4);
}

/* Input styling */
.stNumberInput input,
.stTextInput input {
    background-color: #34495e;
    border: 2px solid #4a5f7f;
    border-radius: 6px;
    color: white !important;
    padding: 0.5rem;
}
.stSelectbox select {
    background-color: #34495e;
    border: 2px solid #4a5f7f;
    border-radius: 6px;
    color: white;
}

/* Checkbox */
.stCheckbox {
    color: white !important;
}

/* Expander styling */
.streamlit-expanderHeader {
    background: linear-gradient(90deg, #ecf0f1 0%, #bdc3c7 100%);
    border-radius: 8px;
    font-weight: 600;
    color: #2c3e50 !important;
}

/* DataFrame */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* Info boxes */
.stAlert {
    border-radius: 8px;
    border-left: 4px solid #3498db;
}

/* Divider */
hr {
    margin: 2rem 0;
    border: none;
    border-top: 2px solid #bdc3c7;
}

/* ⬇️ Download section */
.download-section {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin: 1rem 0;
}

/* 📁 File uploader fix */
/* Remove white background of file uploader dropzone */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background-color: transparent !important;  /* loại bỏ nền trắng */
    border: 2px dashed rgba(52, 152, 219, 0.6) !important; /* hoặc bỏ dòng này nếu không muốn viền */
    border-radius: 10px !important;
    color: #ecf0f1 !important;
    transition: all 0.3s ease;
}

/*  Hover effect để người dùng biết có thể kéo thả */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
    border-color: #3498db !important;
    background-color: rgba(52, 152, 219, 0.1) !important; /* tùy chọn: nền mờ nhẹ khi hover */
}

/* Màu chữ mô tả trong vùng dropzone */
section[data-testid="stSidebar"] [data-testid="stFileUploaderInstructions"] {
    color: rgba(236, 240, 241, 0.8) !important;
}

/* Nút Browse */
section[data-testid="stSidebar"] [data-testid="stFileUploaderBrowseButton"] {
    background: linear-gradient(90deg, #3498db 0%, #2980b9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    box-shadow: 0px 3px 6px rgba(52, 152, 219, 0.3);
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderBrowseButton"]:hover {
    background: linear-gradient(90deg, #2980b9, #3498db) !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    color: #3498db !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] label div {
    color: #2c3e50 !important;
    font-weight: 800;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #2980b9 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, #3498db 0%, #2980b9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    box-shadow: 0px 3px 6px rgba(52, 152, 219, 0.3);
}
[data-testid="stFileUploader"] small {
    color: #2c3e50 !important;
    font-weight: 500 !important;
}

/* ✨ Improve text clarity */
body, p, label, span, div {
    letter-spacing: 0.3px;
    line-height: 1.5;
}


/* Tab styling - make text larger and bold */
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: #2c3e50 !important;
}

.stTabs [data-baseweb="tab-list"] button {
    padding: 1rem 1.5rem !important;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
    color: #3498db !important;
}          
</style>
""", unsafe_allow_html=True)


import pandas as pd
import numpy as np
import os
import glob
import io
import tempfile
import matplotlib.pyplot as plt
import warnings
from openpyxl.styles import Alignment
from reportlab.lib.pagesizes import A4, portrait
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")


def clean_data(data, time_col_name="Time [UTC]"):
    if time_col_name not in data.columns:
        return data, "No time column found"
    
    df = data.copy()
    df[time_col_name] = pd.to_datetime(df[time_col_name], errors="coerce")
    df = df.dropna(subset=[time_col_name]).sort_values(time_col_name).reset_index(drop=True)
    
    if df.empty:
        return df, "Empty after cleaning"
    
    start, end = df[time_col_name].iloc[0], df[time_col_name].iloc[-1]
    start_is_midnight = start.time() == pd.Timestamp("00:00").time()
    end_is_2350 = end.time().hour == 23 and end.time().minute == 50
    
    log_msg = []
    
    if start_is_midnight and end_is_2350:
        df_cut = df.copy()
        log_msg.append("✅ Data already starts at 00:00 and ends at 23:50")
    else:
        start_date = (start + pd.Timedelta(days=1)).normalize()
        end_date = (end.normalize() - pd.Timedelta(minutes=10))
        df_cut = df[(df[time_col_name] >= start_date) & (df[time_col_name] <= end_date)]
        if df_cut.empty:
            return df_cut, "Empty after cutting"
        log_msg.append(f"📅 Cut data: {start_date.date()} to {end_date.date()}, {len(df_cut)} rows")
    
    df_cut = df_cut.sort_values(time_col_name).reset_index(drop=True)
    
    df_cut["date"] = df_cut[time_col_name].dt.date
    
    all_dates = sorted(df_cut["date"].unique())
    days_to_remove = []
    days_report = []
    
    for date in all_dates:
        expected_times = pd.date_range(
            start=pd.Timestamp(date),
            end=pd.Timestamp(date) + pd.Timedelta(hours=23, minutes=50),
            freq="10min"
        )
        
        actual_times = set(df_cut[df_cut["date"] == date][time_col_name])
        
        missing_count = len(set(expected_times) - actual_times)
        
        days_report.append(f"{date}: {missing_count}/144 missing")
        
        if missing_count > 15:
            days_to_remove.append(date)
    
    log_msg.append("\n📊 Daily Missing Report:")
    for report in days_report:
        log_msg.append(f"   {report}")
    
    if days_to_remove:
        df_cut = df_cut[~df_cut["date"].isin(days_to_remove)].copy()
        log_msg.append(f"\n🗑️ Removed {len(days_to_remove)} days with >15 missing timestamps")
        
        if df_cut.empty:
            return df_cut, "\n".join(log_msg) + "\n⚠️ No data remaining after filtering"
    else:
        log_msg.append("\n✅ All days have ≤15 missing timestamps")
    
    # Tính số ngày còn lại
    remaining_dates = sorted(df_cut["date"].unique())
    log_msg.append(f"\n📈 {len(remaining_dates)} valid days remaining")
    
    df_cut = df_cut.drop(columns=["date"]).reset_index(drop=True)
    
    if len(remaining_dates) > 0:
        all_times = []
        for date in remaining_dates:
            day_times = pd.date_range(
                start=pd.Timestamp(date),
                end=pd.Timestamp(date) + pd.Timedelta(hours=23, minutes=50),
                freq="10min"
            )
            all_times.extend(day_times)
        
        all_times = pd.DatetimeIndex(all_times)
        missing_times = all_times.difference(df_cut[time_col_name])
        
        if len(missing_times) > 0:
            log_msg.append(f"⚠️ Filling {len(missing_times)} missing timestamps in valid days")
            missing_df = pd.DataFrame({time_col_name: missing_times})
            df_full = pd.concat([df_cut, missing_df], ignore_index=True)
            df_full = df_full.sort_values(time_col_name).reset_index(drop=True)
            df_full = df_full.ffill().bfill()
        else:
            log_msg.append("✅ No missing timestamps in valid days")
            df_full = df_cut.copy()
        
        df_full["date"] = df_full[time_col_name].dt.date
        samples_per_day = df_full.groupby("date").size()
        days_incomplete = samples_per_day[samples_per_day < 144]
        if len(days_incomplete) > 0:
            log_msg.append(f"⚠️ {len(days_incomplete)} days still have <144 samples")
        else:
            log_msg.append(f"✅ All {len(samples_per_day)} days have exactly 144 samples")
        
        df_full = df_full.drop(columns=["date"])
    else:
        df_full = df_cut.copy()
    
    return df_full, "\n".join(log_msg)

DEFAULT_OUTPUT_FOLDER = os.path.join(os.path.expanduser("~"), "PowerQuality_Output")

os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)
cols_pst = ["Pst1(Avg) []", "Pst2(Avg) []", "Pst3(Avg) []"]
cols_thdu = ["THD U1(AvgOn) [%]", "THD U2(AvgOn) [%]", "THD U3(AvgOn) [%]"]
cols_tddi = ["TDD I1(AvgOn) [%]", "TDD I2(AvgOn) [%]", "TDD I3(AvgOn) [%]"]
cols_plt = ["Plt1(Avg) []", "Plt2(Avg) []", "Plt3(Avg) []"]
cols_uneg = ["u-(Avg) [%]"]

thresholds = {
    "Pst1(Avg) []": 0.8, "Pst2(Avg) []": 0.8, "Pst3(Avg) []": 0.8,
    "THD U1(AvgOn) [%]": 3, "THD U2(AvgOn) [%]": 3, "THD U3(AvgOn) [%]": 3,
    "TDD I1(AvgOn) [%]": 3, "TDD I2(AvgOn) [%]": 3, "TDD I3(AvgOn) [%]": 3,
    "Plt1(Avg) []": 0.6, "Plt2(Avg) []": 0.6, "Plt3(Avg) []": 0.6,
    "u-(Avg) [%]": 3,
}

def update_thresholds(pst_val, plt_val, thdu_val, tddi_val, u_neg_val):
    global thresholds
    thresholds = {
        "Pst1(Avg) []": pst_val, "Pst2(Avg) []": pst_val, "Pst3(Avg) []": pst_val,
        "THD U1(AvgOn) [%]": thdu_val, "THD U2(AvgOn) [%]": thdu_val, "THD U3(AvgOn) [%]": thdu_val,
        "TDD I1(AvgOn) [%]": tddi_val, "TDD I2(AvgOn) [%]": tddi_val, "TDD I3(AvgOn) [%]": tddi_val,
        "Plt1(Avg) []": plt_val, "Plt2(Avg) []": plt_val, "Plt3(Avg) []": plt_val,
        "u-(Avg) [%]": u_neg_val,
    }

try:
    pdfmetrics.registerFont(TTFont('Times-Roman', 'times.ttf'))
except Exception:
    pass

def make_table(data, cols):
    df = data[cols]
    total = df.count()
    passed = pd.Series({c: int(((df[c] <= thresholds[c]) & df[c].notna()).sum()) for c in cols})
    failed = total - passed
    perc = (passed / total.replace(0, np.nan) * 100).round(2).fillna(0)
    status_per_col = perc.apply(lambda x: "PASS" if x >= 95 else "FAIL")
    group_status = "PASS" if (status_per_col == "PASS").all() else "FAIL"
    df_result = pd.DataFrame([
        total.astype(int),
        passed.astype(int),
        failed.astype(int),
        perc,
        status_per_col
    ], index=[
        "No. of samples",
        "No. of pass samples",
        "No. of fail samples",
        "Percentage of pass samples [%]",
        "Status"
    ])
    df_result.loc["Group Status"] = [group_status] * len(df_result.columns)
    return df_result

def make_voltage_harmonics_detail(data, threshold=1.5):
    orders = range(2, 41)
    phases = [("Phase A", "U1"), ("Phase B", "U2"), ("Phase C", "U3")]
    rows = []
    for h in orders:
        row = [h]
        above = []; perc_below = []; status = []
        for pha_name, prefix in phases:
            col = f"{prefix} h{h}(Avg) [%]"
            if col in data.columns:
                values = pd.to_numeric(data[col], errors="coerce")
                total = int(values.count())
                above_val = int((values > threshold).sum())
                below_val = int((values <= threshold).sum())
                perc = round(below_val / total * 100, 2) if total else 0.0
                stat = "PASS" if perc >= 95 else "FAIL"
            else:
                above_val = ""; perc = ""; stat = ""
            above.append(above_val); perc_below.append(perc); status.append(stat)
        row.extend(above); row.extend(perc_below); row.extend(status)
        rows.append(row)
    columns = [
        ("","Harmonic Order"),
        ("Total number of voltage harmonic samples > 1.5%", "Phase A"),
        ("Total number of voltage harmonic samples > 1.5%", "Phase B"),
        ("Total number of voltage harmonic samples > 1.5%", "Phase C"),
        ("Percentage of voltage harmonic samples ≤ 1.5% over total measured samples", "Phase A"),
        ("Percentage of voltage harmonic samples ≤ 1.5% over total measured samples", "Phase B"),
        ("Percentage of voltage harmonic samples ≤ 1.5% over total measured samples", "Phase C"),
        ("Assessment", "Phase A"),
        ("Assessment", "Phase B"),
        ("Assessment", "Phase C"),
    ]
    df_detail = pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(columns))
    return df_detail

def make_current_harmonics_detail(data, threshold=2):
    orders = range(2, 41)
    phases = [("Phase A", "I1"), ("Phase B", "I2"), ("Phase C", "I3")]
    rows = []
    for h in orders:
        row = [h]
        above = []; perc_below = []; status = []
        for pha_name, prefix in phases:
            col = f"{prefix} h{h}(Avg) [%]"
            if col in data.columns:
                values = pd.to_numeric(data[col], errors="coerce")
                total = int(values.count())
                above_val = int((values > threshold).sum())
                below_val = int((values <= threshold).sum())
                perc = round(below_val / total * 100, 2) if total else 0.0
                stat = "PASS" if perc >= 95 else "FAIL"
            else:
                above_val = ""; perc = ""; stat = ""
            above.append(above_val); perc_below.append(perc); status.append(stat)
        row.extend(above); row.extend(perc_below); row.extend(status)
        rows.append(row)
    columns = [
        ("","Harmonic Order"),
        ("Total number of current harmonic samples >2%", "Phase A"),
        ("Total number of current harmonic samples >2%", "Phase B"),
        ("Total number of current harmonic samples >2%", "Phase C"),
        ("Percentage of current harmonic samples ≤ 2% over total measured samples", "Phase A"),
        ("Percentage of current harmonic samples ≤ 2% over total measured samples", "Phase B"),
        ("Percentage of current harmonic samples ≤ 2% over total measured samples", "Phase C"),
        ("Assessment", "Phase A"),
        ("Assessment", "Phase B"),
        ("Assessment", "Phase C"),
    ]
    df_detail_cr = pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(columns))
    return df_detail_cr

def make_voltage_harmonics_by_pdm(data, bins, labels):
    harmonic_orders = range(2, 41)
    phases = [("Phase A", "U1"), ("Phase B", "U2"), ("Phase C", "U3")]
    thd_cols = ["THD U1(AvgOn) [%]", "THD U2(AvgOn) [%]", "THD U3(AvgOn) [%]"]
    rows_vh = []
    for (lower, upper), label in zip(bins, labels):
        if lower <= 0:
            mask = (data["Ptot+(Avg) [W]"] > 0) & (data["Ptot+(Avg) [W]"] <= upper)
        else:
            mask = (data["Ptot+(Avg) [W]"] > lower) & (data["Ptot+(Avg) [W]"] <= upper)
        df_bin = data.loc[mask]
        count = int(mask.sum())
        for h in harmonic_orders:
            row = [label, h]
            for pha_name, prefix in phases:
                col = f"{prefix} h{h}(Avg) [%]"
                if col in data.columns and count > 0:
                    mx = df_bin[col].max(); avg = df_bin[col].mean(); mn = df_bin[col].min()
                else:
                    mx = np.nan; avg = np.nan; mn = np.nan
                row.extend([mx, avg, mn])
            rows_vh.append(row)
        
        thd_row = [label, "THD"]
        for thd_col in thd_cols:
            if thd_col in data.columns and count > 0:
                mx = df_bin[thd_col].max(); avg = df_bin[thd_col].mean(); mn = df_bin[thd_col].min()
            else:
                mx = np.nan; avg = np.nan; mn = np.nan
            thd_row.extend([mx, avg, mn])
        rows_vh.append(thd_row)
        
    columns_vh = pd.MultiIndex.from_tuples([
        ("", "%Pdm"),
        ("", "Harmonic Order"),
        ("Phase A", "Max"), ("Phase A", "AVG"), ("Phase A", "Min"),
        ("Phase B", "Max"), ("Phase B", "AVG"), ("Phase B", "Min"),
        ("Phase C", "Max"), ("Phase C", "AVG"), ("Phase C", "Min"),
    ])
    df_vh = pd.DataFrame(rows_vh, columns=columns_vh)
    return df_vh.round(3)

def make_current_harmonics_by_pdm(data, bins, labels):
    harmonic_orders = range(2, 41)
    phases = [("Phase A", "I1"), ("Phase B", "I2"), ("Phase C", "I3")]
    tdd_cols = ["TDD I1(AvgOn) [%]", "TDD I2(AvgOn) [%]", "TDD I3(AvgOn) [%]"]
    rows_ch = []
    for (lower, upper), label in zip(bins, labels):
        if lower <= 0:
            mask = (data["Ptot+(Avg) [W]"] > 0) & (data["Ptot+(Avg) [W]"] <= upper)
        else:
            mask = (data["Ptot+(Avg) [W]"] > lower) & (data["Ptot+(Avg) [W]"] <= upper)
        df_bin = data.loc[mask]
        count = int(mask.sum())
        for h in harmonic_orders:
            row = [label, h]
            for pha_name, prefix in phases:
                col = f"{prefix} h{h}(Avg) [%]"
                if col in data.columns and count > 0:
                    mx = df_bin[col].max(); avg = df_bin[col].mean(); mn = df_bin[col].min()
                else:
                    mx = np.nan; avg = np.nan; mn = np.nan
                row.extend([mx, avg, mn])
            rows_ch.append(row)
        
        tdd_row = [label, "TDDi"]
        for tdd_col in tdd_cols:
            if tdd_col in data.columns and count > 0:
                mx = df_bin[tdd_col].max(); avg = df_bin[tdd_col].mean(); mn = df_bin[tdd_col].min()
            else:
                mx = np.nan; avg = np.nan; mn = np.nan
            tdd_row.extend([mx, avg, mn])
        rows_ch.append(tdd_row)
        
    columns_ch = pd.MultiIndex.from_tuples([
        ("", "%Pdm"),
        ("", "Harmonic Order"),
        ("Phase A", "Max"), ("Phase A", "AVG"), ("Phase A", "Min"),
        ("Phase B", "Max"), ("Phase B", "AVG"), ("Phase B", "Min"),
        ("Phase C", "Max"), ("Phase C", "AVG"), ("Phase C", "Min"),
    ])
    df_ch = pd.DataFrame(rows_ch, columns=columns_ch)
    return df_ch.round(3)

def make_daily_pdm_distribution(data, Pdm_max_val):
    if "Time [UTC]" not in data.columns or "Ptot+(Avg) [W]" not in data.columns:
        return pd.DataFrame(), "", {}
    
    data_copy = data.copy()
    data_copy["Date"] = pd.to_datetime(data_copy["Time [UTC]"], errors='coerce').dt.date
    data_copy = data_copy.dropna(subset=["Date"])
    
    if len(data_copy) == 0:
        return pd.DataFrame(), "", {}
    
    data_copy["%Pdm"] = (data_copy["Ptot+(Avg) [W]"] / Pdm_max_val * 100)
    pdm_ranges = [(i*10, (i+1)*10) for i in range(10)]
    pdm_labels = [f"{i*10}%" for i in range(1, 11)]
    dates = sorted(data_copy["Date"].unique())
    
    # Phân loại ngày đạt yêu cầu và không đạt
    valid_dates = []
    invalid_dates = []
    valid_dates_samples = {}  # Lưu số mẫu của mỗi ngày đạt yêu cầu
    
    for date in dates:
        date_data = data_copy[data_copy["Date"] == date]
        if (date_data["%Pdm"] >= 50).any():
            valid_dates.append(date)
            valid_dates_samples[date] = len(date_data)  # Tổng số mẫu của ngày đó
        else:
            invalid_dates.append(date)
    
    rows = []
    total_counts = [0] * len(pdm_ranges)
    
    for idx, date in enumerate(dates, 1):
        date_data = data_copy[data_copy["Date"] == date]
        row = [idx, date.strftime("%d/%m/%Y")]
        for i, (lower, upper) in enumerate(pdm_ranges):
            if i == 0:
                count = len(date_data[(date_data["%Pdm"] > 0) & (date_data["%Pdm"] <= upper)])
            else:
                count = len(date_data[(date_data["%Pdm"] > lower) & (date_data["%Pdm"] <= upper)])
            row.append(count)
            total_counts[i] += count
        
        if date in valid_dates:
            row.append(f"✓ Valid ({len(date_data)} samples)")
        else:
            row.append(f"✗ Invalid ({len(date_data)} samples)")
        
        rows.append(row)
    
    # Thêm dòng Total
    total_row = ["", "TOTAL"] + total_counts + [""]
    rows.append(total_row)
    
    columns = ["Index", "Date"] + pdm_labels + ["Status"]
    df_distribution = pd.DataFrame(rows, columns=columns)
    
    total_samples = len(data)
    valid_samples = len(data[data["Ptot+(Avg) [W]"] > 0])
    
    # Tính tổng số mẫu có Pdm >= 50%
    samples_above_50 = sum(total_counts[4:])
    
    # Tính tổng số mẫu của các ngày đạt yêu cầu
    total_valid_days_samples = sum(valid_dates_samples.values())
    
    total_days = len(dates)
    valid_days_count = len(valid_dates)
    invalid_days_count = len(invalid_dates)
    
    summary_text = f"""**Recorded Power Statistics:**
- Total measurement period: **{total_days} days**
- **Valid days (≥ 50% Pdm): {valid_days_count} days** ✅
- Invalid days (< 50% Pdm): {invalid_days_count} days ❌
- **Total samples from valid days: {total_valid_days_samples:,} samples**
- Total recorded samples: {valid_samples:,}/{total_samples:,} samples with power > 0 W (Ptot > 0 W)
- Total samples with P ≥ 50% Pdm: **{samples_above_50:,} samples** 

**Note:** Only data from valid days are used for further analysis."""
    
    # Thông tin chi tiết về các ngày đạt yêu cầu
    detailed_info = {
        'total_days': total_days,
        'valid_days_count': valid_days_count,
        'invalid_days_count': invalid_days_count,
        'total_valid_days_samples': total_valid_days_samples,
        'valid_dates_samples': valid_dates_samples
    }
    
    return df_distribution, summary_text, detailed_info

def plot_and_save(data, file_name, out_folder):
    groups = {"Pst": cols_pst, "THDu (%)": cols_thdu, "TDDi (%)": cols_tddi, "Plt": cols_plt}
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, (title, cols) in zip(axes, groups.items()):
        for col in cols:
            if col in data.columns:
                ax.plot(pd.to_numeric(data[col], errors="coerce"), label=col)
        ax.set_ylabel(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True)
    axes[-1].set_xlabel("Index (sample)", fontsize=10)
    plt.tight_layout()
    fig_path = os.path.join(out_folder, f"Output_{file_name}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path

def save_table_to_excel(writer, df, group_name):
    df_copy = df.copy()
    df_copy.to_excel(writer, sheet_name=group_name + "_summary", index=True, startrow=1)
    worksheet = writer.sheets[group_name + "_summary"]
    last_col = len(df_copy.columns) + 1
    try:
        merge_range = f"A1:{chr(64+last_col)}1"
        worksheet.merge_cells(merge_range)
    except Exception:
        pass
    cell = worksheet.cell(row=1, column=1)
    cell.value = group_name
    cell.alignment = Alignment(horizontal='center', vertical='center')

def save_pdf(tables, out_pdf, base_name):
    PAGE_WIDTH, PAGE_HEIGHT = portrait(A4)
    LEFT_MARGIN = RIGHT_MARGIN = 36
    usable_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
    doc = SimpleDocTemplate(out_pdf, pagesize=portrait(A4),
                           leftMargin=LEFT_MARGIN, rightMargin=RIGHT_MARGIN)
    elements = []
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = 'Times-Roman'
    title = Paragraph(f"Summary Report - {base_name}", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    for group_name, df in tables.items():
        elements.append(Paragraph(group_name, styles['Heading2']))
        elements.append(Spacer(1, 6))

        if isinstance(df.columns, pd.MultiIndex):
            nlevels = df.columns.nlevels
            header = []
            for level in range(nlevels):
                row = []
                for col in df.columns:
                    val = col[level] if isinstance(col, tuple) else ""
                    if level == 0 and val != "":
                        import textwrap
                        wrapped = "<br/>".join(textwrap.wrap(str(val), width=22))
                        val = Paragraph(f"<b><font size=11>{wrapped}</font></b>", ParagraphStyle('header', alignment=1, fontName='Times-Roman', leading=13))
                    elif level == 1 and val != "":
                        val = Paragraph(f"<b><font size=10>{val}</font></b>", ParagraphStyle('header', alignment=1, fontName='Times-Roman', leading=12))
                    else:
                        val = Paragraph(f"<font size=10>{val}</font>", ParagraphStyle('header', alignment=1, fontName='Times-Roman', leading=12))
                    row.append(val)
                header.append(row)
            data_table = header + df.astype(str).values.tolist()
            if df.index.name or df.index.names:
                for i, row in enumerate(data_table[nlevels:]):
                    data_table[nlevels + i] = [str(df.index[i])] + row
                for h in range(len(header)):
                    data_table[h] = [""] + data_table[h]
        else:
            data_table = [[df.index.name or "Metric"] + df.columns.tolist()]
            for idx, row in df.iterrows():
                data_table.append([str(idx)] + row.astype(str).tolist())

        n_cols = len(data_table[0])
        min_col_width = 50
        col_width = max(min_col_width, usable_width // n_cols)
        col_widths = [col_width] * n_cols
        total_width = sum(col_widths)
        if total_width > usable_width:
            scale = usable_width / total_width
            col_widths = [w * scale for w in col_widths]

        t = Table(data_table, hAlign="CENTER", colWidths=col_widths)
        style_list = [
            ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
        ]
        start_row = len(header) if isinstance(df.columns, pd.MultiIndex) else 1
        for r, row in enumerate(data_table[start_row:], start=start_row):
            for c, val in enumerate(row):
                if str(val).strip().upper() in ("PASS", "ĐẠT"):
                    style_list.append(('TEXTCOLOR', (c, r), (c, r), colors.green))
                elif str(val).strip().upper() in ("FAIL", "K.ĐẠT"):
                    style_list.append(('TEXTCOLOR', (c, r), (c, r), colors.red))
        t.setStyle(TableStyle(style_list))
        elements.append(t)
        elements.append(Spacer(1, 10))
    doc.build(elements)
    return out_pdf

# ======================= STREAMLIT UI =======================

# Initialize session state
if "run_processing" not in st.session_state:
    st.session_state.run_processing = False
if "processed_data" not in st.session_state:
    st.session_state.processed_data = {}
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

with st.sidebar:
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        logo_paths = [
            "logo.png",  
            os.path.join(os.path.dirname(__file__), "logo.png"),  
            os.path.join(os.getcwd(), "logo.png"), 
        ]
        
        logo_loaded = False
        for logo_path in logo_paths:
            if os.path.exists(logo_path):
                try:
                    st.image(logo_path, use_container_width=True)
                    logo_loaded = True
                    break
                except Exception:
                    continue

    st.markdown("#### Data Source")
    uploaded_files = st.file_uploader(
        "Upload Excel files (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True,
        help="Maximum 200MB per file"
    )
    
    folder_input = st.text_input(
        "Or folder path:",
        value="",
        help="Full path to folder with .xlsx files"
    )
    
    st.markdown("---")
    st.markdown("#### Output Settings")
    use_local_save = st.checkbox("Save to local folder", value=False,)
    if use_local_save:
        out_folder_input = st.text_input("Output folder:", value=DEFAULT_OUTPUT_FOLDER)
    else:
        out_folder_input = None
    
    
    st.markdown("---")
    st.markdown("#### Processing Options")
    enable_cleaning = st.checkbox("🧹 Clean data before analysis", value=True, 
                                   help="Cut first/last incomplete days, fill missing timestamps")
    
    st.markdown("---")
    st.markdown("#### Voltage Level & Thresholds")
    voltage_level = st.selectbox(
        "Select voltage level:",
        ["110kV", "22kV"],
        help="Different voltage levels have different threshold requirements"
    )
    
    if voltage_level == "110kV":
        threshold_pst = st.number_input("Pst threshold:", value=0.8, step=0.1, format="%.1f")
        threshold_plt = st.number_input("Plt threshold:", value=0.6, step=0.1, format="%.1f")
        threshold_thdu = st.number_input("THD U threshold (%):", value=3.0, step=0.5, format="%.1f")
        threshold_tddi = st.number_input("TDD I threshold (%):", value=3.0, step=0.5, format="%.1f")
        threshold_u_neg = st.number_input("u- threshold (%):", value=3.0, step=0.5, format="%.1f")
        threshold_vh = st.number_input("Voltage harmonic threshold (%):", value=1.5, step=0.1, format="%.1f")
        threshold_ch = st.number_input("Current harmonic threshold (%):", value=2.0, step=0.1, format="%.1f")
    else:  # 22kV
        threshold_pst = st.number_input("Pst threshold:", value=1.0, step=0.1, format="%.1f")
        threshold_plt = st.number_input("Plt threshold:", value=0.8, step=0.1, format="%.1f")
        threshold_thdu = st.number_input("THD U threshold (%):", value=5.0, step=0.5, format="%.1f")
        threshold_tddi = st.number_input("TDD I threshold (%):", value=5.0, step=0.5, format="%.1f")
        threshold_u_neg = st.number_input("u- threshold (%):", value=3.0, step=0.5, format="%.1f")
        threshold_vh = st.number_input("Voltage harmonic threshold (%):", value=3.0, step=0.1, format="%.1f")
        threshold_ch = st.number_input("Current harmonic threshold (%):", value=4.0, step=0.1, format="%.1f")
    
    st.markdown("---")
    st.markdown("#### Rated Power")
    PDM_default = 40000000

    Pdm_max = st.number_input("Pdm (W)", value=PDM_default, step=1000000, format="%d")
    
    st.markdown("---")
    run_button = st.button("▶️ START PROCESSING", use_container_width=True)
    
    if run_button:
        st.session_state.run_processing = True
        st.session_state.processing_complete = False
        st.session_state.processed_data = {}

# MAIN CONTENT AREA
st.markdown('<div class="main-title">⚡ POWER QUALITY ANALYSIS SYSTEM ⚡</div>', unsafe_allow_html=True)

# Processing section
if st.session_state.run_processing and not st.session_state.processing_complete:
    files_to_process = []
    if uploaded_files:
        for f in uploaded_files:
            files_to_process.append(("uploaded", f.name, f))
    elif folder_input:
        if os.path.isdir(folder_input):
            files = glob.glob(os.path.join(folder_input, "*.xlsx"))
            for f in files:
                files_to_process.append(("local", os.path.basename(f), f))
        else:
            st.error("❌ Invalid folder path")
            st.stop()
    else:
        st.error("❌ Please upload files or specify a folder path")
        st.stop()

        update_thresholds(threshold_pst, threshold_plt, threshold_thdu, threshold_tddi, threshold_u_neg)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(files_to_process)}</div>
            <div class="stat-label">Files to Process</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{Pdm_max/1000000:.1f}</div>
            <div class="stat-label">Pdm (MW)</div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("---")
    progress_bar = st.progress(0)
    status_text = st.empty()

    tmp_out_dir = tempfile.mkdtemp(prefix="pq_out_")
    processed_count = 0

    for idx, (ftype, name, fobj) in enumerate(files_to_process, start=1):
        status_text.info(f"🔄 Processing ({idx}/{len(files_to_process)}): **{name}**")
        cleaning_log = ""
        try:
            if ftype == "uploaded":
                xls = pd.ExcelFile(fobj)
            else:
                xls = pd.ExcelFile(fobj)
            
            if "Measurements" in xls.sheet_names:
                sheet_to_read = "Measurements"
            else:
                sheet_to_read = None
                for sheet in xls.sheet_names:
                    try:
                        temp = pd.read_excel(xls, sheet_name=sheet, nrows=1)
                    except Exception:
                        continue
                    if any(col in temp.columns for col in cols_pst):
                        sheet_to_read = sheet
                        break
                if sheet_to_read is None:
                    status_text.warning(f"⚠️ {name}: No matching sheet found — skipping")
                    progress_bar.progress(int(idx/len(files_to_process)*100))
                    continue
            
            data = pd.read_excel(xls, sheet_name=sheet_to_read)

            data_raw = data.copy()
            if enable_cleaning:
                time_col_candidates = ["Time [UTC]", "Time", "Datetime"]
                time_col_found = next((c for c in time_col_candidates if c in data.columns), None)
                if time_col_found:
                    status_text.info(f"🧹 Cleaning data for: **{name}**")
                    data, cleaning_log = clean_data(data, time_col_found)
                    if data.empty:
                        status_text.warning(f"⚠️ {name}: {cleaning_log} — skipping")
                        progress_bar.progress(int(idx/len(files_to_process)*100))
                        continue

            if "Time [UTC]" in data.columns:
                time_col = data["Time [UTC]"].copy()
                data = data.drop(columns=["Time [UTC]"])
            else:
                time_col = None

            data = data.replace("-", np.nan)
            data = data.apply(pd.to_numeric, errors="ignore")
            
            # BƯỚC 1: Lọc theo ngày đạt yêu cầu (có ít nhất 1 mẫu Ptot >= 50% Pdm)
            data_for_plt = None
            data_for_plt_6to18 = None
            
            if "Ptot+(Avg) [W]" in data.columns and time_col is not None:
                # Thêm cột Date và Time để phân loại
                data_temp = data.copy()
                data_temp["Date"] = pd.to_datetime(time_col.loc[data_temp.index], errors='coerce').dt.date
                data_temp["Time"] = pd.to_datetime(time_col.loc[data_temp.index], errors='coerce')
                data_temp = data_temp.dropna(subset=["Date"])
                
                if len(data_temp) > 0:
                    # Tính %Pdm cho mỗi mẫu
                    data_temp["%Pdm"] = (data_temp["Ptot+(Avg) [W]"] / Pdm_max * 100)
                    
                    # Tìm các ngày đạt yêu cầu (có ít nhất 1 mẫu >= 50% Pdm)
                    valid_dates = []
                    for date in data_temp["Date"].unique():
                        date_data = data_temp[data_temp["Date"] == date]
                        if (date_data["%Pdm"] >= 50).any():
                            valid_dates.append(date)
                    
                    # Lọc chỉ giữ lại dữ liệu của các ngày đạt yêu cầu
                    if len(valid_dates) > 0:
                        data_valid_days = data_temp[data_temp["Date"].isin(valid_dates)].copy()
                        status_text.info(f"📅 Found {len(valid_dates)} valid days (with Ptot ≥ 50% Pdm) out of {len(data_temp['Date'].unique())} total days")
                        
                        # Tạo data cho Plt (toàn bộ ngày)
                        data_for_plt = data_valid_days.drop(columns=["Date", "%Pdm", "Time"]).copy()
                        
                        # Tạo data cho Plt 6:00-18:00
                        data_valid_days["Hour"] = data_valid_days["Time"].dt.hour
                        data_6to18 = data_valid_days[(data_valid_days["Hour"] >= 6) & (data_valid_days["Hour"] < 18)].copy()
                        data_for_plt_6to18 = data_6to18.drop(columns=["Date", "%Pdm", "Time", "Hour"]).copy()
                        
                        # BƯỚC 2: Lọc thêm Ptot > 0 cho các bảng còn lại
                        data = data_valid_days.copy()
                        invalid_mask = (data["Ptot+(Avg) [W]"].isna()) | (data["Ptot+(Avg) [W]"] <= 0)
                        if invalid_mask.any():
                            data = data[~invalid_mask]
                        data = data.drop(columns=["Date", "%Pdm", "Time"], errors='ignore')
                        if "Hour" in data.columns:
                            data = data.drop(columns=["Hour"])
                    else:
                        status_text.warning(f"⚠️ {name}: No valid days found (no day with Ptot ≥ 50% Pdm) — skipping")
                        progress_bar.progress(int(idx/len(files_to_process)*100))
                        continue
            
            data = data.fillna(0)
            if time_col is not None:
                time_filtered = time_col.loc[data.index]
                data.insert(0, "Time [UTC]", time_filtered)

            stat_summary_pst = pd.DataFrame()
            stat_summary_plt = pd.DataFrame()
            stat_summary_plt_6to18 = pd.DataFrame()
            table_plt_24h = pd.DataFrame()
            table_plt_6to18 = pd.DataFrame()
            stat_summary_uneg = pd.DataFrame()
            stat_summary_vh = pd.DataFrame()
            stat_summary_ch = pd.DataFrame()
            daily_pdm_table = pd.DataFrame()
            daily_summary_text = ""
            
# TÍNH TOÁN PLT (ưu tiên, chỉ cần Ptot >= 50% Pdm)
            plt_valid_days_count = 0
            plt_total_samples = 0
            plt_divided_by_12 = 0
                        
            if data_for_plt is not None and all(c in data_for_plt.columns for c in cols_plt):
                total_samples_from_valid_days = len(data_for_plt)
                plt_total_samples = total_samples_from_valid_days
                plt_divided_by_12 = total_samples_from_valid_days / 12
                
                # Đếm số ngày đủ điều kiện cho Plt
                if "Date" in data_temp.columns:
                    plt_valid_days_count = len(valid_dates)
                
                # ✅ TÍNH TOÁN BẢNG PLT 24H - Dựa trên tổng mẫu chia 12
                total_plt_values = total_samples_from_valid_days / 12
                
                passed = {}
                failed = {}
                perc = {}
                status_per_col = {}
                
                for col in cols_plt:
                    threshold = thresholds[col]
                    pass_count = ((data_for_plt[col] <= threshold) & data_for_plt[col].notna()).sum() / 12
                    fail_count = total_plt_values - pass_count
                    percentage = (pass_count / total_plt_values * 100) if total_plt_values > 0 else 0
                    
                    passed[col] = pass_count
                    failed[col] = fail_count
                    perc[col] = round(percentage, 2)
                    status_per_col[col] = "PASS" if percentage >= 95 else "FAIL"
                
                group_status = "PASS" if all(s == "PASS" for s in status_per_col.values()) else "FAIL"
                
                table_plt_24h = pd.DataFrame([
                    {col: total_plt_values for col in cols_plt},
                    passed,
                    failed,
                    perc,
                    status_per_col
                ], index=[
                    "No. of samples",
                    "No. of pass samples",
                    "No. of fail samples",
                    "Percentage of pass samples [%]",
                    "Status"
                ])
                table_plt_24h.loc["Group Status"] = [group_status] * len(cols_plt)
                
                plt_sample_count = total_samples_from_valid_days / 12
                
                
                # Plt Statistics (Overall) cho toàn bộ ngày
                plt_stats = []
                for col in cols_plt:
                    mx = data_for_plt[col].max()
                    mn = data_for_plt[col].min()
                    avg = data_for_plt[col].mean()
                    plt_stats.extend([mx, avg, mn])
                
                plt_columns = pd.MultiIndex.from_tuples([
                    ("Phase A", "Max"), ("Phase A", "AVG"), ("Phase A", "Min"),
                    ("Phase B", "Max"), ("Phase B", "AVG"), ("Phase B", "Min"),
                    ("Phase C", "Max"), ("Phase C", "AVG"), ("Phase C", "Min")
                ])
                stat_summary_plt = pd.DataFrame([plt_stats], columns=plt_columns)
                stat_summary_plt.index = ["Overall (24h)"]
                stat_summary_plt.index.name = "Statistic"
                stat_summary_plt = stat_summary_plt.round(3)
            
            if data_for_plt_6to18 is not None and len(data_for_plt_6to18) > 0 and all(c in data_for_plt_6to18.columns for c in cols_plt):
                # ✅ TÍNH TOÁN BẢNG PLT 6-18H - Dựa trên tổng mẫu chia 12
                total_samples_6to18 = len(data_for_plt_6to18)
                total_plt_values_6to18 = total_samples_6to18 / 12
                
                passed_6to18 = {}
                failed_6to18 = {}
                perc_6to18 = {}
                status_per_col_6to18 = {}
                
                for col in cols_plt:
                    threshold = thresholds[col]
                    pass_count = ((data_for_plt_6to18[col] <= threshold) & data_for_plt_6to18[col].notna()).sum() / 12
                    fail_count = total_plt_values_6to18 - pass_count
                    percentage = (pass_count / total_plt_values_6to18 * 100) if total_plt_values_6to18 > 0 else 0
                    
                    passed_6to18[col] = pass_count
                    failed_6to18[col] = fail_count
                    perc_6to18[col] = round(percentage, 2)
                    status_per_col_6to18[col] = "PASS" if percentage >= 95 else "FAIL"
                
                group_status_6to18 = "PASS" if all(s == "PASS" for s in status_per_col_6to18.values()) else "FAIL"
                
                table_plt_6to18 = pd.DataFrame([
                    {col: total_plt_values_6to18 for col in cols_plt},
                    passed_6to18,
                    failed_6to18,
                    perc_6to18,
                    status_per_col_6to18
                ], index=[
                    "No. of samples",
                    "No. of pass samples",
                    "No. of fail samples",
                    "Percentage of pass samples [%]",
                    "Status"
                ])
                table_plt_6to18.loc["Group Status"] = [group_status_6to18] * len(cols_plt)

                
                # Plt Statistics (Overall) cho 6:00-18:00
                plt_stats_6to18 = []
                for col in cols_plt:
                    mx = data_for_plt_6to18[col].max()
                    mn = data_for_plt_6to18[col].min()
                    avg = data_for_plt_6to18[col].mean()
                    plt_stats_6to18.extend([mx, avg, mn])
                
                plt_columns = pd.MultiIndex.from_tuples([
                    ("Phase A", "Max"), ("Phase A", "AVG"), ("Phase A", "Min"),
                    ("Phase B", "Max"), ("Phase B", "AVG"), ("Phase B", "Min"),
                    ("Phase C", "Max"), ("Phase C", "AVG"), ("Phase C", "Min")
                ])
                stat_summary_plt_6to18 = pd.DataFrame([plt_stats_6to18], columns=plt_columns)
                stat_summary_plt_6to18.index = ["Overall (6:00-18:00)"]
                stat_summary_plt_6to18.index.name = "Statistic"
                stat_summary_plt_6to18 = stat_summary_plt_6to18.round(3)

            if "Ptot+(Avg) [W]" in data.columns:
                Pdm_max_val = Pdm_max
                if pd.isna(Pdm_max_val) or Pdm_max_val <= 0:
                    pass
                else:
                    daily_pdm_table, daily_summary_text, pdm_detailed_info = make_daily_pdm_distribution(data, Pdm_max_val)                    
                    bins = []
                    labels = []
                    for i in range(1, 11):
                        lower = (i - 1) * 0.1 * Pdm_max_val
                        upper = i * 0.1 * Pdm_max_val
                        bins.append((lower, upper))
                        labels.append(f"{i*10}%")
                    
                    stat_summary_vh = make_voltage_harmonics_by_pdm(data, bins, labels)
                    stat_summary_ch = make_current_harmonics_by_pdm(data, bins, labels)
                    
                    rows_pst = []
                    rows_uneg = []
                    required_stat_cols = {"Pst": cols_pst, "Uneg": cols_uneg}
                    
                    for (lower, upper), label in zip(bins, labels):
                        if lower <= 0:
                            mask = (data["Ptot+(Avg) [W]"] > 0) & (data["Ptot+(Avg) [W]"] <= upper)
                        else:
                            mask = (data["Ptot+(Avg) [W]"] > lower) & (data["Ptot+(Avg) [W]"] <= upper)
                        df_bin = data.loc[mask]
                        count = int(mask.sum())
                        
                        pst_vals = []
                        for col in required_stat_cols["Pst"]:
                            if col in data.columns and count > 0:
                                mx = df_bin[col].max()
                                mn = df_bin[col].min()
                                avg = df_bin[col].mean()
                            else:
                                mx = np.nan
                                mn = np.nan
                                avg = np.nan
                            pst_vals.append((mx, avg, mn))
                        rows_pst.append([label] + [v for triple in pst_vals for v in triple])
                        
                        ucol = required_stat_cols["Uneg"][0]
                        if ucol in data.columns and count > 0:
                            mx = df_bin[ucol].max()
                            mn = df_bin[ucol].min()
                            avg = df_bin[ucol].mean()
                        else:
                            mx = np.nan
                            mn = np.nan
                            avg = np.nan
                        rows_uneg.append([label, mx, avg, mn])
                    
                    pst_columns = pd.MultiIndex.from_tuples([
                        ("Phase A", "Max"), ("Phase A", "AVG"), ("Phase A", "Min"),
                        ("Phase B", "Max"), ("Phase B", "AVG"), ("Phase B", "Min"),
                        ("Phase C", "Max"), ("Phase C", "AVG"), ("Phase C", "Min")
                    ])
                    pst_df = pd.DataFrame([r[1:] for r in rows_pst], index=[r[0] for r in rows_pst], columns=pst_columns)
                    pst_df.index.name = "%Pdm"
                    
                    uneg_df = pd.DataFrame([r[1:] for r in rows_uneg], index=[r[0] for r in rows_uneg], columns=["Max", "AVG", "Min"])
                    uneg_df.index.name = "%Pdm"
                    
                    stat_summary_pst = pst_df.round(3)
                    stat_summary_uneg = uneg_df.round(3)

            # TÍNH TOÁN CÁC BẢNG CÒN LẠI (sau khi lọc Ptot > 0)
            table_pst = make_table(data, cols_pst) if all(c in data.columns for c in cols_pst) else pd.DataFrame()
            table_thdu = make_table(data, cols_thdu) if all(c in data.columns for c in cols_thdu) else pd.DataFrame()
            table_tddi = make_table(data, cols_tddi) if all(c in data.columns for c in cols_tddi) else pd.DataFrame()
            table_uneg = make_table(data, cols_uneg) if all(c in data.columns for c in cols_uneg) else pd.DataFrame()
            table_vh_order = make_voltage_harmonics_detail(data, threshold_vh)
            table_ch_order = make_current_harmonics_detail(data, threshold_ch)

            if use_local_save and out_folder_input:
                save_folder = out_folder_input
                os.makedirs(save_folder, exist_ok=True)
            elif use_local_save and folder_input:
                save_folder = os.path.join(folder_input, "Output")
                os.makedirs(save_folder, exist_ok=True)
            else:
                save_folder = tmp_out_dir
            
            base_name = os.path.splitext(name)[0]
            out_excel_path = os.path.join(save_folder, f"Output_{base_name}.xlsx")
            out_png_path = os.path.join(save_folder, f"Output_{base_name}.png")
            out_pdf_path = os.path.join(save_folder, f"Output_{base_name}.pdf")

            with pd.ExcelWriter(out_excel_path, engine="openpyxl") as writer:
                if not table_pst.empty:
                    save_table_to_excel(writer, table_pst, "Pst")
                if not table_thdu.empty:
                    save_table_to_excel(writer, table_thdu, "THDu")
                if not table_tddi.empty:
                    save_table_to_excel(writer, table_tddi, "TDDi")
                
                # Plt tables (24h and 6-18)
                if not table_plt_24h.empty:
                    save_table_to_excel(writer, table_plt_24h, "Plt_24h")
                if not table_plt_6to18.empty:
                    save_table_to_excel(writer, table_plt_6to18, "Plt_6to18")
                
                if not table_uneg.empty:
                    save_table_to_excel(writer, table_uneg, "u0Avg")
                table_vh_order.to_excel(writer, sheet_name="Voltage Harmonics by Order")
                table_ch_order.to_excel(writer, sheet_name="Current Harmonics by Order")
                
                if not daily_pdm_table.empty:
                    daily_pdm_table.to_excel(writer, sheet_name="Daily_Pdm_Distribution", index=False)
                
                try:
                    if not stat_summary_pst.empty:
                        stat_summary_pst.to_excel(writer, sheet_name="Power_Stats_Pst", startrow=1)
                    if not stat_summary_plt.empty:
                        stat_summary_plt.to_excel(writer, sheet_name="Plt_Statistics_24h", startrow=1)
                    if not stat_summary_plt_6to18.empty:
                        stat_summary_plt_6to18.to_excel(writer, sheet_name="Plt_Statistics_6to18", startrow=1)
                    if not stat_summary_uneg.empty:
                        stat_summary_uneg.to_excel(writer, sheet_name="Power_Stats_Uneg", startrow=1)
                    if not stat_summary_vh.empty:
                        stat_summary_vh.to_excel(writer, sheet_name="Voltage_Harmonics_by_%Pdm", startrow=1)
                    if not stat_summary_ch.empty:
                        stat_summary_ch.to_excel(writer, sheet_name="Current_Harmonics_by_%Pdm", startrow=1)
                except Exception:
                    pass

            png_saved = plot_and_save(data, base_name, save_folder)
            
            tables_for_pdf = {
                "Pst": table_pst,
                "THDu": table_thdu,
                "TDDi": table_tddi,
                "Plt (24h)": table_plt_24h,
                "Plt (6:00-18:00)": table_plt_6to18,
                "u0Avg": table_uneg,
                "Voltage Harmonics by Order": table_vh_order,
                "Current Harmonics by Order": table_ch_order
            }
            if not daily_pdm_table.empty:
                tables_for_pdf["Daily Pdm Distribution"] = daily_pdm_table
            if not stat_summary_pst.empty:
                tables_for_pdf["Pst by %Pdm"] = stat_summary_pst
            if not stat_summary_plt.empty:
                tables_for_pdf["Plt Statistics (24h)"] = stat_summary_plt
            if not stat_summary_plt_6to18.empty:
                tables_for_pdf["Plt Statistics (6:00-18:00)"] = stat_summary_plt_6to18
            if not stat_summary_uneg.empty:
                tables_for_pdf["Uneg by %Pdm"] = stat_summary_uneg
            if not stat_summary_vh.empty:
                tables_for_pdf["Voltage Harmonics (2-40) by %Pdm"] = stat_summary_vh
            if not stat_summary_ch.empty:
                tables_for_pdf["Current Harmonics (2-40) by %Pdm"] = stat_summary_ch

            save_pdf(tables_for_pdf, out_pdf_path, base_name)

            st.session_state.processed_data[base_name] = {
                'data': data.copy(),
                'data_raw': data_raw,
                'voltage_level': voltage_level,
                'thresholds': {
                    'pst': threshold_pst,
                    'plt': threshold_plt,
                    'thdu': threshold_thdu,
                    'tddi': threshold_tddi,
                    'u_neg': threshold_u_neg,
                    'vh': threshold_vh,
                    'ch': threshold_ch
                },
                'tables': {

                    'table_pst': table_pst,
                    'table_thdu': table_thdu,
                    'table_tddi': table_tddi,
                    'table_plt_24h': table_plt_24h,
                    'table_plt_6to18': table_plt_6to18,
                    'table_uneg': table_uneg,
                    'table_vh_order': table_vh_order,
                    'table_ch_order': table_ch_order,
                    'stat_summary_pst': stat_summary_pst,
                    'stat_summary_plt': stat_summary_plt,
                    'stat_summary_plt_6to18': stat_summary_plt_6to18,
                    'stat_summary_uneg': stat_summary_uneg,
                    'stat_summary_vh': stat_summary_vh,
                    'stat_summary_ch': stat_summary_ch,
                    'daily_pdm_table': daily_pdm_table,
                    'daily_summary_text': daily_summary_text,
                    'pdm_detailed_info': pdm_detailed_info 

                },
                'excel_path': out_excel_path,
                'png_path': out_png_path,
                'pdf_path': out_pdf_path,
                'save_folder': save_folder,
                'plt_info': {
                    'valid_days_count': plt_valid_days_count,
                    'total_samples': plt_total_samples,
                    'divided_by_12': plt_divided_by_12
                },
                'cleaning_log': cleaning_log
            }
            processed_count += 1
        except Exception as e:
            st.error(f"❌ Error processing {name}: {e}")
        progress_bar.progress(int(idx/len(files_to_process)*100))

    st.session_state.processing_complete = True
    st.session_state.run_processing = False
    status_text.success(f"✅ COMPLETED: Successfully processed {processed_count}/{len(files_to_process)} file(s)")

# Display results section
if st.session_state.processing_complete and st.session_state.processed_data:
    st.markdown("---")
    
    # File selector with better styling
    file_names = list(st.session_state.processed_data.keys())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📊 Analysis Results")
    with col2:
        selected_file = st.selectbox("Select file:", file_names, label_visibility="collapsed")
    
    if selected_file:
        result = st.session_state.processed_data[selected_file]
        data = result['data']
        tables = result['tables']
        # Display voltage level and thresholds info
        st.markdown("---")
        st.markdown("### ⚙️ Analysis Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{result.get('voltage_level', 'N/A')}</div>
                <div class="stat-label">Voltage Level</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            thresholds_info = result.get('thresholds', {})
            st.markdown(f"""
            <div class="stat-card" style="text-align: left; padding: 1rem 1.5rem;">
                <div style="font-weight: 600; margin-bottom: 0.5rem;">Applied Thresholds:</div>
                <div style="font-size: 0.9rem; line-height: 1.8;">
                    • Pst: {thresholds_info.get('pst', 'N/A')}<br/>
                    • Plt: {thresholds_info.get('plt', 'N/A')}<br/>
                    • THD U: {thresholds_info.get('thdu', 'N/A')}%<br/>
                    • TDD I: {thresholds_info.get('tddi', 'N/A')}%<br/>
                    • u0: {thresholds_info.get('u0', 'N/A')}%<br/>
                    • Voltage Harmonic: {thresholds_info.get('vh', 'N/A')}%<br/>
                    • Current Harmonic: {thresholds_info.get('ch', 'N/A')}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Download section with cards
        # st.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.markdown(f"#### 📥 Download Results for: **{selected_file}**")
        
        import base64
        def get_download_link(file_path, filename, label, icon):
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            b64 = base64.b64encode(file_bytes).decode()
            if filename.endswith('.xlsx'):
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif filename.endswith('.png'):
                mime = "image/png"
            elif filename.endswith('.pdf'):
                mime = "application/pdf"
            else:
                mime = "application/octet-stream"
            href = f'<a href="data:{mime};base64,{b64}" download="{filename}" target="_blank" style="text-decoration:none;"><button style="background: linear-gradient(90deg, #3498db 0%, #2980b9 100%); color: white; border: none; border-radius: 8px; padding: 0.75rem 1.5rem; font-weight: 600; cursor: pointer; width: 100%; box-shadow: 0px 4px 10px rgba(52, 152, 219, 0.3);">{icon} {label}</button></a>'
            return href
        c1, c2, c3 = st.columns(3)
        c1.markdown(get_download_link(result['excel_path'], f"Output_{selected_file}.xlsx", "Excel Report", "📘"), unsafe_allow_html=True)
        c2.markdown(get_download_link(result['png_path'], f"Output_{selected_file}.png", "Charts (PNG)", "🖼️"), unsafe_allow_html=True)
        c3.markdown(get_download_link(result['pdf_path'], f"Output_{selected_file}.pdf", "PDF Report", "📄"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
# Summary statistics
        if tables['daily_summary_text']:
            st.markdown("---")
            if result.get('cleaning_log'):
                with st.expander("🧹 Data Cleaning Log", expanded=False):
                    st.text(result['cleaning_log'])
            st.info(tables['daily_summary_text'])

            # ✅ HIỂN THỊ THÔNG TIN PLT
            if 'plt_info' in result and result['plt_info']['total_samples'] > 0:
                plt_info = result['plt_info']
                remainder = plt_info['total_samples'] % 12
                
                st.markdown("### 📊 Plt Calculation Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{plt_info['valid_days_count']}</div>
                        <div class="stat-label">Valid Days for Plt<br/>(Ptot≥50%Pdm)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{plt_info['total_samples']:,}</div>
                        <div class="stat-label">Total Samples (Ptot≥50%Pdm)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    if remainder == 0:
                        color = "#27ae60"
                        status = "✅"
                    else:
                        color = "#e74c3c"
                        status = f"⚠️: {remainder}"
                    
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number" style="color: {color};">{plt_info['divided_by_12']:.2f}</div>
                        <div class="stat-label">Plt Samples <br/>{status}</div>
                    </div>
                    """, unsafe_allow_html=True)

        if 'pdm_detailed_info' in tables and tables['pdm_detailed_info']:
            st.markdown("---")            
            info = tables['pdm_detailed_info']
            
            # Statistics cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{info['valid_days_count']}</div>
                    <div class="stat-label">Valid Days</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{info['total_valid_days_samples']:,}</div>
                    <div class="stat-label">Total Samples (All Valid Days)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_samples = info['total_valid_days_samples'] / info['valid_days_count'] if info['valid_days_count'] > 0 else 0
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{avg_samples:.0f}</div>
                    <div class="stat-label">Avg Samples/Day</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed table with expander
            with st.expander("📊 View Detailed Breakdown by Date", expanded=False):
                valid_dates_df = pd.DataFrame([
                    {
                        'No.': idx,
                        'Date': date.strftime("%d/%m/%Y"),
                        'Samples': samples,
                        # 'Status': '✅ Valid (≥50% Pdm)'
                    }
                    for idx, (date, samples) in enumerate(sorted(info['valid_dates_samples'].items()), 1)
                ])
                
                # Add total row
                total_row = pd.DataFrame([{
                    'No.': '',
                    'Date': 'TOTAL',
                    'Samples': info['total_valid_days_samples'],
                    'Status': f"{info['valid_days_count']} days"
                }])
                valid_dates_df = pd.concat([valid_dates_df, total_row], ignore_index=True)
                
                st.dataframe(valid_dates_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Summary Tables", "📊 Harmonics Details", "⚡ Power Statistics", "📈 Raw Data", "🎨 Custom Plot"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                if not tables['table_pst'].empty:
                    st.markdown("##### Pst Summary")
                    st.dataframe(tables['table_pst'], use_container_width=True)
                if not tables['table_plt_24h'].empty:
                    st.markdown("##### Plt Summary (24h) - Divided by 12")
                    st.dataframe(tables['table_plt_24h'], use_container_width=True)
                if not tables['table_plt_6to18'].empty:
                    st.markdown("##### Plt Summary (6:00-18:00) - Divided by 12")
                    st.dataframe(tables['table_plt_6to18'], use_container_width=True)
            with col2:
                if not tables['table_thdu'].empty:
                    st.markdown("##### THD U Summary")
                    st.dataframe(tables['table_thdu'], use_container_width=True)
                if not tables['table_tddi'].empty:
                    st.markdown("##### TDD I Summary")
                    st.dataframe(tables['table_tddi'], use_container_width=True)
            
            if not tables['table_uneg'].empty:
                st.markdown("##### Uneg (u0Avg) Summary")
                st.dataframe(tables['table_uneg'], use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                if not tables['table_vh_order'].empty:
                    st.markdown("##### Voltage Harmonics (Order 2-40)")
                    st.dataframe(tables['table_vh_order'], use_container_width=True, height=500)
            with col2:
                if not tables['table_ch_order'].empty:
                    st.markdown("##### Current Harmonics (Order 2-40)")
                    st.dataframe(tables['table_ch_order'], use_container_width=True, height=500)
        
        with tab3:
            if not tables['daily_pdm_table'].empty:
                st.markdown("##### Daily %Pdm Distribution")
                st.dataframe(tables['daily_pdm_table'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if not tables['stat_summary_pst'].empty:
                    st.markdown("##### Pst by %Pdm")
                    st.dataframe(tables['stat_summary_pst'], use_container_width=True)
            with col2:
                if not tables['stat_summary_uneg'].empty:
                    st.markdown("##### Uneg by %Pdm")
                    st.dataframe(tables['stat_summary_uneg'], use_container_width=True)
            
            if not tables['stat_summary_plt'].empty:
                st.markdown("##### Plt Statistics (24h - Overall)")
                st.dataframe(tables['stat_summary_plt'], use_container_width=True)
            
            if not tables['stat_summary_plt_6to18'].empty:
                st.markdown("##### Plt Statistics (6:00-18:00 - Overall)")
                st.dataframe(tables['stat_summary_plt_6to18'], use_container_width=True)
            
            if not tables['stat_summary_vh'].empty:
                st.markdown("##### Voltage Harmonics by %Pdm")
                st.dataframe(tables['stat_summary_vh'], use_container_width=True, height=400)
            
            if not tables['stat_summary_ch'].empty:
                st.markdown("##### Current Harmonics by %Pdm")
                st.dataframe(tables['stat_summary_ch'], use_container_width=True, height=400)
        
    with tab4:
        # Thêm radio button để chọn
        view_mode = st.radio("View mode:", ["Processed Data", "Raw Data (Original)"], horizontal=True)
        
        if view_mode == "Raw Data (Original)":
            display_data_source = result.get('data_raw', data)
            st.info("📋 Showing original data before any processing")
        else:
            display_data_source = data
            st.info("📋 Showing processed data (filtered by valid days and Ptot > 0)")
        
        st.markdown(f"**Total rows:** {len(display_data_source)} | **Total columns:** {len(display_data_source.columns)}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            show_rows = st.number_input("Rows to display:", min_value=10, max_value=len(display_data_source), value=min(100, len(display_data_source)), step=10)
        with col2:
            start_row = st.number_input("Start from row:", min_value=0, max_value=max(0, len(display_data_source)-1), value=0, step=10)
        with col3:
            search_col = st.text_input("Filter columns:", value="")
        
        if search_col.strip():
            search_terms = [term.strip() for term in search_col.split(",")]
            filtered_cols = [col for col in display_data_source.columns if any(term.lower() in col.lower() for term in search_terms)]
            if filtered_cols:
                display_data = display_data_source[filtered_cols].iloc[start_row:start_row + show_rows]
                st.success(f"✓ Found {len(filtered_cols)} matching columns")
            else:
                display_data = display_data_source.iloc[start_row:start_row + show_rows]
                st.warning("No matching columns found")
        else:
            display_data = display_data_source.iloc[start_row:start_row + show_rows]
        
        st.dataframe(display_data, use_container_width=True, height=400)
        
        if st.checkbox("Show statistics"):
            st.dataframe(display_data_source.describe(), use_container_width=True)
        
        with tab5:
            available_columns = list(data.columns)
            
            if len(available_columns) >= 2:
                col_x, col_y = st.columns(2)
                with col_x:
                    x_col = st.selectbox("X-axis:", available_columns, index=0)
                with col_y:
                    y_cols = st.multiselect(
                        "Y-axis (multiple):",
                        [c for c in available_columns if c != x_col],
                        default=[available_columns[1]] if len(available_columns) > 1 else []
                    )

                plot_type = st.radio("Chart type:", ["Line", "Scatter"], horizontal=True)
                
                if st.button("🎨 Generate Plot", use_container_width=True) and y_cols:
                    try:
                        import plotly.express as px
                        plot_df = data[[x_col] + y_cols].dropna()
                        max_points = 8000
                        if len(plot_df) > max_points:
                            plot_df = plot_df.sample(max_points).sort_values(by=x_col)
                            st.info(f"📊 Sampled to {max_points} points for performance")
                        
                        if plot_type == "Line":
                            fig = px.line(plot_df, x=x_col, y=y_cols)
                        else:
                            fig = px.scatter(plot_df, x=x_col, y=y_cols, opacity=0.7)
                        
                        fig.update_layout(
                            template="plotly_white",
                            height=500,
                            legend=dict(orientation="h", y=-0.2),
                            margin=dict(l=50, r=30, t=50, b=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
                elif y_cols == []:
                    st.warning("⚠️ Select at least one Y-axis column")
            else:
                st.info("Not enough columns for plotting")

elif not st.session_state.processing_complete:
    # Welcome screen
    st.markdown("""
    <style>
    @keyframes typing {
    from { width: 0 }
    to { width: 100% }
    }
    @keyframes blink {
    50% { border-color: transparent }
    }
    .typewriter h2 {
    overflow: hidden;
    border-right: .15em solid #00BCD4;
    white-space: nowrap;
    margin: 0 auto;
    letter-spacing: .05em;
    animation:
        typing 3.5s steps(40, end),
        blink .75s step-end infinite;
    }
    </style>

    <div style="text-align: center; padding: 3rem;">
    <div class="typewriter">
        <h2>Welcome to Power Quality Analysis System</h2>
    </div>
    <p style="font-size: 1.2rem; color: #7f8c8d; margin-top: 1rem;">
        Upload your Excel files or specify a folder path in the sidebar to begin analysis
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .stat-card {
        transition: all 0.3s ease;
        border-radius: 15px;
        padding: 20px;
        background: #ffffff;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        cursor: pointer;
    }
    .stat-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    .stat-card:active {
        transform: scale(1.1);
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)


    with col1:
        st.markdown("""
        <div class="stat-card" style="text-align: center;">
            <div style="font-size: 3rem;">📊</div>
            <div style="font-weight: 600; margin-top: 1rem;">Comprehensive Analysis</div>
            <div style="color: #7f8c8d; margin-top: 0.5rem;">Analyze Pst, Plt, THD, TDD and harmonics</div>
        </div>
        """, unsafe_allow_html=True)


    with col2:
        st.markdown("""
        <div class="stat-card" style="text-align: center;">
            <div style="font-size: 3rem;">⚡</div>
            <div style="font-weight: 600; margin-top: 1rem;">Power Statistics</div>
            <div style="color: #7f8c8d; margin-top: 0.5rem;">Detailed %Pdm distribution analysis</div>
        </div>
        """, unsafe_allow_html=True)


    with col3:
        st.markdown("""
        <div class="stat-card" style="text-align: center;">
            <div style="font-size: 3rem;">📥</div>
            <div style="font-weight: 600; margin-top: 1rem;">Export Results</div>
            <div style="color: #7f8c8d; margin-top: 0.5rem;">Download Excel, PDF, and PNG reports</div>
        </div>
        """, unsafe_allow_html=True)
        
        