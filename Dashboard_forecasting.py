# dashboard_forecasting.py
# Dashboard Forecasting SARIMA + Inventory Optimization (PT Michelin Indonesia)
# Works with sheets: StokGudang, Pembelian, Penggunaan

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="DSS - Forecasting & Inventory Optimization")

st.title("Decision Support System - SARIMA Forecasting & Inventory Optimization")
st.markdown("""
Dashboard ini menggunakan file **Transaction_BahanBaku_2022-2025_FINAL_ADJUSTED.xlsx**  
Data diambil dari tiga sheet utama: `StokGudang`, `Pembelian`, dan `Penggunaan`.
""")

EXCEL_PATH = "Transaction_BahanBaku_2022-2025_FINAL_ADJUSTED.xlsx"

@st.cache_data
def load_data(path):
    try:
        stok = pd.read_excel(path, sheet_name="StokGudang")
        pemb = pd.read_excel(path, sheet_name="Pembelian")
        pakai = pd.read_excel(path, sheet_name="Penggunaan")
        return stok, pemb, pakai
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None, None, None

stok_df, pemb_df, pakai_df = load_data(EXCEL_PATH)
if stok_df is None:
    st.stop()

# --- Preprocessing ---
for df in [stok_df, pemb_df, pakai_df]:
    if "Tahun" in df.columns and "Bulan" in df.columns:
        df["Tanggal"] = pd.to_datetime(df["Tahun"].astype(str) + "-" + df["Bulan"].astype(str) + "-01")

materials = stok_df["Nama Bahan Baku"].dropna().unique().tolist()
selected_material = st.sidebar.selectbox("Pilih Nama Bahan Baku", materials)
selected_gudang = st.sidebar.selectbox("Pilih Gudang", ["(Semua)"] + sorted(stok_df["Gudang"].dropna().unique().tolist()))

# Filter data sesuai bahan baku dan gudang
df_filtered = stok_df[stok_df["Nama Bahan Baku"] == selected_material].copy()
if selected_gudang != "(Semua)":
    df_filtered = df_filtered[df_filtered["Gudang"] == selected_gudang]

# Gabungkan tahun & bulan jadi satu kolom waktu
df_filtered = df_filtered.sort_values("Tanggal")
monthly_usage = df_filtered.groupby("Tanggal")["Usage (kg)"].sum().asfreq("MS")

if monthly_usage.empty:
    st.warning("Tidak ada data usage untuk bahan baku ini.")
    st.stop()

st.subheader(f"Data Usage Bulanan untuk: {selected_material}")
st.line_chart(monthly_usage)

# --- Parameter Sidebar ---
forecast_steps = st.sidebar.number_input("Periode forecast (bulan)", min_value=3, max_value=24, value=12)
lead_time_days = st.sidebar.number_input("Lead Time (hari)", min_value=1, value=14)
service_level_z = st.sidebar.number_input("Z-Value (Service Level)", min_value=0.0, value=1.65)
ordering_cost = st.sidebar.number_input("Biaya Pemesanan (Rp)", min_value=0.0, value=500000.0, step=10000.0)
holding_cost_rate = st.sidebar.number_input("Biaya Penyimpanan per Tahun (%)", min_value=0.0, value=0.20)
unit_cost = st.sidebar.number_input("Harga per kg (Rp)", min_value=0.0, value=10000.0)
holding_cost = holding_cost_rate * unit_cost

# --- Forecasting SARIMA ---
if st.button("Jalankan Forecasting SARIMA"):
    try:
        model = SARIMAX(monthly_usage, order=(1,1,1), seasonal_order=(1,1,1,12))
        res = model.fit(disp=False)
        forecast = res.get_forecast(steps=forecast_steps)
        fc_series = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Plot hasil forecast
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(monthly_usage.index, monthly_usage.values, label="Actual")
        ax.plot(fc_series.index, fc_series.values, label="Forecast", linestyle="--")
        ax.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color="gray", alpha=0.3)
        ax.legend()
        ax.set_title(f"Forecast Usage {selected_material}")
        st.pyplot(fig)

        # --- Hitung EOQ, Safety Stock, ROP ---
        mean_demand = monthly_usage.mean()
        daily_demand = mean_demand / 30
        std_demand = monthly_usage.std() / 30
        safety_stock = service_level_z * std_demand * np.sqrt(lead_time_days)
        reorder_point = daily_demand * lead_time_days + safety_stock
        annual_demand = mean_demand * 12
        if holding_cost > 0:
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        else:
            eoq = np.nan

        st.subheader("📦 Rekomendasi Pengendalian Persediaan")
        st.write(f"Rata-rata permintaan bulanan: **{mean_demand:.2f} kg**")
        st.write(f"Safety Stock: **{safety_stock:.2f} kg**")
        st.write(f"Reorder Point: **{reorder_point:.2f} kg**")
        st.write(f"EOQ (Economic Order Quantity): **{eoq:.2f} kg per order**")

        df_rec = pd.DataFrame({
            "Metric": ["Mean Monthly Demand", "Safety Stock", "Reorder Point", "EOQ"],
            "Value (kg)": [mean_demand, safety_stock, reorder_point, eoq]
        })
        st.dataframe(df_rec.style.format("{:.2f}"))

        # --- Akurasi Model ---
        mape = np.mean(np.abs((monthly_usage - res.fittedvalues) / monthly_usage.replace(0, np.nan))) * 100
        rmse = np.sqrt(np.mean((res.fittedvalues - monthly_usage) ** 2))
        st.subheader("📊 Akurasi Model SARIMA (In-sample)")
        st.write(f"MAPE: {mape:.2f}%")
        st.write(f"RMSE: {rmse:.2f}")

    except Exception as e:
        st.error(f"Gagal menjalankan SARIMA: {e}")

st.markdown("---")
st.caption("© 2025 DSS Forecasting PT Michelin Indonesia — dikembangkan untuk mendukung penelitian tesis.")
