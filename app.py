# app.py — CSV 기반 PV 대시보드 (Streamlit, 1920x1080 1-Page Fit)
# =================================================================
# 변경 요약
# - 전체 레이아웃/여백 압축: 상/하 padding, 카드 간격, 글자 크기 축소
# - 차트 높이 조정: 230 / 165 / 185 px로 고정해 총 높이 1080 내 수렴
# - Plotly 마진/폰트/레전드 축소, Streamlit 기본 여백/푸터/메뉴 숨김
# - 동일 기능/구성 유지 (KPI + ①~⑦ 차트)
# =================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date
import os
import warnings

warnings.filterwarnings('ignore')

# ---- 페이지 설정 ----
st.set_page_config(
    page_title="AI FMS — PV Ops: Performance Monitoring & Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- 커스텀 CSS (다크 테마 & 1080px 맞춤 압축) ----
st.markdown("""
<style>
    /* 전체 컨테이너 여백 최소화 (상단/하단) */
    .block-container {padding-top: 6px !important; padding-bottom: 6px !important;}
    
    /* 기본 메뉴/푸터 숨김으로 높이 절약 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {background-color: #0E1117;}

    /* 카드, KPI, 타이틀 등 컴팩트 스타일 */
    h1, h2, h3, h4 {margin: 0 0 6px 0 !important; padding: 0 !important;}
    .pv-card {
        background: linear-gradient(135deg, #1e2329 0%, #181b20 100%);
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.25);
    }
    .pv-card h3 {color: #E6EDF3; font-size: 14px; font-weight: 700; margin-bottom: 8px; letter-spacing: .2px;}

    .kpi-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-top: 4px;
    }
    .kpi-item {
        background: rgba(255,255,255,0.02);
        padding: 8px;
        border-radius: 6px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .kpi-label {color: #8aa2b2; font-size: 10px; margin-bottom: 2px;}
    .kpi-value {color: #E6EDF3; font-size: 16px; font-weight: 700;}

    .pill {
        display: inline-block; padding: 2px 8px; border-radius: 999px;
        font-size: 11px; font-weight: 700; margin-left: 6px;
    }
    .pill.normal {background: #12202b; color: #7bdff2; border: 1px solid #1e3442;}
    .pill.bad {background: #2b1212; color: #ff7b7b; border: 1px solid #3a1a1a;}

    /* 사이드바 컴팩트 */
    section[data-testid="stSidebar"] .block-container {padding-top: 8px !important;}
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {margin-bottom: 6px !important;}
</style>
""", unsafe_allow_html=True)


# ---- 헬퍼 함수들 ----
def time_labels():
    return [f"{(i * 15) // 60:02d}:{(i * 15) % 60:02d}" for i in range(96)]


def nema_unbalance(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    avg = (a + b + c) / 3
    dev = np.maximum(np.abs(a - avg), np.maximum(np.abs(b - avg), np.abs(c - avg)))
    return 100 * dev / np.maximum(np.abs(avg), 1e-9)


def pick_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None

def create_plotly_theme():
    return {
        'template': 'plotly_dark',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'family': 'Inter, Arial', 'size': 11, 'color': '#F0F6FF'},
        'margin': {'l': 40, 'r': 8, 'b': 28, 't': 22},
        # 'legend': {'font': {'size': 10}, 'itemclick': 'toggleothers', 'itemdoubleclick': 'toggle'},  # <-- 삭제
    }


def load_and_process_data(date_str):
    data_dirs = ["TestData", "/mnt/data", "data", "."]
    filename = f"{date_str}.csv"

    df = None
    for data_dir in data_dirs:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                break
            except Exception as e:
                st.error(f"Error reading {filepath}: {str(e)}")
                continue

    if df is None:
        st.error(f"CSV file not found for date {date_str} in directories: {data_dirs}")
        return None

    col_mappings = {
        'time': ['collect_date', 'ts', 'time', 'datetime'],
        'slot': ['slot', 'idx', 'index'],
        'DC': ['DC', 'Inv_PV_kW', 'Inv_PV_KW'],
        'AC': ['AC', 'Inv_kW', 'Inv_KW'],
        'DC_hat': ['DC_hat', 'Recon_DC', 'DC_recon', 'DC_pred'],
        'AC_hat': ['AC_hat', 'Recon_AC', 'AC_recon', 'AC_pred'],
        'RS': ['RS', 'Inv_RS_V'],
        'ST': ['ST', 'Inv_ST_V'],
        'TR': ['TR', 'Inv_TR_V'],
        'R': ['R', 'Inv_R_A'],
        'S': ['S', 'Inv_S_A'],
        'T': ['T', 'Inv_T_A'],
        'Hz': ['Hz', 'Inv_Hz'],
        'daytime': ['daytime', 'is_day', 'day_mask'],
        'error': ['error', 'mse'],
        'tau': ['tau', 'threshold'],
        'decision': ['decision', 'y_pred', 'label']
    }

    actual_cols = {key: pick_col(df, cand) for key, cand in col_mappings.items()}

    required = ['time', 'DC', 'AC']
    missing = [c for c in required if actual_cols[c] is None]
    if missing:
        st.error(f"Required columns missing: {missing}")
        return None

    processed = {}
    try:
        processed['ts'] = pd.to_datetime(df[actual_cols['time']])
    except:
        processed['ts'] = pd.date_range(start='2025-01-01', periods=len(df), freq='15min')

    numeric = ['DC', 'AC', 'DC_hat', 'AC_hat', 'RS', 'ST', 'TR', 'R', 'S', 'T', 'Hz']
    for col in numeric:
        if actual_cols[col] is not None:
            processed[col] = pd.to_numeric(df[actual_cols[col]], errors='coerce')
        else:
            processed[col] = np.full(len(df), np.nan)

    if actual_cols['slot'] is not None:
        processed['slot'] = pd.to_numeric(df[actual_cols['slot']], errors='coerce')
    else:
        processed['slot'] = np.arange(1, len(df) + 1)

    if actual_cols['daytime'] is not None:
        processed['daytime'] = df[actual_cols['daytime']].astype(bool)
    else:
        processed['daytime'] = processed['DC'] > 0

    summary = {}
    for col in ['error', 'tau', 'decision']:
        if actual_cols[col] is not None:
            vals = df[actual_cols[col]].dropna().unique()
            summary[col] = vals[0] if len(vals) > 0 else np.nan
        else:
            summary[col] = np.nan

    if np.isnan(summary['decision']):
        if not (np.isnan(summary['error']) or np.isnan(summary['tau'])):
            summary['decision'] = 1 if summary['error'] > summary['tau'] else 0
        else:
            summary['decision'] = 0

    return {'series': processed, 'summary': summary, 'date': date_str}


# ---- 메인 앱 ----
def main():
    st.markdown("<div style='height:35px'></div>", unsafe_allow_html=True)  # 얇은 상단 여백
    st.markdown("### AI FMS — PV Ops: Performance Monitoring & Anomaly Detection")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # 사이드바 (컴팩트)
    with st.sidebar:
        st.markdown("### Controls")
        selected_date = st.date_input(
            "Target day",
            value=date(2025, 8, 31),
            min_value=date(2025, 8, 1),
            max_value=date(2025, 8, 31)
        )
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()

    @st.cache_data
    def get_data(date_str):
        return load_and_process_data(date_str)

    data = get_data(selected_date.strftime("%Y-%m-%d"))
    if data is None:
        st.error("데이터를 로드할 수 없습니다.")
        return

    series = data['series']
    summary = data['summary']

    # 계산
    efficiency = np.where(series['DC'] > 1.0, 100 * series['AC'] / series['DC'], 0)
    voltage_unbalance = nema_unbalance(series['RS'], series['ST'], series['TR'])
    current_unbalance = nema_unbalance(series['R'], series['S'], series['T'])

    energy_dc = np.nansum(series['DC']) * 0.25
    energy_ac = np.nansum(series['AC']) * 0.25

    daytime_mask = np.array(series['daytime'])
    daytime_eff = efficiency[daytime_mask]
    avg_efficiency = np.nanmean(daytime_eff) if len(daytime_eff) > 0 else np.nan

    max_vu = np.nanmax(voltage_unbalance) if not np.all(np.isnan(voltage_unbalance)) else np.nan
    max_cu = np.nanmax(current_unbalance) if not np.all(np.isnan(current_unbalance)) else np.nan

    is_anomaly = summary['decision'] == 1
    status_class = "bad" if is_anomaly else "normal"
    status_text = "Anomaly" if is_anomaly else "Normal"

    theme = create_plotly_theme()

    # ====== 레이아웃 (총 2,300 + 3*165 + 3*185 ~= 1,055 내) ======
    # 1행: KPI 요약 + 메인 차트
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("##### KPI — Daily Summary")
        error_str = f"{summary['error']:.5f}" if not np.isnan(summary['error']) else "—"
        tau_str = f"{summary['tau']:.5f}" if not np.isnan(summary['tau']) else "—"
        eff_str = f"{avg_efficiency:.1f}" if not np.isnan(avg_efficiency) else "—"
        vu_str = f"{max_vu:.2f}" if not np.isnan(max_vu) else "—"
        cu_str = f"{max_cu:.2f}" if not np.isnan(max_cu) else "—"

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-item"><div class="kpi-label">Date</div><div class="kpi-value">{data['date']}</div></div>
            <div class="kpi-item"><div class="kpi-label">Decision</div><div class="kpi-value"><span class="pill {status_class}">{status_text}</span></div></div>
            <div class="kpi-item"><div class="kpi-label">Error</div><div class="kpi-value">{error_str}</div></div>
            <div class="kpi-item"><div class="kpi-label">Threshold τ</div><div class="kpi-value">{tau_str}</div></div>
            <div class="kpi-item"><div class="kpi-label">Avg Efficiency (%)</div><div class="kpi-value">{eff_str}</div></div>
            <div class="kpi-item"><div class="kpi-label">Max VU / CU (%)</div><div class="kpi-value">{vu_str} / {cu_str}</div></div>
            <div class="kpi-item"><div class="kpi-label">Energy DC (kWh)</div><div class="kpi-value">{energy_dc:.1f}</div></div>
            <div class="kpi-item"><div class="kpi-label">Energy AC (kWh)</div><div class="kpi-value">{energy_ac:.1f}</div></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("##### ① DC / AC Power")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series['slot'], y=series['DC'], mode='lines', name='DC',
            line=dict(color='#22D3EE', width=2.5),
            fill='tozeroy', fillcolor='rgba(34,211,238,0.18)',
            hovertemplate='Slot: %{x}<br>DC: %{y:.2f} kW<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=series['slot'], y=series['AC'], mode='lines', name='AC',
            line=dict(color='#A78BFA', width=2.5),
            fill='tozeroy', fillcolor='rgba(167,139,250,0.18)',
            hovertemplate='Slot: %{x}<br>AC: %{y:.2f} kW<extra></extra>'
        ))
        fig.update_layout(**theme, height=300, xaxis_title='Time Slot', yaxis_title='Power (kW)',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    # 2행: 효율/전압불균형/전류불균형 (컴팩트)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### ② Inverter Efficiency")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series['slot'], y=efficiency, mode='lines+markers', name='Efficiency (%)',
            line=dict(color='#34D399', width=2.2), marker=dict(size=3),
            hovertemplate='Slot: %{x}<br>Eff: %{y:.2f}%<extra></extra>'
        ))
        fig.update_layout(**theme, height=225, xaxis_title='Time Slot', yaxis_title='Efficiency (%)',
                          yaxis=dict(range=[0, 120]), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### ③ Voltage Unbalance")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series['slot'], y=voltage_unbalance, mode='lines+markers', name='Voltage Unbalance',
            line=dict(color='#F59E0B', width=2.2), marker=dict(size=3),
            hovertemplate='Slot: %{x}<br>VU: %{y:.2f}%<extra></extra>'
        ))
        fig.add_hline(y=2.0, line_dash="dash", line_color="rgba(255,255,255,0.6)", annotation_text="Guide 2%")
        fig.update_layout(**theme, height=225, xaxis_title='Time Slot', yaxis_title='Voltage Unbalance (%)',
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("##### ④ Current Unbalance")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series['slot'], y=current_unbalance, mode='lines+markers', name='Current Unbalance',
            line=dict(color='#F472B6', width=2.2), marker=dict(size=3),
            hovertemplate='Slot: %{x}<br>CU: %{y:.2f}%<extra></extra>'
        ))
        fig.add_hline(y=10.0, line_dash="dash", line_color="rgba(255,255,255,0.6)", annotation_text="Guide 10%")
        fig.update_layout(**theme, height=225, xaxis_title='Time Slot', yaxis_title='Current Unbalance (%)',
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # 3행: 이상탐지 3개 (두 개 라인 + 게이지)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### ⑤ PV Array (DC) Anomaly")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series['slot'], y=series['DC'], mode='lines+markers', name='DC',
            line=dict(color='#60A5FA', width=2.2), marker=dict(size=3),
            hovertemplate='Slot: %{x}<br>DC: %{y:.2f} kW<extra></extra>'
        ))
        if not np.all(np.isnan(series['DC_hat'])):
            fig.add_trace(go.Scatter(
                x=series['slot'], y=series['DC_hat'], mode='lines', name='DC Reconstructed',
                line=dict(color='#93C5FD', width=2, dash='dot'),
                hovertemplate='Slot: %{x}<br>DC Recon: %{y:.2f} kW<extra></extra>'
            ))
        fig.update_layout(**theme, height=225, xaxis_title='Time Slot', yaxis_title='Power (kW)',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### ⑥ Inverter (AC) Anomaly")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series['slot'], y=series['AC'], mode='lines+markers', name='AC',
            line=dict(color='#60A5FA', width=2.2), marker=dict(size=3),
            hovertemplate='Slot: %{x}<br>AC: %{y:.2f} kW<extra></extra>'
        ))
        if not np.all(np.isnan(series['AC_hat'])):
            fig.add_trace(go.Scatter(
                x=series['slot'], y=series['AC_hat'], mode='lines', name='AC Reconstructed',
                line=dict(color='#93C5FD', width=2, dash='dot'),
                hovertemplate='Slot: %{x}<br>AC Recon: %{y:.2f} kW<extra></extra>'
            ))
        fig.update_layout(**theme, height=225, xaxis_title='Time Slot', yaxis_title='Power (kW)',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("##### ⑦ Anomaly Indicator")
        error_val = summary['error'] if not np.isnan(summary['error']) else 0
        tau_val = summary['tau'] if not np.isnan(summary['tau']) else 1
        max_val = max(error_val, tau_val) * 1.4 if max(error_val, tau_val) > 0 else 1
        gauge_color = "#FF6B6B" if is_anomaly else "#34D399"

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=error_val,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Result: {status_text}"},
            delta={'reference': tau_val},
            gauge={
                'axis': {'range': [None, max_val]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, tau_val], 'color': "rgba(52,211,153,0.20)"},
                    {'range': [tau_val, max_val], 'color': "rgba(255,107,107,0.20)"}
                ],
                'threshold': {
                    'line': {'color': "rgba(255,255,255,0.85)", 'width': 3},
                    'thickness': 0.7,
                    'value': tau_val
                }
            }
        ))
        fig.update_layout(**theme, height=225)
        fig.update_layout(margin={'l': 20, 'r': 8, 'b': 16, 't': 16})
        st.plotly_chart(fig, use_container_width=True)

    # 하단 정보(아주 얇게)
    st.markdown(
        "<div style='text-align:center; color:#8aa2b2; font-size:11px; margin-top:2px;'>"
        "AI FMS — PV Operations Dashboard | Streamlit + Plotly"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
