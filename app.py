# app.py — CSV 기반 PV 대시보드 (Streamlit, 1920x1080 1-Page Fit)
# =================================================================
# 변경 요약
# - ①~⑥: 자동재생(loop), x축 제목 "Time (15-min)" + 시간 라벨(HH:MM)
# - ⑦: decision==1에서만 외곽 링 깜박임(iframe CSS), 사이드바 열닫아도 유지, 잘림 방지
# - ⑦: 중앙 숫자 크기 축소
# - ✅ 사이드바 CSV 업로드 지원: 업로드되면 그 CSV로 즉시 시각화, 없으면 날짜 기반 CSV 로드
# =================================================================

import os
import json
import uuid
import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings('ignore')

# ---- 페이지 설정 ----
st.set_page_config(
    page_title="AI FMS — PV Ops: Performance Monitoring & Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- 전역 CSS ----
st.markdown("""
<style>
  .block-container {padding-top: 6px !important; padding-bottom: 6px !important;}
  #MainMenu {visibility: hidden;} footer {visibility: hidden;}
  .stApp {background-color: #0E1117;}
  h1, h2, h3, h4 {margin: 0 0 6px 0 !important; padding: 0 !important;}
  .kpi-grid {display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 4px;}
  .kpi-item {background: rgba(255,255,255,0.02); padding: 8px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);}
  .kpi-label {color: #8aa2b2; font-size: 10px; margin-bottom: 2px;}
  .kpi-value {color: #E6EDF3; font-size: 16px; font-weight: 700;}
  .pill {display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; margin-left: 6px;}
  .pill.normal {background: #12202b; color: #7bdff2; border: 1px solid #1e3442;}
  .pill.bad {background: #2b1212; color: #ff7b7b; border: 1px solid #3a1a1a;}
  section[data-testid="stSidebar"] .block-container {padding-top: 8px !important;}
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {margin-bottom: 6px !important;}
</style>
""", unsafe_allow_html=True)

# ---- 유틸/헬퍼 ----
FPS = 30
FRAME_STEP = 1
FRAME_MS = max(1, int(1000 / FPS))

def fmt_float(x, digits=5):
    try:
        xf = float(x)
        if np.isnan(xf): return "—"
        return f"{xf:.{digits}f}"
    except Exception:
        return "—"

def upd(fig, theme: dict, **overrides):
    safe = {k: v for k, v in theme.items() if k != "margin"}
    safe.update(overrides)
    fig.update_layout(**safe)

def strip_controls(fig):
    fig.update_layout(updatemenus=[])

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
    }

def _fixed_axis_ranges(x_vals, y_vals, pad_y=0.05):
    import math
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    if math.isclose(ymin, ymax):
        ymin -= 1.0; ymax += 1.0
    span = ymax - ymin
    ymin -= span * pad_y; ymax += span * pad_y
    return (xmin, xmax), (ymin, ymax)

# ---- x축 시간 라벨 생성 & 적용 ----
def _time_ticks_from_series(series, every_slots=4):
    """
    x축은 슬롯(1..N), 라벨은 HH:MM.
    every_slots=4 → 15분 데이터에서 1시간 간격 라벨.
    """
    try:
        N = int(np.nanmax(series.get('slot')))
        if N <= 0: N = len(series.get('ts', [])) or 96
    except Exception:
        N = len(series.get('ts', [])) or 96

    tickvals = list(range(1, N + 1, every_slots))
    if tickvals[-1] != N:
        tickvals.append(N)

    ts = pd.to_datetime(series.get('ts')) if 'ts' in series else None
    ticktext = []
    for v in tickvals:
        if ts is not None and len(ts) >= v:
            ticktext.append(pd.to_datetime(ts[v-1]).strftime("%H:%M"))
        else:
            minutes = (v - 1) * 15
            hh = (minutes // 60) % 24
            mm = minutes % 60
            ticktext.append(f"{hh:02d}:{mm:02d}")
    return tickvals, ticktext

def apply_time_axis(fig, series, title="Time (15-min)"):
    """①~⑥ 차트에 x축 제목/시간 라벨 적용(기존 range 보존)"""
    tickvals, ticktext = _time_ticks_from_series(series, every_slots=4)
    xdict = dict(title=title, tickmode="array", tickvals=tickvals, ticktext=ticktext)
    try:
        xr = fig.layout.xaxis.range
        if xr is not None:
            xdict["range"] = list(xr)
    except Exception:
        pass
    fig.update_layout(xaxis=xdict)

# ---- 애니메이션 Figure 생성기 ----
def animated_single_line(x, y, name, color, theme, height, x_title, y_title,
                         yaxis=None, showlegend=False, step=FRAME_STEP):
    x = np.array(x); y = np.array(y)
    idx = np.arange(0, len(x), step)
    x_s, y_s = x[idx], y[idx]
    base = go.Scatter(x=[x_s[0]], y=[y_s[0]], mode='lines+markers', name=name,
                      line=dict(color=color, width=2.2), marker=dict(size=3))
    frames = [go.Frame(name=str(i), data=[go.Scatter(x=x_s[:i+1], y=y_s[:i+1])])
              for i in range(1, len(x_s))]
    fig = go.Figure(data=[base], frames=frames)
    (xr, yr) = _fixed_axis_ranges(x_s, y_s)
    upd(fig, theme, height=height, xaxis_title=x_title, yaxis_title=y_title,
        showlegend=showlegend, xaxis=dict(range=list(xr)))
    if yaxis: fig.update_layout(yaxis=yaxis)
    else:     fig.update_layout(yaxis=dict(range=list(yr)))
    strip_controls(fig)
    return fig

def animated_two_lines(x, y1, y2, name1, name2, color1, color2,
                       theme, height, x_title, y_title, showlegend=True, step=FRAME_STEP,
                       fill1=None, fill2=None):
    x = np.array(x); y1 = np.array(y1); y2 = np.array(y2)
    idx = np.arange(0, len(x), step)
    x_s, y1_s, y2_s = x[idx], y1[idx], y2[idx]
    t1 = go.Scatter(x=[x_s[0]], y=[y1_s[0]], mode='lines', name=name1,
                    line=dict(color=color1, width=2.5),
                    fill=fill1 or None,
                    fillcolor='rgba(34,211,238,0.18)' if fill1 else None)
    t2 = go.Scatter(x=[x_s[0]], y=[y2_s[0]], mode='lines', name=name2,
                    line=dict(color=color2, width=2.5),
                    fill=fill2 or None,
                    fillcolor='rgba(167,139,250,0.18)' if fill2 else None)
    frames = [go.Frame(name=str(i),
                       data=[go.Scatter(x=x_s[:i+1], y=y1_s[:i+1]),
                             go.Scatter(x=x_s[:i+1], y=y2_s[:i+1])])
              for i in range(1, len(x_s))]
    (_, _), (ymin, ymax) = _fixed_axis_ranges(np.r_[x_s, x_s], np.r_[y1_s, y2_s])
    (xmin, xmax), _ = _fixed_axis_ranges(x_s, y1_s)
    fig = go.Figure(data=[t1, t2], frames=frames)
    upd(fig, theme, height=height, xaxis_title=x_title, yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(range=[xmin, xmax]), yaxis=dict(range=[ymin, ymax]))
    strip_controls(fig)
    return fig

def animated_with_recon(x, y, yhat, name, name_hat, color, color_hat,
                        theme, height, x_title, y_title, step=FRAME_STEP):
    if yhat is None or np.all(np.isnan(yhat)):
        return animated_single_line(x, y, name, color, theme, height, x_title, y_title, step=step)
    return animated_two_lines(x, y, yhat, name, name_hat, color, color_hat, theme, height,
                              x_title, y_title, showlegend=True, step=step)

# ---- 자동재생/루프 렌더러 ----
def render_plotly_autoplay(fig, height, fps=FPS):
    strip_controls(fig)
    fig_json = json.loads(pio.to_json(fig))
    div_id = f"plotly-{uuid.uuid4().hex}"
    base = pio.to_html(go.Figure(), include_plotlyjs='inline', full_html=False, div_id=div_id)
    autoplay_js = f"""
    <script>
      (function() {{
        const fig = {json.dumps(fig_json)};
        var gd = document.getElementById("{div_id}");
        Plotly.react(gd, fig.data, fig.layout, {{responsive: true}});
        if (fig.frames && fig.frames.length > 0) {{
          Plotly.addFrames(gd, fig.frames);
          let i = 0;
          const N = fig.frames.length;
          const dur = {FRAME_MS};
          setInterval(function() {{
            const name = String(i);
            Plotly.animate(gd, [name], {{
              mode: "immediate",
              frame: {{duration: dur, redraw: false}},
              transition: {{duration: 0}}
            }});
            i = (i + 1) % N;
          }}, dur);
        }}
      }})();
    </script>
    """
    components.html(base + autoplay_js, height=height+12)

# ---- 정적 렌더러 (⑦ 깜박임용, 잘림 방지) ----
def render_plotly_static(fig, height, blink=False):
    """
    정적 Plotly를 iframe에 렌더링.
    - blink=True: 외곽 링 애니메이션 (콘텐츠 가리지 않음)
    - 초기 0px 문제 방지: min-height/overflow
    - rerun마다 rev 주석 삽입 → remount 보장
    """
    fig_json = json.loads(pio.to_json(fig))
    div_id = f"plotly-{uuid.uuid4().hex}"
    base = pio.to_html(go.Figure(), include_plotlyjs='inline', full_html=False, div_id=div_id)

    blink_style = """
    <style>
      html, body { margin: 0; background: transparent; }
      .anomaly-blink{
        position: relative;
        border-radius: 12px;
        padding: 8px;
        border: 1px solid rgba(255,107,107,.55);
        background: transparent;
        overflow: visible;
        will-change: box-shadow;
        animation: ring-red 1.05s ease-out infinite;
      }
      @keyframes ring-red{
        0%   { box-shadow:0 0 0 0 rgba(255,56,56,0.55); }
        70%  { box-shadow:0 0 0 12px rgba(255,56,56,0.00); }
        100% { box-shadow:0 0 0 0 rgba(255,56,56,0.00); }
      }
    </style>
    """

    js = f"""
    <script>
      (function(){{
        const fig = {json.dumps(fig_json)};
        var gd = document.getElementById("{div_id}");
        Plotly.react(gd, fig.data, fig.layout, {{responsive: true}});
      }})();
    </script>
    """

    rev = st.session_state.get('mount_rev', 0)
    content = base + js
    if blink:
        content = blink_style + f"<div class='anomaly-blink' style='min-height:{height}px'>" + content + "</div>"
    html = f"<!-- rev:{rev} -->" + content
    components.html(html, height=height + 24)

# ---- 공통 파서 (DataFrame -> 내부 포맷) ----
def parse_dataframe(df: pd.DataFrame, date_label: str):
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

    actual = {key: pick_col(df, cand) for key, cand in col_mappings.items()}
    for req in ['time', 'DC', 'AC']:
        if actual[req] is None:
            st.error(f"Required columns missing: {req}")
            return None

    try:
        ts = pd.to_datetime(df[actual['time']])
    except Exception:
        ts = pd.date_range(start='2025-01-01', periods=len(df), freq='15min')

    series = {'ts': ts}
    for k in ['DC', 'AC', 'DC_hat', 'AC_hat', 'RS', 'ST', 'TR', 'R', 'S', 'T', 'Hz']:
        series[k] = (pd.to_numeric(df[actual[k]], errors='coerce')
                     if actual[k] is not None else np.full(len(df), np.nan))

    series['slot'] = (pd.to_numeric(df[actual['slot']], errors='coerce')
                      if actual['slot'] else np.arange(1, len(df) + 1))
    series['daytime'] = (df[actual['daytime']].astype(bool)
                         if actual['daytime'] else (series['DC'] > 0))

    summary = {}
    for col in ['error', 'tau', 'decision']:
        if actual[col] is not None:
            vals = df[actual[col]].dropna().unique()
            summary[col] = vals[0] if len(vals) > 0 else np.nan
        else:
            summary[col] = np.nan

    if np.isnan(summary.get('decision', np.nan)):
        if not (np.isnan(summary.get('error', np.nan)) or np.isnan(summary.get('tau', np.nan))):
            summary['decision'] = 1 if summary['error'] > summary['tau'] else 0
        else:
            summary['decision'] = 0

    return {'series': series, 'summary': summary, 'date': date_label}

# ---- 날짜 기반 CSV 로드 ----
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
                st.error(f"Error reading {filepath}: {str(e)}"); continue
    if df is None:
        st.error(f"CSV file not found for date {date_str} in directories: {data_dirs}")
        return None
    return parse_dataframe(df, date_str)

# ---- 업로드 CSV 로드 ----
def load_and_process_uploaded(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        return None
    name = getattr(uploaded_file, "name", "Uploaded CSV")
    return parse_dataframe(df, name)

# ---- 메인 앱 ----
def main():
    # rerun 카운터(iframe remount 유도)
    if 'mount_rev' not in st.session_state:
        st.session_state['mount_rev'] = 0
    st.session_state['mount_rev'] += 1

    st.markdown("<div style='height:35px'></div>", unsafe_allow_html=True)
    st.markdown("### AI FMS — PV Ops: Performance Monitoring & Anomaly Detection")
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Controls")
        # ✅ CSV 업로더 (업로드되면 업로드 데이터 사용)
        uploaded = st.file_uploader("Upload CSV", type=["csv"], help="CSV를 업로드하면 그 데이터를 즉시 시각화합니다.")
        st.markdown("---")
        selected_date = st.date_input(
            "Target day (fallback when no upload)", value=date(2025, 8, 31),
            min_value=date(2025, 8, 1), max_value=date(2025, 8, 31)
        )
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()

    @st.cache_data
    def get_data_by_date(date_str):
        return load_and_process_data(date_str)

    # ✅ 데이터 선택 로직: 업로드 > 날짜 파일
    if uploaded is not None:
        data = load_and_process_uploaded(uploaded)
    else:
        data = get_data_by_date(selected_date.strftime("%Y-%m-%d"))

    if data is None:
        st.error("데이터를 로드할 수 없습니다."); return

    s = data['series']; summary = data['summary']

    # 계산
    efficiency = np.where(s['DC'] > 1.0, 100 * s['AC'] / s['DC'], 0)
    voltage_unbalance = nema_unbalance(s['RS'], s['ST'], s['TR'])
    current_unbalance = nema_unbalance(s['R'], s['S'], s['T'])
    energy_dc = np.nansum(s['DC']) * 0.25
    energy_ac = np.nansum(s['AC']) * 0.25
    daytime_eff = efficiency[np.array(s['daytime'])]
    avg_efficiency = np.nanmean(daytime_eff) if len(daytime_eff) > 0 else np.nan
    max_vu = np.nanmax(voltage_unbalance) if not np.all(np.isnan(voltage_unbalance)) else np.nan
    max_cu = np.nanmax(current_unbalance) if not np.all(np.isnan(current_unbalance)) else np.nan
    is_anomaly = summary['decision'] == 1
    status_class = "bad" if is_anomaly else "normal"
    status_text = "Anomaly" if is_anomaly else "Normal"
    theme = create_plotly_theme()

    # ====== 1행: KPI + ① DC/AC ======
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("##### KPI — Daily Summary")
        error_str = fmt_float(summary["error"], 5)
        tau_str   = fmt_float(summary["tau"], 5)
        eff_str   = f"{avg_efficiency:.1f}" if not np.isnan(avg_efficiency) else "—"
        vu_str    = f"{max_vu:.2f}" if not np.isnan(max_vu) else "—"
        cu_str    = f"{max_cu:.2f}" if not np.isnan(max_cu) else "—"

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

    with c2:
        st.markdown("##### ① DC / AC Power")
        fig = animated_two_lines(
            s['slot'], s['DC'], s['AC'],
            "DC", "AC", "#22D3EE", "#A78BFA",
            theme, 300, "Time (15-min)", "Power (kW)",
            step=FRAME_STEP, fill1='tozeroy', fill2='tozeroy'
        )
        apply_time_axis(fig, s, title="Time (15-min)")
        render_plotly_autoplay(fig, height=300, fps=FPS)

    # ====== 2행: ②~④ ======
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### ② Inverter Efficiency")
        fig = animated_single_line(
            s['slot'], np.array(efficiency),
            "Efficiency (%)", "#34D399", theme, 225,
            "Time (15-min)", "Efficiency (%)",
            yaxis=dict(range=[0, 120]), showlegend=False, step=FRAME_STEP
        )
        apply_time_axis(fig, s, title="Time (15-min)")
        render_plotly_autoplay(fig, 225, fps=FPS)

    with c2:
        st.markdown("##### ③ Voltage Unbalance")
        fig = animated_single_line(
            s['slot'], np.array(voltage_unbalance),
            "Voltage Unbalance", "#F59E0B", theme, 225,
            "Time (15-min)", "Voltage Unbalance (%)",
            showlegend=False, step=FRAME_STEP
        )
        fig.add_hline(y=2.0, line_dash="dash", line_color="rgba(255,255,255,0.6)", annotation_text="Guide 2%")
        apply_time_axis(fig, s, title="Time (15-min)")
        render_plotly_autoplay(fig, 225, fps=FPS)

    with c3:
        st.markdown("##### ④ Current Unbalance")
        fig = animated_single_line(
            s['slot'], np.array(current_unbalance),
            "Current Unbalance", "#F472B6", theme, 225,
            "Time (15-min)", "Current Unbalance (%)",
            showlegend=False, step=FRAME_STEP
        )
        fig.add_hline(y=10.0, line_dash="dash", line_color="rgba(255,255,255,0.6)", annotation_text="Guide 10%")
        apply_time_axis(fig, s, title="Time (15-min)")
        render_plotly_autoplay(fig, 225, fps=FPS)

    # ====== 3행: ⑤~⑥ (애니메이션), ⑦ 게이지 ======
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### ⑤ PV Array (DC) Anomaly")
        fig = animated_with_recon(
            s['slot'], s['DC'], s.get('DC_hat'),
            "DC", "DC Reconstructed", "#60A5FA", "#93C5FD",
            theme, 225, "Time (15-min)", "Power (kW)", step=FRAME_STEP
        )
        apply_time_axis(fig, s, title="Time (15-min)")
        render_plotly_autoplay(fig, 225, fps=FPS)

    with c2:
        st.markdown("##### ⑥ Inverter (AC) Anomaly")
        fig = animated_with_recon(
            s['slot'], s['AC'], s.get('AC_hat'),
            "AC", "AC Reconstructed", "#60A5FA", "#93C5FD",
            theme, 225, "Time (15-min)", "Power (kW)", step=FRAME_STEP
        )
        apply_time_axis(fig, s, title="Time (15-min)")
        render_plotly_autoplay(fig, 225, fps=FPS)

    with c3:
        st.markdown("##### ⑦ Anomaly Indicator")
        error_val = summary['error'] if not np.isnan(summary['error']) else 0
        tau_val = summary['tau'] if not np.isnan(summary['tau']) else 1
        gauge_h = 225
        max_val = max(error_val, tau_val) * 1.4 if max(error_val, tau_val) > 0 else 1
        gauge_color = "#FF6B6B" if is_anomaly else "#34D399"

        num_size   = int(gauge_h * 0.22)
        delta_size = int(gauge_h * 0.08)
        title_size = int(gauge_h * 0.09)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=error_val,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Result: {status_text}", 'font': {'size': title_size}},
            number={'font': {'size': num_size}},
            delta={'reference': tau_val, 'font': {'size': delta_size}},
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
        upd(fig, theme, height=gauge_h, margin={'l': 20, 'r': 8, 'b': 16, 't': 16})
        render_plotly_static(fig, height=gauge_h, blink=is_anomaly)

    # 하단 정보
    st.markdown(
        "<div style='text-align:center; color:#8aa2b2; font-size:11px; margin-top:2px;'>"
        "AI FMS — PV Operations Dashboard | Streamlit + Plotly"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
