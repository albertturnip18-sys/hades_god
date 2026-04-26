# ============================================================
#  STREAMLIT APP — PEMODELAN ODE PERTUMBUHAN PENDUDUK
#  Kota Tual, Maluku — 2020–2030
#  Model: Eksponensial & Logistik | ODE Numerik & Analitik
#  Referensi: Armin & Michael G.K. Remetwa (JIMAT, Vol.6 No.1, 2025)
# ============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="ODE Populasi Kota Tual",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark github-inspired background */
.stApp {
    background-color: #0D1117;
    color: #E6EDF3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}
[data-testid="stSidebar"] * {
    color: #E6EDF3 !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 12px;
}
[data-testid="metric-container"] label {
    color: #8B949E !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #58A6FF !important;
    font-family: 'Space Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #3FB950 !important;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: #E6EDF3 !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    color: #8B949E !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58A6FF !important;
    border-bottom-color: #58A6FF !important;
}

/* Sliders */
[data-testid="stSlider"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    color: #8B949E !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #30363D !important;
}

/* Divider */
hr {
    border-color: #30363D !important;
}

/* Info/success boxes */
.stAlert {
    border-radius: 6px !important;
    border: 1px solid #30363D !important;
    background-color: #161B22 !important;
}

.hero-banner {
    background: linear-gradient(135deg, #161B22 0%, #1C2128 50%, #161B22 100%);
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #58A6FF, #3FB950, #BC8CFF);
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #E6EDF3;
    margin: 0 0 6px 0;
}
.hero-sub {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    color: #8B949E;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: #21262D;
    border: 1px solid #30363D;
    border-radius: 20px;
    padding: 3px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #58A6FF;
    margin-top: 10px;
    margin-right: 6px;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #58A6FF;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-bottom: 1px solid #30363D;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB STYLE ─────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0D1117',
    'axes.facecolor': '#161B22',
    'axes.edgecolor': '#30363D',
    'axes.labelcolor': '#E6EDF3',
    'xtick.color': '#8B949E',
    'ytick.color': '#8B949E',
    'text.color': '#E6EDF3',
    'grid.color': '#21262D',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8.5,
    'legend.facecolor': '#161B22',
    'legend.edgecolor': '#30363D',
    'figure.dpi': 120,
})

CYAN   = '#58A6FF'
GREEN  = '#3FB950'
ORANGE = '#F78166'
PURPLE = '#BC8CFF'
YELLOW = '#E3B341'
RED    = '#FF7B72'
TEAL   = '#39D353'

# ── DATA & PARAMETER ─────────────────────────────────────────
tahun_historis  = np.array([2020, 2021, 2022, 2023, 2024], dtype=float)
populasi_aktual = np.array([88280, 90322, 93145, 91572, 92744], dtype=float)

P0_hist  = populasi_aktual[0]
P0_pred  = populasi_aktual[-1]
t_fit    = tahun_historis[-1] - tahun_historis[0]

k_analitik = (1 / t_fit) * np.log(P0_pred / P0_hist)
K_default  = 150_000.0

tahun_prediksi = np.array([2026, 2027, 2028, 2029, 2030])
t_pred_rel     = tahun_prediksi - 2024

# ── ODE FUNCTIONS ────────────────────────────────────────────
def ode_eksponensial(P, t, k):
    return k * P

def ode_logistik(P, t, k, K):
    return k * P * (1.0 - P / K)

def solusi_eksponensial(t, P0, k):
    return P0 * np.exp(k * t)

def solusi_logistik(t, P0, k, K):
    return K / (1.0 + ((K - P0) / P0) * np.exp(-k * t))

def hitung_mape(aktual, prediksi):
    return np.mean(np.abs((aktual - prediksi) / aktual)) * 100

def hitung_rmse(aktual, prediksi):
    return np.sqrt(np.mean((aktual - prediksi)**2))

def hitung_r2(aktual, prediksi):
    ss_res = np.sum((aktual - prediksi)**2)
    ss_tot = np.sum((aktual - np.mean(aktual))**2)
    return 1 - ss_res / ss_tot

# ── CURVE FITTING ────────────────────────────────────────────
@st.cache_data
def fit_models():
    t_rel = tahun_historis - tahun_historis[0]
    popt_exp, _ = curve_fit(
        lambda t, k: solusi_eksponensial(t, P0_hist, k),
        t_rel, populasi_aktual,
        p0=[0.012], bounds=(0, 0.2)
    )
    popt_log, _ = curve_fit(
        lambda t, k, K: solusi_logistik(t, P0_hist, k, K),
        t_rel, populasi_aktual,
        p0=[0.05, 150000], bounds=([0, 93000], [1, 500000]),
        maxfev=10000
    )
    return popt_exp[0], popt_log[0], popt_log[1]

k_fit_exp, k_fit_log, K_fit_log = fit_models()

t_rel_hist       = tahun_historis - tahun_historis[0]
pred_exp_di_data = solusi_eksponensial(t_rel_hist, P0_hist, k_fit_exp)
pred_log_di_data = solusi_logistik(t_rel_hist, P0_hist, k_fit_log, K_fit_log)

mape_exp = hitung_mape(populasi_aktual, pred_exp_di_data)
mape_log = hitung_mape(populasi_aktual, pred_log_di_data)
rmse_exp = hitung_rmse(populasi_aktual, pred_exp_di_data)
rmse_log = hitung_rmse(populasi_aktual, pred_log_di_data)
r2_exp   = hitung_r2(populasi_aktual, pred_exp_di_data)
r2_log   = hitung_r2(populasi_aktual, pred_log_di_data)

# ── NUMERICAL METHODS ────────────────────────────────────────
def euler_method(f, P0, t_span, dt, args=()):
    t_vals = np.arange(t_span[0], t_span[1] + dt, dt)
    P_vals = np.zeros(len(t_vals))
    P_vals[0] = P0
    for i in range(1, len(t_vals)):
        P_vals[i] = P_vals[i-1] + dt * f(P_vals[i-1], t_vals[i-1], *args)
    return t_vals, P_vals

def rk4_method(f, P0, t_span, dt, args=()):
    t_vals = np.arange(t_span[0], t_span[1] + dt, dt)
    P_vals = np.zeros(len(t_vals))
    P_vals[0] = P0
    for i in range(1, len(t_vals)):
        t_i = t_vals[i-1]
        h   = dt
        k1  = f(P_vals[i-1],          t_i,       *args)
        k2  = f(P_vals[i-1]+h*k1/2,   t_i+h/2,  *args)
        k3  = f(P_vals[i-1]+h*k2/2,   t_i+h/2,  *args)
        k4  = f(P_vals[i-1]+h*k3,     t_i+h,    *args)
        P_vals[i] = P_vals[i-1] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return t_vals, P_vals


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="section-header">⚙ PARAMETER MODEL</div>', unsafe_allow_html=True)

    k_val = st.slider(
        "k — laju pertumbuhan",
        min_value=0.003, max_value=0.050,
        value=float(round(k_analitik, 4)),
        step=0.001,
        format="%.4f",
        help="Laju pertumbuhan tahunan. Nilai analitik dari data BPS."
    )

    K_val = st.slider(
        "K — kapasitas dukung (jiwa)",
        min_value=100_000, max_value=300_000,
        value=int(K_default),
        step=5_000,
        format="%d",
        help="Daya tampung maksimum lingkungan (model logistik)."
    )

    st.markdown("---")
    st.markdown('<div class="section-header">🔢 METODE NUMERIK</div>', unsafe_allow_html=True)
    dt_val = st.selectbox(
        "Ukuran langkah Δt (tahun)",
        options=[1.0, 0.5, 0.25, 0.1],
        index=1,
        help="Langkah waktu untuk metode Euler dan RK4."
    )

    st.markdown("---")
    if st.button("↺ Reset ke Nilai Jurnal", use_container_width=True):
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace; font-size:10px; color:#8B949E; line-height:1.8;'>
    <b style='color:#58A6FF;'>REFERENSI</b><br>
    Armin & Remetwa, M.G.K.<br>
    JIMAT, Vol.6 No.1, 2025<br><br>
    <b style='color:#58A6FF;'>DATA</b><br>
    BPS Kota Tual 2020–2024
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN — HERO BANNER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🏙️ PEMODELAN ODE PERTUMBUHAN PENDUDUK</div>
  <div class="hero-sub">Kota Tual, Maluku &nbsp;·&nbsp; Model Eksponensial &amp; Logistik &nbsp;·&nbsp; 2020–2030</div>
  <span class="hero-badge">dP/dt = k·P</span>
  <span class="hero-badge">dP/dt = k·P·(1−P/K)</span>
  <span class="hero-badge">RK4 &amp; Euler</span>
  <span class="hero-badge">Scipy odeint</span>
</div>
""", unsafe_allow_html=True)

# ── KPI METRICS ──────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("P₀ (2020)", f"{int(P0_hist):,}", "jiwa")
c2.metric("P (2024)", f"{int(P0_pred):,}", f"+{int(P0_pred-P0_hist):,}")
c3.metric("k analitik", f"{k_analitik:.5f}", f"{k_analitik*100:.3f}%/thn")
c4.metric("MAPE Eksponensial", f"{mape_exp:.4f}%")
c5.metric("MAPE Logistik", f"{mape_log:.4f}%")

st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Kurva Utama",
    "🔬  Sensitivitas",
    "⚙  Metode Numerik",
    "🌐  Phase Portrait",
    "📊  Tabel & Metrik",
])


# ╔══════════════════════════════════════════════╗
# ║  TAB 1 — KURVA UTAMA                        ║
# ╚══════════════════════════════════════════════╝
with tab1:
    st.markdown('<div class="section-header">📈 KURVA PERTUMBUHAN & ANALISIS</div>', unsafe_allow_html=True)

    pred_exp_r = solusi_eksponensial(t_pred_rel, P0_pred, k_val)
    pred_log_r = solusi_logistik(t_pred_rel, P0_pred, k_val, K_val)

    fig = plt.figure(figsize=(18, 18), facecolor='#0D1117')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32,
                            top=0.95, bottom=0.05, left=0.07, right=0.96)

    # ── SP1: Kurva penuh ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    t_full  = np.linspace(0, 15, 1000)
    tahun_f = 2020 + t_full
    P_exp_f = solusi_eksponensial(t_full, P0_hist, k_val)
    P_log_f = solusi_logistik(t_full, P0_hist, k_val, K_val)

    ax1.fill_between(tahun_f, P_exp_f, alpha=0.07, color=CYAN)
    ax1.fill_between(tahun_f, P_log_f, alpha=0.07, color=GREEN)
    ax1.plot(tahun_f, P_exp_f, color=CYAN,  lw=2.5, label=f'Eksponensial  k={k_val:.4f}')
    ax1.plot(tahun_f, P_log_f, color=GREEN, lw=2.5, label=f'Logistik  k={k_val:.4f}, K={K_val:,.0f}')
    ax1.scatter(tahun_historis, populasi_aktual, color=YELLOW, s=90, zorder=6,
                edgecolors='white', lw=0.8, label='Data BPS Aktual')
    ax1.axvline(2024, color=ORANGE, lw=1.2, ls='--', alpha=0.7, label='Batas Prediksi (2024)')
    ax1.axvline(2030, color=PURPLE, lw=1.2, ls='--', alpha=0.7, label='Target 2030')
    ax1.axhline(K_val, color=RED, lw=1, ls=':', alpha=0.6, label=f'Kapasitas K={K_val:,.0f}')

    for yr, pop in zip(tahun_historis, populasi_aktual):
        ax1.annotate(f'{int(pop):,}', (yr, pop),
                     textcoords='offset points', xytext=(0, 12),
                     fontsize=8, color=YELLOW, ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', fc='#161B22', ec=YELLOW, alpha=0.8))

    ax1.set_title('Kurva Pertumbuhan Penduduk Kota Tual (2020–2035)', pad=10)
    ax1.set_xlabel('Tahun')
    ax1.set_ylabel('Jumlah Penduduk (jiwa)')
    ax1.legend(loc='upper left', ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2019.5, 2035.5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # ── SP2: Phase Portrait ──────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    P_range  = np.linspace(50000, min(K_val * 1.2, 200000), 500)
    dPdt_exp = k_val * P_range
    dPdt_log = k_val * P_range * (1 - P_range / K_val)

    ax2.plot(P_range, dPdt_exp, color=CYAN,  lw=2.5, label='dP/dt (Eksponensial)')
    ax2.plot(P_range, dPdt_log, color=GREEN, lw=2.5, label='dP/dt (Logistik)')
    ax2.axhline(0, color='white', lw=0.8, ls='-', alpha=0.4)
    ax2.axvline(K_val/2, color=PURPLE, lw=1, ls='--', alpha=0.6,
                label=f'P=K/2={K_val/2:,.0f} (infleksi)')
    ax2.axvline(K_val, color=RED, lw=1, ls=':', alpha=0.6, label=f'K={K_val:,.0f}')
    ax2.axvline(P0_pred, color=YELLOW, lw=1.2, ls='--', alpha=0.8,
                label=f'P₀={P0_pred:,.0f} (2024)')
    ax2.scatter([P0_pred], [k_val * P0_pred], color=YELLOW, s=80, zorder=6)
    ax2.scatter([P0_pred], [k_val * P0_pred * (1 - P0_pred/K_val)], color=GREEN, s=80, zorder=6)
    ax2.set_title('Phase Portrait: dP/dt vs P')
    ax2.set_xlabel('Populasi P (jiwa)')
    ax2.set_ylabel('Laju dP/dt (jiwa/tahun)')
    ax2.legend(fontsize=7.5)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

    # ── SP3: Slope Field ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    t_mesh = np.linspace(0, 14, 18)
    P_mesh = np.linspace(70000, min(K_val * 1.05, 180000), 18)
    T_m, P_m = np.meshgrid(t_mesh, P_mesh)
    dT = np.ones_like(T_m)
    dP = k_val * P_m * (1 - P_m / K_val)
    magnitude = np.sqrt(dT**2 + dP**2)
    magnitude[magnitude == 0] = 1
    q = ax3.quiver(T_m, P_m, dT/magnitude, dP/magnitude, magnitude,
                   cmap='cool', alpha=0.75, scale=28, headwidth=4, headlength=5)
    plt.colorbar(q, ax=ax3, label='|Kecepatan|', pad=0.01)
    t_solve = np.linspace(0, 14, 300)
    for P_init in [60000, 75000, 88280, 100000, 115000, 130000]:
        sol = odeint(ode_logistik, P_init, t_solve, args=(k_val, K_val))
        color = CYAN if P_init < K_val else ORANGE
        ax3.plot(t_solve, sol[:, 0], color=color, lw=1.5, alpha=0.85)
    ax3.axhline(K_val, color=RED, lw=1.5, ls='--', alpha=0.8,
                label=f'Ekuilibrium K={K_val:,.0f}')
    ax3.set_xlabel('Waktu t (tahun dari 2020)')
    ax3.set_ylabel('Populasi P (jiwa)')
    ax3.set_title('Slope Field + Lintasan Solusi ODE Logistik')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

    # ── SP4: MAPE per tahun ──────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    res_exp = populasi_aktual - pred_exp_di_data
    res_log = populasi_aktual - pred_log_di_data
    mape_py_exp = np.abs(res_exp / populasi_aktual) * 100
    mape_py_log = np.abs(res_log / populasi_aktual) * 100
    x_pos = np.arange(len(tahun_historis))
    w = 0.35
    b1 = ax4.bar(x_pos - w/2, mape_py_exp, w, color=CYAN,  alpha=0.85,
                 label=f'Eksponensial (MAPE={mape_exp:.3f}%)')
    b2 = ax4.bar(x_pos + w/2, mape_py_log, w, color=GREEN, alpha=0.85,
                 label=f'Logistik (MAPE={mape_log:.3f}%)')
    for bar in b1:
        h = bar.get_height()
        ax4.text(bar.get_x()+bar.get_width()/2, h+0.005,
                 f'{h:.3f}%', ha='center', va='bottom', fontsize=7.5, color=CYAN)
    for bar in b2:
        h = bar.get_height()
        ax4.text(bar.get_x()+bar.get_width()/2, h+0.005,
                 f'{h:.3f}%', ha='center', va='bottom', fontsize=7.5, color=GREEN)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([str(int(y)) for y in tahun_historis])
    ax4.set_title('MAPE per Tahun — Eksponensial vs Logistik')
    ax4.set_xlabel('Tahun')
    ax4.set_ylabel('MAPE (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # ── SP5: Bar prediksi ────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    x2 = np.arange(len(tahun_prediksi))
    b3 = ax5.bar(x2 - w/2, pred_exp_r, w, color=CYAN,  alpha=0.85, label='Eksponensial')
    b4 = ax5.bar(x2 + w/2, pred_log_r, w, color=GREEN, alpha=0.85, label='Logistik')
    for bar in b3:
        h = bar.get_height()
        ax5.text(bar.get_x()+bar.get_width()/2, h+100,
                 f'{int(h):,}', ha='center', va='bottom', fontsize=7, color=CYAN)
    for bar in b4:
        h = bar.get_height()
        ax5.text(bar.get_x()+bar.get_width()/2, h+100,
                 f'{int(h):,}', ha='center', va='bottom', fontsize=7, color=GREEN)
    ax5.set_xticks(x2)
    ax5.set_xticklabels([str(y) for y in tahun_prediksi])
    ax5.set_title('Prediksi Jumlah Penduduk 2026–2030')
    ax5.set_xlabel('Tahun')
    ax5.set_ylabel('Jumlah Penduduk (jiwa)')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax5.set_ylim(88000, max(pred_exp_r.max(), pred_log_r.max()) * 1.08)

    st.pyplot(fig)
    plt.close(fig)

    # Tabel prediksi
    st.markdown('<div class="section-header">📋 TABEL PREDIKSI 2026–2030</div>', unsafe_allow_html=True)
    df_pred = pd.DataFrame({
        'Tahun': tahun_prediksi,
        'Eksponensial (jiwa)': pred_exp_r.astype(int),
        'Logistik (jiwa)': pred_log_r.astype(int),
        'Selisih (jiwa)': (pred_exp_r - pred_log_r).astype(int),
    })
    st.dataframe(df_pred.style.format({
        'Eksponensial (jiwa)': '{:,}',
        'Logistik (jiwa)': '{:,}',
        'Selisih (jiwa)': '{:,}',
    }), use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════╗
# ║  TAB 2 — SENSITIVITAS                       ║
# ╚══════════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="section-header">🔬 ANALISIS SENSITIVITAS PARAMETER k DAN K</div>',
                unsafe_allow_html=True)

    fig2, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0D1117')
    t_s = np.linspace(0, 10, 300)
    tahun_s = 2024 + t_s

    # Panel A — variasi k
    ax = axes[0]
    k_values = np.arange(0.005, 0.031, 0.005)
    cmap_k = plt.cm.plasma(np.linspace(0.2, 0.9, len(k_values)))
    for k_v, col in zip(k_values, cmap_k):
        P = solusi_eksponensial(t_s, P0_pred, k_v)
        ax.plot(tahun_s, P, color=col, lw=2, label=f'k={k_v:.3f} ({k_v*100:.1f}%/yr)')
    ax.scatter(tahun_historis[tahun_historis >= 2024], [P0_pred],
               color=YELLOW, s=70, zorder=8, edgecolors='white', lw=0.8)
    ax.axvline(2030, color='white', lw=1, ls='--', alpha=0.5)
    ax.set_title('Sensitivitas k — Model Eksponensial')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Populasi (jiwa)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Panel B — variasi K
    ax2 = axes[1]
    K_values = [110000, 130000, 150000, 175000, 200000, 250000]
    cmap_K = plt.cm.viridis(np.linspace(0.2, 0.9, len(K_values)))
    for K_v, col in zip(K_values, cmap_K):
        P = solusi_logistik(t_s, P0_pred, k_analitik, K_v)
        ax2.plot(tahun_s, P, color=col, lw=2, label=f'K={K_v:,}')
        ax2.axhline(K_v, color=col, lw=0.6, ls=':', alpha=0.5)
    ax2.axvline(2030, color='white', lw=1, ls='--', alpha=0.5)
    ax2.set_title('Sensitivitas K (Kapasitas Dukung) — Model Logistik')
    ax2.set_xlabel('Tahun')
    ax2.set_ylabel('Populasi (jiwa)')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ╔══════════════════════════════════════════════╗
# ║  TAB 3 — METODE NUMERIK                     ║
# ╚══════════════════════════════════════════════╝
with tab3:
    st.markdown('<div class="section-header">⚙ PERBANDINGAN METODE NUMERIK ODE</div>',
                unsafe_allow_html=True)
    st.caption(f"Δt = {dt_val} tahun | Euler · RK4 · odeint vs Solusi Analitik")

    fig3, ax = plt.subplots(figsize=(14, 6), facecolor='#0D1117')

    t_anal   = np.linspace(0, 10, 1000)
    P_anal_e = solusi_eksponensial(t_anal, P0_hist, k_analitik)
    P_anal_l = solusi_logistik(t_anal, P0_hist, k_analitik, K_default)

    t_euler, P_euler_e = euler_method(ode_eksponensial, P0_hist, (0, 10), dt_val,
                                      args=(k_analitik,))
    t_rk4,   P_rk4_e  = rk4_method(ode_eksponensial, P0_hist, (0, 10), dt_val,
                                     args=(k_analitik,))
    t_ode  = np.linspace(0, 10, 500)
    P_ode_l = odeint(ode_logistik, P0_hist, t_ode, args=(k_analitik, K_default))[:, 0]

    ax.plot(t_anal, P_anal_e,  color=CYAN,   lw=2.5, ls='-',  label='Analitik Exp.')
    ax.plot(t_anal, P_anal_l,  color=GREEN,  lw=2.5, ls='-',  label='Analitik Log.')
    ax.plot(t_euler, P_euler_e, color=ORANGE, lw=1.8, ls='--', label=f'Euler (Δt={dt_val})')
    ax.plot(t_rk4,   P_rk4_e,  color=PURPLE, lw=1.8, ls='-.', label=f'RK4 (Δt={dt_val})')
    ax.plot(t_ode,   P_ode_l,  color=TEAL,   lw=1.5, ls=':',  label='odeint (Log.)')

    # Error Euler
    P_anal_at_euler = solusi_eksponensial(t_euler, P0_hist, k_analitik)
    err_euler = np.abs(P_euler_e - P_anal_at_euler)
    ax_twin = ax.twinx()
    ax_twin.fill_between(t_euler, err_euler, alpha=0.12, color=ORANGE)
    ax_twin.plot(t_euler, err_euler, color=ORANGE, lw=0.8, alpha=0.6, label='Error Euler')
    ax_twin.set_ylabel('Error Euler (jiwa)', color=ORANGE, fontsize=9)
    ax_twin.tick_params(axis='y', colors=ORANGE)
    ax_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    ax.set_title(f'Perbandingan Metode Numerik ODE — Δt = {dt_val} tahun')
    ax.set_xlabel('Waktu t (tahun dari 2020)')
    ax.set_ylabel('Populasi (jiwa)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # Tabel error
    st.markdown('<div class="section-header">📋 TABEL ERROR NUMERIK DI TITIK DATA</div>',
                unsafe_allow_html=True)
    t_check = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    P_anal_chk  = solusi_eksponensial(t_check, P0_hist, k_analitik)
    _, Pe_all = euler_method(ode_eksponensial, P0_hist, (0, 4), dt_val, args=(k_analitik,))
    _, Prk_all = rk4_method(ode_eksponensial, P0_hist, (0, 4), dt_val, args=(k_analitik,))
    # Sample at integer points
    idx_e = [round(t/dt_val) for t in t_check]
    idx_e = [min(i, len(Pe_all)-1) for i in idx_e]
    Pe_chk  = Pe_all[idx_e]
    Prk_chk = Prk_all[idx_e]
    df_err = pd.DataFrame({
        'Tahun': tahun_historis.astype(int),
        'Analitik': P_anal_chk.astype(int),
        'Euler': Pe_chk.astype(int),
        'RK4': Prk_chk.astype(int),
        'Error Euler': np.abs(Pe_chk - P_anal_chk).astype(int),
        'Error RK4': np.abs(Prk_chk - P_anal_chk).astype(int),
    })
    st.dataframe(df_err.style.format({
        'Analitik': '{:,}', 'Euler': '{:,}', 'RK4': '{:,}',
        'Error Euler': '{:,}', 'Error RK4': '{:,}',
    }).background_gradient(subset=['Error Euler', 'Error RK4'], cmap='Oranges'),
    use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════╗
# ║  TAB 4 — PHASE PORTRAIT 3D                  ║
# ╚══════════════════════════════════════════════╝
with tab4:
    st.markdown('<div class="section-header">🌐 RUANG FASE 3D — (t, P, dP/dt)</div>',
                unsafe_allow_html=True)

    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig4 = plt.figure(figsize=(16, 7), facecolor='#0D1117')

    for sp_idx, (model_name, ode_fn, col, args) in enumerate([
        ('Eksponensial', ode_eksponensial, CYAN,  (k_val,)),
        ('Logistik',     ode_logistik,     GREEN, (k_val, K_val))
    ]):
        ax = fig4.add_subplot(1, 2, sp_idx+1, projection='3d', facecolor='#161B22')
        ax.set_facecolor('#161B22')

        t_3d = np.linspace(0, 14, 500)
        for P_init in [70000, 88280, 100000, 115000]:
            P_sol = odeint(ode_fn, P_init, t_3d, args=args)[:, 0]
            dPdt  = np.array([ode_fn(p, t, *args) for p, t in zip(P_sol, t_3d)])
            points   = np.array([t_3d, P_sol, dPdt]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm_d   = plt.Normalize(dPdt.min(), dPdt.max())
            lc       = Line3DCollection(segments, cmap='cool', norm=norm_d, lw=1.5, alpha=0.85)
            lc.set_array(dPdt)
            ax.add_collection3d(lc)

        ax.set_xlabel('Waktu t (thn)', labelpad=6, fontsize=8)
        ax.set_ylabel('Populasi P', labelpad=6, fontsize=8)
        ax.set_zlabel('dP/dt', labelpad=6, fontsize=8)
        ax.set_title(f'Model {model_name}', color='#E6EDF3')
        ax.tick_params(colors='#8B949E', labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.info("💡 **Ruang Fase 3D** menunjukkan lintasan solusi pada koordinat (t, P, dP/dt). "
            "Setiap lintasan dimulai dari kondisi awal yang berbeda. "
            "Warna gradien menggambarkan besarnya laju perubahan populasi.")


# ╔══════════════════════════════════════════════╗
# ║  TAB 5 — TABEL & METRIK                     ║
# ╚══════════════════════════════════════════════╝
with tab5:
    st.markdown('<div class="section-header">📊 RINGKASAN METRIK & PERBANDINGAN MODEL</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Metrik Akurasi (Fit ke Data 2020–2024)**")
        df_metrik = pd.DataFrame({
            'Metrik': ['k laju pertumbuhan', 'MAPE (%)', 'RMSE (jiwa)', 'R² (koefisien)'],
            'Eksponensial': [
                f'{k_fit_exp*100:.4f}%/thn',
                f'{mape_exp:.4f}%',
                f'{rmse_exp:,.2f}',
                f'{r2_exp:.6f}',
            ],
            'Logistik': [
                f'{k_fit_log*100:.4f}%/thn',
                f'{mape_log:.4f}%',
                f'{rmse_log:,.2f}',
                f'{r2_log:.6f}',
            ],
        })
        st.dataframe(df_metrik, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**Prediksi Penduduk 2026–2030**")
        pred_exp_tab = solusi_eksponensial(t_pred_rel, P0_pred, k_val)
        pred_log_tab = solusi_logistik(t_pred_rel, P0_pred, k_val, K_val)
        df_full = pd.DataFrame({
            'Tahun': tahun_prediksi,
            'Eksponensial': pred_exp_tab.astype(int),
            'Logistik': pred_log_tab.astype(int),
            'Selisih': (pred_exp_tab - pred_log_tab).astype(int),
        })
        st.dataframe(df_full.style.format({
            'Eksponensial': '{:,}', 'Logistik': '{:,}', 'Selisih': '{:,}',
        }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📈 DATA HISTORIS BPS KOTA TUAL</div>',
                unsafe_allow_html=True)
    df_hist = pd.DataFrame({
        'Tahun': tahun_historis.astype(int),
        'Populasi Aktual (jiwa)': populasi_aktual.astype(int),
        'Pred. Eksponensial': pred_exp_di_data.astype(int),
        'Pred. Logistik': pred_log_di_data.astype(int),
        'Error Eksponen.': (populasi_aktual - pred_exp_di_data).astype(int),
        'Error Logistik': (populasi_aktual - pred_log_di_data).astype(int),
    })
    st.dataframe(df_hist.style.format({
        'Populasi Aktual (jiwa)': '{:,}',
        'Pred. Eksponensial': '{:,}',
        'Pred. Logistik': '{:,}',
        'Error Eksponen.': '{:,}',
        'Error Logistik': '{:,}',
    }), use_container_width=True, hide_index=True)

    # Summary tabel viz
    st.markdown("---")
    st.markdown('<div class="section-header">📋 RINGKASAN ASUMSI MODEL</div>',
                unsafe_allow_html=True)
    df_asumsi = pd.DataFrame({
        'Aspek': ['Persamaan ODE', 'Solusi Analitik', 'Asumsi Pertumbuhan', 'Parameter Utama', 'Titik Ekuilibrium'],
        'Eksponensial': [
            'dP/dt = k·P',
            'P(t) = P₀·e^(kt)',
            'Tak terbatas (∞)',
            f'k = {k_val:.5f}',
            'Tidak ada',
        ],
        'Logistik': [
            'dP/dt = k·P·(1−P/K)',
            'P(t) = K / (1 + ((K−P₀)/P₀)·e^(−kt))',
            f'Terbatas pada K = {K_val:,.0f}',
            f'k = {k_val:.5f}, K = {K_val:,.0f}',
            f'P* = {K_val:,.0f} (stabil)',
        ],
    })
    st.dataframe(df_asumsi, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style='background:#161B22; border:1px solid #30363D; border-radius:8px;
                padding:16px; margin-top:16px; font-family:Space Mono,monospace;
                font-size:11px; color:#8B949E; line-height:2;'>
    <b style='color:#58A6FF;'>INTERPRETASI AKADEMIS</b><br>
    • Model <b style='color:#58A6FF;'>Eksponensial</b> cocok untuk jangka pendek, namun tidak realistis jangka panjang karena mengasumsikan sumber daya tak terbatas.<br>
    • Model <b style='color:#3FB950;'>Logistik</b> lebih realistis karena mempertimbangkan daya dukung lingkungan (K).<br>
    • Kedua model memberikan MAPE &lt; 10%, menunjukkan akurasi yang baik pada rentang historis 2020–2024.<br>
    • Prediksi 2026–2030 menunjukkan perbedaan yang semakin besar antara dua model, terutama saat populasi mendekati K.<br>
    </div>
    """, unsafe_allow_html=True)
