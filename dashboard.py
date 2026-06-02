"""
dashboard.py

Streamlit dashboard for monitoring email classifier performance.

Usage:
    streamlit run dashboard.py

Requirements:
    pip install streamlit pandas plotly
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "email_logs.db"
DEFAULT_THRESHOLD = 0.884695

# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------

st.set_page_config(
    page_title="Email Classifier",
    page_icon="✉️",
    layout="wide"
)

# custom styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main {
        background-color: #0f0f0f;
        color: #e8e8e8;
    }

    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        letter-spacing: -0.02em;
    }

    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 20px 24px;
    }

    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        line-height: 1;
        margin-bottom: 4px;
    }

    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #666;
    }

    .accent-green { color: #4ade80; }
    .accent-red { color: #f87171; }
    .accent-yellow { color: #fbbf24; }
    .accent-blue { color: #60a5fa; }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #444;
        border-bottom: 1px solid #222;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    .stDataFrame {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
    }

    div[data-testid="stMetric"] {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 16px 20px;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem !important;
    }

    .stSlider > div > div {
        background: #4ade80 !important;
    }

    .borderline-row {
        background: #1f1a0a;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

@st.cache_data(ttl=60)  # refresh every 60 seconds
def load_data():
    if not DB_PATH.exists():
        return None, None

    conn = sqlite3.connect(DB_PATH)

    runs = pd.read_sql('''
        SELECT * FROM runs
        ORDER BY timestamp ASC
    ''', conn)

    emails = pd.read_sql('''
        SELECT * FROM emails
        ORDER BY timestamp DESC
    ''', conn)

    conn.close()

    # parse timestamps
    runs['timestamp'] = pd.to_datetime(runs['timestamp'])
    emails['timestamp'] = pd.to_datetime(emails['timestamp'])
    runs['date'] = runs['timestamp'].dt.date
    emails['date'] = emails['timestamp'].dt.date

    return runs, emails

runs, emails = load_data()

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.markdown("# ✉️ email classifier")
st.markdown("<p style='color:#444; font-family:IBM Plex Mono; font-size:0.8rem;'>performance monitor — auto-refreshes every 60s</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if runs is None or runs.empty:
    st.warning("No data yet. Run classify_emails.py first to start collecting data.")
    st.stop()

# ------------------------------------------------------------
# OVERVIEW METRICS
# ------------------------------------------------------------

st.markdown("<div class='section-header'>overview</div>", unsafe_allow_html=True)

total_emails = int(emails.shape[0])
total_junk = int((emails['label'] == 'junk').sum())
total_important = int((emails['label'] == 'important').sum())
avg_junk_pct = round(runs['junk_pct'].mean(), 1) if not runs.empty else 0
last_run = runs['timestamp'].max().strftime('%b %d, %H:%M') if not runs.empty else 'N/A'
total_runs = len(runs)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Classified", f"{total_emails:,}")
with col2:
    st.metric("Junk", f"{total_junk:,}")
with col3:
    st.metric("Important", f"{total_important:,}")
with col4:
    st.metric("Avg Junk Rate", f"{avg_junk_pct}%")
with col5:
    st.metric("Last Run", last_run)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------
# CHARTS ROW 1
# ------------------------------------------------------------

st.markdown("<div class='section-header'>trends</div>", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    # junk % over time
    fig_junk = px.line(
        runs,
        x='timestamp',
        y='junk_pct',
        title='Junk Rate Over Time (%)',
        labels={'junk_pct': 'Junk %', 'timestamp': ''}
    )
    fig_junk.update_traces(line_color='#4ade80', line_width=1.5)
    fig_junk.update_layout(
        plot_bgcolor='#0f0f0f',
        paper_bgcolor='#0f0f0f',
        font_color='#888',
        font_family='IBM Plex Mono',
        title_font_size=12,
        title_font_color='#666',
        xaxis=dict(gridcolor='#1a1a1a', showgrid=True),
        yaxis=dict(gridcolor='#1a1a1a', showgrid=True, range=[0, 100]),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified'
    )
    st.plotly_chart(fig_junk, use_container_width=True)

with col_right:
    # emails per day stacked bar
    daily = emails.groupby(['date', 'label']).size().reset_index(name='count')

    fig_daily = px.bar(
        daily,
        x='date',
        y='count',
        color='label',
        title='Emails Per Day',
        labels={'count': 'Emails', 'date': '', 'label': ''},
        color_discrete_map={'junk': '#f87171', 'important': '#60a5fa'}
    )
    fig_daily.update_layout(
        plot_bgcolor='#0f0f0f',
        paper_bgcolor='#0f0f0f',
        font_color='#888',
        font_family='IBM Plex Mono',
        title_font_size=12,
        title_font_color='#666',
        xaxis=dict(gridcolor='#1a1a1a'),
        yaxis=dict(gridcolor='#1a1a1a'),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(bgcolor='#0f0f0f'),
        barmode='stack'
    )
    st.plotly_chart(fig_daily, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------
# PROBABILITY DISTRIBUTION
# ------------------------------------------------------------

st.markdown("<div class='section-header'>confidence distribution</div>", unsafe_allow_html=True)

fig_hist = px.histogram(
    emails,
    x='probability',
    nbins=50,
    color='label',
    title='Model Confidence Distribution',
    labels={'probability': 'Junk Probability', 'count': 'Emails'},
    color_discrete_map={'junk': '#f87171', 'important': '#60a5fa'},
    opacity=0.8
)
fig_hist.add_vline(
    x=DEFAULT_THRESHOLD,
    line_dash='dash',
    line_color='#fbbf24',
    annotation_text=f'threshold {DEFAULT_THRESHOLD}',
    annotation_font_color='#fbbf24',
    annotation_font_size=10
)
fig_hist.update_layout(
    plot_bgcolor='#0f0f0f',
    paper_bgcolor='#0f0f0f',
    font_color='#888',
    font_family='IBM Plex Mono',
    title_font_size=12,
    title_font_color='#666',
    xaxis=dict(gridcolor='#1a1a1a'),
    yaxis=dict(gridcolor='#1a1a1a'),
    margin=dict(l=0, r=0, t=40, b=0),
    legend=dict(bgcolor='#0f0f0f'),
    barmode='overlay'
)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------
# THRESHOLD SIMULATOR
# ------------------------------------------------------------

st.markdown("<div class='section-header'>threshold simulator</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#555; font-size:0.8rem;'>drag to see how changing the threshold affects classifications</p>", unsafe_allow_html=True)

sim_threshold = st.slider(
    "Threshold",
    min_value=0.5,
    max_value=1.0,
    value=DEFAULT_THRESHOLD,
    step=0.01,
    label_visibility="collapsed"
)

sim_junk = (emails['probability'] >= sim_threshold).sum()
sim_important = (emails['probability'] < sim_threshold).sum()
sim_junk_pct = round(sim_junk / len(emails) * 100, 1) if len(emails) > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    delta = int(sim_junk) - total_junk
    st.metric("Would classify as junk", f"{sim_junk:,}", delta=f"{delta:+,} vs current")
with col2:
    delta2 = int(sim_important) - total_important
    st.metric("Would classify as important", f"{sim_important:,}", delta=f"{delta2:+,} vs current")
with col3:
    st.metric("Junk rate at this threshold", f"{sim_junk_pct}%")

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------
# BORDERLINE DECISIONS
# ------------------------------------------------------------

st.markdown("<div class='section-header'>borderline decisions (0.75 – 0.95)</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#555; font-size:0.8rem;'>emails near the decision boundary — most likely to be misclassified</p>", unsafe_allow_html=True)

borderline = emails[
    (emails['probability'] >= 0.75) &
    (emails['probability'] <= 0.95)
].sort_values('probability', ascending=False)[['timestamp', 'sender', 'subject', 'probability', 'label']]

if borderline.empty:
    st.markdown("<p style='color:#444;'>No borderline decisions yet.</p>", unsafe_allow_html=True)
else:
    st.dataframe(
        borderline.head(50),
        use_container_width=True,
        hide_index=True,
        column_config={
            "probability": st.column_config.ProgressColumn(
                "probability",
                min_value=0,
                max_value=1,
                format="%.3f"
            ),
            "timestamp": st.column_config.DatetimeColumn("time", format="MMM D, HH:mm"),
            "label": st.column_config.TextColumn("label"),
        }
    )

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------
# RECENT CLASSIFICATIONS
# ------------------------------------------------------------

st.markdown("<div class='section-header'>recent classifications</div>", unsafe_allow_html=True)

recent = emails.head(50)[['timestamp', 'sender', 'subject', 'probability', 'label']]

st.dataframe(
    recent,
    use_container_width=True,
    hide_index=True,
    column_config={
        "probability": st.column_config.ProgressColumn(
            "probability",
            min_value=0,
            max_value=1,
            format="%.3f"
        ),
        "timestamp": st.column_config.DatetimeColumn("time", format="MMM D, HH:mm"),
        "label": st.column_config.TextColumn("label"),
    }
)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown(f"""
<p style='color:#333; font-family:IBM Plex Mono; font-size:0.7rem; text-align:center;'>
    {total_runs} runs logged · threshold {DEFAULT_THRESHOLD} · data from {DB_PATH.name}
</p>
""", unsafe_allow_html=True)
