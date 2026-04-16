import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
import warnings
import io

warnings.filterwarnings('ignore')

from streamlit_folium import st_folium
from modules.map_layer import build_map
from modules.lp_solver import get_warehouse_info
from modules.scenario_engine import (
    run_drought, run_festival, run_migration, compute_delta
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='SmartPDS — Intelligence Platform',
    page_icon='🌾',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main { background: #f7f8fa; }

.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 700; }

.metric-card {
    background: white;
    border: 1px solid #e4e6ea;
    border-radius: 8px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    color: #1a1a2e;
    line-height: 1.1;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-card .sub {
    font-size: 0.85rem;
    color: #374151;
    margin-top: 6px;
    font-weight: 500;
}

.header-bar {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 10px;
    padding: 20px 28px;
    color: white;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header-bar h1 { color: white; margin: 0; font-size: 1.6rem; }
.header-bar .sub { color: #94a3b8; font-size: 0.85rem; margin-top: 4px; }

.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.tag-high   { background: #fde8e8; color: #b91c1c; }
.tag-medium { background: #fef3c7; color: #92400e; }
.tag-low    { background: #d1fae5; color: #065f46; }
.tag-fraud  { background: #ede9fe; color: #5b21b6; }

.scenario-box {
    background: white;
    border: 1px solid #e4e6ea;
    border-left: 4px solid #0f3460;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
}

div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e4e6ea;
    border-radius: 8px;
    padding: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        forecast   = pd.read_csv('village_demand_forecast.csv')
        villages   = pd.read_csv('villages.csv')
        allocation = pd.read_csv('allocation_plan.csv')
        fraud      = pd.read_csv('fraud_alerts.csv')
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Place all CSV outputs from Layers 1–3 in the same directory as app.py.")
        st.stop()

    coord_cols = [c for c in ['village_id', 'latitude', 'longitude', 'lat', 'lon', 'district'] if c in villages.columns]
    vdf = villages[coord_cols].copy()

    if 'latitude' in vdf.columns and 'lat' not in vdf.columns:
        vdf = vdf.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

    merged = allocation.merge(vdf, on='village_id', how='left')

    fraud_slim = fraud[['village_id', 'is_fraud_flag', 'anomaly_score', 'top_driver_feature',
                         'collection_ratio', 'ghost_ratio', 'offline_pct',
                         'aadhaar_gap', 'demand_collection_gap', 'alert_reason']].copy()

    merged = merged.merge(fraud_slim, on='village_id', how='left')
    merged['is_fraud_flag'] = merged['is_fraud_flag'].fillna(0).astype(int)

    if 'lat' not in merged.columns or merged['lat'].isnull().all():
        np.random.seed(42)
        merged['lat'] = np.random.uniform(8.0, 13.5, len(merged))
        merged['lon'] = np.random.uniform(76.9, 80.3, len(merged))

    return forecast, villages, allocation, fraud, merged


forecast_df, villages_df, allocation_df, fraud_df, merged_df = load_data()


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <div>
        <h1>SmartPDS</h1>
        <div class="sub">AI-Driven Ration Distribution Intelligence Platform &nbsp;|&nbsp; Team Kahunas &nbsp;|&nbsp; DevsHouse 26</div>
    </div>
    <div style="text-align:right; color:#94a3b8; font-size:0.8rem; font-family:'IBM Plex Mono',monospace;">
        Tamil Nadu &nbsp;|&nbsp; 50 Villages &nbsp;|&nbsp; 3 Warehouses
    </div>
</div>
""", unsafe_allow_html=True)


# ── Top KPI row ──────────────────────────────────────────────────────────────
total_demand    = allocation_df['forecasted_demand_kg'].sum()
total_allocated = allocation_df['allocated_kg'].sum()
total_shortage  = allocation_df['shortage_gap_kg'].sum()
villages_90     = (allocation_df['coverage_pct'] >= 90).sum()
fraud_count     = int(fraud_df['is_fraud_flag'].sum()) if 'is_fraud_flag' in fraud_df.columns else 0
overall_cov     = total_allocated / total_demand * 100 if total_demand > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Total Demand',    f"{total_demand/1000:.1f} MT")
c2.metric('Total Allocated', f"{total_allocated/1000:.1f} MT", f"{overall_cov:.1f}% covered")
c3.metric('Shortage Gap',    f"{total_shortage/1000:.1f} MT")
c4.metric('Villages ≥90%',   f"{villages_90} / {len(allocation_df)}")
c5.metric('Fraud Alerts',    str(fraud_count), delta_color='inverse')

st.markdown("<br>", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '  Live Map  ',
    '  Allocation Dashboard  ',
    '  Digital Twin — Scenarios  ',
    '  Fraud Alerts  ',
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_ctrl, col_map = st.columns([1, 3])

    with col_ctrl:
        st.markdown("#### Map Controls")

        risk_filter = st.multiselect(
            'Shortage risk',
            options=['High', 'Medium', 'Low'],
            default=['High', 'Medium', 'Low'],
        )
        show_fraud_only = st.checkbox('Show fraud-flagged villages only', value=False)

        st.markdown("---")
        st.markdown("**Legend**")
        st.markdown('<span class="tag tag-high">High Risk</span>', unsafe_allow_html=True)
        st.markdown('<span class="tag tag-medium">Medium Risk</span>', unsafe_allow_html=True)
        st.markdown('<span class="tag tag-low">Low Risk</span>', unsafe_allow_html=True)
        st.markdown('<span class="tag tag-fraud">Fraud Flagged</span>', unsafe_allow_html=True)
        st.markdown("🏠 Warehouse depot")

        st.markdown("---")
        st.caption("Click any village marker to see full details including demand, allocation, coverage, and fraud status.")

    with col_map:
        map_df = merged_df.copy()
        if risk_filter:
            map_df = map_df[map_df['shortage_risk'].isin(risk_filter)]
        if show_fraud_only:
            map_df = map_df[map_df['is_fraud_flag'] == 1]

        fmap = build_map(map_df)
        st_folium(fmap, width=None, height=560, returned_objects=[])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ALLOCATION DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Forecasted Demand vs Allocated — All Villages")

        plot_df = allocation_df.copy().sort_values('shortage_gap_kg', ascending=False).head(30)

        fig = go.Figure()
        fig.add_bar(
            x=plot_df['village_id'].astype(str),
            y=plot_df['forecasted_demand_kg'],
            name='Forecasted demand',
            marker_color='#2c7bb6',
            opacity=0.85,
        )
        fig.add_bar(
            x=plot_df['village_id'].astype(str),
            y=plot_df['allocated_kg'],
            name='Allocated',
            marker_color='#1a9641',
            opacity=0.85,
        )
        fig.update_layout(
            barmode='overlay',
            height=340,
            margin=dict(l=0, r=0, t=10, b=80),
            legend=dict(orientation='h', y=1.08),
            xaxis_title='Village ID',
            yaxis_title='Quantity (kg)',
            font=dict(family='IBM Plex Sans'),
            paper_bgcolor='white',
            plot_bgcolor='#f9fafb',
        )
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.markdown("#### Warehouse Utilization")

        warehouses = get_warehouse_info()
        wh_ids   = [wh['warehouse_id'] for wh in warehouses]
        wh_names = [wh['name'].replace(' Depot', '').replace(' Central', '') for wh in warehouses]
        wh_stock = [wh['stock_kg'] for wh in warehouses]

        wh_used = []
        for wh in warehouses:
            col = f"alloc_{wh['warehouse_id']}"
            if col in allocation_df.columns:
                wh_used.append(allocation_df[col].sum())
            else:
                wh_used.append(0)

        fig2 = go.Figure()
        fig2.add_bar(
            y=wh_names, x=wh_stock,
            name='Total stock', orientation='h',
            marker_color='#e5e7eb',
        )
        fig2.add_bar(
            y=wh_names, x=wh_used,
            name='Used', orientation='h',
            marker_color='#0f3460',
        )
        fig2.update_layout(
            barmode='overlay',
            height=200,
            margin=dict(l=0, r=0, t=10, b=10),
            legend=dict(orientation='h', y=1.15),
            xaxis_title='Quantity (kg)',
            font=dict(family='IBM Plex Sans'),
            paper_bgcolor='white',
            plot_bgcolor='#f9fafb',
        )
        st.plotly_chart(fig2, width="stretch")

        for i, wh in enumerate(warehouses):
            used = wh_used[i]
            pct  = used / wh['stock_kg'] * 100 if wh['stock_kg'] > 0 else 0
            st.progress(int(min(pct, 100)), text=f"{wh_names[i]}: {pct:.1f}% utilized")

    st.markdown("---")

    if 'district' in allocation_df.columns:
        st.markdown("#### District-Level Summary")

        dist_df = (
            allocation_df.groupby('district')
            .agg(
                total_demand=('forecasted_demand_kg', 'sum'),
                total_allocated=('allocated_kg', 'sum'),
                total_shortage=('shortage_gap_kg', 'sum'),
                villages=('village_id', 'count'),
            )
            .reset_index()
        )
        dist_df['coverage_pct'] = (dist_df['total_allocated'] / dist_df['total_demand'] * 100).round(1)
        dist_df = dist_df.sort_values('coverage_pct')

        fig3 = px.bar(
            dist_df,
            x='coverage_pct',
            y='district',
            orientation='h',
            color='coverage_pct',
            color_continuous_scale=['#d7191c', '#fdae61', '#1a9641'],
            range_color=[50, 100],
            labels={'coverage_pct': 'Coverage %', 'district': 'District'},
            height=300,
        )
        fig3.update_layout(
            margin=dict(l=0, r=0, t=10, b=10),
            coloraxis_showscale=False,
            font=dict(family='IBM Plex Sans'),
            paper_bgcolor='white',
            plot_bgcolor='#f9fafb',
        )
        fig3.add_vline(x=90, line_dash='dash', line_color='#374151', annotation_text='90% target')
        st.plotly_chart(fig3, width="stretch")

    st.markdown("#### Full Allocation Table")
    display_cols = [c for c in ['village_id', 'village_name', 'district', 'forecasted_demand_kg',
                                 'allocated_kg', 'shortage_gap_kg', 'coverage_pct',
                                 'primary_warehouse', 'shortage_risk'] if c in allocation_df.columns]

    risk_filter_tab2 = st.multiselect(
        'Filter by shortage risk',
        options=['High', 'Medium', 'Low'],
        default=['High', 'Medium', 'Low'],
        key='risk_tab2',
    )
    tbl = allocation_df[allocation_df['shortage_risk'].isin(risk_filter_tab2)][display_cols]
    st.dataframe(tbl.reset_index(drop=True), width="stretch", height=320)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DIGITAL TWIN
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Digital Twin — Scenario Simulator")
    st.caption("Each scenario re-runs the OR-Tools LP optimizer live. Results update the map and delta table below.")

    base_for_scenario = merged_df[[
        'village_id', 'forecasted_demand_kg', 'shortage_risk',
        'lat', 'lon',
    ] + [c for c in ['village_name', 'district', 'bpl_count'] if c in merged_df.columns]].copy()

    scenario_col1, scenario_col2, scenario_col3 = st.columns(3)

    with scenario_col1:
        st.markdown('<div class="scenario-box">', unsafe_allow_html=True)
        st.markdown("**Drought Scenario**")
        st.caption("Scales demand uniformly — reflects summer/drought months.")
        drought_mult = st.slider('Demand multiplier', 1.0, 2.0, 1.45, 0.05, key='drought_slider')
        run_drought_btn = st.button('Run Drought Scenario', width="stretch", key='btn_drought')
        st.markdown('</div>', unsafe_allow_html=True)

    with scenario_col2:
        st.markdown('<div class="scenario-box">', unsafe_allow_html=True)
        st.markdown("**Festival Scenario**")
        st.caption("Spikes demand across all villages for festival months.")
        festival_mult = st.slider('Demand multiplier', 1.0, 1.8, 1.30, 0.05, key='festival_slider')
        run_festival_btn = st.button('Run Festival Scenario', width="stretch", key='btn_festival')
        st.markdown('</div>', unsafe_allow_html=True)

    with scenario_col3:
        st.markdown('<div class="scenario-box">', unsafe_allow_html=True)
        st.markdown("**Migration Scenario**")
        st.caption("Influx into a specific district raises local BPL demand.")
        districts = sorted(merged_df['district'].dropna().unique().tolist()) if 'district' in merged_df.columns else ['All']
        target_dist   = st.selectbox('Target district', districts, key='mig_district')
        influx_pct    = st.slider('Population influx %', 5.0, 50.0, 20.0, 5.0, key='mig_slider')
        run_mig_btn   = st.button('Run Migration Scenario', width="stretch", key='btn_mig')
        st.markdown('</div>', unsafe_allow_html=True)

    scenario_result = None
    scenario_label  = None

    if run_drought_btn:
        with st.spinner('Solving LP for drought scenario...'):
            scenario_result = run_drought(base_for_scenario, drought_mult)
            scenario_label  = f'Drought ({drought_mult:.2f}x)'

    if run_festival_btn:
        with st.spinner('Solving LP for festival scenario...'):
            scenario_result = run_festival(base_for_scenario, festival_mult)
            scenario_label  = f'Festival ({festival_mult:.2f}x)'

    if run_mig_btn:
        with st.spinner('Solving LP for migration scenario...'):
            scenario_result = run_migration(base_for_scenario, target_dist, influx_pct)
            scenario_label  = f'Migration: {target_dist} +{influx_pct:.0f}%'

    if scenario_result is not None:
        st.markdown(f"---")
        st.markdown(f"##### Results — {scenario_label}")

        delta = compute_delta(base_for_scenario.merge(
            allocation_df[['village_id', 'coverage_pct', 'shortage_gap_kg', 'allocated_kg']],
            on='village_id', how='left'
        ), scenario_result)

        s1, s2, s3 = st.columns(3)
        s1.metric('Scenario total demand',    f"{scenario_result['forecasted_demand_kg'].sum()/1000:.1f} MT")
        s2.metric('Scenario allocated',       f"{scenario_result['allocated_kg'].sum()/1000:.1f} MT")
        s3.metric('Villages worsened',        str((delta['coverage_change'] < -5).sum()))

        col_map2, col_delta = st.columns([2, 1])

        with col_map2:
            scenario_map_df = scenario_result.copy()
            if 'lat' not in scenario_map_df.columns:
                lat_lon = merged_df[['village_id', 'lat', 'lon']]
                scenario_map_df = scenario_map_df.merge(lat_lon, on='village_id', how='left')
            smap = build_map(scenario_map_df)
            st_folium(smap, width=None, height=420, returned_objects=[], key='scenario_map')

        with col_delta:
            st.markdown("**Villages with largest coverage drop**")
            delta_cols = [c for c in ['village_name', 'district', 'base_coverage_pct',
                                       'scen_coverage_pct', 'coverage_change'] if c in delta.columns]
            st.dataframe(
                delta[delta_cols].head(15).reset_index(drop=True),
                width="stretch",
                height=400,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FRAUD ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    flagged = fraud_df[fraud_df['is_fraud_flag'] == 1].copy() if 'is_fraud_flag' in fraud_df.columns else pd.DataFrame()

    col_stats, col_chart = st.columns([1, 2])

    with col_stats:
        st.markdown("#### Detection Summary")
        total_vil = len(fraud_df)
        n_flagged = len(flagged)
        st.metric('Villages analysed', total_vil)
        st.metric('Fraud flagged',     n_flagged)
        st.metric('Flag rate',         f"{n_flagged/total_vil*100:.1f}%" if total_vil > 0 else 'N/A')
        st.caption("Algorithm: Isolation Forest (sklearn). Contamination: 5%. No labeled fraud data required.")

    with col_chart:
        if len(flagged) > 0 and 'top_driver_feature' in flagged.columns:
            st.markdown("#### Primary Fraud Driver Distribution")
            driver_counts = flagged['top_driver_feature'].value_counts().reset_index()
            driver_counts.columns = ['Feature', 'Count']
            fig4 = px.bar(
                driver_counts,
                x='Count', y='Feature', orientation='h',
                color='Count',
                color_continuous_scale=['#fde8e8', '#d7191c'],
                height=260,
            )
            fig4.update_layout(
                margin=dict(l=0, r=0, t=10, b=10),
                coloraxis_showscale=False,
                font=dict(family='IBM Plex Sans'),
                paper_bgcolor='white',
                plot_bgcolor='#f9fafb',
            )
            st.plotly_chart(fig4, width="stretch")

    st.markdown("---")

    if len(flagged) > 0:
        st.markdown("#### Flagged Village Alert Table")

        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            if 'district' in flagged.columns:
                dist_opts = ['All'] + sorted(flagged['district'].dropna().unique().tolist())
                dist_sel  = st.selectbox('Filter by district', dist_opts, key='fraud_dist')
            else:
                dist_sel = 'All'

        with col_filter2:
            if 'top_driver_feature' in flagged.columns:
                driver_opts = ['All'] + sorted(flagged['top_driver_feature'].dropna().unique().tolist())
                driver_sel  = st.selectbox('Filter by fraud driver', driver_opts, key='fraud_driver')
            else:
                driver_sel = 'All'

        display_fraud = flagged.copy().sort_values('anomaly_score')
        if dist_sel != 'All' and 'district' in display_fraud.columns:
            display_fraud = display_fraud[display_fraud['district'] == dist_sel]
        if driver_sel != 'All' and 'top_driver_feature' in display_fraud.columns:
            display_fraud = display_fraud[display_fraud['top_driver_feature'] == driver_sel]

        fraud_cols = [c for c in [
            'village_id', 'village_name', 'district', 'anomaly_score',
            'collection_ratio', 'ghost_ratio', 'offline_pct',
            'aadhaar_gap', 'top_driver_feature', 'alert_reason',
        ] if c in display_fraud.columns]

        st.dataframe(display_fraud[fraud_cols].reset_index(drop=True), width="stretch", height=380)

        csv_bytes = display_fraud[fraud_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Download Fraud Alerts CSV',
            data=csv_bytes,
            file_name='fraud_alerts_filtered.csv',
            mime='text/csv',
        )

        if 'collection_ratio' in fraud_df.columns and 'ghost_ratio' in fraud_df.columns:
            st.markdown("#### Key Fraud Signals — Collection Ratio vs Ghost Ratio")

            normal_df  = fraud_df[fraud_df['is_fraud_flag'] == 0]

            fig5 = go.Figure()
            fig5.add_scatter(
                x=normal_df['collection_ratio'],
                y=normal_df['ghost_ratio'],
                mode='markers',
                name='Normal',
                marker=dict(color='#2c7bb6', size=7, opacity=0.6),
            )
            fig5.add_scatter(
                x=flagged['collection_ratio'],
                y=flagged['ghost_ratio'],
                mode='markers',
                name='Fraud flagged',
                marker=dict(color='#d7191c', size=10, symbol='x', opacity=0.9),
                text=flagged.get('village_name', flagged.get('village_id')),
            )
            fig5.add_vline(x=1.0, line_dash='dot', line_color='#9ca3af')
            fig5.add_hline(y=1.0, line_dash='dot', line_color='#9ca3af')
            fig5.update_layout(
                xaxis_title='Collection ratio (collected / allocated)',
                yaxis_title='Ghost ratio (transactions / registered cards)',
                height=360,
                font=dict(family='IBM Plex Sans'),
                paper_bgcolor='white',
                plot_bgcolor='#f9fafb',
                legend=dict(orientation='h', y=1.08),
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig5, width="stretch")

    else:
        st.info("No fraud-flagged villages found in fraud_alerts.csv.")
