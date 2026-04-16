import folium
import pandas as pd
import numpy as np
from modules.lp_solver import get_warehouse_info

RISK_COLORS = {
    'High':   '#d7191c',
    'Medium': '#fdae61',
    'Low':    '#1a9641',
}

FRAUD_BORDER = '#000000'
NORMAL_BORDER = '#ffffff'


def _risk_color(row: pd.Series) -> str:
    if row.get('is_fraud_flag', 0) == 1:
        return '#7b2d8b'
    return RISK_COLORS.get(row.get('shortage_risk', 'Low'), '#1a9641')


def _popup_html(row: pd.Series) -> str:
    village  = row.get('village_name', row.get('village_id', ''))
    district = row.get('district', 'N/A')
    demand   = row.get('forecasted_demand_kg', 0)
    alloc    = row.get('allocated_kg', 0)
    coverage = row.get('coverage_pct', 0)
    risk     = row.get('shortage_risk', 'N/A')
    fraud    = 'YES' if row.get('is_fraud_flag', 0) == 1 else 'No'
    driver   = row.get('top_driver_feature', 'N/A') if row.get('is_fraud_flag', 0) == 1 else 'N/A'
    score    = f"{row.get('anomaly_score', 0):.4f}" if row.get('is_fraud_flag', 0) == 1 else 'N/A'

    color = _risk_color(row)

    return f"""
    <div style="font-family: 'Segoe UI', sans-serif; min-width: 220px;">
        <div style="background:{color}; color:white; padding:8px 12px; border-radius:4px 4px 0 0; font-weight:600; font-size:13px;">
            {village}
        </div>
        <div style="padding:10px 12px; background:#fafafa; border:1px solid #e0e0e0; border-top:none; border-radius:0 0 4px 4px;">
            <table style="width:100%; font-size:12px; border-collapse:collapse;">
                <tr><td style="color:#666; padding:2px 0;">District</td>
                    <td style="font-weight:500; text-align:right;">{district}</td></tr>
                <tr><td style="color:#666; padding:2px 0;">Forecasted demand</td>
                    <td style="font-weight:500; text-align:right;">{demand:,.0f} kg</td></tr>
                <tr><td style="color:#666; padding:2px 0;">Allocated</td>
                    <td style="font-weight:500; text-align:right;">{alloc:,.0f} kg</td></tr>
                <tr><td style="color:#666; padding:2px 0;">Coverage</td>
                    <td style="font-weight:500; text-align:right;">{coverage:.1f}%</td></tr>
                <tr><td style="color:#666; padding:2px 0;">Shortage risk</td>
                    <td style="font-weight:500; text-align:right; color:{color};">{risk}</td></tr>
                <tr><td style="color:#666; padding:2px 0;">Fraud flagged</td>
                    <td style="font-weight:600; text-align:right; color:{'#d7191c' if fraud=='YES' else '#1a9641'};">{fraud}</td></tr>
                {'<tr><td style="color:#666; padding:2px 0;">Fraud driver</td><td style="font-weight:500; text-align:right;">' + driver + '</td></tr>' if fraud == 'YES' else ''}
                {'<tr><td style="color:#666; padding:2px 0;">Anomaly score</td><td style="font-weight:500; text-align:right;">' + score + '</td></tr>' if fraud == 'YES' else ''}
            </table>
        </div>
    </div>
    """


def build_map(merged_df: pd.DataFrame, center: list = None) -> folium.Map:
    """
    Build a Folium map with village circle markers and warehouse markers.

    Parameters
    ----------
    merged_df : DataFrame containing village_id, lat, lon, shortage_risk,
                forecasted_demand_kg, allocated_kg, coverage_pct,
                and optionally is_fraud_flag, anomaly_score, top_driver_feature.
    center    : [lat, lon] for map center. Defaults to Tamil Nadu centroid.

    Returns
    -------
    folium.Map
    """
    if center is None:
        center = [10.85, 78.5]

    m = folium.Map(
        location=center,
        zoom_start=7,
        tiles='CartoDB positron',
        prefer_canvas=True,
    )

    if 'lat' not in merged_df.columns or merged_df['lat'].isnull().all():
        return m

    for _, row in merged_df.iterrows():
        lat = row.get('lat')
        lon = row.get('lon')
        if pd.isna(lat) or pd.isna(lon):
            continue

        color  = _risk_color(row)
        border = FRAUD_BORDER if row.get('is_fraud_flag', 0) == 1 else NORMAL_BORDER
        radius = 7 + min(row.get('forecasted_demand_kg', 0) / 3000, 6)

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=border,
            weight=1.5 if row.get('is_fraud_flag', 0) == 1 else 0.8,
            fill=True,
            fill_color=color,
            fill_opacity=0.82,
            popup=folium.Popup(folium.IFrame(_popup_html(row), width=260, height=220), max_width=260),
            tooltip=row.get('village_name', str(row.get('village_id', ''))),
        ).add_to(m)

    warehouses = get_warehouse_info()
    for wh in warehouses:
        folium.Marker(
            location=[wh['lat'], wh['lon']],
            icon=folium.Icon(color='blue', icon='home', prefix='fa'),
            tooltip=f"{wh['name']} — {wh['stock_kg']:,} kg",
            popup=folium.Popup(
                f"<b>{wh['name']}</b><br>Stock: {wh['stock_kg']:,} kg<br>ID: {wh['warehouse_id']}",
                max_width=200
            ),
        ).add_to(m)

    legend_html = """
    <div style="position:fixed; bottom:40px; left:20px; z-index:1000;
                background:white; border:1px solid #ccc; border-radius:6px;
                padding:12px 16px; font-family:'Segoe UI',sans-serif; font-size:12px; box-shadow:0 2px 8px rgba(0,0,0,0.15);">
        <div style="font-weight:600; margin-bottom:8px; font-size:13px;">Village Risk</div>
        <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <div style="width:12px; height:12px; border-radius:50%; background:#d7191c;"></div> High shortage risk
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <div style="width:12px; height:12px; border-radius:50%; background:#fdae61;"></div> Medium shortage risk
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <div style="width:12px; height:12px; border-radius:50%; background:#1a9641;"></div> Low shortage risk
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <div style="width:12px; height:12px; border-radius:50%; background:#7b2d8b;"></div> Fraud flagged
        </div>
        <div style="margin-top:8px; padding-top:8px; border-top:1px solid #eee; font-weight:600;">
            <span style="color:#2c7bb6;">&#8962;</span> Warehouse depot
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m
