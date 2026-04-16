import pandas as pd
import numpy as np
from modules.lp_solver import solve

FESTIVAL_MONTHS = {10, 11, 1, 4}
DROUGHT_MONTHS  = {4, 5, 6}


def run_drought(base_df: pd.DataFrame, multiplier: float = 1.45) -> pd.DataFrame:
    df = base_df.copy()
    df['forecasted_demand_kg'] = (df['forecasted_demand_kg'] * multiplier).round(1)
    df['shortage_risk'] = df['shortage_risk'].replace({'Low': 'Medium', 'Medium': 'High'})
    result = solve(df)
    result['scenario'] = f'Drought ({multiplier:.2f}x demand)'
    return result


def run_festival(base_df: pd.DataFrame, multiplier: float = 1.30) -> pd.DataFrame:
    df = base_df.copy()
    df['forecasted_demand_kg'] = (df['forecasted_demand_kg'] * multiplier).round(1)
    result = solve(df)
    result['scenario'] = f'Festival ({multiplier:.2f}x demand)'
    return result


def run_migration(base_df: pd.DataFrame, target_district: str, influx_pct: float = 20.0) -> pd.DataFrame:
    df = base_df.copy()

    if 'district' not in df.columns or df['district'].isnull().all():
        scale = 1.0 + influx_pct / 100.0
        df['forecasted_demand_kg'] = (df['forecasted_demand_kg'] * scale).round(1)
        result = solve(df)
        result['scenario'] = f'Migration: all villages +{influx_pct:.0f}% (no district data)'
        return result

    scale = 1.0 + influx_pct / 100.0
    mask = df['district'] == target_district
    df.loc[mask, 'forecasted_demand_kg'] = (df.loc[mask, 'forecasted_demand_kg'] * scale).round(1)
    df.loc[mask, 'shortage_risk'] = df.loc[mask, 'shortage_risk'].replace({'Low': 'Medium', 'Medium': 'High'})
    result = solve(df)
    result['scenario'] = f'Migration: {target_district} +{influx_pct:.0f}%'
    return result


def compute_delta(baseline_df: pd.DataFrame, scenario_df: pd.DataFrame) -> pd.DataFrame:
    needed = {'village_id', 'coverage_pct', 'shortage_gap_kg', 'allocated_kg'}
    for col in needed:
        if col not in baseline_df.columns:
            baseline_df = baseline_df.copy()
            baseline_df[col] = 0.0
        if col not in scenario_df.columns:
            scenario_df = scenario_df.copy()
            scenario_df[col] = 0.0

    base = baseline_df[['village_id', 'coverage_pct', 'shortage_gap_kg', 'allocated_kg']].copy()
    scen = scenario_df[['village_id', 'coverage_pct', 'shortage_gap_kg', 'allocated_kg']].copy()

    base = base.rename(columns={
        'coverage_pct':    'base_coverage_pct',
        'shortage_gap_kg': 'base_shortage_kg',
        'allocated_kg':    'base_allocated_kg',
    })
    scen = scen.rename(columns={
        'coverage_pct':    'scen_coverage_pct',
        'shortage_gap_kg': 'scen_shortage_kg',
        'allocated_kg':    'scen_allocated_kg',
    })

    delta = base.merge(scen, on='village_id', how='inner')
    delta['coverage_change'] = (delta['scen_coverage_pct'] - delta['base_coverage_pct']).round(1)
    delta['shortage_change'] = (delta['scen_shortage_kg']  - delta['base_shortage_kg']).round(1)

    for col in ('village_name', 'district', 'shortage_risk'):
        if col in scenario_df.columns:
            delta = delta.merge(
                scenario_df[['village_id', col]].drop_duplicates('village_id'),
                on='village_id', how='left'
            )

    return delta.sort_values('coverage_change').reset_index(drop=True)
