import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

WAREHOUSES = [
    {'warehouse_id': 'WH_NORTH',   'name': 'Chennai Central Depot',   'lat': 13.0827, 'lon': 80.2707, 'stock_kg': 80000},
    {'warehouse_id': 'WH_CENTRAL', 'name': 'Tiruchirappalli Depot',   'lat': 10.7905, 'lon': 78.7047, 'stock_kg': 65000},
    {'warehouse_id': 'WH_SOUTH',   'name': 'Madurai South Depot',     'lat':  9.9252, 'lon': 78.1198, 'stock_kg': 55000},
]

ALPHA = 1000.0
BETA  = 1.0
RISK_WEIGHTS = {'High': 3.0, 'Medium': 2.0, 'Low': 1.0}


def haversine_km(lat1, lon1, lat2, lon2):
    R    = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a    = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def solve(village_df: pd.DataFrame, stock_overrides: dict = None) -> pd.DataFrame:
    df = village_df.copy().reset_index(drop=True)

    warehouses = []
    for wh in WAREHOUSES:
        wh_copy = wh.copy()
        if stock_overrides and wh['warehouse_id'] in stock_overrides:
            wh_copy['stock_kg'] = stock_overrides[wh['warehouse_id']]
        warehouses.append(wh_copy)

    n_wh  = len(warehouses)
    n_vil = len(df)

    if 'lat' not in df.columns or df['lat'].isnull().all():
        np.random.seed(42)
        df['lat'] = np.random.uniform(8.0, 13.5, n_vil)
        df['lon'] = np.random.uniform(76.9, 80.3, n_vil)

    dist_matrix = np.zeros((n_wh, n_vil))
    for w, wh in enumerate(warehouses):
        for v in range(n_vil):
            dist_matrix[w, v] = haversine_km(
                wh['lat'], wh['lon'],
                df.at[v, 'lat'], df.at[v, 'lon']
            )

    df['risk_weight'] = df['shortage_risk'].map(RISK_WEIGHTS).fillna(1.0)
    demands = df['forecasted_demand_kg'].values
    stocks  = np.array([wh['stock_kg'] for wh in warehouses])
    weights = df['risk_weight'].values

    # 🔥 FIXED SOLVER CREATION
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if solver is None:
        solver = pywraplp.Solver.CreateSolver('CBC')
    if solver is None:
        raise RuntimeError("No solver available. Check OR-Tools installation.")

    x = {}
    for w in range(n_wh):
        for v in range(n_vil):
            x[w, v] = solver.NumVar(0.0, demands[v], f'x_{w}_{v}')

    # 🔥 FIXED CONSTRAINTS (core fix)
    for w in range(n_wh):
        c = solver.Constraint(0, float(stocks[w]))
        for v in range(n_vil):
            c.SetCoefficient(x[w, v], 1.0)

    for v in range(n_vil):
        c = solver.Constraint(0, float(demands[v]))
        for w in range(n_wh):
            c.SetCoefficient(x[w, v], 1.0)

    obj = solver.Objective()
    for v in range(n_vil):
        for w in range(n_wh):
            obj.SetCoefficient(
                x[w, v],
                BETA * dist_matrix[w, v] - ALPHA * weights[v]
            )
    obj.SetMinimization()

    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("Solver did not find optimal solution.")

    records = []
    for v in range(n_vil):
        row         = df.iloc[v]
        total_alloc = sum(x[w, v].solution_value() for w in range(n_wh))

        primary_wh  = warehouses[
            max(range(n_wh), key=lambda w: x[w, v].solution_value())
        ]['warehouse_id']

        shortage    = max(0.0, demands[v] - total_alloc)
        coverage    = (total_alloc / demands[v] * 100) if demands[v] > 0 else 0.0

        rec = {
            'village_id':           row['village_id'],
            'forecasted_demand_kg': round(demands[v], 1),
            'allocated_kg':         round(total_alloc, 1),
            'shortage_gap_kg':      round(shortage, 1),
            'coverage_pct':         round(coverage, 1),
            'primary_warehouse':    primary_wh,
            'shortage_risk':        row.get('shortage_risk', 'Low'),
        }

        for col in ('village_name', 'district', 'bpl_count', 'lat', 'lon'):
            if col in row.index:
                rec[col] = row[col]

        records.append(rec)

    return pd.DataFrame(records)


def get_warehouse_info() -> list:
    return WAREHOUSES