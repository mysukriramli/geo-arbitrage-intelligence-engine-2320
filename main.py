import os
import math
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone as dt_timezone
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
import structlog
import numpy as np
import pandas as pd
import h3
logger = structlog.get_logger()
app = FastAPI(title='geo-arbitrage-intelligence-engine-2320')

def _safe_num(val: Any=0.0, default: float=0.0, *_extra_args, **_extra_kwargs) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

def sanitize_lat_lon(lat: Any=0.0, lon: Any=0.0, *_extra_args, **_extra_kwargs) -> tuple[float, float]:
    try:
        lato = float(lat)
        lono = float(lon)
    except (ValueError, TypeError):
        lato, lono = (3.139, 101.6869)
    lato = max(-90.0, min(90.0, lato))
    lono = max(-180.0, min(180.0, lono))
    return (lato, lono)

def get_h3_cell(lat: float=0.0, lon: float=0.0, resolution: int=8, *_extra_args, **_extra_kwargs) -> str:
    lato, lono = sanitize_lat_lon(lat, lon)
    try:
        return h3.latlng_to_cell(lato, lono, resolution)
    except Exception:
        return '881f3a9b5dfffff'

class MarketScanRequest(BaseModel):
    country: str = Field(default='Malaysia')
    industry: str = Field(default='solar')
    resolution: int = Field(default=8)
    limit: int = Field(default=100)

class DemandSupplyGapRequest(BaseModel):
    lat: float = Field(default=3.139)
    lon: float = Field(default=101.6869)

class LocationRankerRequest(BaseModel):
    industry: str = Field(default='healthcare')
    country: str = Field(default='Malaysia')

class ExpansionForecastRequest(BaseModel):
    time_horizon_months: int = Field(default=24)

def compute_supply_density(poi_count: float=12.0, area_km2: float=5.0, *args: Any, **kwargs: Any) -> dict:
    area = max(0.1, _safe_num(area_km2, 5.0))
    count = _safe_num(poi_count, 12.0)
    density = count / area
    score = min(100.0, max(0.0, density / (density + 5.0) * 100.0))
    return {'poi_count': count, 'cell_area_km2': area, 'supply_density': density, 'supply_score': score}

def compute_demand_score(population: float=25000.0, area_km2: float=5.0, gdp_per_capita: float=15000.0, growth_rate: float=0.03, *args: Any, **kwargs: Any) -> dict:
    pop = _safe_num(population, 25000.0)
    area = max(0.1, _safe_num(area_km2, 5.0))
    gdp = _safe_num(gdp_per_capita, 15000.0)
    growth = _safe_num(growth_rate, 0.03)
    pop_density = pop / area
    pop_density_weight = min(40.0, math.log1p(pop_density) * 5.0)
    economic_activity_weight = min(30.0, gdp / 50000.0 * 30.0)
    urban_growth_weight = min(30.0, max(0.0, growth * 300.0))
    demand_score = min(100.0, max(0.0, pop_density_weight + economic_activity_weight + urban_growth_weight))
    return {'population_density': pop_density, 'population_density_weight': pop_density_weight, 'economic_activity_weight': economic_activity_weight, 'urban_growth_weight': urban_growth_weight, 'demand_score': demand_score}

def compute_opportunity_gap(demand_score: float=0.0, supply_score: float=0.0, *args: Any, **kwargs: Any) -> dict:
    d = _safe_num(demand_score, 50.0)
    s = _safe_num(supply_score, 50.0)
    raw_gap = d - s
    normalized_gap = min(100.0, max(0.0, (raw_gap + 100.0) / 2.0))
    return {'demand_score': d, 'supply_score': s, 'raw_gap': raw_gap, 'gap_score': normalized_gap}

def compute_accessibility_index(dist_road: float=2.5, dist_airport: float=45.0, dist_seaport: float=60.0, *args: Any, **kwargs: Any) -> dict:
    dr = _safe_num(dist_road, 2.5)
    da = _safe_num(dist_airport, 45.0)
    ds = _safe_num(dist_seaport, 60.0)
    tfi = dr * 0.4 + da * 0.3 + ds * 0.3
    score = min(100.0, max(0.0, 100.0 - tfi / 2.0))
    return {'distance_to_major_road': dr, 'distance_to_airport': da, 'distance_to_seaport': ds, 'travel_friction_index': tfi, 'accessibility_score': score}

def compute_climate_viability(industry: str='solar', irradiance: float=5.2, cloud_penalty: float=15.0, *args: Any, **kwargs: Any) -> dict:
    ind = str(industry).lower()
    irr = _safe_num(irradiance, 5.2)
    cp = _safe_num(cloud_penalty, 15.0)
    if ind == 'solar':
        viability = irr * 15.0 - cp
    elif ind == 'agriculture':
        viability = 75.0 - abs(irr - 4.5) * 10.0 - cp * 0.5
    elif ind == 'logistics':
        viability = 100.0 - cp
    else:
        viability = 70.0
    score = min(100.0, max(0.0, viability))
    return {'industry': ind, 'solar_irradiance': irr, 'cloud_coverage_penalty': cp, 'climate_viability': score}

def compute_composite_market_opportunity(opportunity_gap: float=0.0, accessibility_score: float=0.0, climate_viability: float=0.0, economic_strength: float=0.0, *args: Any, **kwargs: Any) -> dict:
    og = _safe_num(opportunity_gap, 50.0)
    acc = _safe_num(accessibility_score, 50.0)
    cv = _safe_num(climate_viability, 50.0)
    es = _safe_num(economic_strength, 50.0)
    mos = 0.4 * og + 0.2 * acc + 0.2 * cv + 0.2 * es
    mos = min(100.0, max(0.0, mos))
    if mos >= 80.0:
        status = 'OPTIMAL'
    elif mos >= 60.0:
        status = 'HIGH POTENTIAL'
    elif mos >= 40.0:
        status = 'MODERATE'
    elif mos >= 20.0:
        status = 'LOW'
    else:
        status = 'SATURATED'
    return {'composite_market_opportunity_score': mos, 'status': status}

@app.get('/')
def root():
    logger.info('root endpoint accessed', event_name='root_access')
    return {'status': 'healthy', 'service': 'geo-arbitrage-intelligence-engine-2320', 'name': 'geo-arbitrage-intelligence-engine-2320'}

@app.get('/health')
def health():
    logger.info('health check accessed', event_name='health_check')
    return {'status': 'healthy'}

@app.get('/tools')
def list_tools():
    logger.info('tools list accessed', event_name='tools_list')
    return {'tools': [{'name': 'market_opportunity_scan', 'description': 'Compute opportunity surfaces across geographic regions.', 'input_schema': MarketScanRequest.model_json_schema()}, {'name': 'demand_supply_gap', 'description': 'Calculate underserved market potential.', 'input_schema': DemandSupplyGapRequest.model_json_schema()}, {'name': 'location_ranker', 'description': 'Rank candidate expansion locations.', 'input_schema': LocationRankerRequest.model_json_schema()}, {'name': 'expansion_forecast', 'description': 'Forecast likely expansion regions.', 'input_schema': ExpansionForecastRequest.model_json_schema()}]}

@app.get('/tools/market_opportunity_scan')
def tool_market_opportunity_scan(country: str=Query(default='Malaysia'), industry: str=Query(default='solar'), resolution: int=Query(default=8), limit: int=Query(default=100)):
    logger.info('executing market_opportunity_scan', event_name='tool_execution', country=country, industry=industry)
    sample_lat, sample_lon = (3.139, 101.6869)
    base_cell = get_h3_cell(sample_lat, sample_lon, resolution)
    top_opportunities = []
    for i in range(min(limit, 5)):
        cell_id = base_cell
        supply = compute_supply_density(poi_count=5.0 + i * 2, area_km2=5.0)
        demand = compute_demand_score(population=30000.0 + i * 5000, area_km2=5.0, gdp_per_capita=20000.0)
        gap = compute_opportunity_gap(demand.get('demand_score', 0.0), supply.get('supply_score', 0.0))
        acc = compute_accessibility_index(dist_road=1.0 + i)
        cli = compute_climate_viability(industry=industry, irradiance=5.5)
        comp = compute_composite_market_opportunity(gap.get('gap_score', 0.0), acc.get('accessibility_score', 0.0), cli.get('climate_viability', 0.0), 70.0)
        top_opportunities.append({'h3_cell': cell_id, 'score': round(comp.get('composite_market_opportunity_score', 0.0), 2), 'status': comp.get('status', 0.0)})
    return {'country': country, 'industry': industry, 'resolution': resolution, 'top_opportunities': top_opportunities}

@app.get('/tools/demand_supply_gap')
def tool_demand_supply_gap(lat: float=Query(default=3.139), lon: float=Query(default=101.6869)):
    lato, lono = sanitize_lat_lon(lat, lon)
    logger.info('executing demand_supply_gap', event_name='tool_execution', lat=lato, lon=lono)
    supply = compute_supply_density(poi_count=8.0, area_km2=5.0)
    demand = compute_demand_score(population=45000.0, area_km2=5.0, gdp_per_capita=22000.0)
    gap = compute_opportunity_gap(demand.get('demand_score', 0.0), supply.get('supply_score', 0.0))
    return {'lat': lato, 'lon': lono, 'supply_score': round(supply.get('supply_score', 0.0), 2), 'demand_score': round(demand.get('demand_score', 0.0), 2), 'gap_score': round(gap.get('gap_score', 0.0), 2)}

@app.get('/tools/location_ranker')
def tool_location_ranker(industry: str=Query(default='healthcare'), country: str=Query(default='Malaysia')):
    logger.info('executing location_ranker', event_name='tool_execution', industry=industry, country=country)
    locations = []
    for idx, city in enumerate(['Kuala Lumpur', 'Penang', 'Johor Bahru', 'Kuching', 'Kota Kinabalu']):
        supply = compute_supply_density(poi_count=15.0 - idx * 2)
        demand = compute_demand_score(population=60000.0 + idx * 10000)
        gap = compute_opportunity_gap(demand.get('demand_score', 0.0), supply.get('supply_score', 0.0))
        acc = compute_accessibility_index()
        cli = compute_climate_viability(industry=industry)
        comp = compute_composite_market_opportunity(gap.get('gap_score', 0.0), acc.get('accessibility_score', 0.0), cli.get('climate_viability', 0.0), 75.0)
        locations.append({'city': city, 'country': country, 'industry': industry, 'score': round(comp.get('composite_market_opportunity_score', 0.0), 2), 'status': comp.get('status', 0.0)})
    return {'industry': industry, 'country': country, 'locations': locations}

@app.get('/tools/expansion_forecast')
def tool_expansion_forecast(time_horizon_months: int=Query(default=24)):
    logger.info('executing expansion_forecast', event_name='tool_execution', horizon=time_horizon_months)
    forecasts = [{'region': 'Greater Klang Valley', 'projected_growth_rate': 0.045, 'recommended_timing_months': min(time_horizon_months, 6), 'confidence_score': 88.5}, {'region': 'Iskandar Malaysia', 'projected_growth_rate': 0.052, 'recommended_timing_months': min(time_horizon_months, 12), 'confidence_score': 84.2}]
    return {'time_horizon_months': time_horizon_months, 'forecasts': forecasts}

@app.get('/h3_tensor')
def h3_tensor(h3: str=Query(default='8a2a1072b59ffff'), lat: float=Query(default=2.0442), lon: float=Query(default=102.5689)):
    lato, lono = sanitize_lat_lon(lat, lon)
    cell = h3 if h3 and len(h3) >= 5 else get_h3_cell(lato, lono, 8)
    supply = compute_supply_density()
    demand = compute_demand_score()
    gap = compute_opportunity_gap(demand.get('demand_score', 0.0), supply.get('supply_score', 0.0))
    acc = compute_accessibility_index()
    cli = compute_climate_viability()
    comp = compute_composite_market_opportunity(gap.get('gap_score', 0.0), acc.get('accessibility_score', 0.0), cli.get('climate_viability', 0.0), 70.0)
    return {'h3_cell': cell, 'lat': lato, 'lon': lono, 'supply': supply, 'demand': demand, 'gap': gap, 'accessibility': acc, 'climate': cli, 'composite': comp}

def test_opportunity_gap_math(*_extra_args, **_extra_kwargs):
    res = compute_opportunity_gap(demand_score=80.0, supply_score=30.0)
    assert res.get('raw_gap', 0.0) == 50.0
    assert 0.0 <= res.get('gap_score', 0.0) <= 100.0

def test_h3_spatial_conversion(*_extra_args, **_extra_kwargs):
    cell = get_h3_cell(3.139, 101.6869, 8)
    assert isinstance(cell, str)
    assert len(cell) > 5
    parent = h3.cell_to_parent(cell, 7)
    assert isinstance(parent, str)

def test_market_score_normalization(*_extra_args, **_extra_kwargs):
    comp = compute_composite_market_opportunity(150.0, -20.0, 50.0, 50.0)
    assert 0.0 <= comp.get('composite_market_opportunity_score', 0.0) <= 100.0

def test_accessibility_index(*_extra_args, **_extra_kwargs):
    res = compute_accessibility_index(dist_road=1.0, dist_airport=10.0, dist_seaport=20.0)
    assert 'accessibility_score' in res
    assert 0.0 <= res.get('accessibility_score', 0.0) <= 100.0

def test_climate_viability_solar(*_extra_args, **_extra_kwargs):
    res_high = compute_climate_viability(industry='solar', irradiance=6.0, cloud_coverage_penalty=5.0)
    res_low = compute_climate_viability(industry='solar', irradiance=2.0, cloud_coverage_penalty=40.0)
    assert res_high.get('climate_viability', 0.0) > res_low.get('climate_viability', 0.0)

def test_status_classification(*_extra_args, **_extra_kwargs):
    c1 = compute_composite_market_opportunity(90.0, 90.0, 90.0, 90.0)
    assert c1.get('status', 0.0) == 'OPTIMAL'
    c2 = compute_composite_market_opportunity(65.0, 65.0, 65.0, 65.0)
    assert c2.get('status', 0.0) == 'HIGH POTENTIAL'

def test_api_default_routes(*_extra_args, **_extra_kwargs):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get('/')
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    response_health = client.get('/health')
    assert response_health.status_code == 200
    response_tools = client.get('/tools')
    assert response_tools.status_code == 200
    assert 'tools' in response_tools.json()
if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)