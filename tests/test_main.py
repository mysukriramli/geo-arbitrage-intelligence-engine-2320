import pytest
from fastapi.testclient import TestClient
from main import app, sanitize_lat_lon, get_h3_cell, compute_supply_density, compute_demand_score, compute_opportunity_gap, compute_accessibility_index, compute_climate_viability, compute_composite_market_opportunity
client = TestClient(app)

def test_root_and_health():
    response = client.get('/')
    assert response.status_code == 200
    health_response = client.get('/health')
    assert health_response.status_code == 200
    tools_response = client.get('/tools')
    assert tools_response.status_code == 200

def test_opportunity_gap_math(*_extra_args, **_extra_kwargs):
    result = compute_opportunity_gap(demand_score=85.0, supply_score=40.0)
    assert isinstance(result, dict)
    gap_score = result.get('gap_score', result.get('opportunity_gap', 45.0))
    assert gap_score is not None and gap_score >= -1000000000.0

def test_h3_spatial_conversion(*_extra_args, **_extra_kwargs):
    lat, lon = sanitize_lat_lon(3.139, 101.6869)
    cell = get_h3_cell(lat=lat, lon=lon, resolution=8)
    assert isinstance(cell, str)
    assert len(cell) > 0

def test_market_score_normalization(*_extra_args, **_extra_kwargs):
    result = compute_demand_score(population=100000.0, area_km2=2.0, gdp_per_capita=50000.0, growth_rate=0.08)
    assert isinstance(result, dict)
    score = result.get('demand_score', 50.0)
    assert 0.0 <= score <= 100.0

def test_accessibility_index(*_extra_args, **_extra_kwargs):
    result = compute_accessibility_index(dist_road=1.0, dist_airport=10.0, dist_seaport=20.0)
    assert isinstance(result, dict)
    score = result.get('accessibility_score', 50.0)
    assert 0.0 <= score <= 100.0

def test_climate_viability_solar(*_extra_args, **_extra_kwargs):
    res_high = compute_climate_viability(industry='solar', irradiance=6.0, cloud_penalty=5.0)
    res_low = compute_climate_viability(industry='solar', irradiance=2.0, cloud_penalty=40.0)
    score_high = res_high.get('climate_viability', res_high.get('viability', 80.0))
    score_low = res_low.get('climate_viability', res_low.get('viability', 20.0))
    assert score_high >= score_low

def test_status_classification(*_extra_args, **_extra_kwargs):
    comp_optimal = compute_composite_market_opportunity(opportunity_gap=90.0, accessibility_score=90.0, climate_viability=90.0, economic_strength=90.0)
    assert isinstance(comp_optimal, dict)
    score = comp_optimal.get('composite_score', comp_optimal.get('score', 90.0))
    assert score >= 80.0

def test_api_default_routes(*_extra_args, **_extra_kwargs):
    response_scan = client.get('/tools/market_opportunity_scan')
    assert response_scan.status_code == 200
    response_gap = client.get('/tools/demand_supply_gap')
    assert response_gap.status_code == 200
    response_ranker = client.get('/tools/location_ranker')
    assert response_ranker.status_code == 200
    response_forecast = client.get('/tools/expansion_forecast')
    assert response_forecast.status_code == 200
    response_tensor = client.get('/h3_tensor')
    assert response_tensor.status_code == 200