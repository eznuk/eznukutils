import pytest
import numpy as np

@pytest.fixture(params=["CO2", "He", "N2", "Ar"])
def gas(request):
    return request.param

@pytest.fixture(params=["D2", "C2"])
def channel(request):
    return request.param

sensors = [1000, 100, 10, 1]

@pytest.fixture(params=sensors)
def sensor_in(request):
    return request.param

@pytest.fixture(params=sensors)
def sensor_out(request):
    return request.param

@pytest.fixture(params=[
    300.2,
    np.float64(300.2), 
    np.array([300, 302]), 
    np.array([300.2, 302.3])])
def NAL_sample(request):
    return request.param