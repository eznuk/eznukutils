import pytest
from pytest import approx
import numpy as np

from eznukutils import physics as ph
from testutils import NAL_sample



def test_type_of_get_mu():
    mu_float = ph.get_mu("He", 200)
    assert isinstance(mu_float, float)
    mu_arr = ph.get_mu("He", np.array([200, 300]))
    assert isinstance(mu_arr, np.ndarray)

def test_type_of_get_M():
    M_float = ph.get_M("He")
    assert isinstance(M_float, float)
    M_arr = ph.get_M(["He", "CO2"])
    assert isinstance(M_arr, np.ndarray)
    M_arr = ph.get_M(np.array(["He", "CO2"]))
    assert isinstance(M_arr, np.ndarray)

def test_type_of_kgstosccm(NAL_sample):
    ret = ph.kgstosccm(NAL_sample, "He")
    assert type(ret) == type(NAL_sample)

def test_type_of_sccmtokgs(NAL_sample):
    ret = ph.sccmtokgs(NAL_sample, "He")
    assert type(ret) == type(NAL_sample)

def test_type_of_mdot_to_pdot(NAL_sample):
    ret = ph.mdot_to_pdot(NAL_sample, "He", NAL_sample, NAL_sample)
    assert type(ret) == type(NAL_sample)