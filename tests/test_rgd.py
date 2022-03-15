import numpy as np
from typing import Tuple

from eznukutils import rgd
from testutils import NAL_sample, gas

def test_type_of_get_pin_pout(NAL_sample, gas):
    ret = rgd.get_pin_pout(
        NAL_sample, 1, NAL_sample, gas
    )
    assert type(ret) == tuple
    assert isinstance(ret[0], type(NAL_sample))
    assert isinstance(ret[1], type(NAL_sample))

def test_type_of_mfp_to_p_visc(NAL_sample, gas):
    ret = rgd.mfp_to_p_visc(NAL_sample, NAL_sample, gas)
    assert isinstance(ret, type(NAL_sample))

def test_type_of_mfp_visc(NAL_sample, gas):
    ret = rgd.mfp_visc(NAL_sample, NAL_sample, gas, NAL_sample, NAL_sample)
    assert isinstance(ret, type(NAL_sample))

def test_type_of_mfp(NAL_sample):
    ret = rgd.mfp(NAL_sample, 1e-12, NAL_sample)
    assert isinstance(ret, type(NAL_sample))

def test_type_of_get_Kn(NAL_sample, gas):
    ret = rgd.get_Kn(NAL_sample, NAL_sample,
                     NAL_sample, 1, gas)
    assert isinstance(ret, type(NAL_sample))
    
def test_type_of_mdot_to_g(NAL_sample, gas):
    ret = rgd.mdot_to_g(NAL_sample, 1, 1, 1, NAL_sample, NAL_sample, gas)
    assert isinstance(ret, type(NAL_sample))

    ret = rgd.mdot_to_g(NAL_sample, 1, 1, 1, NAL_sample, NAL_sample, 
                        ["He", "CO2"])
    assert isinstance(ret, np.ndarray)

def test_type_of_mvel(NAL_sample):
    ret = rgd.mvel(NAL_sample, NAL_sample)
    assert isinstance(ret, type(NAL_sample))

def test_type_of_mdot_to_g_veltzke(NAL_sample, gas):
    ret = rgd.mdot_to_g_veltzke(NAL_sample, 1, 1, 1, NAL_sample,
                                NAL_sample, gas)
    assert isinstance(ret, type(NAL_sample))

def test_type_of_mdot_to_g_graur(NAL_sample, gas):
    ret = rgd.mdot_to_g_graur(NAL_sample, NAL_sample, 1, 1, 1,
                              NAL_sample+1, NAL_sample, 123)
    assert isinstance(ret, type(NAL_sample))

def test_type_of_mdot_to_q_bar_karniadakis(NAL_sample, gas):
    ret = rgd.mdot_to_q_bar_karniadakis(NAL_sample, NAL_sample, 1, 1, 1,
                                        NAL_sample+1, NAL_sample, gas)
    assert isinstance(ret, type(NAL_sample))


def test_type_of_to_mdot(NAL_sample, gas):
    fun = rgd.mdot_to_g
    ret = rgd.g_to_mdot(fun, NAL_sample,
                        1, 1, 1, NAL_sample, NAL_sample, gas)
    assert isinstance(ret, type(NAL_sample))