# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:53:43 2024

@author: iant

TEST DIAGONAL CORRECTION

"""


from instrument.nomad_lno_instrument_v02 import m_aotf


aotf_freq = aotf_hr[70]
spectrum = array_hr[70, :]

aotf_nu = nu0_aotf(aotf_freq)
order = m_aotf(aotf_freq)

px_nus = nu_mp(order+1, px_ixs_hr/HR_SCALER, t)
print(px_nus[0], px_nus[-1])
print(np.abs(px_nus - aotf_nu).argmin())
