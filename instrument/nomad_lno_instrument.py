
import numpy as np
#import matplotlib.pyplot as plt
#import os



##quadratic test values
##MAYBE NOT CORRECT FOR SO!
#Q0=2.50515577e+00
#Q1=-2.09395082e-02
#Q2=6.53155833e-05
#T0 = -6.267734 / 7.299039E-1
#def t_p0_bad(m, t, Q0=Q0, Q1=Q1, Q2=Q2):
#    """instrument temperature to pixel0 shift"""
#    new_Q1 = Q0 + m * (Q1 + m * Q2)
#    p0 = new_Q1 * (t-T0)
#    return p0

#old values
Q0=-6.267734
Q1=-7.299039e-1
Q2=0.0
#T0 = -6.267734 / 7.299039E-1
def t_p0_old(t, Q0=Q0, Q1=Q1, Q2=Q2):
    """instrument temperature to pixel0 shift"""
#    t0 = Q0 / Q1
#    p0 = Q0 + (t0+t) * (Q1 + (t0+t) * Q2)
    p0 = Q0 + t * (Q1 + t * Q2)
    return p0

#new values from mean gradient/offset of best orders: 142, 151, 156, 162, 166, 167, 178, 189, 194
Q0=-10.13785778
Q1=-0.829174444
Q2=0.0
def t_p0(t, Q0=Q0, Q1=Q1, Q2=Q2):
    """instrument temperature to pixel0 shift"""
    p0 = Q0 + t * (Q1 + t * Q2)
    return p0

#temperature required for zero pixel shift
temperature_p0 = -Q0/Q1



#updated for LNO
F0=22.478113
F1=5.508335e-4
F2=3.774791e-8
#F0=22.473422
#F1=5.559526e-4
#F2=1.751279e-8

"""
nu/m = FO + (p+p0) F1 + (p+p0)**2 F2

"""

def nu_mp(m, p, t, F0=F0, F1=F1, F2=F2):
    """pixel number and order to wavenumber calibration. Liuzzi et al. 2018"""
    p0 = t_p0(t)
    f = (F0 + (p+p0)*(F1 + F2*(p+p0)))*m
    return f

def pixel_mnu(m, nu, t, F0=F0, F1=F1, F2=F2):
    """order and wavenumber to pixel calibration. Inverse of function from Liuzzi et al. 2018 using same coefficients"""
    p0 = t_p0(t)
    p = (-F1 + np.sqrt(F1**2 - 4*F2*(F0-nu/m))) / (2*F2) - p0
    return p

def t_nu_mp(m, nu, p, F0=F0, F1=F1, F2=F2, Q0=Q0, Q1=Q1):
    """order, wavenumber and pixel to temperature calibration. Inverse of function from Liuzzi et al. 2018 using same coefficients"""
    #give single nu and p (e.g. absorption band minimum) to find t
    # p0 = t_p0(t)
    p0 = (-F1 + np.sqrt(F1**2 - 4*F2*(F0-nu/m))) / (2*F2) - p
    t = (p0 - Q0)/Q1
    return t

def order_nu0p(nu0, p, t, F0=F0, F1=F1, F2=F2):
    """pixel number and wavenumber to order calibration. Inverse of Liuzzi et al. 2018"""
    p0 = t_p0(t)
    m = nu0 / (F0 + (p+p0)*(F1 + F2*(p+p0)))
    return m


 
bl0=150.80
bl1=0.22
bl2=0.0
def p0_blaze(m, bl0=bl0, bl1=bl1, bl2=bl2):
    """"order to blaze centre in pixels. Liuzzi et al. 2018"""
    p0 = bl0 + m*(bl1 + m*bl2)
    return p0





def wp_blaze(m, t, F0=F0):
    """order to blaze function width (FSR) in pixels. Calculated from F0 in Liuzzi et al. 2018"""
    blaze_centre_px = p0_blaze(m) #get blaze centre (px)
    blaze_centre_nu = nu_mp(m, blaze_centre_px, t) #get blaze centre (cm-1)
    blaze_min_nu = blaze_centre_nu - F0 / 2. #add/subtract half the FSR in cm-1
    blaze_max_nu = blaze_centre_nu + F0 / 2.
    blaze_min_px = pixel_mnu(m, blaze_min_nu, t) #convert to px
    blaze_max_px = pixel_mnu(m, blaze_max_nu, t)
    blaze_width_px = blaze_max_px - blaze_min_px
    return blaze_width_px



def F_blaze(m, p, t, p0=None, wp=None):
    """calculate blaze function in pixels from order and pixel numbers"""
    if p0 == None:
        p0 = p0_blaze(m)
    if wp == None:
        wp = wp_blaze(m, t)
    F = np.sinc((p-p0)/wp)**2
    return F



"""order to aotf frequency"""
#LNO values different to SO
A_aotf = {108:14886, 109:15042, 
    110:15197, 111:15353, 112:15509, 113:15664, 114:15820, 115:15975, 116:16131, 117:16287, 118:16442, 119:16598,
    120:16753, 121:16909, 122:17064, 123:17219, 124:17375, 125:17530, 126:17685, 127:17841, 128:17996, 129:18151, 
    130:18306, 131:18461, 132:18616, 133:18771, 134:18927, 135:19082, 136:19236, 137:19391, 138:19547, 139:19701,
    140:19856, 141:20011, 142:20166, 143:20321, 144:20475, 145:20630, 146:20784, 147:20940, 148:21094, 149:21249,
    150:21403, 151:21557, 152:21712, 153:21867, 154:22021, 155:22175, 156:22330, 157:22484, 158:22639, 159:22793,
    160:22948, 161:23102, 162:23256, 163:23410, 164:23564, 165:23719, 166:23873, 167:24026, 168:24180, 169:24335,
    170:24489, 171:24643, 172:24796, 173:24951, 174:25105, 175:25258, 176:25412, 177:25566, 178:25720, 179:25873,
    180:26027, 181:26181, 182:26335, 183:26488, 184:26642, 185:26795, 186:26949, 187:27103, 188:27256, 189:27409,
    190:27562, 191:27716, 192:27870, 193:28023, 194:28176, 195:28330, 196:28483, 197:28636, 198:28789, 199:28943,
    200:29096, 201:29249, 202:29402, 203:29554, 204:29708, 205:29861, 206:30014, 207:30167, 208:30320, 209:30473,
    210:30625, 0:0
    
}
#A_aotf = {
#    110:14332, 111:14479, 112:14627, 113:14774, 114:14921, 115:15069, 116:15216, 117:15363, 118:15510, 119:15657,
#    120:15804, 121:15951, 122:16098, 123:16245, 124:16392, 125:16539, 126:16686, 127:16832, 128:16979, 129:17126,
#    130:17273, 131:17419, 132:17566, 133:17712, 134:17859, 135:18005, 136:18152, 137:18298, 138:18445, 139:18591,
#    140:18737, 141:18883, 142:19030, 143:19176, 144:19322, 145:19468, 146:19614, 147:19761, 148:19907, 149:20052,
#    150:20198, 151:20344, 152:20490, 153:20636, 154:20782, 155:20927, 156:21074, 157:21219, 158:21365, 159:21510,
#    160:21656, 161:21802, 162:21947, 163:22093, 164:22238, 165:22384, 166:22529, 167:22674, 168:22820, 169:22965,
#    170:23110, 171:23255, 172:23401, 173:23546, 174:23691, 175:23836, 176:23981, 177:24126, 178:24271, 179:24416,
#    180:24561, 181:24706, 182:24851, 183:24996, 184:25140, 185:25285, 186:25430, 187:25575, 188:25719, 189:25864,
#    190:26008, 191:26153, 192:26297, 193:26442, 194:26586, 195:26731, 196:26875, 197:27019, 198:27163, 199:27308,
#    200:27452, 201:27596, 202:27740, 203:27884, 204:28029, 205:28173, 206:28317, 207:28461, 208:28605, 209:28749,
#    210:28893,
#}

aotf_A = ((v,k) for k,v in A_aotf.items())

"""aotf frequency to nearest order"""
def m_aotf(value):
    res_key, res_val = min(A_aotf.items(), key=lambda x: abs(value - x[1]))
    return res_key


#updated for LNO from Liuzzi et al. May not be newest coefficients?
G0=300.67657
G1=0.1422382
G2=9.409476e-8
def nu0_aotf(A, G0=G0, G1=G1, G2=G2):
    """aotf frequency to aotf centre in wavenumbers. Liuzzi 2019"""
    nu0 = G0 + A*(G1 + A*G2)
    return nu0


def nu_grid(m, dnu, t, n=2):
    """calculate wavenumber grid for n orders centred on order m, with resolution dnu"""
    nu_min = nu_mp(m-n, 0, t)
    nu_max = nu_mp(m+n, 319, t)
    nu = np.arange(nu_min, nu_max, dnu)
    return nu


"""AOTF is old formulation from Liuzzi draft"""
I0 = 0.6290016 #from IG/I0 = +0.589821 and I0 + IG = 1
IG = 0.3709984
# I0 = 0.8 #test for LNO dust curvature
# IG = 0.2
W0 = 18.188122
SIGMAG = 12.181137
def F_aotf_goddard19draft(m, nu, t, A=None, nu0=None, iG=IG, i0=I0, w0=W0, sigmaG=SIGMAG):
    """calculate aotf function from order and high resolution wavenumber grid covering all required orders"""
    if A is None:
        A = A_aotf[m]
    if nu0 is None:
        nu0 = nu0_aotf(A)
    F = i0 * np.sinc((nu-nu0)/w0)**2 + iG * np.exp((-1.0 * (nu - nu0)**2)/(sigmaG**2))

    return F


def spec_res_nu_old(nu0):
    """use values from figure in Liuzzi 2019 (resolution too good for LNO)"""
    spectralResolution = np.polyval(np.array([5.80952381e-05, 2.76190476e-03]), nu0) #LNO from Liuzzi et al 2019 figure
    return spectralResolution


def spec_res_nu(nu0):
    """use new values for SO * 2.0"""
    ResolvingPowerCoefficients = [2.15817e-3, -17.3554, 4.59995e4] #SO
    c3 = ResolvingPowerCoefficients
    resolvingPower = np.polyval(c3, nu0)
    spectralResolution = nu0 / resolvingPower * 2.0
    return spectralResolution


def spec_res_order_old(order):
    aotf_frequency = A_aotf[order]
    nu0 = nu0_aotf(aotf_frequency)
    spectral_resolution = spec_res_nu_old(nu0)
    return spectral_resolution


def spec_res_order(order):
    aotf_frequency = A_aotf[order]
    nu0 = nu0_aotf(aotf_frequency)
    spectral_resolution = spec_res_nu(nu0)
    return spectral_resolution





