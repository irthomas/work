
import numpy as np
#import matplotlib.pyplot as plt
#import os





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

#new values from mean gradient/offset of best LNO orders (SO should be same): 142, 151, 156, 162, 166, 167, 178, 189, 194
Q0=-10.13785778
Q1=-0.829174444
Q2=0.0
def t_p0(t, Q0=Q0, Q1=Q1, Q2=Q2):
    """instrument temperature to pixel0 shift"""
    p0 = Q0 + t * (Q1 + t * Q2)
    return p0






F0=22.473422
F1=5.559526e-4
F2=1.751279e-8
def nu_mp(m, p, t, p0=None, F0=F0, F1=F1, F2=F2):
    """pixel number and order to wavenumber calibration. Liuzzi et al. 2018"""
    if p0 == None:
        p0 = t_p0(t)
    f = (F0 + (p+p0)*(F1 + F2*(p+p0)))*m
    return f


def pixel_mnu(m, nu, t, F0=F0, F1=F1, F2=F2):
    """order and wavenumber to pixel calibration. Inverse of function from Liuzzi et al. 2018 using same coefficients"""
    p0 = t_p0(t)
    p = (-F1 + np.sqrt(F1**2 - 4*F2*(F0-nu/m))) / (2*F2) - p0
    return p


def p0_nu_mp(m, nu, p, F0=F0, F1=F1, F2=F2):
    """order, wavenumber and pixel to first pixel calibration. Inverse of function from Liuzzi et al. 2018 using same coefficients"""
    #give single nu and p (e.g. absorption band minimum) to find p0
    p0 = (-F1 + np.sqrt(F1**2 - 4*F2*(F0-nu/m))) / (2*F2) - p
    return p0


def t_nu_mp(m, nu, p, F0=F0, F1=F1, F2=F2, Q0=Q0, Q1=Q1):
    """order, wavenumber and pixel to temperature calibration. Inverse of function from Liuzzi et al. 2018 using same coefficients"""
    #give single nu and p (e.g. absorption band minimum) to find t
    p0 = p0_nu_mp(m, nu, p)
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



#wp0=811.133822
#wp1=-5.29102188
#wp2=0.0111650642
##def wp_blaze(m, wp0=730.02044, wp1=-4.76191969, wp2=0.0100485578): #0.9 scalar
#def wp_blaze(m, wp0=wp0, wp1=wp1, wp2=wp2):
#    """order to blaze function width (FSR) in pixels. Quadratic fit to data from EXM-NO-SNO-AER-00026-iss0rev3-Spectral_Calibration_Comparison-180515"""
#    wp = wp0 + m*(wp1 + m*wp2)
#    return wp



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
A_aotf = {
    110:14332, 111:14479, 112:14627, 113:14774, 114:14921, 115:15069, 116:15216, 117:15363, 118:15510, 119:15657,
    120:15804, 121:15951, 122:16098, 123:16245, 124:16392, 125:16539, 126:16686, 127:16832, 128:16979, 129:17126,
    130:17273, 131:17419, 132:17566, 133:17712, 134:17859, 135:18005, 136:18152, 137:18298, 138:18445, 139:18591,
    140:18737, 141:18883, 142:19030, 143:19176, 144:19322, 145:19468, 146:19614, 147:19761, 148:19907, 149:20052,
    150:20198, 151:20344, 152:20490, 153:20636, 154:20782, 155:20927, 156:21074, 157:21219, 158:21365, 159:21510,
    160:21656, 161:21802, 162:21947, 163:22093, 164:22238, 165:22384, 166:22529, 167:22674, 168:22820, 169:22965,
    170:23110, 171:23255, 172:23401, 173:23546, 174:23691, 175:23836, 176:23981, 177:24126, 178:24271, 179:24416,
    180:24561, 181:24706, 182:24851, 183:24996, 184:25140, 185:25285, 186:25430, 187:25575, 188:25719, 189:25864,
    190:26008, 191:26153, 192:26297, 193:26442, 194:26586, 195:26731, 196:26875, 197:27019, 198:27163, 199:27308,
    200:27452, 201:27596, 202:27740, 203:27884, 204:28029, 205:28173, 206:28317, 207:28461, 208:28605, 209:28749,
    210:28893,
}


"""aotf frequency to nearest order"""
def m_aotf(value):
    res_key, res_val = min(A_aotf.items(), key=lambda x: abs(value - x[1]))
    return res_key

aw0=2.18543e1
aw1=5.82007e-4
aw2=-2.18387e-7
def w0_aotf(nu0, aw0=aw0, aw1=aw1, aw2=aw2):
    """aotf wavenumber to aotf sinc2 width in wavenumbers. Goddard SWT13 presentation"""
    w0 = aw0 + nu0*(aw1 + aw2*nu0)
    return w0



as0=4.24031
as1=-2.24849e-3
as2=4.25071e-7
def slf_aotf(nu0, as0=as0, as1=as1, as2=as2):
    """aotf wavenumber to aotf sidelobe ratio (no units). Goddard SWT13 presentation"""
    slf = as0 + nu0*(as1 + as2*nu0)
    return slf



ao0=-4.99704e-1
ao1=2.80952e-4
ao2=-3.51707e-8
def offset_aotf(nu0, ao0=ao0, c1=ao1, c2=ao2):
    """aotf wavenumber to aotf offset (+- 2 orders) i.e. straylight leak. Goddard SWT13 presentation"""
    offset = np.max([ao0 + nu0*(ao1 + ao2*nu0), 0.])
    return offset



G0=305.0604
G1=0.1497089
G2=1.34082E-7
#def nu0_aotf(A, G0=313.91768, G1=0.1494441, G2=1.340818E-7): #SWT13
def nu0_aotf(A, G0=G0, G1=G1, G2=G2):
    """aotf frequency to aotf centre in wavenumbers. Update from team telecon Feb 2019"""
    nu0 = G0 + A*(G1 + A*G2)
    return nu0



tau=-6.5278E-5
def delta_nu0_aotf(nu0, t, tau=tau): #T in celsius
    """calculate aotf shift in wavenumbers from temperature and old aotf centre"""
    delta_nu0 = nu0 * t * tau
    return delta_nu0



def nu_grid(m, dnu, t, n=2):
    """calculate wavenumber grid for n orders centred on order m, with resolution dnu"""
    nu_min = nu_mp(m-n, 0, t)
    nu_max = nu_mp(m+n, 319, t)
    nu = np.arange(nu_min, nu_max, dnu)
    return nu



def F_aotf_goddard18b(m, nu, t, A=None, nu0=None, w0=None, slf=None, offset=None, shift=None):
    """calculate aotf function from order and high resolution wavenumber grid covering all required orders"""
    if A is None:
        A = A_aotf[m]
    if nu0 is None:
        nu0 = nu0_aotf(A)
    if w0 is None:
        w0 = w0_aotf(nu0)
    if slf is None:
        slf = slf_aotf(nu0)
    if offset is None:
        offset = offset_aotf(nu0)
    if shift is None:
        shift = delta_nu0_aotf(nu0, t)

    nu0 += shift
    F = np.sinc((nu-nu0)/w0)**2
    F[np.abs(nu-nu0)>=w0] *= slf

    """add aotf offset (straylight leak) and scale to renormalise peak to 1"""
    F = offset + (1.-offset)*F

    return F


    


def spec_res_nu(nu0):
    rp_coefficients = [2.15817e-3, -17.3554, 4.59995e4]
    rp = np.polyval(rp_coefficients, nu0)
    spectral_resolution = nu0 / rp
    return spectral_resolution



def spec_res_order(order):
    aotf_frequency = A_aotf[order]
    nu0 = nu0_aotf(aotf_frequency)
    spectral_resolution = spec_res_nu(nu0)
    return spectral_resolution




"""2021 new functions"""
def F_blaze_goddard21(m, p, t):
    # Calibration coefficients (Liuzzi+2019 with updates in Aug/2019)
    cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])  # Blaze free-spectral-range (FSR) [cm-1 from pixel]
    tcoeff  = np.array([-0.736363, -6.363908])                   # Blaze frequency shift due to temperature [pixel from Celsius]
    blazep  = [0.22,150.8]                                       # Blaze pixel location with order [pixel from order]

    xdat  = np.polyval(cfpixel, p) * m
    dpix = np.polyval(tcoeff, t)
    xdat += dpix * (xdat[-1] - xdat[0]) / 320.0
    
    
    blazep0 = round(np.polyval(blazep, m)) # Center location of the blaze  in pixels
    blaze0 = xdat[blazep0]                    # Blaze center frequency [cm-1]
    blazew = np.polyval(cfpixel, blazep0)      # Blaze width [cm-1]
    dx = xdat - blaze0
    dx[blazep0] = 1.0e-6
    F = (blazew*np.sin(np.pi*dx/blazew)/(np.pi*dx))**2

    return F


def F_aotf_goddard21(m, nu, t, A=None, wd=None, sl=None, af=None, silent=True):
    """don't set m, use A instead to specify the AOTF frequency"""

    if m != 0.0:
        if not A:
            A = A_aotf[int(m)]
        else:
            return None #error if order and AOTF freq supplied

    # AOTF shape parameters
    aotfwc  = [1.11085173e-06, -8.88538288e-03,  3.83437870e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
    aotfsc  = [2.87490586e-06, -1.65141511e-02,  2.49266314e+01] # sidelobes factor [scaler from AOTF frequency cm-1]
    aotfaf  = [-5.47912085e-07, 3.60576934e-03, -4.99837334e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
    
    # Calibration coefficients (Liuzzi+2019 with updates in Aug/2019)
    cfaotf  = np.array([1.34082e-7, 0.1497089, 305.0604])        # Frequency of AOTF [cm-1 from kHz]
    aotfts  = -6.5278e-5                                         # AOTF frequency shift due to temperature [relative cm-1 from Celsius]


    # def sinc(dx, amp, width, lobe, asym):
    #  	sinc = amp*(width*np.sin(np.pi*dx/width)/(np.pi*dx))**2
    #  	ind = (abs(dx)>width).nonzero()[0]
    #  	if len(ind)>0: sinc[ind] = sinc[ind]*lobe
    #  	ind = (dx<=-width).nonzero()[0]
    #  	if len(ind)>0: sinc[ind] = sinc[ind]*asym
    #  	return sinc

    """reverse AOTF asymmetry"""
    def sinc(dx, amp, width, lobe, asym):
        # """asymetry switched 
     	sinc = amp*(width*np.sin(np.pi*dx/width)/(np.pi*dx))**2

     	ind = (abs(dx)>width).nonzero()[0]
     	if len(ind)>0: 
            sinc[ind] = sinc[ind]*lobe

     	ind = (dx>=width).nonzero()[0]
     	if len(ind)>0: 
            sinc[ind] = sinc[ind]*asym

     	return sinc

    nu0 = np.polyval(cfaotf, A)
    nu0 += aotfts * t * nu0

    if not wd:
        wd = np.polyval(aotfwc, nu0)
    if not sl:
        sl = np.polyval(aotfsc, nu0)
    if not af:
        af = np.polyval(aotfaf, nu0)

    if not silent:
        print("nu0:", nu0)
        print("sinc width:", wd)
        print("sidelobe factor:", sl)
        print("asymmetry:", af)
    dx = nu - nu0
    F = sinc(dx, 1.0, wd, sl, af)
    
    return F







##m=134
#m=167
#t = T0
##t = 0.0
#pixels = np.arange(320)
#resp_pixels = [0] * 320
#radiance_pixels = [0] * 320
#
#nu_aotf = nu_grid(m, 0.01, t, n=2)
#bb_radiance = planck(nu_aotf, 425, "cm-1")
#
#plt.figure()
#plt.title("AOTF and blaze function shapes for order %i \u00B12 orders" %m)
#plt.xlabel("Wavenumber (cm-1)")
#plt.ylabel("Relative response")
#
#
#aotf = F_aotf_goddard18b(m, nu_aotf, 0.0)
#plt.plot(nu_aotf, aotf, "k", label="AOTF")
#
#nColours = 5
#cmap = plt.get_cmap('brg')
#colours = [cmap(i) for i in np.arange(nColours)/nColours]
#
#
#for order_index, order in enumerate(range(m-2, m+3, 1)):
#    nu_pixels = nu_mp(order, pixels, t)
#    blaze = F_blaze(order, pixels, t)
#    plt.plot(nu_pixels, blaze, color=colours[order_index], label="Blaze order %i" %order)
#    
#    detector_centre_nu = nu_mp(order, 159.5, t)
#    detector_start_nu = nu_mp(order, 0, t)
#    detector_end_nu = nu_mp(order, 319, t)
#    plt.axvline(x=detector_centre_nu, color=colours[order_index])
#
#    plt.fill_betweenx([0,1], detector_start_nu, x2=detector_end_nu, alpha=0.2, color=colours[order_index])
#
##    if order==m:
#    for pixel in range(320):
#        index = np.abs(nu_aotf - nu_pixels[pixel]).argmin()
#        resp = aotf[index] * blaze[pixel]
#        resp_pixels[pixel] += resp
#        
#        radiance_pixels[pixel] += bb_radiance[index] * resp #find radiance value closest to pixel wavenumber
#        
#
#
#resp_pixels = np.asfarray(resp_pixels) / np.max(resp_pixels)
#
#radiance_pixels = np.asfarray(radiance_pixels)
#
#plt.plot(nu_mp(m, pixels, t), resp_pixels, "k--", label="Combined response")
#
#plt.plot(nu_aotf, bb_radiance/np.max(bb_radiance), label="BB radiance")
#
#window_transmission = cslWindow(nu_aotf)
#plt.plot(nu_aotf, window_transmission, label="TVAC window transmission")
#plt.legend()
#
#
#
#plt.figure()
#plt.plot(resp_pixels)
#plt.title("Detector pixel relative responses due to blaze and aotf functions")
#plt.ylabel("Relative response")
#plt.xlabel("Pixel Number")
#
#plt.figure()
#plt.plot(radiance_pixels)
#plt.title("423K BB radiance contribution to detector pixels")
#plt.ylabel("Radiance")
#plt.xlabel("Pixel Number")



