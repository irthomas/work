# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:18:50 2019

@author: iant

TEST FOV CALCULATIONS IN PYTHON

"""

import numpy as np

pi = np.pi
dpr = 180.0 / pi
ampr = dpr * 60.0

def planck_sun(xscale): #planck function W/m2/sr/spectral unit
#    units=="wavenumbers"
        temp = 5778.0
        c1=1.191042e-5
        c2=1.4387752
        bb_radiance = ((c1*xscale**3.0)/(np.exp(c2*xscale/temp)-1.0)) / 1000.0 #mW to W
        return bb_radiance


dMars = 215.7e9 #m for 20180611 obs; mean= 227.9e6 km
rSun = 695510.0e3 #m
dEarth = 149.6e9 #m 1 AU

area_sphere_at_mars = 4.0 * pi * dMars**2
area_sphere_at_earth = 4.0 * pi * dEarth**2
surface_area_sun = 4.0 * pi * rSun**2





sun_total_irradiance_earth = 1361.0 #W/m2
sun_total_irradiance_mars = 1361.0 * (dEarth / dMars)**2 #W/m2
sun_total_luminosity = 3.828e26 #W
sun_total_luminosity_calculated = sun_total_irradiance_earth * area_sphere_at_earth #W
sun_total_emittance = sun_total_irradiance_earth * (dEarth / rSun)**2 #W/m2 sun surface

print("sun_total_luminosity", sun_total_luminosity)
print("sun_total_luminosity_calculated",sun_total_luminosity_calculated )

sun_spectral_irradiance_earth = 2.583e-2 #W/m2/cm-1 approx @ order 189 at top of Earth atmosphere
sun_spectral_irradiance_mars = sun_spectral_irradiance_earth * (dEarth / dMars)**2 #W/m2/cm-1 mars
sun_spectral_luminosity = sun_spectral_irradiance_earth * area_sphere_at_earth #W/cm-1
sun_spectral_emittance = sun_spectral_irradiance_earth * (dEarth / rSun)**2 #W/m2/cm-1 sun surface


#get irradiance from calculation W/m2
w_per_m2_earth = sun_total_luminosity / area_sphere_at_earth #W/m2
w_per_m2_mars = sun_total_luminosity / area_sphere_at_mars #W/m2

print("sun_total_irradiance_earth", sun_total_irradiance_earth)
print("w_per_m2_earth", w_per_m2_earth)
print("sun_total_irradiance_mars", sun_total_irradiance_mars)
print("w_per_m2_mars", w_per_m2_mars)


wavenumbers = np.arange(1.0, 55000.0, 1.0)
planck = planck_sun(wavenumbers)
integrated_planck = np.trapz(planck, x=wavenumbers) #W/m2/sr
integrated_planck_sb = (5.67e-8 / pi) * 5778.0**4 #radiance requires factor pi

ratio_plancks = integrated_planck / integrated_planck_sb #all good


sun_integrated_emittance = integrated_planck * pi #W/m2 (radiant emittance = pi * radiance)
sun_integrated_luminosity = sun_integrated_emittance * surface_area_sun #W should be approx same as ref value above
sun_integrated_irradiance_mars = sun_integrated_luminosity / area_sphere_at_mars

print("sun_integrated_irradiance_mars", sun_integrated_irradiance_mars)

print("sun_total_emittance", sun_total_emittance)
print("sun_integrated_emittance", sun_integrated_emittance)

#check planck matches spectral luminosity
sun_spectral_radiance = planck_sun(3860.0) #W/m2/sr/cm-1
sun_spectral_emittance_calculated = sun_spectral_radiance * pi #W/m2 # radiant emittance = pi * radiance
sun_spectral_luminosity_calculated = sun_spectral_emittance_calculated * surface_area_sun

print("sun_spectral_luminosity", sun_spectral_luminosity)
print("sun_spectral_luminosity_calculated", sun_spectral_luminosity_calculated)

print("sun_spectral_emittance", sun_spectral_emittance)
print("sun_spectral_emittance_calculated", sun_spectral_emittance_calculated)



"""nomad calculations"""
incidence_angle = 0.0
cos_incidence_angle = np.cos(incidence_angle / dpr)
dTgo = 400e3 #m orbit

fov_nadir = (dTgo * np.tan(144.0 / ampr)) * (dTgo * np.tan(4.0 / ampr)) * cos_incidence_angle #m2 instantaneous for all detector rows
lno_nadir_total_luminosity = fov_nadir * sun_total_irradiance_mars

fov_sun = (dMars * np.tan(1.0 / ampr)) * (dMars * np.tan(4.0 / ampr)) #sr, assuming centre of sun
lno_sun_total_luminosity = fov_sun * sun_total_emittance

ratio_sun_nadir = lno_sun_total_luminosity / lno_nadir_total_luminosity

##find 1 arcmin on sun in km
#d1arcmin = dMars * np.tan((1.0 / 60.0) * (pi / 180.0))
#
#angleSolar = np.pi * (rSun / dMars) **2
#ratio_fov_full_sun = (pi * rSun**2) / (d1arcmin * d1arcmin * 4.0)




#check signal levels
sun_counts = 4.0e6
nadir_counts = 15.5

ratio_sun_nadir_counts = sun_counts / nadir_counts

rMars = 3389.5e3 #m
dLimb = np.sqrt((rMars+400e3)**2 - (rMars)**2) #m

fovSize = dLimb * np.tan(43.0 / ampr)
print(fovSize)
fovSize = dLimb * np.tan(144.0 / ampr)
print(fovSize)



orbit_seconds = 2 * 60.0 * 60.0
orbit_degrees = 360.0
orbit_degrees_per_second = orbit_degrees / orbit_seconds

circumference = 2.0 * pi * rMars
orbit_km_per_second = circumference / orbit_seconds / 1000.0
