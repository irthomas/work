import os
import json
import sys
import numpy as np
import glob
import h5py
from nomad_ops.core.hdf5.l1p0a_to_1p0b.paths import NOMADParams



DEFAULT_GEM_VERSION = 'fd-a585'
GEM_PATHS = {
  'fd-a585' : {
    'version_string':'gem-mars-a585', 
    'output_dir':os.path.join(NOMADParams["GEM_OUTPUT_DIR"], 'fd-a585', "hdf5")
    },
  'hl-a758' : {
    'version_string':'gem-mars-a758', 
    'output_dir':os.path.join(NOMADParams["GEM_OUTPUT_DIR"], 'hl-a758', "hdf5")
    }
  }



def get_obs_geometry(hdf5File, zind=None):

    geom = {}
    geom['Ls'] = np.mean(hdf5File["Geometry/LSubS"][...], axis=1)
    geom['Lat'] = np.mean(hdf5File["Geometry/Point0/Lat"][...], axis=1)
    geom['Lon'] = np.mean(hdf5File["Geometry/Point0/Lon"][...], axis=1)
    geom['LST'] = np.mean(hdf5File["Geometry/Point0/LST"][...], axis=1)
    geom['SZA'] = np.ones_like(geom['Ls'])*90.
    geom['CSZA'] = np.zeros_like(geom['Ls'])
    geom['TangentAlt'] = np.mean(hdf5File["Geometry/Point0/TangentAltAreoid"][...], axis=1)

    if zind is not None:
        for key in geom:
            geom[key] = geom[key][zind]

    return geom


def get_observation_atmolist_filename(hdf5File, hdf5Filename, gem_version=DEFAULT_GEM_VERSION):
    """ get the atmosphere filelist for a observation file 
    
    Args:
        obs_filename (str): observation filename or shortname

    Returns:
        atmosphere filelist associated with obs_filename

    """

    obs_filename = hdf5Filename
    #
    if gem_version in GEM_PATHS:
        gem_version_string = GEM_PATHS[gem_version]['version_string']
    else:
        raise Exception("GEM_VERSION '%s' not implemented"%gem_version)

    #
    obs_shortname = os.path.splitext(os.path.basename(obs_filename))[0]
    obs_year = obs_shortname[0:4]
    obs_month = obs_shortname[4:6]
    obs_day = obs_shortname[6:8]
    atmo_subdir = os.path.join(gem_version_string, obs_year, obs_month, obs_day, obs_shortname)
    atmolist_filename = os.path.join(atmo_subdir, gem_version_string + '_' + obs_shortname + '_list.dat')

    if not os.path.exists(os.path.join(NOMADParams["APRIORI_FILE_DESTINATION"], atmolist_filename)):
        print(os.path.join(NOMADParams["APRIORI_FILE_DESTINATION"], atmolist_filename), "not found")
        print("Creating atmo files for %s" % obs_filename)
        create_observation_zpt_dir(hdf5File, hdf5Filename)

    return atmolist_filename



# set up target arrays (using dictionary for that, it makes naming flexible)
GEM_molecules = ('Ar', 'CO2', 'CO', 'H', 'H2', 'H2O', 'H2O2', 'HO2', 'N2', 'OH', 'O2', 'O3', 'O_3p', 'O_1d')
GEM_aerosols = ('d0.1', 'd1.5', 'd10', 'H2O_ice')
GEM_hdf_fields = {'z': 'height_t', 'T': 'temperature', 'P': 'pressure_t', 'NT': 'air_nd'}
for key in GEM_molecules+GEM_aerosols:
    GEM_hdf_fields[key] = key.lower() + '_vmr'

# set up some arrays to help with atmofile output
atmofile_header1 = '%%%11s%12s%16s%16s' % ('Z(km)', 'T(K)', 'P(mb)', 'NT(cm^-3)')
atmofile_header2 = '%%#%10s%12s%16s%16s' % ('Z', 'T', 'P', 'N')
atmofile_printfmt = ['%12.6f', '%12.6f', '%16.6e', '%16.6e']
atmofile_vars = ['z', 'T', 'P', 'NT']
atmofile_fact = [1.e-3, 1., 0.01, 1.e-6]

for key in ('Ar', 'CO', 'CO2', 'H2O', 'N2', 'O2', 'O3', 'd0.1', 'd1.5', 'd10', 'H2O_ice'):
    if key in GEM_molecules:
        atmofile_header1 += '%16s' % (key+'(ppmv)')
        atmofile_header2 += '%16s' % key
        atmofile_printfmt.append('%16.6e')
        atmofile_vars.append(key)
        atmofile_fact.append(1.e6)
    if key in GEM_aerosols:
        atmofile_header1 += '%16s' % (key+'(cm^-3)')
        atmofile_header2 += '%16s' % key
        atmofile_printfmt.append('%16.6e')
        atmofile_vars.append(key)
        atmofile_fact.append(1.e-6)


def create_observation_zpt_dir(hdf5File, hdf5Filename, atmo_dir=None, gem_version=DEFAULT_GEM_VERSION, **kwargs):
    """ create directory with filelist corresponding to observation 
    
    Loops through the bins and spectra, calling compute_interpolated_zpt_file for each. 
    Creates a individual filelist, along with the atmosphere files corresponding
    to the complete observation.

    Args:
        obs_filename (string): the observation .h5 file, relative or absolute accepted.

    Examples:
        >> python gem-mars_tool.py --observation 20161122_153906_0p3a_LNO_D_169.h5
        will make (if needed) the directory '20161122_153906_0p3a_LNO_D_169' within the GEM_VERSION_STRING
        subdirectory of the NOMADParams["APRIORI_FILE_DESTINATION"], and populate all the files for the observation. To
        include these in the asimut input, assume the Atmosphere directory is set correctly, then set:
        atmofile=gem-mars-1585/220161122_153906_0p3a_LNO_D_169/gem-mars-a461_20161122_153906_0p3a_LNO_D_169_list.dat
        filetype=list
    """

    obs_filename = hdf5Filename

    # set obs_absfilename and obs_shortname
    obs_shortname = os.path.splitext(os.path.basename(obs_filename))[0]
    print(obs_shortname)

    # get geometry information
    geom = get_obs_geometry(hdf5File)
    NbSpectra = len(geom['Lat'])
    print(NbSpectra)

    # set umask so files and directories are accessable by group
    os.umask(0o007)

    #
    if gem_version in GEM_PATHS:
        gem_version_string = GEM_PATHS[gem_version]['version_string']
    else:
        raise Exception("GEM_VERSION '%s' not implemented"%gem_version)

    # create subfolder
    if atmo_dir is None:
        obs_year = obs_shortname[0:4]
        obs_month = obs_shortname[4:6]
        obs_day = obs_shortname[6:8]
        atmo_dir = os.path.join(NOMADParams["APRIORI_FILE_DESTINATION"], gem_version_string, obs_year, obs_month, obs_day, obs_shortname)
        if not os.path.isdir(atmo_dir):
            os.makedirs(atmo_dir)
        atmo_relpath = os.path.relpath(atmo_dir, NOMADParams["APRIORI_FILE_DESTINATION"])
    else:
        atmo_dir = os.path.join(atmo_dir, obs_shortname)
        if not os.path.isdir(atmo_dir):
            os.makedirs(atmo_dir)
        atmo_relpath = obs_shortname

    print(atmo_dir)
    #sys.exit()

    # loop through spectra to create files and filelists
    atmolist_filename = gem_version_string + '_' + obs_shortname + '_list.dat'
    atmolist_contents = ''
    for ns in range(NbSpectra):
        # may need to consider fringe cases where geometry is -999
        print("ns=%d/%d:"%(ns, NbSpectra))
        atmo_filename = gem_version_string + '_' + obs_shortname + '_sp%04d.dat' % ns
        create_interpolated_zpt_file(geom['Ls'][ns], geom['Lat'][ns], geom['Lon'][ns], geom['LST'][ns],
            atmo_filename=os.path.join(atmo_dir, atmo_filename), gem_version=gem_version, **kwargs)
        atmolist_contents += os.path.join(atmo_relpath, atmo_filename) + '\n'
    with open(os.path.join(atmo_dir, atmolist_filename), 'w') as f:
        f.write(atmolist_contents)


def create_interpolated_zpt_file(Ls, Lat, Lon, LST, atmo_filename=None, order_Ls=3, order_LST=3, ZMETHOD='MOLA', gem_version=DEFAULT_GEM_VERSION):
    """ create an interpolate atmo file using GEM 
    
    Interpolates a atmosphere from the GEM-Mars database. From a particular GEM file,
    spherical bilinear interpolation is used to compute a local atmosphere and time. 
    A Lagrange polynomial of order "order_LST" is used to interpolate several time-steps
    in a week to the correct LST. Finally, A Lagrange polynomial of order "order_Ls" is 
    used to interpolate several weeks to the correct Ls. 

    Args:
        Ls (float): observation geometry parameters
        Lat (float): observation geometry parameters
        Lon (float): observation geometry parameters
        LST (float): observation geometry parameters
        atmo_filename (str): output file name (default is None)
        order_Ls (int): polynomial order for interpolation for Ls (1 is linear, 3 is cubic, default is 3)
        order_LST (int): polynomial order for interpolation for LST (1 is linear, 3 is cubic, default is 3)
        ZMETHOD (None, GEM-MOLA, MOLA): method to adapt to surface topography

    """


    # I assume a lot about the naming and organization of the gem files here, but read in names and Ls into a 36x48 array
    NbLs_gem, NbLST_gem = 36, 48
    gem_filelist = np.empty([NbLs_gem, NbLST_gem], dtype='a128')
#    print("output dir=", os.path.join(GEM_PATHS[gem_version]['output_dir'], '*.h5'))
    if gem_version == 'fd-a585':
        # standard senario
        filelist = glob.glob(os.path.join(GEM_PATHS[gem_version]['output_dir'], '*.h5'))
        for iLs in range(NbLs_gem):
            gem_filelist[iLs,:] = filelist[iLs*NbLST_gem:(iLs+1)*NbLST_gem]
    elif gem_version == 'hl-a758':
        # dusty senario (use 0-170 from standard senario, and 180-350 from dusty senario)
        filelist = glob.glob(os.path.join(GEM_PATHS['fd-a585']['output_dir'], '*.h5'))
        for iLs in range(18):
            gem_filelist[iLs,:] = filelist[iLs*NbLST_gem:(iLs+1)*NbLST_gem]
        filelist = glob.glob(os.path.join(GEM_PATHS['hl-a758']['output_dir'], '*.h5'))
        for iLs in range(18):
            gem_filelist[iLs+18,:] = filelist[iLs*NbLST_gem:(iLs+1)*NbLST_gem]
    else:
        raise Exception("The GEM version '%s' is not supported"%gem_version)

    #
    gem_Ls = np.array([[float(name[-11:-3]) for name in row] for row in gem_filelist])

    # normally should read in lat and lon, assume they are the same in each file
    abs_filename = gem_filelist[0,0]
    with h5py.File(abs_filename, 'r') as f:
        gem_Lat = f['lat'][:]
        gem_Lon = f['lon'][:]
        gem_Z = f['height_t'][0,0,:]
    NbLat = len(gem_Lat)
    """NbLon = len(gem_Lon)"""
    NbZ = len(gem_Z)

    #
    if Lon < 0.:
       Lon = Lon + 360. 
    iLon = np.searchsorted(gem_Lon, Lon) - 1
    if iLon < 0:
        iLon = 0
    print("Lon=%.1f is in the range"%Lon, gem_Lon[iLon:iLon+2])

    #
    if (Lat <= gem_Lat[0]):
        #print("Warning, Lat=%.1f is outside of GEM range, using lower boundary (Lat=%.1f)."%(Lat,gem_Lat[0]))
        iLat = 0
        #Lat = gem_Lat[0]
        w_Lon_Lat = bilinear_spherical_interp_weights(gem_Lat[iLat:iLat+2], gem_Lon[iLon:iLon+2], gem_Lat[0], Lon).T
    elif (Lat >= gem_Lat[-1]):
        #print("Warning, Lat=%.1f is outside of GEM range, using upper boundary (Lat=%1f)."%(Lat,gem_Lat[-1]))
        iLat = NbLat - 2
        #Lat = gem_Lat[-1]
        w_Lon_Lat = bilinear_spherical_interp_weights(gem_Lat[iLat:iLat+2], gem_Lon[iLon:iLon+2], gem_Lat[-1], Lon).T
    else:
        iLat = np.searchsorted(gem_Lat, Lat) - 1
        w_Lon_Lat = bilinear_spherical_interp_weights(gem_Lat[iLat:iLat+2], gem_Lon[iLon:iLon+2], Lat, Lon).T
    print("Lat=%.1f is in the range"%Lat, gem_Lat[iLat:iLat+2])
    #  build weights for spherical bilinear interpolation
    #print('  w_Lon_Lat = ', w_Lon_Lat.tolist())
    
    #
    #print("gem_Ls =", gem_Ls[:,0])
    if order_Ls == 0:
        # find nearest
        if (Ls < gem_Ls[0,0]) or (Ls > gem_Ls[-1,0]):
            iLs = 0
            if Ls <= gem_Ls[0]:
                iLs = 0 if (abs(gem_Ls[0,0]-Ls) <= abs(gem_Ls[-1,0]-360.-Ls)) else NbLs_gem-1
            else:
                iLs = 0 if (abs(gem_Ls[0,0]+360.-Ls) <= abs(gem_Ls[-1,0]-Ls)) else NbLs_gem-1
        else:
            iLs = np.argmin((gem_Ls[:,0]-Ls)**2)
        indLs = [iLs]
        #print("Ls=%.2f is nearest"%Ls, gem_Ls[indLs,0])
    else:
        if (Ls < gem_Ls[0,0]) or (Ls > gem_Ls[-1,0]):
            iLs = NbLs_gem-1
        else:
            iLs = np.searchsorted(gem_Ls[:,0], Ls) - 1
        k = order_Ls
        indLs = [int(num) for num in np.mod(np.linspace(np.ceil(iLs-k/2.),np.ceil(iLs+k/2),k+1), NbLs_gem)]
        #print("Ls=%.2f is in the range"%Ls, gem_Ls[indLs,0])

    #
    """Ls_array2  = np.zeros((order_Ls+1,order_LST+1))"""
    Ls_array   = np.zeros(order_Ls+1)
    LST_array2 = np.zeros((order_Ls+1,order_LST+1))

    #
    val_array2 = {}
    val_array = {}
    val_new = {}
    for key in atmofile_vars:
        val_array2[key] = np.zeros((order_Ls+1,order_LST+1,NbZ))
        val_array[key] = np.zeros((order_Ls+1,NbZ))
        val_new[key] = np.zeros(NbZ)

    # outer loop over Ls
    for i, iLs in enumerate(indLs):

        # interpolate LST_array to Lat and Lon
        LST_array = np.zeros(NbLST_gem)
        for iLST in range(NbLST_gem):
            abs_filename = gem_filelist[iLs,iLST]
            with h5py.File(abs_filename, 'r') as f:
                tarray = f['localtime'][iLon:iLon+2,iLat:iLat+2]
                tarray = np.where(tarray >= tarray[0,0], tarray, tarray+24.)   # shift to avoid wrap around 24h
                LST_array[iLST] = (w_Lon_Lat * tarray).sum()   # interpolate to Lat and Lon
                if LST_array[iLST] > 24.:
                    LST_array[iLST] -= 24.   # shift to avoid wrap around 24h
                
        # determine indLST
        if order_LST == 0:
            iLST = np.argmin((LST_array-LST)**2)
            indLST = [iLST]
            #print("  LST=%.1f is nearest"%LST, LST_array[indLST])
            LST_array2[i,:] = LST_array[indLST]
        else:
            if (LST < LST_array[0]) and (LST > LST_array[-1]):
                iLST = NbLST_gem - 1
                # avoid 24h wrap near edges
                for j in range(order_LST):
                    if LST_array[j+1] < LST_array[j]:
                        LST_array[j+1] += 24.
                    if LST_array[-(j+1)] > LST_array[-j]:
                        LST_array[j+1] -= 24.
            else:
                if LST < LST_array[0]:
                    LST_array = np.where(LST_array<LST_array[0], LST_array, LST_array-24.) # shift values to make list sorted
                    iLST = np.searchsorted(LST_array, LST) - 1
                else:
                    LST_array = np.where(LST_array<LST_array[0], LST_array+24., LST_array) # shift values to make list sorted
                    iLST = np.searchsorted(LST_array, LST) - 1
    
                for j in range(order_LST):
                    if LST_array[np.mod(iLST+j+1,NbLST_gem)] < LST_array[np.mod(iLST+j,NbLST_gem)]:
                        LST_array[np.mod(iLST+j+1,NbLST_gem)] += 24.
                    if LST_array[np.mod(iLST-(j+1),NbLST_gem)] > LST_array[np.mod(iLST-j,NbLST_gem)]:
                        LST_array[np.mod(iLST-(j+1),NbLST_gem)] -= 24.
            #print(LST, LST_array)
            k = order_LST
            indLST = [int(num) for num in np.mod(np.linspace(np.ceil(iLST-k/2.),np.ceil(iLST+k/2),k+1), NbLST_gem)]
            #print("  LST=%.1f is in the range"%LST, LST_array[indLST])
            LST_array2[i,:] = LST_array[indLST]

        # lagrange polynomial coefficients (of degree order_LST)
        w_LST = lagrange_interp_weights(LST_array[indLST], LST)
        #print("    w_LST =", w_LST)

        #
        Ls_array[i] = (w_LST * gem_Ls[iLs,indLST]).sum()
        #print("Ls[%d] = %.3f" % (i, Ls_array[i]))

        # inner loop 
        for j, iLST in enumerate(indLST):
            abs_filename = gem_filelist[iLs,iLST]
#            print('Reading from : ', abs_filename)

            data = {}
            with h5py.File(abs_filename, 'r') as f:
                for key, hdf_name in {k: GEM_hdf_fields[k] for k in atmofile_vars}.items():
                    #print(key, hdf_name)
                    data[key] = f[hdf_name][iLon:iLon+2,iLat:iLat+2,:]

            for key in atmofile_vars:
                if key in ('P', 'NT'):
                    val_new[key] =  np.exp(np.sum(w_Lon_Lat[:,:,None] * np.log(data[key]), axis=(0,1,)))
                #if key in GEM_aerosols:
                #    val_new[key] =  np.exp(np.sum(w_Lon_Lat[:,:,None] * np.log(data[key]*data['NT']), axis=(0,1,)))
                else:
                    val_new[key] =  np.sum(w_Lon_Lat[:,:,None] * data[key], axis=(0,1,))



            for key in val_new:
                val_array[key][i,:] += w_LST[j] * val_new[key]
                val_array2[key][i,j,:] = val_new[key]

    #print('LST_array2 = ', LST_array2)
    #print('Ls_array = ', Ls_array)

    # wrap Ls_array around 360, and compute lagrange weights
    if order_Ls != 0:
        if (Ls <= Ls_array[0]):
            Ls_array = np.where(Ls_array < Ls_array[0], Ls_array, Ls_array-360.)
        else:
            Ls_array = np.where(Ls_array < Ls_array[0], Ls_array+360., Ls_array)
        #print("Ls=%.1f in range"%Ls, Ls_array)
    w_Ls = lagrange_interp_weights(Ls_array, Ls)
    #print("  w_LS =", w_Ls)

    #
    for key in val_new:
        val_new[key] = np.sum(w_Ls[:,None] * val_array[key], axis=0)

    # get GEM-MOLA surface heights
    mola_filename = os.path.join(NOMADParams["GEM_OUTPUT_DIR"], "gem-mola-topo.h5")
    with h5py.File(mola_filename, 'r') as f:
        zs_gem = np.sum(w_Lon_Lat[:,:] * f['gem_topo_wneg'][iLon:iLon+2,iLat:iLat+2])

    # get MOLA surface height
    mola_filename = os.path.join(NOMADParams['MOLA_DIR'], "MOLA_32.h5")
    with h5py.File(mola_filename, 'r') as f:
        Lons_mola = f['lon_centers'][()]
        Lats_mola = f['lat_centers'][()]
    if (Lon < Lons_mola[0] or Lon > Lons_mola[-1]):
        indLonm = [len(Lons_mola)-1, 0]
    else:
        iLonm = np.argmax(Lon<Lons_mola[()])
        indLonm = [iLonm-1, iLonm]
    if (Lat > Lats_mola[0]):
        #print("warning: lat=%.2f is outside MOLA range, using MOLA boundary")
        indLatm = [0,1]
        w_Lat_Lon_m = bilinear_spherical_interp_weights(Lats_mola[0:2], Lons_mola[indLonm], Lats_mola[0], Lon)
    elif (Lat < Lats_mola[-1]):
        #print("warning: lat=%.2f is outside MOLA range, using MOLA boundary")
        indLatm = [-2,-1]
        w_Lat_Lon_m = bilinear_spherical_interp_weights(Lats_mola[-2:], Lons_mola[indLonm], Lats_mola[-1], Lon)
    else:
        iLatm = np.argmax(Lat>Lats_mola)
        indLatm = [iLatm, iLatm+1]
        w_Lat_Lon_m = bilinear_spherical_interp_weights(Lats_mola[indLatm], Lons_mola[indLonm], Lat, Lon)
    Topo_mola = np.zeros((2,2))
    with h5py.File(mola_filename, 'r') as f:
        dset = f['topo']
        for iLat in range(2):
            for iLon in range(2):
                Topo_mola[iLat,iLon] = dset[indLatm[iLat],indLonm[iLon]]
    zs_mola = np.sum(w_Lat_Lon_m * Topo_mola)
    #print(Lon, Lons_mola[indLonm])
    #print(Lat, Lats_mola[indLatm])
    #print(w_Lat_Lon_m)
    #print(Topo_mola)

    #
    val_new['zs_gem'] = zs_gem
    val_new['zs_mola'] = zs_mola
    #print('zs_gem = %.1f, zs_mola = %.1f' % (zs_gem, zs_mola))

    # here we should compute z_s and scale p
    if ZMETHOD == 'GEM-MOLA':
        val_new['z'] += zs_gem

    elif ZMETHOD == 'MOLA':
        val_new['z'] += zs_gem
        val_new = zs_pressure_scaling(val_new, zs_mola)

    if True:

        val_extend = {}

        for key in atmofile_vars:
            val_extend[key] = np.zeros(NbZ+1)
            val_extend[key][0] = val_new[key][0]
            val_extend[key][1:] = val_new[key][:]

        H = - (val_new['z'][0]-val_new['z'][1])/np.log(val_new['P'][0]/val_new['P'][1])
        val_extend['z'][0] = 250.e3
        fact = np.exp(-(val_extend['z'][0]-val_new['z'][0])/H)
        val_extend['P'][0] = val_new['P'][0] * fact
        val_extend['NT'][0] = val_new['NT'][0] * fact

        for key in atmofile_vars:
            val_new[key] = val_extend[key]


    # convert H2O_ice from molecule to 1 micron particle 
    val_new['H2O_ice'] /= 1.284e11
    # convert vmr to ND for aerosols
    for key in GEM_aerosols:
        val_new[key][val_new[key]<0.0] = 0.0
        val_new[key] = val_new[key] * val_new['NT']

    #
    if atmo_filename is not None:
        with open(atmo_filename, 'w') as f:
            f.write(atmofile_header1 + '\n')
            f.write(atmofile_header2 + '\n')
            for iz in range(NbZ):
                vals = [val_new[key][iz] for key in atmofile_vars]
                f.write(''.join([sfmt%(val*fact) for sfmt, val, fact in zip(atmofile_printfmt, vals, atmofile_fact)]) +'\n')

    #
    #print('done...')
    return val_new


def lagrange_interp_weights(xi, x):
    """ Compute the weights for a Langrange interpolation """
    m = len(xi)
    w = np.ones(m)

    for i in range(m):
        for j in range(m):
            if i != j:
                w[i] *= (x-xi[j])/(xi[i]-xi[j])

    return w


def bilinear_spherical_interp_weights(Lats, Lons, lat, lon):
    """ Compute the weights for a Langrange interpolation """

    Thetas = np.pi/2. - np.deg2rad(Lats)
    theta = np.pi/2. - np.deg2rad(lat)
    Phis = np.deg2rad(Lons)
    phi = np.deg2rad(lon)

    if Phis[0] > Phis[1]:
        if phi >= Phis[1]:
            phi = phi - 2*np.pi
        Phis[0] = Phis[0] - 2*np.pi
    #print(Thetas, theta, Phis, phi)

    w = np.outer([abs(np.cos(theta)-np.cos(Thetas[1])), abs(np.cos(theta)-np.cos(Thetas[0]))],
                    [abs(phi-Phis[1]), abs(phi-Phis[0])])
    w /= w.sum()

    return w


def zs_pressure_scaling(atmo_in, zs, H=None):
    """  perform pressure scaling due to changes in sureface height

    Args:
        atmo_in (dict): dictionary with field z, T, P, NT and others (in SI units)
        zs (float): new surface height (in m)
        H (float, optional): atmospheric scale height

    """

    a0 = 3396.e3
    g0 = 3.7257964
    R = 8.3144598

    nz = len(atmo_in['z'])
    zs_in = atmo_in['z'][-1]
    Ps_in = atmo_in['P'][-1]
    #print('')
    #print('Shifting from zs_GCM=%.1f to zs=%.1f'%(zs_in, zs))
    #print('  Ps_in=%.4e'%Ps_in)

    # get reference for these values
    mass_spec = {'Ar':0.039948, 'CO':0.028010, 'CO2':0.04401, 'H2O':0.018015, 'N2':0.028006, 'O2':0.031990, 'O3':0.047998}
    mass = sum([atmo_in[key]*mass_spec[key] for key in mass_spec]) / sum([atmo_in[key] for key in mass_spec])

    #
    if H is None:
        g = g0*(a0/(a0+zs_in))**2
        #H = R*atmo_in['T'][-5]/g/mass[-5]
        H = -(atmo_in['z'][-2]-zs_in)/np.log(atmo_in['P'][-2]/Ps_in)
    #print('  Using H=%.0f m '%H)

    #
    Ps = Ps_in*np.exp(-(zs-zs_in)/H)
    #print('  Ps=%.4e '%Ps)

    # copy initial atmo
    atmo = {}
    for key in atmo_in:
        atmo[key] = atmo_in[key]


    if True:

        a = 0.3
        bp = 1.0
        bn = 0.0
        K = 6.0

        Gamma = 4.4e-3

        # set up parameters
        Delz_s = zs - zs_in
        if Delz_s > 0.:
            z_L = a*H + bp*Delz_s
        else:
            z_L = a*H - bn*Delz_s
        #z_L = a*H + b*Delz_s if Delz_s>0 else a*H
        #z_L = a*H + b*Delz_s
        #z_L = a*H if a*H > -b*Delz_s else -b*Delz_s
        f = Ps/Ps_in
        #print('  Delz_s=%.1f, z_L=%.1f, f=%.3f' % (Delz_s, z_L, f))

        # adjust P
        P = atmo_in['P']*(f + (1.-f)*0.5*(1.+np.tanh(K*(-H*np.log(atmo_in['P']/Ps_in)/z_L-1.))) )
        #print('  Ps=%.2e, Ps=%.2e'%(Ps, P[-1]))

        # adjust z
        T = atmo_in['T']
        z = np.zeros(nz)
        z[-1] = zs
        for l in range(nz-1,0,-1):
            g = g0*(a0/(a0+z[l]))**2
            if T[l-1] != T[l]:
                Tm = (T[l-1]-T[l])/np.log(T[l-1]/T[l])
            else :
                Tm = T[l-1]
            dz = R/(mass[l]*g) * Tm * np.log(P[l]/P[l-1])
            z[l-1] = z[l] + dz
            #print(l, z[l], l-1, z[l-1], dz, g, Tm)
#        for l in range(nz):
#            print('%5d%15.5e%15.5e%10.1f%10.1f' % (l, atmo_in['P'][l], P[l], atmo_in['z'][l], z[l]) )

        for step in range(0):
            print('    step %d:' % step)
            """T_p = T.copy()"""
            """z_p = z.copy()"""

            delta_z = z-atmo_in['z']
            delta_T = -Gamma*delta_z
            T = atmo_in['T'] + delta_T

            z[-1] = zs
            for l in range(nz-1,0,-1):
                g = g0*(a0/(a0+z[l-1]))**2
                if T[l-1] != T[l]:
                    Tm = (T[l-1]-T[l])/np.log(T[l-1]/T[l])
                else :
                    Tm = T[l-1]
                dz = R/(mass[l-1]*g) * Tm * np.log(P[l]/P[l-1])
                z[l-1] = z[l] + dz

        #
        atmo['z'] = z
        atmo['T'] = T
        atmo['P'] = P
        atmo['NT'] = P/(1.38064852e-23*T)

    else:  #MCD method

        # set up parameters
        Delz = -10.*np.log(Ps/Ps_in)
        z = Delz+3. if Delz>0. else 3.
        print('  Delz=%.3f, z=%.3f'%(Delz, z))
        x = 0.0
        if Delz > 1.0:
            x = np.max([-0.8, -0.12*(np.abs(Delz)-1.)])
        elif Delz < 1.0:
            x = np.min([0.8, 0.12*(np.abs(Delz)-1.)])
        print('  x=%.2f'%x)
        fl = Ps/Ps_in*(atmo_in['P']/Ps_in)**x
        print('  f[-1]=%.3f, f[0]=%.3f' % (fl[-1], fl[0]))

        # adjust P
        P = atmo_in['P']*(fl + (1.-fl)*0.5*(1.+np.tanh(6.*(-10.*np.log(atmo_in['P']/Ps_in)-z)/z)) )
        print('  Ps=%.2e, Ps=%.2e'%(Ps, P[-1]))

        # adjust z
        T = atmo_in['T']
        z = np.zeros(nz)
        z[-1] = zs
        for l in range(nz-1,0,-1):
            g = g0*(a0/(a0+z[l-1]))**2
            if T[l-1] != T[l]:
                Tm = (T[l-1]-T[l])/np.log(T[l-1]/T[l])
            else :
                Tm = T[l-1]
            dz = R/(mass[l-1]*g) * Tm * np.log(P[l]/P[l-1])
            z[l-1] = z[l] + dz
            #print(l, z[l], l-1, z[l-1], dz, g, Tm)
#        for l in range(nz):
#            print('%5d%15.5e%15.5e%10.1f%10.1f' % (l, atmo_in['P'][l], P[l], atmo_in['z'][l], z[l]) )

        atmo['P'] = P
        atmo['z'] = z

    return atmo



