""" global path definitions """

import os
from nomad_ops.config import APRIORI_FILE_DESTINATION


NOMADParams = {}


#Home computer
#NOMADParams['OBSFILE_DIR'] = r"C:\Users\ithom\Dropbox\NOMAD\Python\retrievals"
#NOMADParams['RADTRAN_DIR'] = r"D:\Radiative_Transfer"
#NOMADParams["GEM_OUTPUT_DIR"] = r"D:\gem-mars\output" #some data available offline
#Windows server
#NOMADParams["GEM_OUTPUT_DIR"] = r"X:\projects\planetary\gem-mars\output"

#Work computer
#NOMADParams['OBSFILE_DIR'] = r'C:\Users\iant\Dropbox\NOMAD\Python\retrievals'
#NOMADParams['RADTRAN_DIR'] = r"C:\Users\iant\Documents\DATA\Radiative_Transfer"  #data available offline
#NOMADParams["GEM_OUTPUT_DIR"] = r"C:\Users\iant\Documents\DATA\gem-mars\output" #data available offline



#linux server
NOMADParams['OBSFILE_DIR'] = '/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5'
NOMADParams["APRIORI_FILE_DESTINATION"] = APRIORI_FILE_DESTINATION
NOMADParams['RADTRAN_DIR'] = '/bira-iasb/projects/NOMAD/Science/Radiative_Transfer'
NOMADParams["GEM_OUTPUT_DIR"] = "/bira-iasb/projects/planetary/gem-mars/output"


NOMADParams['AUXILIARY_DIR'] = os.path.join(NOMADParams['RADTRAN_DIR'], "Auxiliary_files")
NOMADParams['ATMOSPHERE_DIR'] = os.path.join(NOMADParams['AUXILIARY_DIR'], "Atmosphere")
NOMADParams['PLANET_DIR'] = os.path.join(NOMADParams['AUXILIARY_DIR'], "Planet")
NOMADParams['SOLAR_DIR'] = os.path.join(NOMADParams['AUXILIARY_DIR'], "Solar")
NOMADParams['LIDORT_DIR'] = os.path.join(NOMADParams['AUXILIARY_DIR'], "Lidort")
NOMADParams['HITRAN_DIR'] = os.path.join(NOMADParams['AUXILIARY_DIR'], "Spectroscopy")
NOMADParams['MOLA_DIR'] = os.path.join(NOMADParams['AUXILIARY_DIR'], "MOLA")


def read_dict_from_file(filename):
  """ Can read in dict from some files (.txt,.yml)"""

  file_ext = os.path.splitext(filename)[-1]
  print("file_ext=", file_ext)

  if file_ext == '.txt':
    dict_in = {}
    with open(filename, 'r') as f:
      for line in [l.strip() for l in f.readlines()]:
        #print(line)
        if (len(line) == 0) or (line[0] in ('!','#','%',)):
          continue
        key, val = [w.strip() for w in line.split(':')]
        dict_in[key] = val

  elif file_ext == '.yml':
    import yaml
    with open(filename, 'r') as f:
      dict_in = yaml.load(f)

  elif file_ext == '.json':
    import json
    with open(filename, 'r') as f:
      dict_in = json.load(f)

  else:
    raise Exception("cannot read in dict from ''" % filename)

  return dict_in

def set_NOMADParams(*args, **kwargs):
    """ set the global NOMADParams (either from file or dict or by kwargs directly) """

    if len(args) > 0:
        if os.paths.isfile(args[0]):
            # read dict from file
            new_dict = read_dict_from_file(args[0])
        elif type(args[0]) == dict:
            new_dict = args[0]
        else:
            raise Exception("set_NOMADParams, should pass filename or dict")

        NOMADParams.update(new_dict)

    if len(kwargs) > 0:
        NOMADParams.update(**kwargs)


def set_global_obs_dir(global_obs_dir):
    """ set the global observation data directory 
    Args:
        global_obs_dir (str): dir path

    """
    #global NOMADParams['OBSFILE_DIR']
    #NOMADParams['OBSFILE_DIR'] = os.path.realpath(global_obs_dir)
    NOMADParams['OBSFILE_DIR'] = global_obs_dir

def get_obs_dir(obs_filename):
    """ determine directory for observation file 
    Args:
        obs_filename (str): observation file name

    Returns:
        absolute obs_dir (str): path to observation file  
    
    """

    # check if filename is an absolute address
    if os.path.isfile(obs_filename):
        obs_dir = os.path.abspath(os.path.dirname(obs_filename))
        return obs_dir

    # check if NOMADParams['OBSFILE_DIR'] is a flat directory structure
    if os.path.isfile(os.path.join(NOMADParams['OBSFILE_DIR'],obs_filename)):
        obs_dir = NOMADParams['OBSFILE_DIR']
        return obs_dir

    # check if NOMADParams['OBSFILE_DIR'] is a nested directory structure
    obs_shortname = os.path.splitext(obs_filename)[0]
    if obs_shortname.count('_') == 5:
        obs_date, obs_time, obs_level, obs_channel, obs_type_code, obs_order = obs_shortname.split('_')
    elif obs_shortname.count('_') == 6:
        obs_date, obs_time, obs_level, obs_channel, obs_sci_num, obs_type_code, obs_order = obs_shortname.split('_')
    else:
        raise RuntimeError('obs_shortname is not a recognized format')
    obs_dir = os.path.join(NOMADParams['OBSFILE_DIR'], 'hdf5_level_'+obs_level, obs_date[0:4], obs_date[4:6], obs_date[6:8])
    if os.path.isfile(os.path.join(obs_dir, obs_filename)):
        return obs_dir

    # filename does not exit
    raise IOError("observation file '" + obs_filename + "' not found in "+obs_dir)


def get_obs_absfilename(obs_filename):
    """ determine absolute file name  

    Args:
        obs_filename (str): observation file name

    Returns:
        absolute file name    
    
    """

    # check if filename is an absolute address
    if os.path.isfile(obs_filename):
        return os.path.abspath(obs_filename)

    # check if NOMADParams['OBSFILE_DIR'] is a flat directory structure
    obs_absfilename = os.path.join(NOMADParams['OBSFILE_DIR'],obs_filename)
    if os.path.isfile(obs_absfilename):
        return obs_absfilename

    # check if NOMADParams['OBSFILE_DIR'] is a nested directory structure
    obs_shortname = os.path.splitext(obs_filename)[0]
    #if obs_shortname.count('_') == 4:
    #    obs_date, obs_time, obs_level, obs_channel, obs_type_code = obs_shortname.split('_')
    #elif obs_shortname.count('_') == 5:
    #    obs_date, obs_time, obs_level, obs_channel, obs_type_code, obs_order = obs_shortname.split('_')
    #elif obs_shortname.count('_') == 6:
    #    obs_date, obs_time, obs_level, obs_channel, obs_sci_num, obs_type_code, obs_order = obs_shortname.split('_')
    #else:
    #    raise RuntimeError('obs_shortname is not a recognized format')
    obs_date, obs_time, obs_level = obs_shortname.split('_')[:3]
    obs_dir = os.path.join(NOMADParams['OBSFILE_DIR'], 'hdf5_level_'+obs_level, obs_date[0:4], obs_date[4:6], obs_date[6:8])
    obs_absfilename = os.path.join(obs_dir, obs_filename)
    if os.path.isfile(obs_absfilename):
        return obs_absfilename

    # filename does not exit
    raise IOError("observation file '" + obs_filename + "' not found (%s)" %obs_absfilename)


def get_retrieval_dir(obs_filename):
    """ get rerieval directory based on channel and type """

    pass





