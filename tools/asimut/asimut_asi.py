# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:02:43 2022

@author: iant

WRITE ASIMUT ASI FILE
"""

def asimut_asi(asi_filepath, inp_filename, h5_dir, atmos_dir, input_dir, save_dir, results_dir):
    
    
    asi = f"""
    [Run]
    verbose        = 3
    save           = transmittance, atmosphere, radiance, FullRetrieval
    saveOUTformat  = hdf5
    UseMatlab      = no
    plotvisible    = no
    
    [Directories]
    dirSpectra=    {h5_dir}
    dirAtmosphere= {atmos_dir}
    dirHitran=     /bira-iasb/projects/NOMAD/Science/Radiative_Transfer/Auxiliary_files/Spectroscopy
    dirLP=         /bira-iasb/projects/NOMAD/Science/Radiative_Transfer/Auxiliary_files/Spectroscopy
    dirInput=      {input_dir}
    dirSolar=      /bira-iasb/projects/NOMAD/Science/Radiative_Transfer/Auxiliary_files/Solar
    dirLidort=     /bira-iasb/projects/NOMAD/Science/Radiative_Transfer/Auxiliary_files/
    dirAerosol=    /bira-iasb/projects/NOMAD/Science/Radiative_Transfer/Auxiliary_files/Aerosols
    dirPlanet=     /bira-iasb/projects/NOMAD/Science/Radiative_Transfer/Auxiliary_files/Surface
    dirSave=       {save_dir}
    dirResults=    {results_dir}
    
    [List]
    {inp_filename}
    
    [RadiativeCode]
    Code= Asimut
    """
    
    with open(asi_filepath, "w") as f:
        f.writelines(asi)
    
