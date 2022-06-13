# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:41:25 2022

@author: iant

WRITE ASIMUT INPUT FILE
"""


def asimut_inp(h5, indices, aotf_nu, inp_filepath, aotf_filepath, ils_filepath, wavenb_filepath, atmos_filename):
    """write basic asimut input"""
    
    id_list = "val[" + " ".join(str(i + 1) for i in indices) + "]" #asimut is 1 indexed
    
    
    inp = f"""

[Set]
zType                    = range
zScale                   = [0.0, 150.0, 2.0]
nbSpectra                = 1

[Planet]
Planet                   = mars
PlanetRadius             = 3389.5
RefractiveIndex          = mars

[SP1]
FenList                  = [1]
InstrumentType           = aotf
ConsiderBlaze            = yes
source                   = sun
Refraction               = no
NeglectThermalSource     = no
NeglectThermalReflection = no
FileType                 = Hdf5_generic
Filename                 = {h5}.h5
DataType                 = Transmittance
list                     = yes
SpectraID_List           = {id_list}
NumberOrders             = 9
AOTFFunction             = File
FileAotfFilter           = {aotf_filepath}
DataYErrorSelect         = YError
aotfCentralWnb           = [{aotf_nu:#0.5f}]
AltitudeTypeSelect       = areoid
Ils                      = nomad6param
IlsConstant              = 0
FileIlsParam             = {ils_filepath}
Geometry=limbZtg

RedoSpectralCalibration  = no
FileCalibList            = no
FileCalibWavenb          = {wavenb_filepath}


[SP1_FEN1]
pass                     = 1
noise                    = fromSpectrumFile
rayleigh                 = yes
Emissivity               = 0.85
molecules                = [H2O]
FitMolecules             = [-1000.0]
aPrioriMolecules         = [model]
% FitShift                 = 1.01
% FitShiftLimit            = 0.5

FitBaselineMethod        = Rodgers
FitBaseline              = 1.9
FitBaselineOrderPolynomial=4


[AtmosphericModels]
model                    = 0
atmFile                  = {atmos_filename}
atmFileType              = col
atmFileIsList            = no
zptType                  = std
density                  = nowater

[Solar]
FileSolar                = Solar_irradiance_ACESOLSPEC_2015.dat

[Continua]
Rayleigh                 = shdom

[Molecules]
fileHitran               = HITRAN2012.par

[CO2LP]
ATMname                  = CO2
File                     = 02_hit16_2000-5000_CO2broadened.par
isotope                  = 999
type                     = Hitran

[H2OLP]
ATMname                  = H2O
File                     = 01_hit16_2000-5000_CO2broadened.par
isotope                  = 999
type                     = Hitran

[COLP]
ATMname                  = CO
File                     = 05_hit16_2000-5000_CO2broadened.par
isotope                  = 999
type                     = Hitran
"""

#     inp = f"""


# [Set]
#   zType=Values
#   zScale= [33.0,35.0,37.0,39.0,41.0,43.0,45.0,47.0,49.0,51.0,53.0,55.0,57.0,59.0,61.0,63.0,65.0,67.0,69.0,71.0,73.0,77.0,81.0,85.0,89.0,93.0,103.0,113.0]
#   nbSpectra= 1

# [Planet]
#   Planet= Mars
#   PlanetRadius= 3396
#   RefractiveIndex= mars

# [Solar]
#   FileSolar= Solar_irradiance_ACESOLSPEC_2015.dat

# [SP1]
#   Fenlist=[1] 
#   Geometry=limbZtg
#   FileName= {h5}.h5
#   DataType= Transmittance
#   FileType= hdf5_generic
#   List= yes
#   SpectraID_List= {id_list}

#   Refraction= no
#   NeglectThermalSource     = no
#   NeglectThermalReflection = no
#   source= sun

#   DataYSelect=Y
#   DataYErrorSelect=YError
#   AltitudeTypeSelect=areoid

#   zerofilling = 2
  
#   InstrumentType= aotf
#   NumberOrders             = 9
#   AOTFFunction             = File
#   FileAotfFilter           = {aotf_filepath}
#   ConsiderBlaze= yes
#   BlazeFunction=Gaussian
#   FileBlazeFunction=/bira-iasb/projects/work/NOMAD/Science/loict/CO2/2022/03/04/20220304_222120_1p0a_SO_A_I/Order148/Bin2/SBS/Loop0/Ind117/blaze_asym.dat

#   aotfCentralWnb=[{aotf_nu:#0.5f}]

#   FileCalibList=no
#   FileCalibWavenb=/bira-iasb/projects/work/NOMAD/Science/loict/CO2/2022/03/04/20220304_222120_1p0a_SO_A_I/Order148/Bin2/SBS/Loop0/Ind117/CoeffsInd117.dat 

#   Ils=Nomad6param 
#   IlsConstant=0 
#   FileIlsParam=/bira-iasb/projects/work/NOMAD/Science/loict/CO2/2022/03/04/20220304_222120_1p0a_SO_A_I/Order148/Bin2/SBS/Loop0/Ind117/ILS_test.dat

# [SP1_FEN1]
#   pass= 1
#   Molecules=[   CO2_626  ] 
#   FitMolecules=[   -1.9  ] 
#   aPrioriMolecules=[   model  ] 

#   FitShift=1.000001

#   Noise= fromSpectrumFile

#   FitBaselineMethod=Rodgers
#   FitBaseline= 1.9
#   FitBaselineOrderPolynomial=4


  
# [AtmosphericModels]
#   model= 0
#   atmFileIsList=0
#   atmFile = /bira-iasb/projects/work/NOMAD/Science/Radiative_Transfer/Auxiliary_files/Atmosphere/gem-mars-a652-MY35/2022/03/04/20220304_222120_1p0a_SO_A_I_148/gem-mars-a652-MY35_20220304_222120_1p0a_SO_A_I_148_tangentatmo_bin0.dat
#   atmFileType= col
#   zptType= std
#   zptrelative= no
#   density= nowater
#   hydrostatic= 0
#   Space= 113.0

  
# [CO2_626LP]
#   DBName=CO2
#   ATMName=CO2
#   File= HITRAN2020/20220113_CO2_2000-5000.par
#   type= Hitran
#   isotope= 626
#   atmfile= /bira-iasb/projects/work/NOMAD/Science/Radiative_Transfer/Auxiliary_files/Atmosphere/gem-mars-a585/apriori_1_1_1_GEMZ_wz_VMR_hydro/gem-mars-a585_AllSeasons_AllHemispheres_AllTime_mean_CO2.dat
#   LProfile= VoigtFaddeeva
  
# [H2O_161LP]
#   DBName=H2O
#   ATMName=H2O
#   File= 01_hit16_2000-5000_CO2broadened.par
#   type= Hitran
#   isotope= 161
#   fact=1.0
#   atmfile= /bira-iasb/projects/NOMAD/Science/Radiative_Transfer/Auxiliary_files/Atmosphere/H2O_atm_uniform_1ppm.dat
#   LProfile= VoigtFaddeeva

# """
    
    with open(inp_filepath, "w") as f:
        f.writelines(inp)

