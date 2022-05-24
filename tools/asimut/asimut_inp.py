# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:41:25 2022

@author: iant

WRITE ASIMUT INPUT FILE
"""


def asimut_inp(h5, indices, aotf_nu, inp_filepath, aotf_filepath, ils_filepath, atmos_filename):
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
aotfCentralWnb           = {aotf_nu:#0.5f}
AltitudeTypeSelect       = areoid
RedoSpectralCalibration  = yes
Ils                      = nomad6param
IlsConstant              = 0
FileIlsParam             = {ils_filepath}
UsePreviousFit           = 1
ReverseOrder             = 1

[SP1_FEN1]
pass                     = 1
noise                    = fromSpectrumFile
FitBaseline                = 1.5
FitBaselineOrderPolynomial = 4
rayleigh                 = yes
Emissivity               = 0.85
molecules                = [H2O]
FitMolecules             = [-1000.]
aPrioriMolecules         = [model model]

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
    
    with open(inp_filepath, "w") as f:
        f.writelines(inp)

