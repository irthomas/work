# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 09:27:16 2022

@author: iant

WRITE PUBLICLY AVAILABLE DATASET TABLE

    {"title":"",
     "year":2022,
     "inst":"NOMAD",
     "lead":"",
     "pdoi":"",
     "ddoi":""},

"""


obs_paper_list = [
    {"title":"Retrieved dust and water ice concentration and particle size over time from solar occultation measurements",
     "year":2019,
     "inst":"NOMAD-SO",
     "lead":"Giuliano Liuzzi",
     "pdoi":"https://doi.org/10.1029/2019JE006250",
     "ddoi":"https://psg.gsfc.nasa.gov/apps/exomars.php"},

    {"title":"NOMAD-UVIS ozone and aerosol vertical profile retrievals for MY 34-35 from solar occultation measurements",
     "year":2021,
     "inst":"NOMAD-UVIS",
     "lead":"Manish Patel",
     "pdoi":"https://doi.org/10.1029/2021JE006834",
     "ddoi":"https://ordo.open.ac.uk/articles/dataset/NOMAD-UVIS_ozone_vertical_profile_retrievals_for_Mars_Year_34-35/13580336/1"},

    {"title":"Water vapour vertical profiles in the atmosphere of Mars retrieved from 3.5 years of solar occultation measurements",
     "year":2022,
     "inst":"NOMAD-SO",
     "lead":"Shohei Aoki",
     "pdoi":"https://doi.org/10.1029/2022JE007231",
     "ddoi":"https://dx.doi.org/10.18758/71021072"},

    {"title":"CO/CO2 ratio in MY 35 retrieved from solar occultation measurements",
     "year":2022,
     "inst":"NOMAD-SO",
     "lead":"Nao Yoshida",
     "pdoi":"https://doi.org/10.1029/2022GL098485",
     "ddoi":"https://dx.doi.org/10.18758/71021076"},

    {"title":"Martian atmospheric temperature and density profiles during the 1st year of NOMAD/TGO solar occultation measurements",
     "year":2022,
     "inst":"NOMAD-SO",
     "lead":"Miguel-Angel Lopez Valverde",
     "pdoi":"https://doi.org/10.1029/2022JE007278",
     "ddoi":"https://zenodo.org/record/7086187#.Y4xDQb_MJX0"},

    {"title":"Limb profiles, brightness and altitude of the emission peak of oxygen green and red lines from dayside limb observations",
     "year":2022,
     "inst":"NOMAD-UVIS",
     "lead":"Lauriane Soret",
     "pdoi":"https://doi.org/10.1029/2022JE007220",
     "ddoi":"https://dx.doi.org/10.18758/71021077"},

    {"title":"Density and temperature of the upper mesosphere and lower thermosphere of Mars retrieved from the OI 557.7 nm dayglow",
     "year":2022,
     "inst":"NOMAD-UVIS",
     "lead":"Shohei Aoki",
     "pdoi":"https://doi.org/10.1029/2022JE007206",
     "ddoi":"https://dx.doi.org/10.18758/71021073"},

    {"title":"Retrievals of ozone profiles the 2018 GDS and for the same season one MY later, and GEM-Mars GCM ozone simulations including comparisons to observations",
     "year":2022,
     "inst":"NOMAD-UVIS and GEM-Mars",
     "lead":"Frank Daerden and Michael Wolff",
     "pdoi":"https://doi.org/10.1029/2022GL098821",
     "ddoi":"https://dx.doi.org/10.18758/71021070"},

    {"title":"Carbon dioxide retrievals and temperature profiles retrievals with the hydrostatic equilibrium equation from solar occultation measurements",
     "year":2022,
     "inst":"NOMAD-SO",
     "lead":"Loic Trompet",
     "pdoi":"-",
     "ddoi":"https://dx.doi.org/10.18758/71021074"},

    {"title":"Water vapor vertical distribution on Mars during perihelion season of MY 34 and MY 35 from solar occultation measurements",
     "year":2022,
     "inst":"NOMAD-SO",
     "lead":"Adrian Brines",
     "pdoi":"-",
     "ddoi":"https://doi.org/10.5281/zenodo.7085454"},

    {"title":"Retrieval of Martian atmospheric CO vertical profiles from NOMAD observations during the 1st year of TGO solar occultation measurements",
     "year":2022,
     "inst":"NOMAD-SO",
     "lead":"Ashimananda Modak",
     "pdoi":"-",
     "ddoi":"https://zenodo.org/record/7268447#.Y4xFIL_MJX0"},

    {"title":"Global distribution of ozone vertical profiles retrieved from solar occultation measurements",
     "year":2022,
     "inst":"NOMAD-UVIS",
     "lead":"Arianna Piccialli",
     "pdoi":"-",
     "ddoi":"https://dx.doi.org/10.18758/71021079"},

    {"title":"Temperature, water VMR and saturation profiles from NIR observations and aerosol extinctions from TIRVIM and MIR observations",
     "year":2020,
     "inst":"ACS",
     "lead":"Anna Fedorova",
     "pdoi":"https://doi.org/10.1126/science.aay9522",
     "ddoi":"http://exomars.cosmos.ru/ACS_Results_stormy_water_vREzUd4pxG/"},

    {"title":"Isotopic composition of H2O and CO2 from solar occultation measurements",
     "year":2021,
     "inst":"ACS-MIR",
     "lead":"Juan Alday",
     "pdoi":"https://doi.org/10.1038/s41550-021-01389-x",
     "ddoi":"https://zenodo.org/record/5100449#.Y4jMCH3MJPY"},

    {"title":"H2O VMR measured during the second halves of MY 34 and 35 from solar occultation measurements",
     "year":2021,
     "inst":"ACS-MIR",
     "lead":"Denis Belyaev",
     "pdoi":"https://doi.org/10.1029/2021GL093411",
     "ddoi":"https://data.mendeley.com/datasets/995y7ymdgm/1"},

    {"title":"Pressure, temperature, dust, water vapour, water ice and CO2 ice", 
     "year":2009,
     "inst":"MCS",
     "lead":"Armin Kleinböhl",
     "pdoi":"https://doi.org/10.1029/2009JE003358",
     "ddoi":"https://atmos.nmsu.edu/data_and_services/atmospheres_data/MARS/mcs.html"},


    # {"title":"",
    #  "year":2022,
    #  "inst":"NOMAD",
    #  "lead":"",
    #  "pdoi":"",
    #  "ddoi":""},

]

sim_paper_list = [
    {"title":"GEM-Mars GCM simulations of O3 for NOMAD-UVIS solar occultation profiles",
     "year":2021,
     "inst":"GEM-Mars",
     "lead":"Frank Daerden",
     "pdoi":"https://doi.org/10.1029/2021JE006837",
     "ddoi":"https://repository.aeronomie.be/?doi=10.18758/71021066"},

    {"title":"Mars GCM output masked to UVIS aerosol opacity profiles for MY 34-35",
     "year":2021,
     "inst":"MGCM",
     "lead":"Paul Streeter",
     "pdoi":"https://doi.org/10.1029/2021JE007065",
     "ddoi":"https://ordo.open.ac.uk/articles/dataset/Mars_Global_Climate_Model_output_masked_to_UVIS_aerosol_opacity_profiles_Mars_Year_34-35/16616680/1"},


    {"title":"Temperature, dust, water vapour and ozone from multiple different spacecraft, combined with a Mars GCM",
      "year":2022,
      "inst":"OpenMars",
      "lead":"James Holmes",
      "pdoi":"https://doi.org/10.1016/j.pss.2020.104962",
      "ddoi":"https://doi.org/10.21954/ou.rd.c.4278950"},

    # {"title":"",
    #  "year":2022,
    #  "inst":"NOMAD",
    #  "lead":"",
    #  "pdoi":"",
    #  "ddoi":""},

]


headers = ["Dataset description", "Year", "Instrument(s)", "Dataset lead author(s)", "Article DOI", "Link to dataset"]

h = ""

h += "<h1>This page aims to keep a record of all published datasets that may be relevant to the NOMAD team</h1><p></p>"
h += "<h2>Observational datasets linked to published articles</h2>"
h += "<table border='2'><tbody>\n"

h += "<tr>\n"
for header in headers:
    h += "<th>%s</th>" %header
h += "</tr>\n"


for paper in obs_paper_list:
    h += "<tr>\n"
    for key, value in paper.items():
        if key in ["pdoi", "ddoi"]:
            h += "<td><a href='%s'>%s</a></td>" %(value, value)
        else:
            h += "<td>%s</td>" %(value)
    h += "\n</tr>\n"


h += "</tbody></table>\n"



h += "<p></p><h2>GCM datasets linked to published articles</h2>"
h += "<table border='2'><tbody>\n"


for paper in sim_paper_list:
    h += "<tr>\n"
    for key, value in paper.items():
        if key in ["pdoi", "ddoi"]:
            h += "<td><a href='%s'>%s</a></td>" %(value, value)
        else:
            h += "<td>%s</td>" %(value)
    h += "\n</tr>\n"


h += "</tbody></table>\n"


