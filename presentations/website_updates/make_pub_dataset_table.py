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
from datetime import datetime


nomad_obs_paper_list = [

    # {"title": "",
    #  "year": 2026,
    #  "inst": "NOMAD",
    #  "lead": "",
    #  "pdoi": "",
    #  "ddoi": ""},

    {"title": "Dust and ice cloud optical depths in MY34-38",
     "year": 2026,
     "inst": "NOMAD-UVIS (nadir)",
     "lead": "Yannick Willame",
     "pdoi": "https://doi.org/10.1029/2025JE009633",
     "ddoi": "https://doi.org/10.18758/69qazhkk"},

    {"title": "Water vapour nadir column abundances in MY34-38",
     "year": 2026,
     "inst": "NOMAD-LNO",
     "lead": "Francois Hendrick",
     "pdoi": "https://doi.org/10.1029/2025JE009592",
     "ddoi": "https://doi.org/10.18758/txmf1fej"},

    {"title": "Mars water vapour vertical distribution in MY34-37: Part 2",
     "year": 2026,
     "inst": "NOMAD-SO",
     "lead": "Adrian Brines",
     "pdoi": "https://doi.org/10.1029/2025JE009191",
     "ddoi": "https://zenodo.org/records/14384256"},

    {"title": "Mars water vapour vertical distribution in MY34-37: Part 1",
     "year": 2026,
     "inst": "NOMAD-SO",
     "lead": "Adrian Brines",
     "pdoi": "https://doi.org/10.1029/2024JE008916",
     "ddoi": "https://zenodo.org/records/15146458"},

    {"title": "Atmospheric water escape from Mars triggered by a dust storm in MY37",
     "year": 2026,
     "inst": "NOMAD-SO",
     "lead": "Adrian Brines",
     "pdoi": "https://doi.org/10.1038/s43247-025-03157-5",
     "ddoi": "https://zenodo.org/records/17797042"},

    {"title": "Upper limits of CH4 and OCS in MY34-36",
     "year": 2025,
     "inst": "NOMAD-SO",
     "lead": "Miguel Escudero-Jiménez",
     "pdoi": "https://doi.org/10.1016/j.icarus.2025.116923",
     "ddoi": "https://zenodo.org/records/17368530"},

    {"title": "Mars dust and aerosol climatology from UVIS solar occultations for MY34-36",
     "year": 2025,
     "inst": "NOMAD-UVIS (occultation)",
     "lead": "Zachary Flimon",
     "pdoi": "https://doi.org/10.1029/2024JE008303",
     "ddoi": "https://doi.org/10.18758/71021091"},

    {"title": "Changes in green line dayglow emission due to an ICME",
     "year": 2025,
     "inst": "NOMAD-UVIS (limb)",
     "lead": "Aadarsh Sharma",
     "pdoi": "https://doi.org/10.1029/2024GL111745",
     "ddoi": "https://doi.org/10.5281/zenodo.12690459"},

    {"title": "NO and O2 nightglow emissions from Mars southern winter polar region",
     "year": 2024,
     "inst": "NOMAD-UVIS (limb)",
     "lead": "Lauriane Soret",
     "pdoi": "https://doi.org/10.1029/2024JE008620",
     "ddoi": "https://doi.org/10.58119/ULG/U19BNE"},


    {"title": "Mars water vapour vertical profiles in the perihelion seasons of MY34-36",
     "year": 2024,
     "inst": "NOMAD-SO",
     "lead": "Adrian Brines",
     "pdoi": "https://doi.org/10.1029/2023GL107224",
     "ddoi": "https://zenodo.org/records/10779225"},

    {"title": "Mars ozone column abundances for MY34-36",
     "year": 2024,
     "inst": "NOMAD-UVIS (nadir)",
     "lead": "Jon Mason",
     "pdoi": "https://doi.org/10.1029/2023JE008270",
     "ddoi": "https://doi.org/10.21954/ou.rd.24877095.v1"},

    {"title": "Ultraviolet and visible reflectance spectra of Phobos and Deimos",
     "year": 2023,
     "inst": "NOMAD-UVIS",
     "lead": "Jon Mason",
     "pdoi": "https://doi.org/10.1029/2023JE008002",
     "ddoi": "https://doi.org/10.21954/ou.rd.23676801.v1"},

    {"title": "Dust vertical profiles for MY34-35",
     "year": 2023,
     "inst": "NOMAD-SO",
     "lead": "Aurelien Stolzenbach",
     "pdoi": ["https://doi.org/10.1029/2022JE007276", "https://doi.org/10.1029/2023JE007835"],
     "ddoi": "https://zenodo.org/records/7753821"},

    {"title": "Mars O2 visible nightglow limb profiles",
     "year": 2023,
     "inst": "NOMAD-UVIS (limb)",
     "lead": "Jean-Claude Gerard",
     "pdoi": "https://doi.org/10.1038/s41550-023-02104-8",
     "ddoi": "https://doi.org/10.18758/71021084"},

    {"title": "Mars ultraviolet dayglow limb profiles",
     "year": 2023,
     "inst": "NOMAD-UVIS (limb)",
     "lead": "Lauriane Soret",
     "pdoi": "https://doi.org/10.1029/2023JE007762",
     "ddoi": "https://doi.org/10.18758/71021081"},

    {"title": "Mars atmospheric CO vertical profiles for MY34-35",
     "year": 2023,
     "inst": "NOMAD-SO",
     "lead": "Ashimananda Modak",
     "pdoi": "https://doi.org/10.1029/2022JE007282",
     "ddoi": "https://zenodo.org/records/7268447"},

    {"title": "Mars ozone vertical profiles for MY34-36",
     "year": 2023,
     "inst": "NOMAD-UVIS (occultation)",
     "lead": "Arianna Piccialli",
     "pdoi": "https://doi.org/10.1029/2022EA002429",
     "ddoi": "https://doi.org/10.18758/71021079"},

    {"title": "Mars CO2 density and temperature vertical profiles for MY35-36",
     "year": 2023,
     "inst": "NOMAD-SO",
     "lead": "Loic Trompet",
     "pdoi": ["https://doi.org/10.1029/2022JE007277", "https://doi.org/10.1029/2022JE007279"],
     "ddoi": "https://doi.org/10.18758/71021074"},

    {"title": "Mars water vapour vertical profiles in the perihelion season MY34-35",
     "year": 2022,
     "inst": "NOMAD-SO",
     "lead": "Adrian Brines",
     "pdoi": "https://doi.org/10.1029/2022JE007273",
     "ddoi": "https://doi.org/10.5281/zenodo.7085454"},

    {"title": "Mars CO2 temperature and density vertical profiles for MY34",
     "year": 2022,
     "inst": "NOMAD-SO",
     "lead": "Miguel-Angel Lopez Valverde",
     "pdoi": "https://doi.org/10.1029/2022JE007278",
     "ddoi": "https://zenodo.org/record/7086187#.Y4xDQb_MJX0"},

    {"title": "Mars water vapour vertical profiles for MY34-36",
     "year": 2022,
     "inst": "NOMAD-SO",
     "lead": "Shohei Aoki",
     "pdoi": "https://doi.org/10.1029/2022JE007231",
     "ddoi": "https://doi.org/10.18758/71021072"},

    {"title": "(1) Mars ozone vertical profiles for MY34-35 and (2) comparable GEM-Mars ozone simulations",
     "year": 2022,
     "inst": "NOMAD-UVIS (occultation) and GEM-Mars",
     "lead": "(1) Michael Wolff and (2) Frank Daerden",
     "pdoi": "https://doi.org/10.1029/2022GL098821",
     "ddoi": "https://dx.doi.org/10.18758/71021070"},

    {"title": "Mars oxygen green and red line dayglow limb profiles",
     "year": 2022,
     "inst": "NOMAD-UVIS (limb)",
     "lead": "Lauriane Soret",
     "pdoi": "https://doi.org/10.1029/2022JE007220",
     "ddoi": "https://dx.doi.org/10.18758/71021077"},

    {"title": "Mars oxygen dayglow density and temperature limb profiles of the upper mesosphere and lower thermosphere",
     "year": 2022,
     "inst": "NOMAD-UVIS (limb)",
     "lead": "Shohei Aoki",
     "pdoi": "https://doi.org/10.1029/2022JE007206",
     "ddoi": "https://doi.org/10.18758/71021073"},

    {"title": "CO/CO2 vertical profile ratios for MY35",
     "year": 2022,
     "inst": "NOMAD-SO",
     "lead": "Nao Yoshida",
     "pdoi": "https://doi.org/10.1029/2022GL098485",
     "ddoi": "https://dx.doi.org/10.18758/71021076"},

    {"title": "Mars CO2 ice seasonal maps for MY34-35",
     "year": 2022,
     "inst": "NOMAD-LNO",
     "lead": "Fabrizio Oliva",
     "pdoi": "https://doi.org/10.1029/2021JE007083",
     "ddoi": "https://dx.doi.org/10.17632/cs72xnsz7k.2"},

    {"title": "Mars CO2 ice cloud vertical profiles for MY34-35",
     "year": 2021,
     "inst": "NOMAD-SO",
     "lead": "Guiliano Liuzzi",
     "pdoi": "https://doi.org/10.1029/2021GL095895",
     "ddoi": "https://zenodo.org/records/5572859"},

    {"title": "Mars ozone and aerosol vertical profile retrievals for MY34-35",
     "year": 2021,
     "inst": "NOMAD-UVIS (occultation)",
     "lead": "Manish Patel",
     "pdoi": "https://doi.org/10.1029/2021JE006834",
     "ddoi": "https://doi.org/10.21954/ou.rd.13580336.v1"},

    {"title": "Mars water vapour column abundances for MY34-35",
     "year": 2021,
     "inst": "NOMAD-LNO",
     "lead": "Matteo Crismani",
     "pdoi": "https://doi.org/10.1029/2021JE006878",
     "ddoi": "https://zenodo.org/records/5510605"},

    {"title": "Mars oxygen green line dayglow limb profiles",
     "year": 2020,
     "inst": "NOMAD-UVIS (limb)",
     "lead": "Jean-Claude Gerard",
     "pdoi": "https://doi.org/10.1038/s41550-020-1123-2",
     "ddoi": "https://doi.org/10.18758/71021055"},

    {"title": "Retrieved dust and water ice concentration and particle size over time from solar occultation measurements",
     "year": 2019,
     "inst": "NOMAD-SO",
     "lead": "Giuliano Liuzzi",
     "pdoi": "https://doi.org/10.1029/2019JE006250",
     "ddoi": "https://psg.gsfc.nasa.gov/apps/exomars.php"},

    {"title": "Mars water vapour vertical profiles in MY34 dust storms",
     "year": 2019,
     "inst": "NOMAD-SO",
     "lead": "Shohei Aoki",
     "pdoi": "https://doi.org/10.1029/2019JE006109",
     "ddoi": "https://doi.org/10.18758/71021054"},


]


other_obs_paper_list = [

    {"title": "H2O VMR measured during the second halves of MY 34 and 35 from solar occultation measurements",
     "year": 2021,
     "inst": "ACS-MIR",
     "lead": "Denis Belyaev",
     "pdoi": "https://doi.org/10.1029/2021GL093411",
     "ddoi": "https://data.mendeley.com/datasets/995y7ymdgm/1"},

    {"title": "Isotopic composition of H2O and CO2 from solar occultation measurements",
     "year": 2021,
     "inst": "ACS-MIR",
     "lead": "Juan Alday",
     "pdoi": "https://doi.org/10.1038/s41550-021-01389-x",
     "ddoi": "https://zenodo.org/record/5100449#.Y4jMCH3MJPY"},

    {"title": "Temperature, water VMR and saturation profiles from NIR observations and aerosol extinctions from TIRVIM and MIR observations",
     "year": 2020,
     "inst": "ACS",
     "lead": "Anna Fedorova",
     "pdoi": "https://doi.org/10.1126/science.aay9522",
     "ddoi": "http://exomars.cosmos.ru/ACS_Results_stormy_water_vREzUd4pxG/"},

    {"title": "Pressure, temperature, dust, water vapour, water ice and CO2 ice",
     "year": 2009,
     "inst": "MCS",
     "lead": "Armin Kleinböhl",
     "pdoi": "https://doi.org/10.1029/2009JE003358",
     "ddoi": "https://atmos.nmsu.edu/data_and_services/atmospheres_data/MARS/mcs.html"},

]

sim_paper_list = [

    # {"title": "",
    #  "year": 2024,
    #  "inst": "",
    #  "lead": "",
    #  "pdoi": "",
    #  "ddoi": ""},

    {"title": "Simulations of ozone for different heterogeneous processes for MY35",
     "year": 2024,
     "inst": "Mars PCM",
     "lead": "Megan Brown",
     "pdoi": "https://doi.org/10.1029/2024JE008302",
     "ddoi": "https://zenodo.org/records/10514426"},

    {"title": "Simulations of H2O2 and ozone for different heterogeneous processes for MY35",
     "year": 2023,
     "inst": "GEM-Mars",
     "lead": "Frank Daerden",
     "pdoi": "https://doi.org/10.1029/2023JE008014",
     "ddoi": "https://doi.org/10.18758/71021085"},

    {"title": "Simulations of temperature, dust, water vapour and ozone from assimilations",
     "year": 2022,
     "inst": "OpenMars (assimilation)",
     "lead": "James Holmes",
     "pdoi": "https://doi.org/10.1016/j.pss.2020.104962",
     "ddoi": "https://doi.org/10.21954/ou.rd.c.4278950"},

    {"title": "Simulations of ozone with heterogeneous processes for MY35",
     "year": 2022,
     "inst": "Mars PCM",
     "lead": "Megan Brown",
     "pdoi": "https://doi.org/10.1029/2022JE007346",
     "ddoi": "https://doi.org/10.21954/ou.rd.19646784.v1"},

    {"title": "Simulations of ozone vertical profiles for the MY34 global dust storm",
     "year": 2022,
     "inst": "GEM-Mars",
     "lead": "Frank Daerden",
     "pdoi": "https://doi.org/10.1029/2022GL098821",
     "ddoi": "https://doi.org/10.18758/71021070"},

    {"title": "Simulations masked to aerosol opacity vertical profiles for MY34-35",
     "year": 2022,
     "inst": "Mars PCM",
     "lead": "Paul Streeter",
     "pdoi": "https://doi.org/10.1029/2021JE007065",
     "ddoi": "https://doi.org/10.21954/ou.rd.16616680.v1"},

    {"title": "Simulations of D/H observations with cloud-induced fractionation for MY34",
     "year": 2022,
     "inst": "GEM-Mars",
     "lead": "Frank Daerden",
     "pdoi": "https://doi.org/10.1029/2021JE007079",
     "ddoi": "https://doi.org/10.18758/71021067"},


]


headers = ["Dataset description", "Year", "Instrument(s)", "Dataset lead author(s)", "Article DOI", "Link to dataset"]

h = ""

h += "<h1>This page aims to keep a record of all published datasets that may be relevant to the NOMAD team</h1><p></p>"

h += "<h2>NOMAD observation datasets linked to published articles</h2>"
h += "<table border='2'><tbody>\n"
h += "<tr>\n"
for header in headers:
    h += "<th>%s</th>" % header
h += "</tr>\n"

for row_ix, paper in enumerate(nomad_obs_paper_list):
    h += "<tr>\n"
    for col_ix, (key, value) in enumerate(paper.items()):
        if col_ix == col_ix // 2 * 2:
            colour = "#f0f4fb"
        else:
            colour = "#ffffff"
        if row_ix == row_ix // 2 * 2:
            colour = "#a3bbe6"
        # else:
        #     colour = "#ffffff"

        if key in ["pdoi", "ddoi"]:
            if isinstance(value, list):

                h += "<td bgcolor=%s>" % colour
                for each_value in value:
                    h += "<a href='%s'>%s</a> " % (each_value, each_value)
                h += "</td>"
            else:
                h += "<td bgcolor=%s><a href='%s'>%s</a></td>" % (colour, value, value)

        else:
            h += "<td bgcolor=%s>%s</td>" % (colour, value)
    h += "\n</tr>\n"
h += "</tbody></table>\n"


h += "<h2>Observational datasets from other instruments linked to published articles</h2>"
h += "<i>This table is not complete - team members can edit the page by logging in and clicking 'Edit' in the upper right hand corner of this page"
h += "<table border='2'><tbody>\n"
h += "<tr>\n"
for header in headers:
    h += "<th>%s</th>" % header
h += "</tr>\n"

for row_ix, paper in enumerate(other_obs_paper_list):
    h += "<tr>\n"
    for col_ix, (key, value) in enumerate(paper.items()):
        if col_ix == col_ix // 2 * 2:
            colour = "#f0f4fb"
        else:
            colour = "#ffffff"
        if row_ix == row_ix // 2 * 2:
            colour = "#a3bbe6"
        # else:
        #     colour = "#ffffff"

        if key in ["pdoi", "ddoi"]:
            h += "<td bgcolor=%s><a href='%s'>%s</a></td>" % (colour, value, value)
        else:
            h += "<td bgcolor=%s>%s</td>" % (colour, value)
    h += "\n</tr>\n"
h += "</tbody></table>\n"


h += "<p></p><h2>GCM datasets linked to published articles</h2>"
h += "<table border='2'><tbody>\n"


for row_ix, paper in enumerate(sim_paper_list):
    h += "<tr>\n"
    for col_ix, (key, value) in enumerate(paper.items()):
        if col_ix == col_ix // 2 * 2:
            colour = "#f0f4fb"
        else:
            colour = "#ffffff"
        if row_ix == row_ix // 2 * 2:
            colour = "#a3bbe6"
        # else:
        #     colour = "#ffffff"

        if key in ["pdoi", "ddoi"]:
            h += "<td bgcolor=%s><a href='%s'>%s</a></td>" % (colour, value, value)
        else:
            h += "<td bgcolor=%s>%s</td>" % (colour, value)
    h += "\n</tr>\n"


h += "</tbody></table>\n"

h += "Made with presentations.website_updates.make_pub_dataset_table.py<br>\n"
h += "Page last updated %s<br>\n" % str(datetime.now())[:10]

with open("publicly_available_datasets.html", "w") as f:
    f.write(h)
