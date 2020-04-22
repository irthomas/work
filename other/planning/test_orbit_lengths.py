# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:13:36 2019

@author: iant

"""

for orbit in orbitList:

    if "dayside" in orbit["irMeasuredObsTypes"]:
        dayside = orbit["dayside"]
        
        print(orbit["orbitNumber"])
        print(f"latStart={dayside['latStart']}, latMidpoint={dayside['latMidpoint']}, latEnd={dayside['latEnd']}")
        print(f"incStart={dayside['incidenceStart']}, incMidpoint={dayside['incidenceMidpoint']}, incEnd={dayside['incidenceEnd']}")
        
        relStart = dayside['etMidpoint'] - dayside['etStart']
        relObsStart = dayside['etMidpoint'] - dayside['obsStart'] - 610.0
        relEnd = dayside['etEnd'] - dayside['etMidpoint']
        relObsEnd = dayside['obsEnd'] - dayside['etMidpoint']
        
        if np.abs(relStart - relEnd) > 1.0 or np.abs(relObsStart - relObsEnd) > 1.0 or np.abs(relStart - relObsEnd) > 1.0:
            print(f"relStart={relStart}, relObsStart={relObsStart}, relEnd={relEnd}, relObsEnd={relObsEnd}")
            
        print(f"incidence0={dayside['incidences'][0]}, incidence-1={dayside['incidences'][-1]}")    
        midIndex = int(len(dayside['incidences']) /2)
        print(f"latmid={dayside['lats'][midIndex]}")
    else:
        print("No dayside")
    print("####################")