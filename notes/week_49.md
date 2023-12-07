# Notes Week 47
---
### Dependence Diagram:
A dependency graph is a data structure formed by a directed graph that describes the dependency of an entity in the system on the other entities of the same system. The underlying structure of a dependency graph is a directed graph where each node points to the node on which it depends.

In our project we will be looking at the score/reward function variable interdependence. 

![Dependency graph](images/dependency_diagram.svg)

---

### Observations used in reward calculation:

Other variables not used directly in score/reward calculation can possibly be used in prediction of various things, for example: warming/cooling energy loads based on outdoor dry bulb temperature.

##### Dynamics independent variables:
1. Neigborhood level variables:
   - Calendar type variables: 
     - [ ] Month
     - [ ] Hour
     - [ ] Day type
     - [ ] Daylight saving status
   - Weather type variables: 
     - [ ] Outdoor dry bulb temp (0h)
     - [ ] Outdoor dry bulb temp (6h, 12h, 24h)
     - [ ] Outdoor relative humidity (0h, 6h, 12h, 24h)
     - [ ] Diffuse solar irradiance (0h, 6h, 12h, 24h)
     - [ ] Direct solar irradiance (0h, 6h, 12h, 24h)
   - [x] District level carbon intensity (district = neighborhood)

2. Building specific variables:
   - [x] Average unmet cooling setpoint difference
   - [ ] Indoor relative humidity
   - [x] Non shiftable load
   - [ ] Solar generation
   - [ ] Electricity pricing (0h, 6h, 12h, 24h)
   - [x] Indoor dry bulb temperature set point

##### Dynamics dependent variables:
1. Neighborhood level variables:
   - None
2. Building specific variables:
   - [x] Cooling storage state of charge
   - [x] Heating storage state of charge
   - [x] Domestic hot water storage state of charge
   - [x] Electrical storage state of charge
   - [x] Net electricity consumption
   - [x] Cooling device coefficient of performance
   - [x] Heating device coefficient of performance
   - [x] Cooling electricity consumption
   - [x] Heating electricity consumption
   - [x] Domestic hot water electricty consumption

##### Either taken from .csv file or calculated during runtime:
1. Neighborhood level variables:
   - None
2. Building level variables:
   - [x] Indoor dry bulb temperature
   - [x] Cooling demand
   - [x] Heating demand
   - [x] Domestic hot water demand
   - [x] Indoor dry bulb temperature difference to set point
   - [x] Power outage boolean (?) 
