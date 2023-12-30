# Notes Week 52
---

### Observations to use in Reward calculation

**Given in citylearn challenge 2023:**

{'day_type': 5,
 'hour': 1,
 'outdoor_dry_bulb_temperature': 24.66,
 'outdoor_dry_bulb_temperature_predicted_6h': 24.910639,
 'outdoor_dry_bulb_temperature_predicted_12h': 38.41596,
 'outdoor_dry_bulb_temperature_predicted_24h': 27.611464,
 'diffuse_solar_irradiance': 0.0,
 'diffuse_solar_irradiance_predicted_6h': 54.625927,
 'diffuse_solar_irradiance_predicted_12h': 116.84289,
 'diffuse_solar_irradiance_predicted_24h': 0.0,
 'direct_solar_irradiance': 0.0,
 'direct_solar_irradiance_predicted_6h': 143.32434,
 'direct_solar_irradiance_predicted_12h': 1020.7561,
 'direct_solar_irradiance_predicted_24h': 0.0,
 'carbon_intensity': 0.40248835,
 'indoor_dry_bulb_temperature': 23.098652,
 'non_shiftable_load': 0.35683933,
 'solar_generation': 0.0,
 'dhw_storage_soc': 0.0,
 'electrical_storage_soc': 0.2,
 'net_electricity_consumption': 0.67788136,
 'electricity_pricing': 0.02893,
 'electricity_pricing_predicted_6h': 0.02893,
 'electricity_pricing_predicted_12h': 0.02915,
 'electricity_pricing_predicted_24h': 0.02893,
 'cooling_demand': 1.1192156,
 'dhw_demand': 0.055682074,
 'occupant_count': 3.0,
 'indoor_dry_bulb_temperature_set_point': 23.222221,
 'power_outage': 0}

 **All possible observations for citylearn environment:**

 {'month': 6,
 'hour': 1,
 'day_type': 5,
 'daylight_savings_status': 0,
 'indoor_dry_bulb_temperature': 23.098652,
 'average_unmet_cooling_setpoint_difference': -0.1235699,
 'indoor_relative_humidity': 61.086185,
 'non_shiftable_load': 0.35683933,
 'dhw_demand': 0.055682074,
 'cooling_demand': 1.1192156,
 'heating_demand': 0.0,
 'solar_generation': 0.0,
 'occupant_count': 3.0,
 'indoor_dry_bulb_temperature_set_point': 23.222221,
 'power_outage': 0,
 'indoor_dry_bulb_temperature_without_control': 23.098652,
 'cooling_demand_without_control': 1.1192156,
 'heating_demand_without_control': 0.0,
 'dhw_demand_without_control': 0.055682074,
 'non_shiftable_load_without_control': 0.35683933,
 'indoor_relative_humidity_without_control': 61.086185,
 'indoor_dry_bulb_temperature_set_point_without_control': 23.222221,
 'hvac_mode': 1,
 'outdoor_dry_bulb_temperature': 24.66,
 'outdoor_relative_humidity': 77.56,
 'diffuse_solar_irradiance': 0.0,
 'direct_solar_irradiance': 0.0,
 'outdoor_dry_bulb_temperature_predicted_6h': 24.910639,
 'outdoor_dry_bulb_temperature_predicted_12h': 38.41596,
 'outdoor_dry_bulb_temperature_predicted_24h': 27.611464,
 'outdoor_relative_humidity_predicted_6h': 72.98027,
 'outdoor_relative_humidity_predicted_12h': 41.82236,
 'outdoor_relative_humidity_predicted_24h': 83.230995,
 'diffuse_solar_irradiance_predicted_6h': 54.625927,
 'diffuse_solar_irradiance_predicted_12h': 116.84289,
 'diffuse_solar_irradiance_predicted_24h': 0.0,
 'direct_solar_irradiance_predicted_6h': 143.32434,
 'direct_solar_irradiance_predicted_12h': 1020.7561,
 'direct_solar_irradiance_predicted_24h': 0.0,
 'electricity_pricing': 0.02893,
 'electricity_pricing_predicted_6h': 0.02893,
 'electricity_pricing_predicted_12h': 0.02915,
 'electricity_pricing_predicted_24h': 0.02893,
 'carbon_intensity': 0.40248835,
 'cooling_storage_soc': 0.0,
 'heating_storage_soc': 0.0,
 'dhw_storage_soc': 0.0,
 'electrical_storage_soc': 0.2,
 'net_electricity_consumption': 0.67788136,
 'cooling_electricity_consumption': 0.26175198,
 'heating_electricity_consumption': 0.0,
 'dhw_electricity_consumption': 0.059290096,
 'cooling_storage_electricity_consumption': 0.0,
 'heating_storage_electricity_consumption': 0.0,
 'dhw_storage_electricity_consumption': 0.0,
 'electrical_storage_electricity_consumption': 0.0,
 'cooling_device_cop': array(4.27586316),
 'heating_device_cop': array(3.12831856),
 'indoor_dry_bulb_temperature_delta': 0.12356949}

 **Difference:**

 'daylight_savings_status': 0,
 'average_unmet_cooling_setpoint_difference': -0.1235699,
 'cooling_demand_without_control': 1.1192156,
 'heating_demand_without_control': 0.0,
 'dhw_demand_without_control': 0.055682074,
 'non_shiftable_load_without_control': 0.35683933,
 'indoor_relative_humidity_without_control': 61.086185,
 'indoor_dry_bulb_temperature_set_point_without_control': 23.222221,
 'hvac_mode': 1,
 'cooling_storage_electricity_consumption': 0.0,
 'heating_storage_electricity_consumption': 0.0,
 'dhw_storage_electricity_consumption': 0.0,
 'electrical_storage_electricity_consumption': 0.0,
 'indoor_dry_bulb_temperature_delta': 0.12356949
 'indoor_relative_humidity': 61.086185,
 'outdoor_relative_humidity_predicted_6h': 72.98027,
 'outdoor_relative_humidity_predicted_12h': 41.82236,
 'outdoor_relative_humidity_predicted_24h': 83.230995,

---

### First results

#### Relaxed:

Relaxed after 20 epoch/ 500 epoch: 

name |	Building_1 | Building_2 |	Building_3 | District|
-----|-------------|-----------|-----------|---------|
cost_function				
annual_normalized_unserved_energy_total	|0.028435|	0.021880|	0.026368|	0.025561
| |0.043612|	0.034839	|0.022205|	0.033552
annual_peak_average	|NaN	|NaN	|NaN	|0.824677
||NaN	|NaN|	NaN	|0.680509
carbon_emissions_total	|0.673362	|0.895182|	0.830995	|0.799846
||0.368648	|0.388671	|0.478098	|0.411806
cost_total|	0.645915	|0.851934|	0.800669|	0.766173
||	0.362068	|0.370061	|0.453815	|0.395315
daily_one_minus_load_factor_average|	NaN|	NaN|	NaN|	1.099002
||NaN	|NaN	|NaN|	1.224610
daily_peak_average	|NaN|	NaN|	NaN|	0.869741
||	NaN	|NaN|	NaN|	0.606193
discomfort_delta_average	|1.014534|	0.973325|	0.268143|	0.752001
||9.239667	|7.000887|	7.906140|	8.048898
discomfort_delta_maximum	|8.919123	|11.037985|	5.922489|	8.626532
||16.679779	|14.778639|	13.923975|	15.127464
discomfort_delta_minimum|	-9.583345|	-8.084770|	-4.065001|	-7.244372
||-0.123569|	-0.581318|	-0.372210|	-0.359032
discomfort_proportion	|0.600281|	0.516791|	0.286190|	0.467754
||	0.981767	|0.977612|	0.978369|	0.979250
discomfort_too_cold_proportion	|0.173913|	0.147388|	0.088186|	0.136496
||	0.000000|	0.000000|	0.000000|	0.000000
discomfort_too_hot_proportion	|0.426367|	0.369403|	0.198003|	0.331258
||	0.981767|	0.977612|	0.978369|	0.979250
electricity_consumption_total	|0.690403|	0.910707|	0.848112|	0.816407
||	0.369596|	0.393878|	0.483236|	0.415570
monthly_one_minus_load_factor_average	|NaN|	NaN|	NaN|	1.028323
||	NaN	|NaN|	NaN	|1.102825
one_minus_thermal_resilience_proportion	|0.333333	|0.571429|	0.200000|	0.368254
||	1.000000|	1.000000|	1.000000|	1.000000
power_outage_normalized_unserved_energy_total	|0.713369|	0.569394|	0.702040|	0.661601
||	0.672167|	0.625904|	0.624914|	0.640995
ramping_average	|NaN	|NaN	|NaN|	1.410598
||	NaN	|NaN|	NaN|	0.773802
zero_net_energy	|0.580739|	0.794432|	0.815752|	0.730308
||	0.281915	|0.359632|	0.449702|	0.363750


#### Complete reward func:


name	|Building_1	|Building_2|	Building_3|	District|
-------|----------|------------|------------|-----------|
cost_function				
annual_normalized_unserved_energy_total	|0.040994|	0.031383|	0.016687|	0.029688
||	0.047288|	0.068959|	0.037210|	<span style="color:red;">0.051152 </span>
annual_peak_average	|NaN|	NaN|	NaN|	0.916526
||NaN|	NaN|	NaN|	<span style="color:green;">0.642677</span>
carbon_emissions_total|	0.664530|	0.950685|	0.768076|	0.794430
||	0.375099|	0.424919|	0.485740|	<span style="color:green;">0.428586</span>
cost_total|	0.621010	|0.941299	|0.736163|	0.766157
||	0.356556|	0.412925|	0.465554|	<span style="color:green;">0.411678</span>
daily_one_minus_load_factor_average	|NaN|	NaN|	NaN|	1.148110
||	NaN|	NaN|	NaN|	<span style="color:green;">1.205944</span>
daily_peak_average|	NaN|	NaN|	NaN|	0.977748
||NaN|	NaN|	NaN|	<span style="color:green;">0.610686</span>
discomfort_delta_average	|1.664222	|0.106197|	0.715239|	0.828553
||9.206611	|6.959458|	7.833532|	<span style="color:red;">7.999867!!!</span>
discomfort_delta_maximum|	10.693090|	7.757879|	7.290037|	8.580336
||	16.692406|	14.765867|	13.961416|	<span style="color:red;">15.139896</span>
discomfort_delta_minimum|	-8.249487	|-9.666939|	-4.148111|	-7.354846
||	-0.123569|	-0.581318|	-0.372210|	<span style="color:green;">-0.359032</span>
discomfort_proportion|	0.572230|	0.533582|	0.409318|	0.505043
||	0.980365|	0.977612|	0.978369|	<span style="color:green;">0.978782</span>
discomfort_too_cold_proportion|	0.110799|	0.251866	|0.099834|	0.154166
||	0.000000|	0.000000|	0.000000|	<span style="color:green;">0.000000</span>
discomfort_too_hot_proportion|	0.461431|	0.281716|	0.309484|	0.350877
||	0.980365|	0.977612|	0.978369|	<span style="color:red;">0.978782!!!</span>
electricity_consumption_total	|0.670024|	0.953164|	0.778043|	0.800410
||	0.377034	|0.426918|	0.488286	| <span style="color:green;">0.430746</span>
monthly_one_minus_load_factor_average|	NaN	|NaN|	NaN|	1.047417
||	NaN|	NaN|	NaN|	<span style="color:green;">1.094343</span>
one_minus_thermal_resilience_proportion|	0.333333|	0.785714|	0.266667|	0.461905
||	0.933333|	1.000000|	1.000000	| <span style="color:green;">0.977778!!!</span>
power_outage_normalized_unserved_energy_total|	0.760572|	0.843412|	0.595447|	0.733144
||	0.651892	|0.737790|	0.679437|	<span style="color:green;">0.689706</span>
ramping_average|	NaN|	NaN|	NaN|	1.357430
||	NaN|	NaN|	NaN|	<span style="color:green;">0.736733</span>
zero_net_energy|	0.575715|	0.902347|	0.747688|	0.741916
||	0.279682|	0.369444	|0.456251|	<span style="color:green;">0.368459</span>
