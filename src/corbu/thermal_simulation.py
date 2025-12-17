from batem.core.data import DataProvider
from batem.core.building import Building, ContextData, BuildingData, SideMaskData
from batem.core.inhabitants import Preference

def _run_single_thermal_simulation(args):
    """Helper function to run a single thermal simulation - used for parallelization."""
    i, row_dict, context_data, floor_compositions = args

    # Create BuildingData
    building_data = BuildingData(
        length=row_dict['building_length'],
        width=row_dict['building_width'],
        n_floors=round(row_dict['nb_floors']),
        floor_height=2.5,
        base_elevation=0.1, # 10 cm
        z_rotation_angle_deg=.0,
        glazing_ratio=0.15,
        glazing_solar_factor=0.56,
        # ref_glazing_ratio=0.15,
        # opposite_glazing_ratio=0.15,
        # left_glazing_ratio=0.15,
        # right_glazing_ratio=0.15,
        compositions={
            'wall': (
                ('plaster', 13e-3),
                ('polystyrene', 15e-2),
                ('concrete', 20e-2)
            ),
            'intermediate_floor': floor_compositions[i],
            'roof': floor_compositions[i],
            'glazing': (
                ('glass', 4e-3),
                ('air', 8e-3),
                ('glass', 4e-3),
                ('air', 8e-3),
                ('glass', 4e-3)
            ),
            'ground_floor': floor_compositions[i],
            'basement_floor': (
                ('concrete', 5e-2),
                ('polystyrene', 30e-2),
                ('concrete', 20e-2),
                ('gravels', 20e-2)
            ),
        },
        low_heating_setpoint=16,
        normal_heating_setpoint=20,
        normal_cooling_setpoint=24,
        initial_temperature=20,
        heating_period=('1/11', '1/5'),
        cooling_period=('1/5', '30/9'),
        max_heating_power=100000,
        max_cooling_power=100000,
        occupant_consumption=150,
        body_PCO2=7,
        density_occupants_per_100m2=7,
        long_absence_period=('1/8', '15/8'),
        regular_air_renewal_rate_vol_per_hour=1,
        super_air_renewal_rate_vol_per_hour=None, # uncomment when batem upgraded
        # shutter_closed_temperature=26.0,
        state_model_order_max=None,
        periodic_depth_seconds=3600,
    )
    
    # Perform simulation
    building = Building(context_data=context_data, building_data=building_data)
    dp = building.simulate(suffix='sim')
    
    total_energy_needs = 0
    cooling_energy_needs = 0
    heating_energy_needs = 0
    floor_surface = building.building_data.length * building.building_data.width

    # Single pass through floors - cache series data
    for floor in building.floors:
        phvac_key = f'PHVAC:{floor.name}#sim'
        if phvac_key in dp:
            phvac_series = dp.series(phvac_key)
            floor_energy_needs = sum(abs(x) for x in phvac_series)
            floor_cooling = sum(abs(x) for x in phvac_series if x < 0)
            floor_heating = sum(abs(x) for x in phvac_series if x > 0)
            
            total_energy_needs += floor_energy_needs
            cooling_energy_needs += floor_cooling
            heating_energy_needs += floor_heating

    return i, {
        'cooling': cooling_energy_needs/1000,
        'heating': heating_energy_needs/1000,
        'total': total_energy_needs/1000,
    }