import os
import tqdm
import time
import argparse
import json
from pathlib import Path
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Batem
from batem.core.data import DataProvider
from batem.core.building import Building, ContextData, BuildingData, SideMaskData
from batem.core.inhabitants import Preference

# Custom imports
from cvae_generation import *
from geometry import *
from structure import *
from thermal_simulation import _run_single_thermal_simulation
from lca import *

def load_context(root_path, context_name="default_context.json"):
    
    # Load context from file
    with open(root_path.joinpath(f"./data/context/{context_name}")) as json_file:
        context = json.load(json_file)
    
    return context

def generate_structures(
        parcel_length, parcel_width, floor_area_target, max_gwp, path, n_sol=15
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and model
    X_scaler, y_scaler, cat_onehot_enc, disc_onehot_enc, cat_cols, disc_cols, \
        conti_cols, target_cols, n_cat, n_disc, n_cont, y_target_orig, \
            y_target_scaled = load_data(
                path, max_gwp, floor_area_target, device
            )
    model = load_model(path, device)
    
    # Generate raw designs
    NUM_GEN = 1000
    TAU = 1.
    df_gen = generate_designs(
        model, NUM_GEN, TAU, y_target_orig, y_target_scaled,
        X_scaler, y_scaler, cat_onehot_enc, disc_onehot_enc,
        n_cat, n_disc, n_cont, cat_cols, disc_cols, conti_cols, target_cols,
        device
    )

    # Filter generated structures and select 15 diverse solutions
    selected_solutions = filter_designs(
        df_gen, parcel_length, parcel_width, floor_area_target, max_gwp,
        n_sol=n_sol
    )

    # Calc total impact predicted
    selected_solutions["Pred GWP"] = selected_solutions["Mean GWP / Floor area"] \
        * selected_solutions["building_width"] * \
            selected_solutions["building_length"] * \
                selected_solutions["nb_floors"]

    return selected_solutions

def structural_design(generated_structures, context, path, keep_sol=10):

    # Load db of pre-dimensioned structural elements
    structural_elements_db = StructuralElementsDB(
        path.joinpath('data/predim_struct_elements')
    )

    # Iterate over each design proposed
    structure_material_quantities = {}
    non_struct_mat_quantities = {}
    floor_compositions = {}
    for i in tqdm.tqdm(range(len(generated_structures))):

        input_params = {
            "sample_id": i,
            "nb_floors": generated_structures.iloc[i]['nb_floors'].item(),
            "building_width": generated_structures.iloc[i][
                'building_width'
            ].item(),
            "building_length": generated_structures.iloc[i][
                'building_length'
            ].item(),
            "column_height": 2.5,
            "core_length": generated_structures.iloc[i]['core_length'].item(),
            "core_width": generated_structures.iloc[i]['core_width'].item(),
            "max_stair_dist": generated_structures.iloc[i][
                'max_stair_distance'
            ].item(),
            "core_y_location": generated_structures.iloc[i][
                'core_location_y'
            ].item(),
            "column_spacing_x": generated_structures.iloc[i][
                'column_spacing_x'
            ].item(),
            "column_spacing_y": generated_structures.iloc[i][
                'column_spacing_y'
            ].item(),
            "floor_material": generated_structures.iloc[i][
                'floor_material'
            ],
            "beam_material": generated_structures.iloc[i][
                'beam_material'
            ],
            "column_material": generated_structures.iloc[i][
                'column_material'
            ]
        }

        # Generate raw geometries (node, lines, surfs) not dimensioned
        core_pls = get_core_locations(input_params, plot=False)
        cell_pls, cell_surfs = create_base_surfaces(
            input_params, core_pls, plot=False
        )
        column_locations = generate_column_locations(
            input_params, cell_surfs, plot=False
        )
        floor_locations = generate_floors(column_locations, plot=False)
        beam_locations, span_dirs = generate_beams(
            floor_locations, core_pls, plot=False
        )
        columns_1floor = generate_columns_1floor(
            input_params, beam_locations, floor_locations, plot=False
        )
        walls_1floor = generate_walls_1floor(
            input_params, core_pls, columns_1floor, floor_locations, plot=False
        )
        nodes, floors, beams, span_dirs, columns, walls, foundation_floors, \
        foundation_beams, foundation_span_dirs = generate_full_building(
            input_params, floor_locations, beam_locations, span_dirs,
            columns_1floor, walls_1floor, core_pls
        )

        # Perform structural design
        # in a try/except block for cases in which no suitable pre-designed
        # element can be found in a specific design
        try:
            structure = Structure(
                path,
                i,
                'Logement',
                'Argile et limons mous',
                structural_elements_db,
                geom_objects={
                    "nodes": nodes,
                    "floors": floors,
                    "span_dirs": span_dirs,
                    "beams": beams,
                    "columns": columns,
                    "walls": walls,
                    "foundation_floors": foundation_floors,
                    "foundation_span_dirs": foundation_span_dirs,
                    "foundation_beams": foundation_beams
                },
                params=input_params
            )
            structure.design()
            structure.compute_material_quantities()
            structure_material_quantities[i] = structure.material_quantities

            # Calculate material quantities for envelope and non-structural
            # material for floors
            envelope_total_surface = (
                input_params["building_length"] + input_params["building_width"]
            ) * input_params["column_height"] * input_params["nb_floors"] * 2
            wall_surf = envelope_total_surface * (1-context["glazing_ratio"])
            glazing_surf = envelope_total_surface * (context["glazing_ratio"])

            # vertical surfaces (external walls and windows)
            non_struct_mat_quantities[i] = {
                "Plâtre (kg)": wall_surf * 13e-3 * 950,
                "Polystyrène (kg)": wall_surf * 15e-2 * 25,
                "Mur-Béton CEMII (kg)": wall_surf * 0.2 * 2400,
                "Triple vitrage (m2)": glazing_surf,
            }

            # horizontal surfaces
            groundfloor_surface = input_params["building_length"] \
                * input_params["building_width"]
            total_floor_surface = groundfloor_surface \
                * (input_params["nb_floors"] + 1)
            # levels
            non_struct_mat_quantities[i]["Plâtre (kg)"] += total_floor_surface \
                * 3e-2 * 950
            non_struct_mat_quantities[i]["Polystyrène (kg)"] += \
                total_floor_surface * 4e-2 * 25

            # Work out floor compositions
            average_floor_thick = np.mean(
                [
                    f.height for f in list(structure.all_floors.values()) \
                        + list(structure.all_foundation_floors.values())
                ]
            ).item()
            if input_params["floor_material"] == "Coulee-En-Place":
                level_composition = (
                    ('plaster', 1e-2),
                    ('polystyrene', 2e-2),
                    ("concrete", average_floor_thick),
                    ('polystyrene', 2e-2),
                    ('plaster', 1e-2)
                )
            elif input_params["floor_material"] == "CLT":
                level_composition = (
                    ('plaster', 1e-2),
                    ('polystyrene', 2e-2),
                    ("concrete", 0.06),
                    ("wood", average_floor_thick - 0.06),
                    ('polystyrene', 2e-2),
                    ('plaster', 1e-2)
                )
            elif input_params["floor_material"] == "Collaborant":
                level_composition = (
                    ('plaster', 1e-2),
                    ('polystyrene', 2e-2),
                    ("concrete", average_floor_thick - 0.00075),
                    ("steel", 0.00075),
                    ('polystyrene', 2e-2),
                    ('plaster', 1e-2)
                )
            floor_compositions[i] = level_composition

        except ValueError as e:
            pass

    # Remove failed designs from list of possible solutions, and renumber
    # objects
    generated_structures = generated_structures.iloc[
        list(structure_material_quantities.keys())
    ].reset_index(drop=True)
    structure_material_quantities = {
        i:structure_material_quantities[
            sorted(list(structure_material_quantities.keys()))[i]
        ] for i in range(len(generated_structures))
    }
    non_struct_mat_quantities = {
        i:non_struct_mat_quantities[
            sorted(list(non_struct_mat_quantities.keys()))[i]
        ] for i in range(len(generated_structures))
    }

    # Keep only the specified number of solutions
    generated_structures = generated_structures.iloc[:keep_sol]
    structure_material_quantities = {
        i:structure_material_quantities[i] for i in range(keep_sol)
    }
    non_struct_mat_quantities = {
        i:non_struct_mat_quantities[i] for i in range(keep_sol)
    }

    return (
        generated_structures, structure_material_quantities, \
            non_struct_mat_quantities, floor_compositions
    )

def thermal_simulation(
        generated_structures, floor_compositions, context, plot=False,
        verbose=False
    ):

    # Define context
    context_data: ContextData = ContextData(
        latitude_north_deg=context["latitude_north_deg"],
        longitude_east_deg=context["longitude_east_deg"],
        starting_stringdate=context["starting_stringdate"],
        ending_stringdate=context["ending_stringdate"],
        location=context["location"],
        albedo=context["albedo"],
        pollution=context["pollution"],
        number_of_levels=context["number_of_levels"],
        ground_temperature=context["ground_temperature"],
        side_masks=[
            SideMaskData(
                x_center=v["x_center"], y_center=v["y_center"],
                width=v["width"], height=v["height"], elevation=v["elevation"],
                exposure_deg=v["exposure_deg"], slope_deg=v["slope_deg"],
                normal_rotation_angle_deg=v["normal_rotation_angle_deg"],
            ) for v in context["side_masks"].values()
        ]
    )

    # Iterate over each solution
    hvac_consumptions = {}
    for i, sample in tqdm.tqdm(
        enumerate(generated_structures.to_numpy().tolist()),
        total=len(generated_structures)
    ):

        # define building
        building_data: BuildingData = BuildingData(
            length=sample[5],
            width=sample[4],
            n_floors=round(sample[3]),
            floor_height=2.5,
            base_elevation=0.1, # 10 cm
            z_rotation_angle_deg=.0,
            glazing_ratio=0.15,
            glazing_solar_factor=0.56,
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
            super_air_renewal_rate_vol_per_hour=None,
            state_model_order_max=None,  # None means no reduction
            periodic_depth_seconds=3600,  # 1 hour for a 1h time step is a good compomize: it removes the higher dynamical behaviors
        )

        # Perform simulation
        building: Building = Building(
            context_data=context_data, building_data=building_data
        )
        time_start: float = time.time()
        dp: DataProvider = building.simulate(suffix='sim')
        time_end: float = time.time()
        
        total_energy_needs: float = 0
        cooling_energy_needs: float = 0
        heating_energy_needs: float = 0
        floor_surface: float = building.building_data.length \
            * building.building_data.width
        preference: Preference = Preference()
        for floor in building.floors:
            print(f'\n##### Zone {floor.name} ######\n')
            # Only floors with HVAC controllers have PHVAC variables
            # (floor0/basement doesn't have HVAC)
            if f'PHVAC:{floor.name}#sim' in dp:
                floor_energy_needs: float = sum(
                    [abs(_) for _ in dp.series(f'PHVAC:{floor.name}#sim')]
                )
                cooling_energy_needs += sum(
                    [
                        abs(_) for _ in dp.series(
                            f'PHVAC:{floor.name}#sim'
                        ) if _ < 0
                    ]
                )
                heating_energy_needs += sum(
                    [
                        abs(_) for _ in dp.series(
                            f'PHVAC:{floor.name}#sim'
                        ) if _ > 0
                    ]
                )
                print(
                    f'Floor {floor.name} HVAC system consumption:' + \
                    f' {round(floor_energy_needs/1000/floor_surface)}' + \
                    f' kWh/m2.year (cooling: ' + \
                    f'{round(cooling_energy_needs/1000/floor_surface)} ' + \
                    f'kWh/m2.year, heating: ' + \
                    f'{round(heating_energy_needs/1000/floor_surface)} ' + \
                    'kWh/m2.year)'
                )
                total_energy_needs += floor_energy_needs
                preference.print_assessment(
                    dp.datetimes,
                    dp.series(f'PHVAC:{floor.name}#sim'),
                    dp.series(f'TZ_OP:{floor.name}#sim'),
                    dp.series(f'CCO2:{floor.name}#sim'),
                    dp.series(f'OCCUPANCY:{floor.name}')
                )
            else:
                print(f'Floor {floor.name} has no HVAC system')
        
        print(
            'Total HVAC system consumption: ' + \
            f'{round(total_energy_needs/1000)} kWh/year'
        )

        if plot:
            # plot window heliodons
            for floor in building.floors:
                # Only plot heliodon for floors with windows (skip basement)
                if len(floor.window_masks()) > 0:
                    building.plot_heliodon(floor.floor_number)

            # plot resultss
            dp.plot(
                'PHVAC:floor1#sim', 'PHVAC:floor2#sim', 'PHVAC:floor3#sim',
                'PHVAC:floor4#sim', 'PHVAC:floor5#sim',
                'GAIN_OCCUPANCY:floor1#sim', 'GAIN_SOLAR:floor1#sim',
                'GAIN_SOLAR:floor2#sim', 'GAIN_SOLAR:floor3#sim',
                'GAIN_SOLAR:floor4#sim', 'GAIN_SOLAR:floor5#sim',
                'TZ:floor1#sim', 'weather_temperature', 'SETPOINT:floor1#sim',
                'MODE:floor1#sim', 'TZ_OP:floor1#sim', 'TZ_OP:floor2#sim',
                'TZ_OP:floor3#sim', 'TZ_OP:floor4#sim', 'TZ_OP:floor5#sim'
            )
            dp.plot(
                'TZ_OP:floor0#sim', 'TZ_OP:floor1#sim', 'TZ_OP:floor2#sim',
                'TZ_OP:floor3#sim', 'TZ_OP:floor4#sim', 'TZ_OP:floor5#sim',
                'weather_temperature', threshold=1
            )
            dp.plot(
                'TZ_OP:floor5#sim', 'weather_temperature', 'Q:floor5-outdoor',
                'OCCUPANCY:floor5', 'MODE:floor5', threshold=1
            )
            plt.show()
            building.draw()  # Call this LAST if you want to visualize the 3D building

        
        total_hvac_consumption = total_energy_needs/1000 # in kWh/year
        total_cooling_needs = cooling_energy_needs/1000 # in kWh/year
        total_heating_needs = heating_energy_needs/1000 # in kWh/year
        # print(f'Total HVAC system consumption: {round(total_energy_needs/1000)} kWh/year')
        hvac_consumptions[i] = {
            'cooling': total_cooling_needs,
            'heating': total_heating_needs,
            'total': total_hvac_consumption,
        }

    return hvac_consumptions

def thermal_simulation_parallel(
        generated_structures, floor_compositions, context, plot=False, n_jobs=2,
        verbose=False
    ):

    # Define context
    context_data: ContextData = ContextData(
        latitude_north_deg=context["latitude_north_deg"],
        longitude_east_deg=context["longitude_east_deg"],
        starting_stringdate=context["starting_stringdate"],
        ending_stringdate=context["ending_stringdate"],
        location=context["location"],
        albedo=context["albedo"],
        pollution=context["pollution"],
        number_of_levels=context["number_of_levels"],
        ground_temperature=context["ground_temperature"],
        side_masks=[
            SideMaskData(
                x_center=v["x_center"], y_center=v["y_center"],
                width=v["width"], height=v["height"], elevation=v["elevation"],
                exposure_deg=v["exposure_deg"], slope_deg=v["slope_deg"],
                normal_rotation_angle_deg=v["normal_rotation_angle_deg"],
            ) for v in context["side_masks"].values()
        ]
    )

    # Pre-load weather data by accessing it through ContextData
    # This forces weather data to be loaded once before the loop
    # Try multiple ways to access weather data depending on batem implementation
    try:
        # Method 1: Direct weather attribute
        if hasattr(context_data, 'weather'):
            _ = context_data.weather
        # Method 2: Weather data provider
        elif hasattr(context_data, 'weather_data'):
            _ = context_data.weather_data
        # Method 3: Access through a property that triggers loading
        elif hasattr(context_data, '_weather'):
            _ = context_data._weather
        # Method 4: Create minimal building to trigger weather loading
        elif len(generated_structures) > 0:
            first_row = generated_structures.iloc[0]
            dummy_building_data = BuildingData(
                length=first_row['building_length'],
                width=first_row['building_width'],
                n_floors=round(first_row['nb_floors']),
                floor_height=2.5,
                base_elevation=1.0,
                z_rotation_angle_deg=.0,
                glazing_ratio=0.15,
                glazing_solar_factor=0.56,
                compositions={
                    'wall': (
                        ('plaster', 13e-3),
                        ('polystyrene', 15e-2),
                        ('concrete', 20e-2)
                    ),
                    'intermediate_floor': floor_compositions[0],
                    'roof': floor_compositions[0],
                    'glazing': (
                        ('glass', 4e-3),
                        ('air', 8e-3),
                        ('glass', 4e-3),
                        ('air', 8e-3),
                        ('glass', 4e-3)
                    ),
                    'ground_floor': floor_compositions[0],
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
                max_heating_power=8000,
                max_cooling_power=8000,
                occupant_consumption=150,
                body_PCO2=7,
                density_occupants_per_100m2=7,
                long_absence_period=('1/8', '15/8'),
                regular_air_renewal_rate_vol_per_hour=1,
                super_air_renewal_rate_vol_per_hour=None,
                state_model_order_max=None,
                periodic_depth_seconds=3600,
            )
            dummy_building = Building(
                context_data=context_data, building_data=dummy_building_data
            )
            # Access a property that would trigger weather loading
            if hasattr(dummy_building, 'context'):
                _ = dummy_building.context
            del dummy_building, dummy_building_data
    except (AttributeError, Exception):
        # Weather data loading might happen differently, continue anyway
        # The context_data object should still be reused across iterations
        pass

    # Prepare arguments for parallel execution
    tasks = [
        (i, row.to_dict(), context_data, floor_compositions)
        for i, (_, row) in enumerate(generated_structures.iterrows())
    ]
    
    # Run simulations in parallel or sequentially
    hvac_consumptions = {}
    # Parallel execution using ProcessPoolExecutor
    # Using processes for better CPU-bound performance
    n_jobs = min(n_jobs, len(tasks), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(_run_single_thermal_simulation, task): task[0]
            for task in tasks
        }

        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Thermal simulation"
        ):
            try:
                i, result = future.result()
                hvac_consumptions[i] = result
            except Exception as e:
                if verbose:
                    print(f"Error in simulation {futures[future]}: {e}")
                # Continue with other simulations
                pass

    print(f"Completed {len(hvac_consumptions)} thermal simulation(s)")
    
    return hvac_consumptions

def perform_lca(
        struct_mat_quantities, non_struct_mat_quant, hvac_consumption, path
    ):

    INDICATORS = {
        "('ecoinvent-3.9.1', 'ReCiPe 2016 v1.03, midpoint (H)', " \
            + "'climate change', 'global warming potential (GWP100)')" : "GWP100"
    }
    
    # Perform LCA for each design
    lca_results = {}
    for i in tqdm.tqdm(range(len(list(struct_mat_quantities.keys())))):
        mc_lca = LCA(
            i, struct_mat_quantities[i], non_struct_mat_quant[i],
            hvac_consumption[i]["total"], life_span=50
        )
        mc_lca.manual_MClca(
            path.joinpath("./data/prerun_lca/prerun_lca_MC10000.csv"),
            indicators=list(INDICATORS.keys()),
            n_runs=10000
        ) # Continue manual MClca func from line 121 in lca.py

        lca_results[i] = mc_lca.monte_carlo_results
    
    # Make LCA results df
    rows = []
    for num, res in lca_results.items():
        embodied_structure = res.get("embodied_structure", {})
        embodied_non_structure = res.get("embodied_non_structure", {})
        operational = res.get("operational", {})
        total = res.get("total", {})

        for indicator in list(INDICATORS.keys()):
            emb_struct_list = embodied_structure.get(indicator, [])
            emb_nonstruct_list = embodied_non_structure.get(indicator, [])
            op_list = operational.get(indicator, [])
            tot_list = total.get(indicator, [])
            n = max(len(emb_struct_list), len(op_list), len(tot_list))
            for i in range(n):
                rows.append(
                    {
                        "building_id": num,
                        "MC_run": i,
                        "indicator": INDICATORS[indicator],
                        "embodied_structure": emb_struct_list[i] if i < len(emb_struct_list) else None,
                        "embodied_nonstruct": emb_nonstruct_list[i] if i < len(emb_nonstruct_list) else None,
                        "operational": op_list[i] if i < len(op_list) else None,
                        "total": tot_list[i] if i < len(tot_list) else None,
                    }
                )

    return pd.DataFrame(rows)

def plot_lca_uncertainties(df, indicator="GWP100", result_set="total"):
    subset = df[(df["indicator"] == indicator) & df[result_set].notna()].copy()

    all_values = subset[result_set].values
    bins = np.histogram_bin_edges(all_values, bins="auto")

    plt.figure()

    for b_id, group in subset.groupby("building_id"):
        median_val = group[result_set].median()

        sns.histplot(
            data=group,
            x=result_set,
            bins=bins,
            element="step",
            stat="count",
            common_norm=False,
            alpha=0.4,
            label=f"{b_id} (median={median_val:.2f})",
        )

    plt.legend(title="building_id (median)")
    plt.title(f"Histogram of {result_set} impacts for indicator = {indicator}")
    plt.xlabel(f"{result_set} {indicator}")
    plt.tight_layout()
    plt.show()

def run_pipeline(floor_target, max_gwp, root_path, n_sol, n_jobs):

    # 0. Load context
    context = load_context(root_path)

    # 1. Generate n_sol * 2 solutions using cVAE
    print("Generating structures with cVAE")
    start_time = time.time()
    generated_designs = generate_structures(
        context["parcel_length"], context["parcel_width"], floor_target,
        max_gwp, root_path, n_sol=n_sol * 2 # Generate more solutions than needed at first
    )
    print(f"Generation time = {time.time() - start_time}")

    generated_designs.to_csv(
        root_path.joinpath("./data/results/cvae_designs.csv"), index=False
    )

    # 2. Perform dimensioning and compute total material quantities
    print("Structural design and material quantities calculation")
    start_time = time.time()
    generated_designs, struct_mat_quantities, non_struct_mat_quantities,\
        floor_compos = structural_design(
            generated_designs, context, root_path, keep_sol=n_sol
    )
    print(f"Structural design time = {time.time() - start_time}")

    generated_designs.to_csv(
        root_path.joinpath("./data/results/structure_designs.csv"), index=False
    )

    # 3. Perform energy simulation for the 10 best structures
    start_time = time.time()
    print("Performing thermal simulation")
    if n_jobs != 1:
        hvac_consumptions = thermal_simulation(
            generated_designs, floor_compos, context, plot=False
        )
    else:
        hvac_consumptions = thermal_simulation_parallel(
            generated_designs, floor_compos, context, plot=False, n_jobs=n_jobs
        )
    
    print(f"Thermal simulation time = {time.time() - start_time}")



    # 4. Compute LCA
    lca_results = perform_lca(
        struct_mat_quantities, non_struct_mat_quantities, hvac_consumptions,
        root_path
    )

    print(lca_results)

    # 5. Plot LCA results as histograms
    plot_lca_uncertainties(lca_results)

def main():

    # Collect inputs from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor_area", type=float, default=2000.)
    parser.add_argument("--max_gwp", type=float, default=200.)
    parser.add_argument("--n_solutions", type=int, default=3)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()
    FLOOR_AREA_TARGET = args.floor_area
    MAX_GWP = args.max_gwp
    N_SOL = args.n_solutions
    N_JOBS = args.n_jobs
    ROOT_PATH = Path(__file__).resolve().parents[2]

    results = run_pipeline(FLOOR_AREA_TARGET,MAX_GWP, ROOT_PATH, N_SOL, N_JOBS)

if __name__ == "__main__":
    main()