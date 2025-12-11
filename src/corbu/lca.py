import os
import pandas as pd
import numpy as np

class LCA:
    def __init__(
            self, id, struct_mat_quantities, non_struct_mat_quantities,
            hvac_consumption, life_span
    ):
        self.id = id
        self.struct_mat_quantities = struct_mat_quantities
        self.nonstruct_mat_quantities = non_struct_mat_quantities
        self.hvac_consumption = hvac_consumption
        self.life_span = life_span
    
    def manual_MClca(self, prerun_lca_path, indicators, n_runs=10):

        # Set upo self.processes without using Brightway
        # Store processes functional unit conversion coeffs
        # (to translate kg to m3 when needed)
        # Get weights of for 1 m3 of each material
        p_glt = 490
        p_clt = 480
        p_bois_massif = 480
        p_cemii = 2400
        p_osb = 616
        p_acier = 7850
        self.structure_material_processes = {
            'standard_steel': {
                'names': ['Acier-S235'],
                'process_name': 'standard steel, A-C',
                'FU_conversion_coeff': 1
            },
            'glt': {
                'names': ['Bois-GL24h'],
                'process_name': 'glued laminated timber, A-C',
                'FU_conversion_coeff': 1/p_glt
            },
            'standard_concrete': {
                'names': ['Béton CEMII', 'Dalle Béton'],
                'process_name': 'concrete, cement CEM II/A, building' \
                    + ' construction, A-C, economic allocation',
                'FU_conversion_coeff': 1/p_cemii
            },
            'reinforcing_steel': {
                'names': ['Ferraillage'],
                'process_name': 'reinforcing steel, A-C',
                'FU_conversion_coeff': 1
            },
            'sheet_steel': {
                'names': ['Acier-Assemblage', 'Tôle'],
                'process_name': 'sheet steel, A-C',
                'FU_conversion_coeff': 1
            },
            'concrete_screed': {
                'names': ['Chape Béton'],
                'process_name': 'concrete, cement CEM II/A, for lean concrete' \
                    + ', A-C, economic allocation',
                'FU_conversion_coeff': 1/p_cemii
            },
            'clt': {
                'names': ['-CLT'],
                'process_name': 'cross laminated timber, A-C',
                'FU_conversion_coeff': 1/p_clt
            },
            'wood_joist': {
                'names': ['Bois-C18', 'Solives bois'],
                'process_name': 'wood joist, A-C',
                'FU_conversion_coeff': 1/p_bois_massif
            },
            # 'plaster': {
            #     'names': ['Platre'],
            #     'process_name': 'plaster',
            #     'FU_conversion_coeff': 1
            # },
            'concrete_foundation': {
                'names': ['Fondation-Beton'],
                'process_name': 'concrete, cement CEM II/A, civil engineering' \
                    ', A-C, economic allocation',
                'FU_conversion_coeff': 1/p_cemii
            },
            'stone_wool': {
                'names': ['Flocage'],
                'process_name': 'stone wool',
                'FU_conversion_coeff': 1
            }
        }

        self.nonstruct_material_processes = {
            'polystyrene': {
                'names': ['Polystyrène'],
                'process_name': 'Polystyrène expansé',
                'FU_conversion_coeff': 1
            },
            'triple_glazing': {
                'names': ['Triple vitrage'],
                'process_name': 'Triple vitrage',
                'FU_conversion_coeff': 1
            },
            'plaster': {
                'names': ['Plâtre'],
                'process_name': 'Plâtre',
                'FU_conversion_coeff': 1
            },
            'standard_concrete': {
                'names': ['Béton CEMII', 'Dalle Béton'],
                'process_name': 'concrete, cement CEM II/A, building' \
                    + ' construction, A-C, economic allocation',
                'FU_conversion_coeff': 1/p_cemii
            },
        }

        self.hvac_process = {
            'process_name': 'heat production, air-water heat pump 10kW',
            'FU_conversion_coeff': 3.6
        }

        # Load materials lca
        prerun_lca_results = pd.read_csv(prerun_lca_path).iloc[:n_runs]

        # Calculate quantity for each structural material process
        structure_material_process_quantities = {}
        for material_process_dict in self.structure_material_processes.values():
            material_quantity = 0
            # iterate over possible names for this process
            for material in material_process_dict['names']:
                # Iterate over material quantities in structure and check if names
                # match
                for elem, quantity in self.struct_mat_quantities['Total'].items():
                    if material in elem:
                        material_quantity += quantity
            # If material quantity is 0, then move to next iteration directly
            if material_quantity == 0:
                continue
            # Convert kg to m3 if needed
            material_quantity *= material_process_dict['FU_conversion_coeff']
            structure_material_process_quantities[
                material_process_dict['process_name']
            ] = material_quantity
        
        # Calculate quantity for each non-structural material process
        nonstruct_material_process_quantities = {}
        for material_process_dict in self.nonstruct_material_processes.values():
            material_quantity = 0
            # iterate over possible names for this process
            for material in material_process_dict['names']:
                # Iterate over material quantities in structure and check if names
                # match
                for elem, quantity in self.nonstruct_mat_quantities.items():
                    if material in elem:
                        material_quantity += quantity
            # If material quantity is 0, then move to next iteration directly
            if material_quantity == 0:
                continue
            # Convert kg to m3 if needed
            material_quantity *= material_process_dict['FU_conversion_coeff']
            nonstruct_material_process_quantities[
                material_process_dict['process_name']
            ] = material_quantity

        # Calculate scores for embodied and operational
        embodied_struct_dict = {}
        embodied_nonstruct_dict = {}
        operational_results_dict = {}
        # Iterate over methods
        for m in indicators:

            # Structure
            embodied_total_indicator = np.zeros(
                shape=prerun_lca_results['Run ID'].shape
            )
            # Accumulate result over all materials involved
            for material, quantity in \
                structure_material_process_quantities.items():
                embodied_total_indicator += np.array(
                    (
                        prerun_lca_results[f'{material}_{m}'] * quantity
                    ).values
                )
            embodied_struct_dict[f'{m}'] = embodied_total_indicator.tolist()

            # Non structural
            embodied_total_indicator = np.zeros(
                shape=prerun_lca_results['Run ID'].shape
            )
            # Accumulate result over all materials involved
            for material, quantity in \
                nonstruct_material_process_quantities.items():
                embodied_total_indicator += np.array(
                    (
                        prerun_lca_results[f'{material}_{m}'] * quantity
                    ).values
                )
            embodied_nonstruct_dict[f'{m}'] = embodied_total_indicator.tolist()

            operational_total_indicator = np.zeros(
                shape=prerun_lca_results['Run ID'].shape
            )
            operational_total_indicator += np.array(
                (
                    prerun_lca_results[
                        f'heat production, air-water heat pump 10kW_{m}'
                    ] * self.hvac_consumption * self.life_span
                ).values
            )
            operational_results_dict[f'{m}'] = operational_total_indicator.tolist()
        
        self.monte_carlo_results = {
            "embodied_structure": embodied_struct_dict,
            "embodied_non_structure": embodied_nonstruct_dict,
            "operational": operational_results_dict,
            "total": {
                m: [
                    i+j+k for i,j,k in zip(
                        embodied_struct_dict[m], embodied_nonstruct_dict[m], \
                            operational_results_dict[m]
                    )
                ] for m in indicators
            }
        }

    def save_results(self):
        # Create results folders if they don't exist
        try:
            os.mkdir(self.path.joinpath('./lca_results'))
        except FileExistsError:
            pass
        try:
            os.mkdir(self.path.joinpath('./lca_results/mc_results'))
        except FileExistsError:
            pass
        
        # Save Monte Carlo raw runs
        self.monte_carlo_results.to_parquet(
            self.path.joinpath(
                f'./lca_results/mc_results/sample_{self.id}.parquet'
            ),
            index=False
        )