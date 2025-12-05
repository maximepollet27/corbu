# CORBU: Life Cycle Analysis-informed generation of buildings for conceptual design

This repository contains a Python pipeline to generate building designs that satisfy environmental performance and geometric constraints.
The pipeline is organised as follows:
1. **cVAE-based generation** of building parameters from a trained PyTorch model.
2. **Geometry generation** from parameter sets.
3. **Structural design** derived from the building geometries.
4. **Thermal simulation** to estimate energy needs.
5. **Life Cycle Assessment (LCA)** based on geometry, structure, and thermal performance.
The design parameters are generated using a conditional VAE (cVAE), construct their geometry, design their structure, run thermal simulations, and perform a Life Cycle Assessment (LCA) of the resulting buildings.

## Installation

To install a local copy of the project, run:
```
git clone https://github.com/maximepollet27/corbu.git
cd corbu
```

Then, create and activate a virtual environment:
```
python virtualenv venv
# Linux / macOS
source venv/bin/activate
# Windows
# venv\Scripts\activate
```

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

To run the full pipeline from the project root:
```
python src\corbu\main.py
```
Additional arguments can be added to specify custom targets (the above command uses default values):
- ```--parcel_width``` and ```--parcel_length``` describe the dimensions of a rectangular parcel (default width = 30, length = 50).
- ```--floor_area``` describes the target total floor area to generate (default 2000 m2).
- ```--max_gwp``` describes the target GWP100 per m2 for the structure (default 200 kgCO2/m2).
- ```--n_solutions``` describes the number of solutions to generate (default 3).

## Repository structure

```
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ data/                    # to be downloaded separately
│
├─ notebooks/
│  ├─ monte_carlo_lca.ipynb # nb to prerun monte carlo runs for LCA uncertainty
│  └─ quick_hvac_lca.ipynb  # nb for quick LCA testing
│
├─ src/
│  └─ corbu/
│     ├─ __init__.py
│     ├─ main.py            # run_pipeline() orchestrator
│     ├─ generation.py      # cVAE loading + parameter generation and selection
│     ├─ geometry.py        # geometry creation (nodes, lines, surface, etc)
│     ├─ structure.py       # structural design and calculation of material quantities
│     ├─ thermal.py         # thermal simulations and calculation of energy needs
│     └─ lca.py             # LCA calculations
```