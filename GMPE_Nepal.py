""""
This script computes Peak Ground Acceleration (PGA) values using the Boore et al. (2014) GMPE.
It has to be run with the python interpreter from OpenQuake.
"""



import numpy as np
import pandas as pd
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
from openquake.hazardlib.imt import PGA
from openquake.hazardlib.contexts import SitesContext, RuptureContext, DistancesContext
import os

# -----------------------
# 1. Define the GMPE
# -----------------------
gmpe = BooreEtAl2014()

# -----------------------
# 2. Load data from data2.csv
# -----------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'Data', 'data2.csv')

print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()[:5]}...")  # Print first 5 columns

# -----------------------
# 3. Extract epicentral distance (convert from miles to km)
# -----------------------
# First column is district_distance_to_earthquakecenter(mi)
R_epi_miles = df.iloc[:, 0].values
R_epi_km = R_epi_miles * 1.60934  # Convert miles to km

print(f"Distance range (km): {R_epi_km.min():.2f} to {R_epi_km.max():.2f}")

# -----------------------
# 4. Define fixed rupture parameters (assuming Nepal earthquake characteristics)
# -----------------------
mag = 7.8                         # Mw - similar to 2015 Nepal earthquake
rake = 108.0                      # reverse fault
hypo_depth = 8.2                  # km
ztor = 7.5                        # zero-dip depth of top of rupture
dip = 7.0                         # dip angle
width = 150.0                     # rupture width in km

# -----------------------
# 5. Compute PGA for each UNIQUE distance
# -----------------------
# Get unique distances
unique_distances = np.unique(R_epi_km)
print(f"Found {len(unique_distances)} unique distances")

# Create a dictionary to store PGA values for each unique distance
pga_dict = {}

for idx, r_epi in enumerate(unique_distances):
    if (idx + 1) % 100 == 0:
        print(f"Processing unique distance {idx + 1}/{len(unique_distances)}")
    
    # Create site context
    sites = SitesContext()
    sites.sids = np.array([0])             # site ID
    sites.vs30 = np.array([250.0])         # m/s (default for Nepal)
    sites.vs30measured = np.array([False])
    sites.z1pt0 = np.array([np.nan])
    sites.z2pt5 = np.array([np.nan])
    
    # Create rupture context
    rup = RuptureContext()
    rup.mag = mag
    rup.rake = rake
    rup.hypo_depth = hypo_depth
    rup.ztor = ztor
    rup.dip = dip
    rup.width = width
    
    # Create distances context
    dists = DistancesContext()
    dists.rjb = np.array([r_epi])         # Rjb ≈ Repi
    dists.rrup = np.array([np.sqrt(r_epi**2 + hypo_depth**2)])
    dists.rx = np.array([0.0])
    
    # Compute PGA
    mean, sigma = gmpe.get_mean_and_stddevs(
        sites,
        rup,
        dists,
        PGA(),
        stddev_types=["total"]
    )
    
    # Convert from log scale to g (acceleration in units of gravity)
    pga_g = np.exp(mean[0])
    pga_dict[r_epi] = pga_g

print(f"\nSuccessfully computed PGA for all {len(pga_dict)} unique distances")

# -----------------------
# 6. Map PGA values back to original data
# -----------------------
# Create a list of PGA values corresponding to each row's distance
pga_values = [pga_dict[distance] for distance in R_epi_km]

# Keep the first 7 digits for precision
pga_values = [float(f"{pga:.7f}") for pga in pga_values]

# -----------------------
# 7. Create new dataframe with PGA column
# -----------------------
df_new = df.copy()
df_new['PGA_g'] = pga_values

# -----------------------
# 8. Save to new file
# -----------------------
output_path = os.path.join(script_dir, 'Data', 'new_data2.csv')
df_new.to_csv(output_path, index=False)
print(f"\nSaved new data to: {output_path}")
print(f"New file shape: {df_new.shape}")
print(f"PGA range (g): {np.array(pga_values).min():.6f} to {np.array(pga_values).max():.6f}")
print(f"Sample PGA values (first 5): {pga_values[:5]}")
