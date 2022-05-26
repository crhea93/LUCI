"""
Using the ascii transmission curve information from the SITELLE website, create
.dat files with averaged transmission curves for each filter
"""
import pandas as pd
import numpy as np


def create_new(filter):
    """
    Primary function to create .dat file of averaged transmission curve
    """
    transmission = pd.read_csv('%s_Transmission.dat'%filter, ",")
    # Calculate average transmission
    avg = transmission.iloc[:, [1,2,3,4,5,6,7]].mean(axis=1)
    # Get x-axis
    trans_axis = transmission.iloc[:, [0]].mean(axis=1)
    # Translate to cm-1 from nm
    trans_axis = [1e7/trans for trans in trans_axis]
    # Combine into stacked 2D array
    combined = np.column_stack((trans_axis, avg.values))
    # Save as .dat file
    np.savetxt('%s_filter.dat'%filter, combined, fmt='%.2f')

#for filter_ in ['SN1', 'SN2', 'SN3', 'C4']:
for filter_ in ['C1', 'C2']:
    create_new(filter_)
