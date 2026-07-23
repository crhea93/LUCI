"""
Using the ascii transmission curve information from the SITELLE website, create
.dat files with averaged transmission curves for each filter.

The input files are the raw CFHT curves (e.g. http://www.cfht.hawaii.edu/Instruments/Filters/curves/cfh3601.dat
for SN3) saved here as `<FILTER>_Transmission.dat`. They are comma separated with the
wavelength in nm in the first column followed by one column per measurement of the
transmission in percent (the number of measurement columns differs from filter to filter --
7 for SN1/SN2/SN3, 2 for SN4).

The output `<FILTER>_filter.dat` files have the axis in cm-1 and the transmission in
percent; this is what `LUCI.LuciUtility.read_in_transmission` expects.

Run this from inside the `Data` directory: `python create_filters.py`
"""
import pandas as pd
import numpy as np

# Filters for which we have a `<FILTER>_Transmission.dat` file
FILTERS = ['SN1', 'SN2', 'SN3', 'SN4', 'C3', 'C4']


def create_new(filter_):
    """
    Primary function to create .dat file of averaged transmission curve
    """
    transmission = pd.read_csv('%s_Transmission.dat' % filter_, sep=',', header=None,
                               encoding='utf-8-sig')
    # Calculate average transmission over every measurement column
    avg = transmission.iloc[:, 1:].mean(axis=1)
    # Get x-axis and translate to cm-1 from nm
    trans_axis = [1e7 / trans for trans in transmission.iloc[:, 0]]
    # Combine into stacked 2D array
    combined = np.column_stack((trans_axis, avg.values))
    # Save as .dat file
    np.savetxt('%s_filter.dat' % filter_, combined, fmt='%.2f')


if __name__ == '__main__':
    for filter_ in FILTERS:
        create_new(filter_)
