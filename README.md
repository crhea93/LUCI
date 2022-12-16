# Welcome to LUCI
In this document, you will find a brief description of what `LUCI` is, how to install it (her), where to go to find examples, documentation, and more!

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5385351.svg)](https://doi.org/10.5281/zenodo.5385351)

You can find the RNAAS article here:
[Arxiv Paper: 2108.12428](https://arxiv.org/abs/2108.12428)

![LuciLogo.jpg](LuciLogo.jpg)


## What is LUCI
`LUCI` is a general purpose fitting pipeline built specifically with [SITELLE IFU](https://www.cfht.hawaii.edu/Instruments/Sitelle/)
data cubes in mind; however, if you need to fit any emission line spectra, LUCI
will be able to help! Although other codes exist, we built `LUCI` specifically with the user
in mind. Thanks to the clear coding practices used to create `LUCI` and her detailed documentation,
users can easily modify the code by changing numerical solvers. Additionally, we implemented
a Bayesian Inference algorithm using MCMC (using `emcee`) to derive uncertainty estimates.


## Installing `LUCI`
We have tried to make the installation of `LUCI` as smooth and painless as possible; however, if you have suggestions, please reach out to us.

Below are instructions for installing on a linux distribution (only tested on Ubuntu and Pop-OS!).

1. **Clone** this repository. I suggest cloning it in Documents or Applications.
    ```git clone https://github.com/crhea93/LUCI.git```
2. **Enter repository** wherever you cloned it.
    ```cd LUCI```
3. **Create** luci environment using the following command: `conda env create -f luci.yml`. Now, whenever you wish to use `LUCI`, you can load the environment by simply typing the following into your terminal: `conda activate luci`.  


If you are on a Mac, you will need to change step 3 slightly:
3.  **Create** luci environment with `conda create -n luci` and then install the requirements with `pip install -r requirements.txt`.
    
    **OR**
    
    **Create** luci environment with `conda env create -f luci_macOS-11.6` which 
    was kindly created by Johanna Hartke.


Now you are all set to use Luci! To load the module into a python file or jupyter notebook, simply add the following lines:
```
import sys
sys.path.insert(0, '/the/path/to/LUCI/')
import Luci
```

You can quickly test that everything is working for your system by entering the LUCI directory and running `pytest`. You may receive some warnings, but everything should pass! If not, please let me know :)


## What to use `LUCI` for
`LUCI` is - of course - first and foremost, a fitting algorithm. In order to ease the pain of fitting IFU data cubes, we have built several wrappers for the fitting functions (these can be found in **Luci/LuciFit.py**). These functionalities include, but are not limited to: reading in a data cube in the HDF5 format (`LUCI()`), fitting the entire data cube (`LUCI.fit_entire_cube`), fitting a region or masked region of the cube (`LUCI.fit_region()`), extracting and fitting an integrated region (`LUCI.fit_spectrum_region`), creating a *deep image* (`LUCI.create_deep_image`), and building a Signal-to-Noise ratio map of the cube (`LUCI.create_snr_map`).

You can also access the fitting directly by circumventing the `LUCI` wrappers and directly accessing the fitting algorithms (`Luci.LuciFitting`). With this, you can fit any spectrum using the `Luci.LuciFitting.fit` and `Luci.LuciFitting.bayes_fit` algorithms. We also provide basic plotting functionality which can be found in `Luci.LuciPlotting`.

### Where to find examples
Examples are paramount to the success of any open source code. Therefore, we have tried to make our examples as complete as possible. That said, we surely have forgotten something! If you wish to see an example that does not exist (or for an example to be better explained), please shoot us an email or open up an issue!

All examples can be found in two locations. Read-through examples can be found on our [read the docs (https://crhea93.github.io/LUCI/index.html)](https://crhea93.github.io/LUCI/index.html) page while jupyter notebooks can be found in the **Examples** folder.
I suggest starting with [https://crhea93.github.io/LUCI/example_basic_lite.html](https://crhea93.github.io/LUCI/example_basic_lite.html).

### Where to find documentation
Documentation can also be found on our [read the docs (https://crhea93.github.io/LUCI/index.html)](https://crhea93.github.io/LUCI/index.html) page. In addition to documentation on each function in `LUCI`, you can also find a description of what `LUCI` calculates and how she does what she does!


## Contributing
If you wish to contribute, that's awesome! Please shoot me an email at [carter.rhea@umontreal.ca](mailto:carter.rhea@umontreal.ca).
The easiest way to get involved is to make an issue or fork the repo, make your changes, and submit a well-documented pull request.

## Contact
If you have any questions about how to install, use, or modify `LUCI`, please send an email to [Carter Rhea](mailto:carter.rhea@umontreal.ca).

## Copyright & License
2021 Carter Rhea ([carter.rhea@umontreal.ca](mailto:carter.rhea@umontreal.ca))

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see [https://www.gnu.org/licenses/](https://www.gnu.org/licenses/).


## Citing LUCI
If you cite LUCI, please use both the following citations.

Software Citation: Carter Lee Rhea, Laurie Rousseau-Nepton, Jessie Covington, Leo Alcorn, Benjamin Vigneron, Julie Hlavacek-Larrondo, & Louis-Simon Guité. (2021). crhea93/LUCI: Luci Updates (v1.1). Zenodo. https://doi.org/10.5281/zenodo.5730149

Paper Citation:  Carter Rhea et al 2021 Res. Notes AAS 5 208

If you use the mixture density network (MDN) implementation, please include the following citation:
Carter Rhea et al 2021 Res. Notes AAS 12 276