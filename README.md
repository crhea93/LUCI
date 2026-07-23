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

`LUCI` uses [uv](https://docs.astral.sh/uv/) to manage its environment. The exact dependency set is
pinned in `uv.lock`, so everyone gets an identical, reproducible install on every platform.

1. **Install uv** (once, if you do not already have it):
    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. **Clone** this repository and enter it:
    ```
    git clone https://github.com/crhea93/LUCI.git
    cd LUCI
    ```
3. **Create the environment** — this reads `uv.lock` and installs `LUCI` itself in editable mode:
    ```
    uv sync
    ```

That's it. There is no conda environment to activate: prefix commands with `uv run` and they use the
project environment automatically.

```
uv run python my_script.py
uv run jupyter lab          # for the notebooks in Examples/
uv run pytest               # run the test suite
```

`LUCI` is now a proper installed package, so **no `sys.path` juggling is required**. Load it with:

```python
from LuciBase import Luci
```

(Older documentation and notebooks may still show `sys.path.insert(0, '/the/path/to/LUCI/')` before
the import. That line is no longer needed and can be deleted.)

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
If you wish to contribute, that's awesome! Please shoot me an email at [carter.rhea@umontreal.ca](mailto:carterrhea93@gmail.com).
The easiest way to get involved is to make an issue or fork the repo, make your changes, and submit a well-documented pull request.

## Contact
If you have any questions about how to install, use, or modify `LUCI`, please send an email to [Carter Rhea](mailto:carterrhea93@gmail.com).

## Copyright & License
2021 Carter Rhea ([carter.rhea@umontreal.ca](mailto:carterrhea93@gmail.com))

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see [https://www.gnu.org/licenses/](https://www.gnu.org/licenses/).


## Citing LUCI
If you cite LUCI, please use both the following citations.

Software Citation: Carter Lee Rhea, Laurie Rousseau-Nepton, Jessie Covington, Leo Alcorn, Benjamin Vigneron, Julie Hlavacek-Larrondo, & Louis-Simon Guité. (2021). crhea93/LUCI: Luci Updates (v1.1). Zenodo. https://doi.org/10.5281/zenodo.5730149

Paper Citation:  Carter Rhea et al 2021 Res. Notes AAS 5 208

If you use the mixture density network (MDN) implementation, please include the following citation:
Carter Rhea et al 2021 Res. Notes AAS 12 276
