
# FreeGS4E: Free-boundary Grad-Shafranov for Evolution

FreeGS4E is a package forked from [FreeGS](https://github.com/freegs-plasma/freegs) (v0.6.1), which has the capability to solve the static inverse free-boundary Grad-Shafranov problem for plasma equilibria in tokamak devices.

Its intended usage is as an underlying solver for the dynamic (time-dependent) free-boundary equilibrium solver [FreeGSNKE](https://github.com/FusionComputingLab/freegsnke).

The addtion and removal of certain features within FreeGS, as well as some performance optimisation, were neccesary to enable this and so FreeGS4E has now diverged significantly from original FreeGS codebase.

Therefore, FreeGS4E is **not intended to be a drop in replacement solver for FreeGS** but rather is designed for use explicitly **within** [FreeGSNKE](https://github.com/FusionComputingLab/freegsnke).


## Installation

Given FreeGS4E is not a standalone equilibrium solver, we recommend following the [installation instructions for FreeGSNKE](https://docs.freegsnke.com/#installation) (which will install FreeGS4E automatically). 

If you would, however, like to contribute to FreeGS4E directly, please see the installation instructions in the section on contributing below.

## Getting started

All of the examples for getting started can be found within the `freegsnke/examples` directory.


## Contributing

We welcome contributions including **bug fixes** or **new feature requests** for FreeGS4E, though we would suggest making these via issues on the FreeGSNKE homepage.

If you would, however, like to install FreeGS4E separately for development purposes, clone this repository, and install the package in editable mode with the development dependencies:

```bash

git clone git@github.com:FusionComputingLab/freegs4e.git

cd freegs4e

pip install  -e  ".[dev]"

```

Changes to the `main` branch must be made via pull request. If you don't have write access to the repository, pull requests through GitHub forks are welcome.

Pre-commit hooks are used to ensure code quality so do make sure you install the following pre-commit hooks and run them prior submitting pull requests:

```bash

pre-commit install

```

## License

    Copyright 2024 Nicola C. Amorisco, George K. Holt, Adriano Agnello, and other contributors.

    FreeGS4E is licensed under the GNU Lesser General Public License version 3. The license text is included in the file LICENSE.

    The license text for FreeGS is reproduced below:

    Copyright 2016-2021 Ben Dudson, University of York, and other contributors.
    Email: benjamin.dudson@york.ac.uk

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
