
# FreeGS4E: Free-boundary Grad-Shafranov for Evolution

FreeGS4E is a package forked from [FreeGS](https://github.com/freegs-plasma/freegs) (v0.6.1). It retains substantial code inherited from that general-purpose equilibrium solver, but not all inherited workflows remain supported.

Its intended usage is as an underlying solver for the dynamic (time-dependent) free-boundary equilibrium solver [FreeGSNKE](https://github.com/FusionComputingLab/freegsnke).

The addition and removal of features, together with performance optimisations needed by FreeGSNKE, mean that FreeGS4E has diverged significantly from the original FreeGS codebase.

Therefore, FreeGS4E is **not intended to be a drop in replacement solver for FreeGS** but rather is designed for use explicitly **within** [FreeGSNKE](https://github.com/FusionComputingLab/freegsnke).

The presence of an inherited module does not by itself mean that its original
FreeGS workflow is supported. In particular, the standalone Picard inverse
solver and its original control-constraint interface are not supported entry
points for FreeGSNKE. See the
[support scope](https://github.com/FusionComputingLab/freegs4e/blob/main/SUPPORT.md)
for the backend API used by FreeGSNKE, known unsupported behaviour, and the
status of retained legacy code.


## Installation

Because FreeGS4E is not a standalone equilibrium solver, we recommend following the [installation instructions for FreeGSNKE](https://docs.freegsnke.com/#installation) (which will install FreeGS4E automatically).

If you would, however, like to contribute to FreeGS4E directly, please see the installation instructions in the section on contributing below.

### Supported Python and dependencies

FreeGS4E supports Python 3.10 through 3.14. Its runtime dependency bounds are
kept compatible with FreeGSNKE, which uses a subset of the same scientific
Python envelope. Changes to shared lower or upper bounds should therefore be
tested in both repositories before release.

## Getting started

Supported examples can be found in the `freegsnke/examples` directory. Examples
from the original FreeGS project should not be assumed to run with FreeGS4E.


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
