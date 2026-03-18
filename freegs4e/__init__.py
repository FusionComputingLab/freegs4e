"""
FreeGS4E
========

Free boundary Grad-Shafranov solver for time evolution


License
-------

FreeGS4E is derived from FreeGS and distributed under the same license: the GNU Lesser General Public License version 3.

Copyright 2025 Nicola C. Amorisco, George K. Holt, Adriano Agnello and other contributors.

The original FreeGS license is as follows:

Copyright 2016-2021 Ben Dudson, University of York and other contributors. 
Email: benjamin.dudson@york.ac.uk

FreeGS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS.  If not, see <http://www.gnu.org/licenses/>.

"""

from importlib import metadata

__version__ = metadata.version("freegs4e")

from . import control, jtor, machine, multigrid, plotting
from .dump import OutputFile
from .equilibrium import Equilibrium
from .picard import solve
