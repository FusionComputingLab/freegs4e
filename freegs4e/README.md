# FreeGS4E package internals

This directory contains the LGPL-licensed equilibrium backend used by
FreeGSNKE. The supported surface and the status of retained FreeGS modules are
documented in the repository-level [support scope](../SUPPORT.md).

## Testing

Run the complete backend suite from the repository root with:

```shell
python -m pytest
```

## Linear solver

`multigrid.createVcycle` retains both direct and iterative implementations.
The supported FreeGS4E and FreeGSNKE equilibrium paths currently use a
single-level direct solve (`direct=True`).

## Historical iterative multigrid demonstration

> [!WARNING]
> The iterative multigrid path is retained legacy functionality. It is not used
> by the supported FreeGS4E or FreeGSNKE equilibrium workflows and is not
> covered by representative correctness tests. Its numerical behaviour is
> therefore not warranted.

`multigrid.py` contains a demonstration for a two-dimensional Poisson problem
on a square domain with fixed-value boundary conditions. It uses second-order
central differences and compares Jacobi smoothing on the full mesh with a
hierarchy of successively coarser meshes. Run the current demonstration from
the repository root with:

```shell
python -m freegs4e.multigrid
```

The original demonstration started both solves with a maximum residual of
`1.0`. In its historical configuration, 100 Jacobi iterations on the full mesh
reduced the residual only to approximately `0.87`, whereas two multigrid
V-cycles reported:

```text
Cycle 0: 0.0338261789164
Cycle 1: 0.0022779802307
```

This comparison was illustrative rather than a controlled performance
benchmark: interpolation adds overhead, while most multigrid smoothing occurs
on cheaper coarse meshes. The current script configuration and output have
since changed, so the values above are retained as historical context and are
not expected test results or evidence that the iterative implementation remains
correct.
