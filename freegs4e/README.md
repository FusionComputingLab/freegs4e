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
