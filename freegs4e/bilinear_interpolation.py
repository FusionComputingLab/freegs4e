import numpy as np


#
def biliint(R, Z, psi, points):
    """Simple bilinear interpolation of 2d map

    Parameters
    ----------
    R : np.array
        R coordinates on 2d grid
    Z : np.array
        Z coordinates on 2d grid
    psi : np.array
        function values on 2d grid
    points : np.array
        coordinates where the interpolation is sought
        shape (2, whatever)

    Returns
    -------
    np.array
        interpolated values, same shape as points: (1, whatever)
    """

    nx, ny = np.shape(psi)

    R1d = R[:, :1]
    Z1d = Z[:1, :]
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    dRdZ = dR * dZ

    points_shape = np.shape(points)
    points = points.reshape(2, -1)
    len_points = np.shape(points)[1]

    points_R = R1d - points[:1, :]
    points_Z = Z1d.T - points[1:2, :]

    idxs_R = np.sum(points_R < 0, axis=0)
    idxs_Z = np.sum(points_Z < 0, axis=0)

    idxs_R = np.where(idxs_R < nx, idxs_R, nx - 1)
    idxs_Z = np.where(idxs_Z < ny, idxs_Z, ny - 1)

    iR = idxs_R[:, np.newaxis, np.newaxis]
    iZ = idxs_Z[:, np.newaxis, np.newaxis]
    qq = psi[
        np.concatenate(
            (
                np.concatenate((iR - 1, iR - 1), axis=2),
                np.concatenate((iR, iR), axis=2),
            ),
            axis=1,
        ),
        np.concatenate(
            (
                np.concatenate((iZ - 1, iZ), axis=2),
                np.concatenate((iZ - 1, iZ), axis=2),
            ),
            axis=1,
        ),
    ]

    iR = idxs_R[:, np.newaxis]
    iZ = idxs_Z[:, np.newaxis]
    xx = points_R[
        np.concatenate((iR, iR - 1), axis=1),
        np.arange(len_points)[:, np.newaxis],
    ] * (np.array([[1, -1]]))
    yy = points_Z[
        np.concatenate((iZ, iZ - 1), axis=1),
        np.arange(len_points)[:, np.newaxis],
    ] * (np.array([[1, -1]]))

    vals = (
        np.sum(np.sum(qq * yy[:, np.newaxis, :], axis=-1) * xx, axis=-1) / dRdZ
    )
    return vals.reshape(points_shape[1:])
