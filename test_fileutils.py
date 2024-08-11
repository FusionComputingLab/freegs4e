from freegs4e._fileutils import next_value
from numpy import allclose


def test_next_value():
    data = [
        '+9.500000000e-01+2.000000000e+00+1.000000000e+00+2.500000000e+01+0.000000000e+00\n',
        ' 9.500000000E-01 2.000000000E+00 1.000000000E+00 2.500000000E+01 0.000000000E+00\n',
        '+3.563359524e+02+2.337058846e-02-1.212203732e-02+1.953790839e-03-7.116987284e-03',
        ' 3.563359524E+02 2.337058846e-02-1.212203732E-02 1.953790839E-03-7.116987284e-03',
        '0 0 0\n',
        '   0   0\n',
    ]

    expected = [
        0.95, 2.0, 1.0, 25.0, 0.0,
        0.95, 2.0, 1.0, 25.0, 0.0,
        356.3359524, 0.02337058846, -0.01212203732, 0.001953790839, -0.007116987284,
        356.3359524, 0.02337058846, -0.01212203732, 0.001953790839, -0.007116987284,
        0, 0, 0,
        0, 0
    ]

    values = next_value(data)

    actual = [val for val in values]

    assert allclose(expected, actual)
