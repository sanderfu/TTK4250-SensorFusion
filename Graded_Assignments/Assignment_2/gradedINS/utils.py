import numpy as np
from mytypes import ArrayLike


def cross_product_matrix(n: ArrayLike, debug: bool = True) -> np.ndarray:
    assert len(n) == 3, f"utils.cross_product_matrix: Vector not of length 3: {n}"
    vector = np.array(n, dtype=float).reshape(3)

    S = np.zeros((3, 3))  # TODO: Create the cross product matrix
    S[0,1] = -n[2]
    S[0,2] = n[1]
    S[1,2] = -n[0]
    S[1,0] = n[2]
    S[2,0] = -n[1]
    S[2,1] = n[0]


    if debug:
        assert S.shape == (
            3,
            3,
        ), f"utils.cross_product_matrix: Result is not a 3x3 matrix: {S}, \n{S.shape}"
        assert np.allclose(
            S.T, -S
        ), f"utils.cross_product_matrix: Result is not skew-symmetric: {S}"

    return S

def test_cross_product_matrix():
    vec = np.array([0,4,-2])
    vec2 = np.array([23, 34, -1])

    cross_prod = cross_product_matrix(vec)@vec2

    assert((cross_prod == np.cross(vec, vec2)).all()), "Cross product not the same."
    print("Test passed, cross product matrix works")

test_cross_product_matrix()
