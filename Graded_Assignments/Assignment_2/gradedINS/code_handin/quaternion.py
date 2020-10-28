import numpy as np
import utils


def quaternion_product(ql: np.ndarray, qr: np.ndarray) -> np.ndarray:
    """Perform quaternion product according to either (10.21) or (10.34).

    Args:
        ql (np.ndarray): Left quaternion of the product of either shape (3,) (pure quaternion) or (4,)
        qr (np.ndarray): Right quaternion of the product of either shape (3,) (pure quaternion) or (4,)

    Raises:
        RuntimeError: Left or right quaternion are of the wrong shape
        AssertionError: Resulting quaternion is of wrong shape

    Returns:
        np.ndarray: Quaternion product of ql and qr of shape (4,)s
    """
    if ql.shape == (4,):
        eta_left = ql[0]
        epsilon_left = ql[1:].reshape((3, 1)).squeeze()
        
    elif ql.shape == (3,):
        eta_left = 0
        epsilon_left = ql.reshape((3, 1)).squeeze()
    else:
        raise RuntimeError(
            f"utils.quaternion_product: Quaternion multiplication error, left quaternion shape incorrect: {ql.shape}"
        )

    if qr.shape == (4,):
        eta_right = qr[0]
        epsilon_right = qr[1:].reshape((3,1)).squeeze()
        q_right = qr.copy()
    elif qr.shape == (3,):
        eta_right = 0
        epsilon_right = qr.reshape((3,1)).squeeze()
        q_right = np.concatenate(([0], qr))
    else:
        raise RuntimeError(
            f"utils.quaternion_product: Quaternion multiplication error, right quaternion wrong shape: {qr.shape}"
        )

    quaternion = np.zeros((4,))  # Done: Implement quaternion product


    quaternion[0] = eta_left*eta_right - np.dot(epsilon_left.T, epsilon_right)
    quaternion[1:4] = eta_right*epsilon_left + eta_left*epsilon_right + np.cross(epsilon_left, epsilon_right)


    # Ensure result is of correct shape
    quaternion = quaternion.ravel()
    assert quaternion.shape == (
        4,
    ), f"utils.quaternion_product: Quaternion multiplication error, result quaternion wrong shape: {quaternion.shape}"
    return quaternion

def test_quaternion_product():
    quat1 = np.array([-0.321, -0.003, -0.899, 0.300])
    quat2 = np.array([0.713, -0.003, -0.665, 0.222])

    res = quaternion_product(quat1, quat2)
    true_product = np.array([-0.89332,-0.00125,-0.42776,0.14194])
    assert(np.allclose(res, true_product, atol=0.0001)), "Does not return correct quaternion product."
    print("Test passed, quaternion product")
#test_quaternion_product()

def quaternion_to_rotation_matrix(
    quaternion: np.ndarray, debug: bool = True
) -> np.ndarray:
    """Convert a quaternion to a rotation matrix

    Args:
        quaternion (np.ndarray): Quaternion of either shape (3,) (pure quaternion) or (4,)
        debug (bool, optional): Debug flag, could speed up by setting to False. Defaults to True.

    Raises:
        RuntimeError: Quaternion is of the wrong shape
        AssertionError: Debug assert fails, rotation matrix is not element of SO(3)

    Returns:
        np.ndarray: Rotation matrix of shape (3, 3)
    """
    if quaternion.shape == (4,):
        eta = quaternion[0]
        epsilon = quaternion[1:]
    elif quaternion.shape == (3,):
        eta = 0
        epsilon = quaternion.copy()
    else:
        raise RuntimeError(
            f"quaternion.quaternion_to_rotation_matrix: Quaternion to multiplication error, quaternion shape incorrect: {quaternion.shape}"
        )

    R = np.zeros((3, 3))  # Done: Convert from quaternion to rotation matrix

    R = np.eye(3) + 2*eta*utils.cross_product_matrix(epsilon) + 2*utils.cross_product_matrix(epsilon) @ utils.cross_product_matrix(epsilon)

    if debug:
        assert np.allclose(
            np.linalg.det(R), 1, atol = 0.0001 #Added tolerance boundaries here as the default is 0.00001 for rtol. 
        ), f"quaternion.quaternion_to_rotation_matrix: Determinant of rotation matrix not close to 1"
        assert np.allclose(
            R.T, np.linalg.inv(R), atol = 0.0001
        ), f"quaternion.quaternion_to_rotation_matrix: Transpose of rotation matrix not close to inverse"

    return R

def test_quaternion_to_rotation_matrix():
    quat = np.array([0.7071, 0.7071, 0, 0])
    rotation_mat = quaternion_to_rotation_matrix(quat)

    true_rot = np.array([[1, 0, 0,], [0, 0, -1], [0, 1, 0]])
    assert(np.allclose(rotation_mat, true_rot, atol=0.0001)), "Does not return correct quaternion product."
    print("Test passed, quaternion_to_rotation")
#test_quaternion_to_rotation_matrix()

def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion into euler angles

    Args:
        quaternion (np.ndarray): Quaternion of shape (4,)

    Returns:
        np.ndarray: Euler angles of shape (3,)
    """

    assert quaternion.shape == (
        4,
    ), f"quaternion.quaternion_to_euler: Quaternion shape incorrect {quaternion.shape}"

    #quaternion_squared = quaternion ** 2
    eta = quaternion[0]
    eps1 = quaternion[1]
    eps2 = quaternion[2]
    eps3 = quaternion[3]

    #10.38 in the book
    phi = np.arctan2(2*(eps3*eps2 + eta*eps1), eta**2-eps1**2-eps2**2 + eps3**2)  # Done: Convert from quaternion to euler angles
    theta = np.arcsin(2*(eta*eps2 - eps1*eps3))  # Done: Convert from quaternion to euler angles
    psi = np.arctan2(2*(eps1*eps2 + eta*eps3), eta**2+eps1**2-eps2**2 - eps3**2)  # Done: Convert from quaternion to euler angles

    euler_angles = np.array([phi, theta, psi])
    assert euler_angles.shape == (
        3,
    ), f"quaternion.quaternion_to_euler: Euler angles shape incorrect: {euler_angles.shape}"

    return euler_angles



def euler_to_quaternion(euler_angles: np.ndarray) -> np.ndarray:
    """Convert euler angles into quaternion

    Args:
        euler_angles (np.ndarray): Euler angles of shape (3,)

    Returns:
        np.ndarray: Quaternion of shape (4,)
    """

    assert euler_angles.shape == (
        3,
    ), f"quaternion.euler_to_quaternion: euler_angles shape wrong {euler_angles.shape}"

    half_angles = 0.5 * euler_angles
    c_phi2, c_theta2, c_psi2 = np.cos(half_angles)
    s_phi2, s_theta2, s_psi2 = np.sin(half_angles)

    quaternion = np.array(
        [
            c_phi2 * c_theta2 * c_psi2 + s_phi2 * s_theta2 * s_psi2,
            s_phi2 * c_theta2 * c_psi2 - c_phi2 * s_theta2 * s_psi2,
            c_phi2 * s_theta2 * c_psi2 + s_phi2 * c_theta2 * s_psi2,
            c_phi2 * c_theta2 * s_psi2 - s_phi2 * s_theta2 * c_psi2,
        ]
    )

    assert quaternion.shape == (
        4,
    ), f"quaternion.euler_to_quaternion: Quaternion shape incorrect {quaternion.shape}"

    return quaternion

def test_quaternion_to_euler():
    phi = np.pi/2
    theta = np.pi/3
    psi = -np.pi/6
    
    expected_euler = np.array([phi, theta, psi])

    quat = euler_to_quaternion(expected_euler)
    euler = quaternion_to_euler(quat)
    assert(np.allclose(expected_euler, euler)), "Received euler angles are wrong"
    print("Quaternion to euler angles test passed")
#test_quaternion_to_euler()
