from HomoTopiContinuation.DataStructures.datastructures import Homography
import numpy as np
import jax.numpy as jnp
from jax import jit
import jax
from scipy.optimize import minimize


class SKSDecomposer():
    """
    Class to decompose a homography into a set of homographies using the SKS decomposition.
    """

    @jit
    def createS(a, b, c, d):
        return jnp.array([
            [a, -b, c],
            [b, a, d],
            [0, 0, 1]
        ])

    @jit
    def createK(a, b, c, d):
        return jnp.array([
            [a, c, b],
            [0, 1, 0],
            [b, d, a]
        ])

    @jit
    def computeErr(H, theta):
        S_1 = SKSDecomposer.createS(theta[0], theta[1], theta[2], theta[3])
        K = SKSDecomposer.createK(theta[4], theta[5], theta[6], theta[7])
        S_2 = SKSDecomposer.createS(theta[8], theta[9], theta[10], theta[11])
        H_recov = jnp.linalg.inv(S_2) @ K @ S_1
        return jnp.linalg.norm(H - H_recov, 'fro')

    errGradient = jit(jax.grad(computeErr, argnums=1))

    def decompose(H: Homography) -> tuple:
        """
        Decompose a homography into a set of homographies using the SKS decomposition.

        Args:
            H (Homography): The homography to decompose

        Returns:
            list: The set of homographies
        """

    def initializeParameters() -> list:
        """
        Initialize the parameters for the optimization.

        Returns:
            list: The parameters for the optimization
        """
        s1_params = np.array([1, 0, 0, 0])
        k_params = np.array([1, 0, 0, 0])
        s2_params = np.array([1, 0, 0, 0])

        params = np.concatenate((s1_params, k_params, s2_params))
        return jnp.array(params)

    def optimizeBFGS(H: Homography) -> list:
        """
        Optimize the parameters using the BFGS optimization algorithm.

        Args:
            H (Homography): The homography to decompose

        Returns:
            list: The optimized parameters
        """

        H_jax = jnp.array(H(), dtype=jnp.float32)

        # Implement the BFGS optimization algorithm
        def objective(theta):
            return SKSDecomposer.computeErr(H_jax, theta)

        initial_theta = SKSDecomposer.initializeParameters()
        result = minimize(objective, initial_theta, method='BFGS',
                          jac=lambda x: SKSDecomposer.errGradient(H_jax, x), options={'disp': True}, tol=1e-60)

        S_1 = SKSDecomposer.createS(
            result.x[0], result.x[1], result.x[2], result.x[3])
        K = SKSDecomposer.createK(
            result.x[4], result.x[5], result.x[6], result.x[7])
        S_2 = SKSDecomposer.createS(
            result.x[8], result.x[9], result.x[10], result.x[11])
        H_recov = np.linalg.inv(S_2) @ K @ S_1

        error = SKSDecomposer.computeErr(H(), result.x)
        return S_1, K, S_2, H_recov, error

    def decompose(H: Homography) -> tuple:
        """
        Decompose a homography into a set of homographies using the SKS decomposition.

        Args:
            H (Homography): The homography to decompose

        Returns:
            list: The set of homographies
        """

        return SKSDecomposer.optimizeBFGS(H)


if __name__ == "__main__":
    H = Homography(np.array([
        [1, 0, 3],
        [0, 1, 6],
        [0, 0, 1]
    ]))

    decomposer = SKSDecomposer
    S_1, K, S_2, H_rec, error = decomposer.optimizeBFGS(H)

    print(f'Original Homography: {H()}')
    print(f'S_1: {S_1}')
    print(f'K: {K}')
    print(f'S_2: {S_2}')
    print(f'H_rec: {H_rec}')
    print(f'Error: {error}')

    # decomposer = SKSDecomposer
    # theta = decomposer.initializeParameters()
    # # Convert Homography to JAX array with float type
    # H_jax = jnp.array(H(), dtype=jnp.float32)

    # print(f"Initial parameters: {theta}")
    # print(f"Initial error: {decomposer.computeErr(H_jax, theta)}")
    # print(f"Initial gradient: {decomposer.errGradient(H_jax, theta)}")

    # iterations = 100000
    # alpha = 0.001
    # for i in range(iterations):
    #     theta = theta - alpha * decomposer.errGradient(H_jax, theta)
    #     print(
    #         f"Iteration {i}, error: {decomposer.computeErr(H_jax, theta)}")
    #     alpha = alpha * 0.999999
    # print(f"Final parameters: {theta}")

    # print(f'Original Homography: {H()}')
    # S_1 = decomposer.createS(theta[0], theta[1], theta[2], theta[3])
    # K = decomposer.createK(theta[4], theta[5], theta[6], theta[7])
    # S_2 = decomposer.createS(theta[8], theta[9], theta[10], theta[11])
    # H_recov = np.linalg.inv(S_2) @ K @ S_1
    # print(f'S_1: {S_1}')
    # print(f'K: {K}')
    # print(f'S_2: {S_2}')
    # print(f'Recovered Homography: {H_recov}')
