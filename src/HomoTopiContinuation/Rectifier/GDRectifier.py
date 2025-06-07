import numpy as np
import jax.numpy as jnp
import jax
from HomoTopiContinuation.DataStructures.datastructures import Circle, ConicJax, ConicsJax, Homography, Conics
from tqdm import tqdm
from .rectifier import Rectifier


class GDRectifier(Rectifier):
    def loss(H_inv: jnp.array, conic: ConicJax) -> float:
        warpedConic = conic.applyHomographyFromInv(H_inv)
        axes = warpedConic.computeEigvals22()
        # maxAxis = jnp.max(axes)
        # minAxis = jnp.min(axes)
        tot = jnp.sum(axes)
        # return maxAxis - minAxis
        # return jnp.max(jnp.array([axes[0]/axes[1], axes[1]/axes[0]])) - 1.0
        return jnp.square((axes[0] - axes[1])/tot)
        # return jnp.sqrt(1 - jnp.min(axes)/jnp.max(axes))
        # return jnp.abs(jnp.log(axes[0]/axes[1]))  # log ratio of axes
        # return jnp.square(jnp.log(axes[0]/axes[1]))  # log ratio of axes
    loss = jax.jit(loss, static_argnames=['conic'])

    def lossConics(H_inv: jnp.array, conics: ConicsJax, weights: jnp.array) -> float:
        weights /= jnp.sum(weights)
        loss1 = GDRectifier.loss(H_inv, conics.C1)
        loss2 = GDRectifier.loss(H_inv, conics.C2)
        loss3 = GDRectifier.loss(H_inv, conics.C3)
        losses = jnp.array([loss1, loss2, loss3])
        weightedLosses = losses * weights
        softMax = jnp.exp(weightedLosses)
        softMax /= jnp.sum(softMax)
        softMax = softMax * weights
        softMax /= jnp.sum(softMax)  # normalize again

        softweightedAverage = jnp.sum(
            softMax * losses
        )
        return softweightedAverage

        # return jnp.sum(
        #     jnp.array([
        #         GDRectifier.loss(H_inv, conics.C1) * weights[0],
        #         GDRectifier.loss(H_inv, conics.C2) * weights[1],
        #         GDRectifier.loss(H_inv, conics.C3) * weights[2]
        #     ])
        # )
    lossConics = jax.jit(lossConics, static_argnames=['conics'])

    gradient = jax.grad(lossConics, argnums=0)
    gradient = jax.jit(gradient, static_argnames=['conics'])

    def rectify(C_img: Conics, iterations: int = 20000, alpha: float = 1e-3, beta1: float = 0.99, beta2: float = 0.999, epsilon: float = 1e-12, weights=jnp.array([1.0, 1.0, 1.0]), gradientCap=jnp.inf, early_stopping: bool = True, patience: int = 300, min_delta: float = 1e-6) -> Homography:
        """
        Performs rectification of conics using gradient descent with Adam optimization and early stopping.
        Args:
            C_img (Conics): The input conics to be rectified.
            iterations (int, optional): Number of optimization iterations. Default is 20000.
            alpha (float, optional): Learning rate for the Adam optimizer. Default is 1e-8.
            beta1 (float, optional): Exponential decay rate for the first moment estimates in Adam. Default is 0.9.
            beta2 (float, optional): Exponential decay rate for the second moment estimates in Adam. Default is 0.999.
            epsilon (float, optional): Small constant for numerical stability in Adam. Default is 1e-8.
            weights (jnp.ndarray, optional): Weights for the loss function. Default is jnp.array([1.0, 1.0, 1.0]).
            gradientCap (float, optional): Maximum allowed gradient norm for gradient clipping. Default is np.inf.
            early_stopping (bool, optional): Whether to use early stopping. Default is True.
            patience (int, optional): Number of iterations to wait for improvement before stopping. Default is 100.
            min_delta (float, optional): Minimum change in loss to qualify as improvement. Default is 1e-8.
        Returns:
            Homography: The final estimated homography after rectification.
            list[Homography]: List of homographies at each iteration.
            list[float]: List of loss values at each iteration.
            jnp.ndarray: Array of gradients at each iteration.
            jnp.ndarray: Array of first moment vectors (m) at each iteration.
            jnp.ndarray: Array of second moment vectors (v) at each iteration.
        Notes:
            - Uses Adam optimizer for updating the inverse homography matrix.
            - Tracks the optimization process by storing homographies, losses, gradients, and optimizer states.
            - Prints the loss at each iteration for monitoring convergence.
            - Stops early if loss does not improve for 'patience' iterations.
        """
        warpedConics = ConicsJax(C_img)
        H_inv = jnp.eye(3)
        m = jnp.zeros_like(H_inv)
        v = jnp.zeros_like(H_inv)

        Hs = []
        losses = []
        grads = []
        ms = []
        vs = []

        best_loss = float('inf')
        patience_counter = 0

        with tqdm(range(1, iterations+1)) as pbar:
            for i in pbar:
                grad = GDRectifier.gradient(
                    H_inv, warpedConics, weights=weights)
                gradNorm = jnp.linalg.norm(grad)
                gradNormMinned = jnp.minimum(gradNorm, gradientCap)
                grad = grad / (gradNorm + epsilon) * gradNormMinned

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)

                m_hat = m / (1 - beta1 ** i)
                v_hat = v / (1 - beta2 ** i)

                current_loss = GDRectifier.lossConics(
                    H_inv, warpedConics, weights=weights)

                pbar.set_postfix({"Loss": float(current_loss)})

                H = jnp.linalg.inv(H_inv)
                Hs.append(Homography(np.array(H)))
                losses.append([
                    GDRectifier.loss(H_inv, warpedConics.C1),
                    GDRectifier.loss(H_inv, warpedConics.C2),
                    GDRectifier.loss(H_inv, warpedConics.C3),
                    current_loss])
                grads.append(grad)
                ms.append(m)
                vs.append(v)

                H_inv = H_inv - alpha * m_hat / (jnp.sqrt(v_hat) + epsilon)

                # Early stopping logic
                if early_stopping:
                    if float(best_loss) - float(current_loss) > min_delta:
                        best_loss = float(current_loss)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= patience:
                        print(
                            f"Early stopping at iteration {i} with loss {current_loss}")
                        break

        H = jnp.linalg.inv(H_inv)
        return Homography(np.array(H)), Hs, np.array(losses), jnp.array(grads), jnp.array(ms), jnp.array(vs)

    def computeImagesOfCircularPoints(self, C_img: Conics) -> np.ndarray:
        """
        Compute the image of the circular points.
        """
        pass
