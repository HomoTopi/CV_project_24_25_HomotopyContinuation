import numpy as np
import jax.numpy as jnp
import jax
from HomoTopiContinuation.DataStructures.datastructures import Circle, ConicJax, ConicsJax, Homography, Conics

from .rectifier import Rectifier


class GDRectifier(Rectifier):
    def loss(H_inv: jnp.array, conic: ConicJax) -> float:
        warpedConic = conic.applyHomographyFromInv(H_inv)
        axes = warpedConic.computeSemiAxes()
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
        return jnp.sum(
            jnp.array([
                GDRectifier.loss(H_inv, conics.C1) * weights[0],
                GDRectifier.loss(H_inv, conics.C2) * weights[1],
                GDRectifier.loss(H_inv, conics.C3) * weights[2]
            ])
        )
    lossConics = jax.jit(lossConics, static_argnames=['conics'])

    gradient = jax.grad(lossConics, argnums=0)
    gradient = jax.jit(gradient, static_argnames=['conics'])

    def rectify(C_img: Conics, iterations: int = 20000, alpha: float = 0.00000001, beta: float = 0.9, weights=jnp.array([1.0, 1.0, 1.0]), gradientCap=5.0) -> Homography:
        warpedConics = ConicsJax(C_img)
        H_inv = jnp.eye(3)
        # # add noise to the initial homography
        # key = jax.random.PRNGKey(0)
        # noise = 0.00001 * jax.random.normal(key, H_inv.shape)
        # H_inv += noise
        v = jnp.zeros_like(H_inv)

        Hs = []
        losses = []
        grads = []
        vs = []

        for i in range(iterations):
            grad = GDRectifier.gradient(H_inv, warpedConics, weights=weights)
            gradNorm = jnp.linalg.norm(grad)
            gradNormMinned = jnp.min(jnp.array([gradNorm, gradientCap]))
            grad = grad / gradNorm * gradNormMinned
            v = beta * v + (1 - beta) * grad
            H_inv = H_inv - alpha * v
            current_loss = GDRectifier.lossConics(
                H_inv, warpedConics, weights=weights)
            print(f"Iteration {i}, Loss: {current_loss}")
            H = jnp.linalg.inv(H_inv)
            Hs.append(Homography(np.array(H)))
            losses.append(current_loss)
            grads.append(grad)
            vs.append(v)

        H = jnp.linalg.inv(H_inv)
        return Homography(np.array(H)), Hs, losses, jnp.array(grads), jnp.array(vs)
