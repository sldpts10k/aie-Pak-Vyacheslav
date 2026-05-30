import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats


def _is_jax_value(value):
    if value is None:
        return False
    if isinstance(value, (jax.Array, jax.core.Tracer)):
        return True
    return type(value).__module__.startswith("jax")


def _array_module(*values):
    return jnp if any(_is_jax_value(value) for value in values) else np


def calc_prior(dist_prior, val):
    if dist_prior is None:
        return 0.0

    prior, p1, p2 = dist_prior

    if prior == "normal":
        return stats.norm.logpdf(val, loc=p1, scale=p2)

    if prior == "gamma":
        real_val = jnp.exp(val)
        return stats.gamma.logpdf(real_val, a=p1, scale=p2) + val

    return 0.0


class SumKernels:
    def __init__(self, kernels_list, scale_priors=None):
        if len(kernels_list) == 0:
            raise ValueError("SumKernels requires at least one kernel.")

        self.kernels = kernels_list
        self.specs = []
        self.param_names = []
        self.alpha_indices = []
        self.alpha_idx = None

        self.scale_priors = (
            scale_priors if scale_priors is not None else [None] * len(self.kernels)
        )
        if len(self.scale_priors) != len(self.kernels):
            raise ValueError("scale_priors must have the same length as kernels_list.")

        idx = 0
        for i, kernel in enumerate(self.kernels):
            names = list(kernel.free_params())
            n_params = len(names)
            slice_ = slice(idx, idx + n_params)
            scale_idx = idx + n_params if i != 0 else None

            for offset, name in enumerate(names):
                self.param_names.append(name)
                if name == "alpha":
                    alpha_idx = idx + offset
                    self.alpha_indices.append(alpha_idx)
                    if self.alpha_idx is None:
                        self.alpha_idx = alpha_idx

            if scale_idx is not None:
                self.param_names.append(f"scale_{i}")

            self.specs.append((kernel, slice_, scale_idx))
            idx += n_params + (1 if i != 0 else 0)

    def return_bounds(self, k):
        bounds = []
        bounds.extend(self.kernels[0].return_bounds(k))

        for kernel in self.kernels[1:]:
            bounds.extend(kernel.return_bounds(k))
            bounds.append((1e-3, 1000.0))

        return bounds

    def calc(self, D, params, ignore_nugget=False):
        xp = _array_module(D, params)
        matrix = xp.zeros_like(D, dtype=float)

        for kernel, slic, scale_idx in self.specs:
            if ignore_nugget and getattr(kernel, "is_noise", False):
                continue

            param = params[slic] if slic.start != slic.stop else None
            scale_i = 1.0 if scale_idx is None else params[scale_idx]
            matrix = matrix + scale_i * kernel.calc(D, param)

        return matrix

    def get_lprior(self, lparams):
        lprior = 0.0

        for i, (kernel, slic, scale_idx) in enumerate(self.specs):
            if slic.start != slic.stop:
                lprior += kernel.get_lprior(lparams[slic])
            if scale_idx is not None:
                lprior += calc_prior(self.scale_priors[i], lparams[scale_idx])

        return lprior


class ProductKernels:
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2
        self.is_noise = getattr(k1, "is_noise", False) or getattr(k2, "is_noise", False)
        self.is_periodic = getattr(k1, "is_periodic", False) or getattr(
            k2, "is_periodic", False
        )

    def free_params(self):
        return self.k1.free_params() + self.k2.free_params()

    def return_bounds(self, k):
        return self.k1.return_bounds(k) + self.k2.return_bounds(k)

    def calc(self, D, params):
        params = [] if params is None else params
        params_k1 = len(self.k1.free_params())
        params_1 = params[:params_k1]
        params_2 = params[params_k1:]

        return self.k1.calc(D, params_1) * self.k2.calc(D, params_2)

    def get_lprior(self, lparams):
        params_k1 = len(self.k1.free_params())
        return self.k1.get_lprior(lparams[:params_k1]) + self.k2.get_lprior(
            lparams[params_k1:]
        )


class PowerLaw:
    is_periodic = False

    def __init__(self, alpha=None, is_noise=False):
        if is_noise:
            raise ValueError(
                "PowerLaw is an IRF generalized covariance, not an observation-noise "
                "kernel. Use Nugget(is_noise=True) or another PSD covariance for noise."
            )
        self.alpha = alpha
        self.is_noise = False

    def free_params(self):
        return ["alpha"] if self.alpha is None else []

    def return_bounds(self, k):
        return [(1e-6, 2 * (k + 1) - 1e-6)] if self.alpha is None else []

    def calc(self, D, params):
        alpha = params[0] if self.alpha is None else self.alpha
        xp = _array_module(D, alpha)
        sign = xp.where(xp.mod(xp.floor(alpha / 2.0), 2.0) == 0.0, -1.0, 1.0)
        if xp is jnp:
            sign = jax.lax.stop_gradient(sign)
        return sign * xp.power(D, alpha)

    def get_lprior(self, lparams):
        return 0.0


class Periodic:
    is_periodic = True

    def __init__(self, T=None, l=None,
                 prior_T=None, prior_l=None,
                 len_scal_bounds=(0.1, 100.0),
                 is_noise=False):
        self.T = T
        self.l = l
        self.len_scal_bounds=len_scal_bounds
        self.prior_T = prior_T
        self.prior_l = prior_l
        self.is_noise = is_noise

    def free_params(self):
        free = []
        if self.T is None:
            free.append("T")
        if self.l is None:
            free.append("l")
        return free

    def return_bounds(self, k=None):
        bounds = []
        if self.T is None:
            bounds.append((1e-3, 100.0))
        if self.l is None:
            bounds.append(self.len_scal_bounds)
        return bounds

    def calc(self, D, params, ignore_nugget=True):
        i = 0

        if self.T is None:
            T = params[i]
            i += 1
        else:
            T = self.T

        if self.l is None:
            l = params[i]
        else:
            l = self.l

        xp = _array_module(D, T, l)
        return xp.exp(-2.0 * xp.sin(xp.pi * D / T) ** 2 / l**2)

    def get_lprior(self, lparams):
        lprior = 0.0
        i = 0
        if self.T is None:
            lprior += calc_prior(self.prior_T, lparams[i])
            i += 1
        if self.l is None:
            lprior += calc_prior(self.prior_l, lparams[i])
        return lprior


Perodic = Periodic


class Nugget:
    is_periodic = False

    def __init__(self, is_noise=True):
        self.is_noise = is_noise

    def return_bounds(self, k):
        return []

    def free_params(self):
        return []

    def calc(self, D, params):
        xp = _array_module(D)
        return (D == 0).astype(float) if xp is jnp else np.asarray(D == 0, dtype=float)

    def get_lprior(self, lparams):
        return 0.0


class RBF:
    is_periodic = False

    def __init__(self, len_scale=None, len_scale_prior=None, is_noise=False):
        self.len_scale = len_scale
        self.len_scale_prior = len_scale_prior
        self.is_noise = is_noise

    def return_bounds(self, k):
        return [(1e-5, 1000.0)] if self.len_scale is None else []

    def free_params(self):
        return ["len_scale"] if self.len_scale is None else []

    def calc(self, D, params):
        len_scale = params[0] if self.len_scale is None else self.len_scale
        xp = _array_module(D, len_scale)
        return xp.exp(-0.5 * (D / len_scale) ** 2)

    def get_lprior(self, lparams):
        return 0.0 if self.len_scale is not None else calc_prior(
            self.len_scale_prior, lparams[0]
        )


class RationalQuadratic:
    is_periodic = False

    def __init__(self, power_scale=None, len_scale=None,
                 ps_bounds = (1e-3, 100.0),
                 len_scal_bounds = (1e-3, 100.0),
                 is_noise=False):
        self.power_scale = power_scale
        self.len_scale = len_scale
        self.ps_bounds = ps_bounds
        self.len_scal_bounds=len_scal_bounds
        self.is_noise = is_noise

    def return_bounds(self, k):
        bounds = []
        if self.power_scale is None:
            bounds.append(self.ps_bounds)
        if self.len_scale is None:
            bounds.append(self.len_scal_bounds)
        return bounds

    def free_params(self):
        free = []
        if self.power_scale is None:
            free.append("power_scale")
        if self.len_scale is None:
            free.append("len_scale")
        return free

    def calc(self, D, params, ignore_nugget=False):
        i = 0

        if self.power_scale is None:
            power_scale = params[i]
            i += 1
        else:
            power_scale = self.power_scale

        if self.len_scale is None:
            len_scale = params[i]
        else:
            len_scale = self.len_scale

        xp = _array_module(D, power_scale, len_scale)
        base = 1.0 + D**2 / (2.0 * power_scale * len_scale**2)
        return base ** (-power_scale)

    def get_lprior(self, lparams):
        return 0.0
