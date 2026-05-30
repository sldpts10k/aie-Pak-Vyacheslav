from itertools import product
import warnings

import numpy as np
from scipy.linalg import (
    LinAlgError,
    LinAlgWarning,
    cholesky,
    lu_factor,
    lu_solve,
    null_space,
    solve_triangular,
)
from scipy.optimize import differential_evolution, minimize
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import cdist, pdist, squareform

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular as jax_solve_triangular
try:
    from jaxopt import ScipyBoundedMinimize
except ImportError:  # jaxopt is required only for optim_type="grad"
    ScipyBoundedMinimize = None


jax.config.update("jax_enable_x64", True)


class IRFKriging:
    def __init__(
        self,
        kernel,
        k,
        period=None,
        exp_term=None,
        exp_periodic_terms=None,
        optim_type="DE",
        n_starts=50,
        maxiter=50,
        warm_start=False,
        popsize=9,
        maxiter_de=10,
        jitter=1e-10,
        extract_signal=True,
        variance_tol=1e-8,
    ):
        self.x = None
        self.y = None
        self.k = int(k)
        self.period = period
        self.exp_term=[] if exp_term is None else list(map(float, exp_term))
        self.exp_periodic_terms = [] if exp_periodic_terms is None else [tuple(map(float, term)) for term in exp_periodic_terms]

        self.optim_type = optim_type
        self.kernel = kernel
        self.n_starts = int(n_starts)
        self.maxiter = int(maxiter)
        self.warm_start = warm_start
        self.popsize = int(popsize)
        self.maxiter_de = int(maxiter_de)
        self.jitter = float(jitter)
        self.variance_tol = float(variance_tol)

        self.extract_signal = extract_signal
        self.matrix = None
        self.matrix_lu = None
        self.params = None
        self.scale0 = None
        self.ll = None
        self.D = None
        self.F = None
        self.y_diff = None
        self.m = None
        self._last_cholesky_jitter = None

        if self.k < 0:
            raise ValueError("k must be non-negative.")
        if self.period is not None and self.period <= 0:
            raise ValueError("period must be positive when provided.")
        if self.jitter <= 0:
            raise ValueError("jitter must be positive.")
        if self.variance_tol < 0:
            raise ValueError("variance_tol must be non-negative.")

        self._validate_kernel()

    def _validate_kernel(self):
        kernels = getattr(self.kernel, "kernels", [self.kernel])
        for kernel in kernels:
            if type(kernel).__name__ == "PowerLaw" and getattr(kernel, "is_noise", False):
                raise ValueError(
                    "PowerLaw(is_noise=True) is not valid: PowerLaw is an IRF "
                    "generalized covariance, not an observation-noise covariance. "
                    "Use Nugget(is_noise=True) or another PSD noise kernel."
                )

    def _alpha_indices(self):
        indices = getattr(self.kernel, "alpha_indices", None)
        if indices is None:
            idx = getattr(self.kernel, "alpha_idx", None)
            indices = [] if idx is None else [idx]
        return tuple(indices)

    def _alpha_interval(self, bounds):
        l, h = bounds
        intervals = []
        breakpoints = list(range(0, int(np.ceil(h)) + 2, 2))

        for i in range(len(breakpoints) - 1):
            a = max(breakpoints[i] + 1e-3, l)
            b = min(breakpoints[i + 1] - 1e-3, h)
            if a < b:
                intervals.append((a, b))

        return intervals if intervals else [(l, h)]

    def _iter_alpha_bound_sets(self, bounds):
        alpha_indices = self._alpha_indices()
        if not alpha_indices:
            yield list(bounds)
            return

        interval_groups = [self._alpha_interval(bounds[idx]) for idx in alpha_indices]
        for selected_intervals in product(*interval_groups):
            cur_bounds = list(bounds)
            for idx, interval in zip(alpha_indices, selected_intervals):
                cur_bounds[idx] = interval
            yield cur_bounds

    def _make_log_bounds(self, bounds):
        alpha_indices = set(self._alpha_indices())
        log_bounds = []

        for i, (lo, hi) in enumerate(bounds):
            if i in alpha_indices:
                log_bounds.append((lo, hi))
            else:
                log_bounds.append((np.log(max(lo, 1e-12)), np.log(max(hi, 1e-12))))

        return log_bounds

    def _to_log_space(self, start):
        start = np.asarray(start, dtype=float)
        opt = np.log(np.maximum(start, 1e-300))

        for idx in self._alpha_indices():
            opt[idx] = start[idx]

        return opt

    def _to_normal_space(self, params):
        params = np.asarray(params, dtype=float)
        rez = np.exp(params)

        for idx in self._alpha_indices():
            rez[idx] = params[idx]

        return rez

    def _to_normal_space_jax(self, params):
        rez = jnp.exp(params)

        for idx in self._alpha_indices():
            rez = rez.at[idx].set(params[idx])

        return rez

    def _validate_xy(self, x, y):
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float).reshape(-1)

        if x_arr.size != y_arr.size:
            raise ValueError("x and y must have the same length.")
        if x_arr.size == 0:
            raise ValueError("x and y must be non-empty.")
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("x and y must contain only finite values.")
        if np.unique(x_arr).size != x_arr.size:
            raise ValueError("x must not contain duplicate coordinates.")

        order = np.argsort(x_arr)
        x_arr = x_arr[order]
        y_arr = y_arr[order]

        window_size = self._basis_count() + 1
        if x_arr.size < window_size:
            raise ValueError(
                f"Need at least {window_size} points for k={self.k} "
                f"and period={self.period!r}."
            )

        return x_arr, y_arr

    #def _basis_count(self):
    #    return self.k + 1 + (2 if self.period is not None else 0)
    '''
    def _basis_count(self):
        return (
            self.k + 1
            + (2 if self.period is not None else 0)
            + len(self.exp_term)
        )
    '''
    def _basis_count(self):
        return (
            self.k + 1
            + (2 if self.period is not None else 0)
            + len(self.exp_term)
            + 2 * len(self.exp_periodic_terms)
        )

    @staticmethod
    def _scaled_window(window):
        center = np.mean(window)
        scale = np.max(np.abs(window - center))
        if scale <= 0:
            raise ValueError("ALC window has zero coordinate spread.")
        return (window - center) / scale

    @staticmethod
    def _one_dim_null_vector(basis, window_start):
        alk = null_space(basis.T)
        if alk.shape[1] != 1:
            raise ValueError(
                "ALC basis is rank-deficient at window "
                f"{window_start}: expected one null vector, got {alk.shape[1]}."
            )
        return alk[:, 0]
    '''
    def _calc_alk_coefs(self, x, k):
        n = len(x)
        #window_size = k + 2
        window_size = self._basis_count() + 1
        m = n - window_size + 1
        rows, cols, data = [], [], []

        for i in range(m):
            window = x[i : i + window_size]
            scaled = self._scaled_window(window)
            #basis = np.vander(scaled, N=k + 1, increasing=True)
            polynomial = np.vander(scaled, N=k + 1, increasing=True)
            blocks = [polynomial]

            center = np.mean(window)
            scale = np.max(np.abs(window - center))

            for a in self.exp_term:
                blocks.append(np.exp((a * scale) * scaled).reshape(-1, 1))

            for a, omega in self.exp_periodic_terms:
                env = np.exp((a * scale) * scaled)
                phase = (omega * scale) * scaled

                blocks.append((env * np.sin(phase)).reshape(-1, 1))
                blocks.append((env * np.cos(phase)).reshape(-1, 1))

            basis = np.hstack(blocks)
            alk = self._one_dim_null_vector(basis, i)

            rows.extend([i] * window_size)
            cols.extend(range(i, i + window_size))
            data.extend(alk)

        return csr_matrix((data, (rows, cols)), shape=(m, n))

    def _calc_alk_coefs_with_period(self, x, k, period):
        n = len(x)
        #window_size = k + 4
        window_size = self._basis_count() + 1
        m = n - window_size + 1
        rows, cols, data = [], [], []

        for i in range(m):
            window = x[i : i + window_size]
            scaled = self._scaled_window(window)
            polynomial = np.vander(scaled, N=k + 1, increasing=True)
            periodic_sin = np.sin((2.0 * np.pi * window) / period).reshape(-1, 1)
            periodic_cos = np.cos((2.0 * np.pi * window) / period).reshape(-1, 1)
            #basis = np.hstack([polynomial, periodic_sin, periodic_cos])
            blocks = [polynomial, periodic_sin, periodic_cos]

            for a in self.exp_term:
                blocks.append(np.exp((a * scale) * scaled).reshape(-1, 1))

            basis = np.hstack(blocks)
            alk = self._one_dim_null_vector(basis, i)

            rows.extend([i] * window_size)
            cols.extend(range(i, i + window_size))
            data.extend(alk)

        return csr_matrix((data, (rows, cols)), shape=(m, n))
    '''
    def _calc_alk_coefs(self, x, k):
        n = len(x)
        window_size = self._basis_count() + 1
        m = n - window_size + 1
        rows, cols, data = [], [], []

        for i in range(m):
            window = x[i : i + window_size]
            scaled = self._scaled_window(window)

            polynomial = np.vander(scaled, N=k + 1, increasing=True)
            blocks = [polynomial]

            center = np.mean(window)
            scale = np.max(np.abs(window - center))

            if self.period is not None:
                periodic_sin = np.sin((2.0 * np.pi * window) / self.period).reshape(-1, 1)
                periodic_cos = np.cos((2.0 * np.pi * window) / self.period).reshape(-1, 1)
                blocks.extend([periodic_sin, periodic_cos])

            for a in self.exp_term:
                blocks.append(np.exp((a * scale) * scaled).reshape(-1, 1))

            for a, omega in self.exp_periodic_terms:
                env = np.exp((a * scale) * scaled)
                phase = (omega * scale) * scaled

                blocks.append((env * np.sin(phase)).reshape(-1, 1))
                blocks.append((env * np.cos(phase)).reshape(-1, 1))

            basis = np.hstack(blocks)
            alk = self._one_dim_null_vector(basis, i)

            rows.extend([i] * window_size)
            cols.extend(range(i, i + window_size))
            data.extend(alk)

        return csr_matrix((data, (rows, cols)), shape=(m, n))

    def _good_start(self, bounds, y_diff, D):
        def get_fft(values, num_periods=2):
            if num_periods <= 0 or len(values) < 4:
                return []
            yf = np.fft.rfft(values)
            power = np.abs(yf) ** 2
            power[:2] = 0
            freqs = np.fft.rfftfreq(len(values))
            peak_idx = np.argsort(power)[::-1][:num_periods]
            return [1.0 / freqs[i] for i in peak_idx if freqs[i] > 0]

        kernels = getattr(self.kernel, "kernels", [self.kernel])
        y_diff = np.asarray(y_diff, dtype=float)
        D = np.asarray(D, dtype=float)
        periodic_cnt = sum(1 for k in kernels if getattr(k, "is_periodic", False))

        start = []
        T_spectr = get_fft(y_diff, periodic_cnt)
        T_ptr = 0

        dists = D[D > 0]
        median_dist = float(np.median(dists)) if dists.size else 1.0
        varr = float(np.var(y_diff)) if y_diff.size else 1.0

        for i, kernel_i in enumerate(kernels):
            free = kernel_i.free_params()
            cur_T = None

            for name in free:
                lo, hi = bounds[len(start)]

                if getattr(kernel_i, "is_periodic", False) and name == "T":
                    if T_spectr:
                        T = T_spectr[T_ptr % len(T_spectr)]
                        T_ptr += 1
                    else:
                        T = max(4.0 * median_dist, lo)
                    cur_T = T
                    start.append(np.clip(T * np.random.uniform(0.8, 1.2), lo, hi))
                elif getattr(kernel_i, "is_periodic", False) and name == "l":
                    base = cur_T / 3.0 if cur_T is not None else median_dist
                    start.append(np.clip(base * np.random.uniform(0.8, 1.2), lo, hi))
                elif name == "alpha":
                    start.append((lo + hi) / 2.0)
                elif name in ("l", "len_scale"):
                    start.append(np.clip(median_dist * np.random.uniform(0.8, 1.2), lo, hi))
                elif name == "power_scale":
                    start.append(np.clip(1.0, lo, hi))
                else:
                    start.append(np.random.uniform(lo, hi))

            if i != 0:
                lo, hi = bounds[len(start)]
                start.append(np.clip(varr / max(len(kernels), 1), lo, hi))

        return np.asarray(start, dtype=float)

    def _build_R_numpy(self, params, D, F):
        G = np.asarray(self.kernel.calc(D, params), dtype=float)
        if not np.all(np.isfinite(G)):
            return None

        if issparse(F):
            R = (F @ G) @ F.T
            R = R.toarray() if issparse(R) else np.asarray(R)
        else:
            R = np.asarray(F @ G @ F.T, dtype=float)

        R = 0.5 * (R + R.T)
        return R

    def _cholesky_with_adaptive_jitter(self, R):
        if R is None or not np.all(np.isfinite(R)):
            raise LinAlgError("R contains non-finite values.")

        diag_scale = max(1.0, float(np.mean(np.abs(np.diag(R)))))
        eye = np.eye(R.shape[0])
        jitter = max(self.jitter, np.finfo(float).eps) * diag_scale

        for _ in range(8):
            try:
                L = cholesky(R + jitter * eye, lower=True, check_finite=False)
                self._last_cholesky_jitter = jitter
                return L
            except LinAlgError:
                jitter *= 10.0

        raise LinAlgError("R is not positive definite after adaptive jitter.")

    def _compute_ll(self, params, D, F, y_diff, m):
        bad = 1e100
        params = np.asarray(params, dtype=float)
        y_diff = np.asarray(y_diff, dtype=float).reshape(-1)
        D = np.asarray(D, dtype=float)

        try:
            R = self._build_R_numpy(params, D, F)
            L = self._cholesky_with_adaptive_jitter(R)
        except (LinAlgError, ValueError, FloatingPointError):
            return bad, np.inf

        ln_det_R = 2.0 * np.sum(np.log(np.diag(L)))
        v = solve_triangular(L, y_diff, lower=True, check_finite=False)
        Q = float(v @ v)

        if not np.isfinite(Q) or Q <= 0.0:
            return bad, np.inf

        scale0 = Q / m
        ll = m * np.log(scale0) + ln_det_R

        if not np.isfinite(ll):
            return bad, np.inf

        return float(ll), float(scale0)

    def _objective(self, params, D, F, y_diff, m):
        ll, _ = self._compute_ll(params, D, F, y_diff, m)
        return ll

    def _fit_de(self, bounds, D, F, y_diff, m):
        if len(bounds) == 0:
            params = np.asarray([], dtype=float)
            return params, self._objective(params, D, F, y_diff, m)

        best_ll = np.inf
        best_params = None

        for cur_bounds in self._iter_alpha_bound_sets(bounds):
            log_bounds = self._make_log_bounds(cur_bounds)
            start = self._good_start(cur_bounds, y_diff, D)
            start_log = self._to_log_space(start)

            for i, (lo, hi) in enumerate(log_bounds):
                start_log[i] = np.clip(start_log[i], lo + 1e-8, hi - 1e-8)

            def objective(log_params):
                real_params = self._to_normal_space(log_params)
                return self._objective(real_params, D, F, y_diff, m)

            rez_de = differential_evolution(
                objective,
                bounds=log_bounds,
                x0=start_log,
                popsize=self.popsize,
                maxiter=self.maxiter,
                polish=False,
                workers=1,
            )

            candidates = [(float(rez_de.fun), np.asarray(rez_de.x, dtype=float))]

            rez_local = minimize(
                objective,
                np.asarray(rez_de.x, dtype=float),
                method="L-BFGS-B",
                bounds=log_bounds,
                options={"maxiter": self.maxiter},
            )
            if np.isfinite(rez_local.fun):
                candidates.append((float(rez_local.fun), np.asarray(rez_local.x, dtype=float)))

            interval_ll, interval_log_params = min(candidates, key=lambda item: item[0])
            if interval_ll < best_ll:
                best_ll = interval_ll
                best_params = self._to_normal_space(interval_log_params)

        if best_params is None:
            raise RuntimeError("DE optimization failed to find finite kernel parameters.")

        return best_params, best_ll

    def _build_loss(self, D, F, y_diff, m):
        def loss_fn(log_params):
            all_params = self._to_normal_space_jax(log_params)

            G = self.kernel.calc(D, all_params)
            R = F @ G @ F.T
            R = 0.5 * (R + R.T)
            diag_scale = jnp.maximum(1.0, jnp.mean(jnp.abs(jnp.diag(R))))
            R = R + self.jitter * diag_scale * jnp.eye(R.shape[0])

            L = jnp.linalg.cholesky(R)
            ln_det_R = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

            v = jax_solve_triangular(L, y_diff, lower=True)
            Q = jnp.dot(v, v)

            log_prior = jnp.nan_to_num(
                self.kernel.get_lprior(log_params),
                nan=-1e10,
                posinf=-1e10,
                neginf=-1e10,
            )
            ll = m * jnp.log(Q / m) + ln_det_R

            return jnp.nan_to_num(
                ll - log_prior,
                nan=1e100,
                posinf=1e100,
                neginf=1e100,
            )

        return jax.jit(loss_fn)

    def _fit_grad(self, num_starts, bounds, D, F, y_diff, m):
        if len(bounds) == 0:
            params = np.asarray([], dtype=float)
            return params, self._objective(params, np.asarray(D), np.asarray(F), y_diff, m)

        best_loss = np.inf
        best_params = None
        num_starts = max(1, int(num_starts))

        for cur_bounds in self._iter_alpha_bound_sets(bounds):
            log_bounds = self._make_log_bounds(cur_bounds)
            bounds_jax = (
                jnp.asarray([b[0] for b in log_bounds]),
                jnp.asarray([b[1] for b in log_bounds]),
            )
            bound_widths = np.asarray([hi - lo for lo, hi in log_bounds], dtype=float)
            std_pop = 0.05 * bound_widths

            loss_func = self._build_loss(D, F, y_diff, m)
            if ScipyBoundedMinimize is None:
                raise ImportError("optim_type=\"grad\" requires jaxopt. Install it with: pip install jaxopt")
            solver = ScipyBoundedMinimize(fun=loss_func, method="L-BFGS-B")

            start = self._good_start(cur_bounds, np.asarray(y_diff), np.asarray(D))
            start_log = self._to_log_space(start)

            for i, (lo, hi) in enumerate(log_bounds):
                start_log[i] = np.clip(start_log[i], lo + 1e-8, hi - 1e-8)

            if self.warm_start:
                def objective(log_params):
                    params_norm = self._to_normal_space(log_params)
                    return self._objective(
                        params_norm,
                        np.asarray(D),
                        np.asarray(F),
                        np.asarray(y_diff),
                        m,
                    )

                rez_de = differential_evolution(
                    objective,
                    bounds=log_bounds,
                    x0=start_log,
                    popsize=self.popsize,
                    maxiter=self.maxiter_de,
                    polish=False,
                )
                start_log = np.asarray(rez_de.x, dtype=float)

                population = getattr(rez_de, "population", None)
                if population is not None:
                    std_pop = np.maximum(np.std(population, axis=0), 0.01 * bound_widths)

            for strt in range(num_starts):
                if strt == 0:
                    start_i = start_log.copy()
                elif strt < max(2, num_starts // 2):
                    start_i = start_log + np.random.normal(0.0, std_pop, size=start_log.shape)
                else:
                    start_i = np.asarray(
                        [np.random.uniform(lo, hi) for lo, hi in log_bounds],
                        dtype=float,
                    )

                for i, (lo, hi) in enumerate(log_bounds):
                    start_i[i] = np.clip(start_i[i], lo, hi)

                rez = solver.run(jnp.asarray(start_i), bounds=bounds_jax)
                loss_opt = float(rez.state.fun_val)
                params_opt = np.asarray(self._to_normal_space_jax(rez.params), dtype=float)

                if np.isfinite(loss_opt) and loss_opt < best_loss:
                    best_loss = loss_opt
                    best_params = params_opt

        if best_params is None:
            raise RuntimeError("Gradient optimization failed to find finite kernel parameters.")

        return best_params, best_loss

    def fit(self, x, y):
        x_arr, y_arr = self._validate_xy(x, y)
        n = len(y_arr)
        
        '''
        if self.period is not None:
            F_sparse = self._calc_alk_coefs_with_period(x_arr, self.k, period=self.period)
        else:
            F_sparse = self._calc_alk_coefs(x_arr, self.k)
        '''
        F_sparse = self._calc_alk_coefs(x_arr, self.k)

        m = F_sparse.shape[0]
        y_diff_np = np.asarray(F_sparse @ y_arr, dtype=float)

        self.x = x_arr.reshape(-1, 1)
        self.y = y_arr

        distances = pdist(self.x, metric="cityblock")
        D_np = squareform(distances)

        self.D = D_np
        self.F = F_sparse
        self.y_diff = y_diff_np
        self.m = m

        bounds = self.kernel.return_bounds(self.k)
        param_names = getattr(self.kernel, "param_names", None)
        if param_names is not None and len(param_names) != len(bounds):
            raise ValueError("Kernel parameter metadata and bounds have different lengths.")

        if self.optim_type == "DE":
            rez, _ = self._fit_de(bounds, D_np, F_sparse, y_diff_np, m)
        elif self.optim_type == "grad":
            F_dense = jnp.asarray(F_sparse.toarray())
            D_jax = jnp.asarray(D_np)
            y_diff_jax = jnp.asarray(y_diff_np)
            rez, _ = self._fit_grad(self.n_starts, bounds, D_jax, F_dense, y_diff_jax, m)
        else:
            raise ValueError(f"Unknown optim_type: {self.optim_type}")

        self.params = np.asarray(rez, dtype=float)
        ll, self.scale0 = self._compute_ll(self.params, D_np, F_sparse, y_diff_np, m)
        if not np.isfinite(ll) or not np.isfinite(self.scale0):
            raise RuntimeError("Fitted kernel produced a non-positive-definite REML matrix.")
        self.ll = ll

        basic_func_cnt = self._basis_count()
        K = np.zeros((n + basic_func_cnt, n + basic_func_cnt), dtype=float)
        K[:n, :n] = (
            np.asarray(self.kernel.calc(D_np, self.params, ignore_nugget=False), dtype=float)
            * self.scale0
        )
        K[:n, :n] = 0.5 * (K[:n, :n] + K[:n, :n].T)

        idx_trend = n
        x_flat = self.x.ravel()

        # 1) polynomial: 1, x, ..., x^k
        for i in range(self.k + 1):
            col = x_flat ** i

            K[:n, idx_trend] = col
            K[idx_trend, :n] = col
            idx_trend += 1

        # 2) ordinary periodic: sin(2*pi*x/T), cos(2*pi*x/T)
        if self.period is not None:
            col = np.sin(2.0 * np.pi * x_flat / self.period)
            K[:n, idx_trend] = col
            K[idx_trend, :n] = col
            idx_trend += 1

            col = np.cos(2.0 * np.pi * x_flat / self.period)
            K[:n, idx_trend] = col
            K[idx_trend, :n] = col
            idx_trend += 1

        # 3) plain exponential: exp(a*x)
        for a in self.exp_term:
            col = np.exp(a * x_flat)

            K[:n, idx_trend] = col
            K[idx_trend, :n] = col
            idx_trend += 1

        # 4) exp-periodic: exp(a*x)*sin(omega*x), exp(a*x)*cos(omega*x)
        for a, omega in self.exp_periodic_terms:
            env = np.exp(a * x_flat)

            col = env * np.sin(omega * x_flat)
            K[:n, idx_trend] = col
            K[idx_trend, :n] = col
            idx_trend += 1

            col = env * np.cos(omega * x_flat)
            K[:n, idx_trend] = col
            K[idx_trend, :n] = col
            idx_trend += 1

        if idx_trend != n + basic_func_cnt:
            raise RuntimeError("Trend basis size mismatch in fit().")

        self.matrix = K

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", LinAlgWarning)
                self.matrix_lu = lu_factor(K, check_finite=False)
        except (LinAlgError, LinAlgWarning) as exc:
            raise RuntimeError("Kriging system is singular and cannot be factorized.") from exc

        return self

    def predict(self, grid):
        if self.matrix is None or self.params is None or self.scale0 is None:
            raise RuntimeError("Call fit() before predict().")

        grid = np.asarray(grid, dtype=float).reshape(-1)
        m = len(grid)
        n = len(self.x)

        x_grided = grid.reshape(-1, 1)
        D = cdist(x_grided, self.x, metric="cityblock")

        basic_func_cnt = self._basis_count()
        right_part = np.zeros((n + basic_func_cnt, m), dtype=float)
        right_part[:n, :] = (
            np.asarray(
                self.kernel.calc(D.T, self.params, ignore_nugget=self.extract_signal),
                dtype=float,
            )
            * self.scale0
        )

        idx_trend = n

        # 1) polynomial: 1, x, ..., x^k
        for i in range(self.k + 1):
            right_part[idx_trend, :] = grid ** i
            idx_trend += 1

        # 2) ordinary periodic
        if self.period is not None:
            right_part[idx_trend, :] = np.sin(2.0 * np.pi * grid / self.period)
            idx_trend += 1

            right_part[idx_trend, :] = np.cos(2.0 * np.pi * grid / self.period)
            idx_trend += 1

        # 3) plain exponential
        for a in self.exp_term:
            right_part[idx_trend, :] = np.exp(a * grid)
            idx_trend += 1

        # 4) exp-periodic
        for a, omega in self.exp_periodic_terms:
            env = np.exp(a * grid)

            right_part[idx_trend, :] = env * np.sin(omega * grid)
            idx_trend += 1

            right_part[idx_trend, :] = env * np.cos(omega * grid)
            idx_trend += 1

        if idx_trend != n + basic_func_cnt:
            raise RuntimeError("Trend basis size mismatch in predict().")
        '''
        idx_per = n
        for i in range(self.k + 1):
            col_monom = x_grided.ravel() ** i
            right_part[n + i, :] = col_monom
            idx_per += 1
    
         
        if self.period is not None:
            right_part[idx_per, :] = np.sin(2.0 * np.pi * grid / self.period)
            right_part[idx_per + 1, :] = np.cos(2.0 * np.pi * grid / self.period)
        '''

        if self.matrix_lu is not None:
            weights = lu_solve(self.matrix_lu, right_part, check_finite=False)
        else:
            weights = np.linalg.solve(self.matrix, right_part)

        pred = self.y @ weights[:n, :]
        zero_dist = (
            self.scale0
            * np.asarray(
                self.kernel.calc(
                    np.zeros((1, 1), dtype=float),
                    self.params,
                    ignore_nugget=self.extract_signal,
                ),
                dtype=float,
            ).item()
        )
        pred_var = zero_dist - np.sum(weights * right_part, axis=0)
        pred_var = np.asarray(pred_var, dtype=float)

        if not np.all(np.isfinite(pred_var)):
            raise RuntimeError("Kriging variance contains non-finite values.")

        scale = max(1.0, abs(float(zero_dist)), float(np.max(np.abs(pred_var))))
        tol = self.variance_tol * scale
        bad_negative = pred_var < -tol
        if np.any(bad_negative):
            worst = float(np.min(pred_var))
            raise RuntimeError(
                "Kriging variance is materially negative "
                f"(min={worst:.6g}, tol={tol:.6g}). This usually indicates an "
                "invalid covariance/noise split or an ill-conditioned system."
            )

        pred_var = np.where(pred_var < 0.0, 0.0, pred_var)
        return pred, pred_var
