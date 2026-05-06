"""
Swing-aware payload-MPC baseline.

A real receding-horizon planner (replacing the 7x7 grid search in
evaluate_closed_loop.py) that uses the same suspended-load dynamics the
neural method is trained against. The point of this baseline is to give
reviewers a credible "well-tuned classical MPC" comparison, not a strawman.

Method:
- 5th-order polynomial parameterization over the segment, identical to the
  one the YOPO head outputs, so the optimizer searches the SAME action
  space.
- Cost = w_goal * goal_term + w_dyn * pendulum_swing_cost + w_smooth * jerk
        + w_safe * obstacle_distance_penalty (if ESDF callback provided).
- Pendulum swing cost is computed by integrating the spherical pendulum ODE
  along the candidate trajectory, exactly the same surrogate the YOPO loss
  uses. So this baseline gets the same physics knowledge YOPO does — the
  only difference is *online* optimization vs *amortized* learning.
- Optimizer: scipy.optimize.minimize (L-BFGS-B) over the 6 free parameters
  per axis (end pos / vel / acc), so 18 decision variables. Multistart from
  a small grid of warm-starts, plus the previous-segment solution.

Usage:
    from baselines.swing_aware_mpc import SwingAwareMPC

    mpc = SwingAwareMPC(L=0.8, m=0.3, sgm_time=2.5)
    end_state, traj_acc = mpc.plan(start_state, goal, pendulum_state,
                                   esdf=esdf_callback)
"""

import math
import numpy as np
from scipy.optimize import minimize


G = 9.81


def _poly_coefficients(start_pva, end_pva, T):
    """5th-order polynomial coefficients from boundary conditions.

    start_pva, end_pva: arrays shaped (3, 3) — rows = [pos, vel, acc] per axis.
    Returns: (3, 6) array, rows = [c0..c5] per axis.
    """
    # Per-axis 6x6 mapping. p0, v0, a0, pT, vT, aT -> [c0,c1,c2,c3,c4,c5]
    # p(t) = c0 + c1 t + c2 t^2 + c3 t^3 + c4 t^4 + c5 t^5
    c = np.zeros((3, 6))
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    M = np.array([
        [1, 0,  0,    0,     0,     0],
        [0, 1,  0,    0,     0,     0],
        [0, 0,  2,    0,     0,     0],
        [1, T,  T2,  T3,    T4,    T5],
        [0, 1, 2*T, 3*T2,  4*T3,  5*T4],
        [0, 0,  2,  6*T,  12*T2, 20*T3],
    ], dtype=float)
    Minv = np.linalg.inv(M)
    for ax in range(3):
        bc = np.array([
            start_pva[0, ax], start_pva[1, ax], start_pva[2, ax],
            end_pva[0, ax],   end_pva[1, ax],   end_pva[2, ax],
        ])
        c[ax] = Minv @ bc
    return c


def _eval_acc(c, t):
    """Acceleration from polynomial coefficients at times t (vectorized)."""
    # a(t) = 2 c2 + 6 c3 t + 12 c4 t^2 + 20 c5 t^3
    t = np.asarray(t)
    out = (
        2 * c[:, 2:3]
        + 6 * c[:, 3:4] * t
        + 12 * c[:, 4:5] * (t ** 2)
        + 20 * c[:, 5:6] * (t ** 3)
    )
    return out  # (3, len(t))


def _eval_jerk_integral(c, T):
    """Closed-form integral of jerk^2 over [0, T] for 5th-order polynomial.

    j(t) = 6 c3 + 24 c4 t + 60 c5 t^2  (per axis)
    """
    j_int = 0.0
    for ax in range(3):
        c3, c4, c5 = c[ax, 3], c[ax, 4], c[ax, 5]
        # Integral of (a + b t + c t^2)^2 dt from 0 to T
        a = 6 * c3
        b = 24 * c4
        cc = 60 * c5
        j_int += (
            a * a * T
            + a * b * T ** 2
            + (b * b / 3 + 2 * a * cc / 3) * T ** 3
            + b * cc / 2 * T ** 4
            + cc * cc / 5 * T ** 5
        )
    return j_int


def _rollout_swing(acc_traj, dt, L, p_state):
    """Forward-integrate spherical pendulum under a UAV acc profile.

    acc_traj: (N, 3) — UAV accelerations in world frame at uniform spacing dt.
    p_state: [theta, phi, dtheta, dphi].
    Returns: dict with peak_swing_deg, mean_swing_deg, final_swing_deg.
    """
    theta, phi, dtheta, dphi = p_state
    theta = float(np.clip(theta, 1e-4, math.pi * 0.9))
    swings = []
    for k in range(len(acc_traj)):
        ax, ay, az = acc_traj[k]
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        sin_p, cos_p = math.sin(phi), math.cos(phi)
        a_eff = (ax, ay, az + G)
        a_dot_e_theta = (a_eff[0] * cos_t * cos_p
                         + a_eff[1] * cos_t * sin_p
                         + a_eff[2] * sin_t)
        a_dot_e_phi = -a_eff[0] * sin_p + a_eff[1] * cos_p

        ddtheta = dphi ** 2 * sin_t * cos_t - (1.0 / L) * a_dot_e_theta
        # Standard spherical-pendulum azimuth equation (Euler-Lagrange):
        #   φ̈ = -2cosθ·θ̇·φ̇/sinθ - (1/(L·sinθ))·(a_eff·ê_φ)
        # Same form used in loss/differentiable_pendulum.py.
        sin_safe = max(abs(sin_t), 0.1) * (1.0 if sin_t >= 0 else -1.0)
        ddphi = (-2.0 * dtheta * dphi * cos_t / sin_safe
                 - (1.0 / (L * sin_safe)) * a_dot_e_phi
                 if abs(sin_t) > 0.15 else -0.5 * dphi)

        ddtheta = max(min(ddtheta, 20.0), -20.0)
        ddphi = max(min(ddphi, 20.0), -20.0)

        dtheta = max(min(dtheta + ddtheta * dt, 5.0), -5.0)
        dphi = max(min(dphi + ddphi * dt, 5.0), -5.0)
        theta = max(min(theta + dtheta * dt, math.pi * 0.8), 0.01)
        phi = phi + dphi * dt

        swings.append(theta)

    swings = np.array(swings)
    return {
        "peak": float(np.max(swings)),
        "rms":  float(np.sqrt(np.mean(swings ** 2))),
        "final": float(swings[-1]),
    }


class SwingAwareMPC:
    """Receding-horizon swing-aware MPC sharing YOPO's polynomial action space."""

    def __init__(
        self,
        L: float = 0.8,
        m: float = 0.3,
        sgm_time: float = 2.5,
        n_steps: int = 20,
        vel_max: float = 4.0,
        acc_max: float = 6.0,
        w_goal: float = 1.0,
        w_dyn: float = 0.6,
        w_smooth: float = 1e-3,
        w_safe: float = 5.0,
    ):
        self.L = L
        self.m = m
        self.T = sgm_time
        self.n_steps = n_steps
        self.dt = sgm_time / n_steps
        self.vel_max = vel_max
        self.acc_max = acc_max
        self.w_goal = w_goal
        self.w_dyn = w_dyn
        self.w_smooth = w_smooth
        self.w_safe = w_safe
        self._last_solution = None  # warm-start across calls

    # ---- public api ---------------------------------------------------------

    def plan(self, start_pva, goal_pos, pendulum_state, esdf=None):
        """Plan a 5th-order polynomial trajectory.

        start_pva: (3, 3) — rows = [pos, vel, acc] per axis (world frame).
        goal_pos:  (3,)   — goal position (world).
        pendulum_state: (4,) — [theta, phi, dtheta, dphi].
        esdf: optional callable (3,) -> distance to obstacles (negative inside).

        Returns: end_pva (3, 3), traj_acc (n_steps, 3).
        """
        start_pva = np.asarray(start_pva, dtype=float)
        goal = np.asarray(goal_pos, dtype=float)
        p_state = np.asarray(pendulum_state, dtype=float)

        # Decision variable: end_pva flattened (9 dims).
        # Initial guess: hold velocity & accel constant, end_pos = start + v*T.
        warm_starts = [
            np.array([
                start_pva[0] + start_pva[1] * self.T,
                np.clip(start_pva[1] * 0.5, -self.vel_max, self.vel_max),
                np.zeros(3),
            ]).reshape(-1),
        ]
        if self._last_solution is not None:
            warm_starts.append(self._last_solution.copy())
        # Warm-start aimed at goal
        delta = goal - start_pva[0]
        norm = np.linalg.norm(delta) + 1e-6
        v_to_goal = delta / norm * min(self.vel_max, norm / self.T)
        warm_starts.append(np.array([
            start_pva[0] + v_to_goal * self.T,
            v_to_goal * 0.5,
            np.zeros(3),
        ]).reshape(-1))

        best = None
        for x0 in warm_starts:
            res = minimize(
                self._cost,
                x0,
                args=(start_pva, goal, p_state, esdf),
                method="L-BFGS-B",
                options={"maxiter": 60, "ftol": 1e-5},
            )
            if best is None or res.fun < best.fun:
                best = res

        end_pva = best.x.reshape(3, 3)
        self._last_solution = best.x.copy()
        c = _poly_coefficients(start_pva, end_pva, self.T)
        t_pts = np.linspace(self.dt, self.T, self.n_steps)
        acc_traj = _eval_acc(c, t_pts).T  # (n_steps, 3)
        return end_pva, acc_traj

    # ---- internals ----------------------------------------------------------

    def _cost(self, x, start_pva, goal, p_state, esdf):
        end_pva = x.reshape(3, 3)
        c = _poly_coefficients(start_pva, end_pva, self.T)

        # Goal term
        goal_err = np.linalg.norm(end_pva[0] - goal)
        goal_term = goal_err

        # Acceleration / vel limits as soft penalties
        v_excess = np.maximum(np.abs(end_pva[1]) - self.vel_max, 0.0).sum()
        a_excess = np.maximum(np.abs(end_pva[2]) - self.acc_max, 0.0).sum()
        limit_term = 5.0 * (v_excess + a_excess)

        # Smoothness term — closed form
        smooth_term = _eval_jerk_integral(c, self.T)

        # Dynamics (swing) term — rollout pendulum under traj acceleration
        t_pts = np.linspace(self.dt, self.T, self.n_steps)
        acc_traj = _eval_acc(c, t_pts).T  # (n_steps, 3)
        acc_traj = np.clip(acc_traj, -self.acc_max, self.acc_max)
        swing = _rollout_swing(acc_traj, self.dt, self.L, p_state)
        dyn_term = swing["peak"] ** 2 + 0.5 * swing["rms"] ** 2

        # Safety term — sample positions along trajectory, query ESDF
        safe_term = 0.0
        if esdf is not None:
            # Sample 8 points along the path
            ts = np.linspace(0, self.T, 8)
            for t in ts:
                # Evaluate position polynomial at t
                pos_t = np.array([
                    sum(c[ax, k] * t ** k for k in range(6))
                    for ax in range(3)
                ])
                d = esdf(pos_t)
                safe_term += np.exp(-max(d, 0.0))

        return (self.w_goal * goal_term
                + self.w_dyn * dyn_term
                + self.w_smooth * smooth_term
                + self.w_safe * safe_term
                + limit_term)
