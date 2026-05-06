# Surrogate A/B: Spherical vs Cartesian

- Trials per cell: 50    (total: 900)
- Reference: scipy RK45 rtol 1e-6, atol 1e-8 (`policy/pendulum_simulator.py`)
- Spherical surrogate: `loss/differentiable_pendulum.py` (clamp 0.01 ≤ θ ≤ 0.8π)
- Cartesian surrogate: `loss/cartesian_pendulum.py` (no clamp; explicit |q|=1 projection)
- Same polynomial-derived UAV acc(t) fed to all three integrators.

| L (m) | θ₀ (°) | dφ̇₀ | sph peak err mean (°) | cart peak err mean (°) | Δ (°) |
|------:|------:|------:|------:|------:|------:|
| 0.5 | 5.0 | 0.0 | 9.24 | 1.94 | +7.30 |
| 0.5 | 5.0 | 0.8 | 10.70 | 2.25 | +8.45 |
| 0.5 | 15.0 | 0.0 | 11.88 | 3.95 | +7.93 |
| 0.5 | 15.0 | 0.8 | 11.73 | 3.79 | +7.94 |
| 0.5 | 30.0 | 0.0 | 17.21 | 5.58 | +11.63 |
| 0.5 | 30.0 | 0.8 | 20.16 | 6.12 | +14.05 |
| 0.8 | 5.0 | 0.0 | 23.97 | 3.17 | +20.80 |
| 0.8 | 5.0 | 0.8 | 19.81 | 2.88 | +16.93 |
| 0.8 | 15.0 | 0.0 | 17.40 | 6.06 | +11.34 |
| 0.8 | 15.0 | 0.8 | 21.29 | 8.48 | +12.81 |
| 0.8 | 30.0 | 0.0 | 17.85 | 11.54 | +6.31 |
| 0.8 | 30.0 | 0.8 | 20.09 | 12.16 | +7.93 |
| 1.2 | 5.0 | 0.0 | 28.91 | 4.40 | +24.51 |
| 1.2 | 5.0 | 0.8 | 24.39 | 3.74 | +20.66 |
| 1.2 | 15.0 | 0.0 | 22.25 | 9.47 | +12.78 |
| 1.2 | 15.0 | 0.8 | 25.69 | 10.10 | +15.59 |
| 1.2 | 30.0 | 0.0 | 23.41 | 12.63 | +10.79 |
| 1.2 | 30.0 | 0.8 | 23.38 | 19.17 | +4.21 |

**Overall mean peak error:** spherical 19.41° → cartesian 7.08° (Δ = +12.33°). Positive Δ means Cartesian closer to RK45.
