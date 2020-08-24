Fitting to forces without any constraints that parameters are shared across molecules.

Intent:
* Observe behavior of optimizers and samplers on force regression, without the added difficulty of a message-passing model or the representational limits imposed by sharing parameters across molecules.
* Identify initialization x parameterization x optimizer choices that allow to recover a potential energy model from forces

Contents: scripts and outputs for fitting MM parameters to forces from a reference MM model.
* fit_bonds_to_forces.py
    * and fit_bonds_to_forces_alt_param.py -- uses "linearized" parameterization
* fit_angles_to_forces.py
    * fit_angles_to_forces_alt_param.py -- uses "linearized parameterization"
* fit_torsions_to_forces.py
* fit_valence_to_forces.py
* plots.py -- reused 
* plots/
    * loss vs. iteration trajectory plots
    * pred vs. target, and pred vs. (target - pred) scatter plots
    
Optimizers/samplers used:
* BFGS + basin-hopping
* Inertial Langevin

Parameterizations used:
* "original" u(x, k, x0) = k * (x - x0)^2
* "linearized" u(x, k1, k2) = k1 * (x - x1)^2 + k2 * (x - x2)^2, for fixed x1, x2

Initializations used:
* At a constant per force type (ballpark the right magnitude for harmonic bonds, etc.)
* Near the reference values, multiplied by uniform noise in interval [1-eps, 1+eps]

TODO:
* Toggle different parameterizations, including making k log-scale
* Repeat many times
* fit_valence_to_ani_minus_mm_nonbond.py
* Refactor to reduce duplication among scripts for testing these conditions
