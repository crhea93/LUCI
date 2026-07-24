"""
Constraints handed to ``scipy.optimize.minimize`` for a line fit.

Extracted from ``Fit`` so the constraint algebra can be read, tested and changed
without wading through the fitter.  Each builder returns a list of the dicts
SLSQP expects.

Every lambda binds its loop variable through a default argument
(``ind=ind``).  That is not stylistic: a bare closure over a loop variable is
evaluated when the constraint is *called*, long after the loop has finished, so
all of them would see the final index.  ``sigma_bounds`` documents a live
instance of exactly that mistake.
"""

from __future__ import annotations

import numpy as np
import scipy.special as sps

SPEED_OF_LIGHT = 299792  # km/s


def sigma_group_constraints(sigma_rel):
    """Tie the velocity dispersion of lines that share a group in ``sigma_rel``."""
    constraints = []
    for unique_ in np.unique(sigma_rel):
        inds_unique = [i for i, e in enumerate(sigma_rel) if e == unique_]
        if len(inds_unique) > 1:
            ind_0_ = inds_unique[0]
            for ind_unique_ in inds_unique[1:]:
                constraints.append(
                    {
                        "type": "eq",
                        "fun": lambda x, ind_unique=ind_unique_, ind_0=ind_0_: (
                            (SPEED_OF_LIGHT * x[3 * ind_0 + 2]) / x[3 * ind_0 + 1]
                            - (SPEED_OF_LIGHT * x[3 * ind_unique + 2]) / x[3 * ind_unique + 1]
                        ),
                    }
                )
    return constraints


def sigma_bounds(sigma_rel, _preserve_late_binding_bug=False):
    """
    Keep each line's sigma in ``(0, 10)``.

    ``_preserve_late_binding_bug`` reproduces the original behaviour bit for bit.
    It defaults to False -- the bug is fixed -- and is kept only so the defect
    stays demonstrable in tests.  The original wrote::

        for i in range(len(self.sigma_rel)):
            sigma_dict_list.append({'type': 'ineq', 'fun': lambda x: x[3*i+2]})
            sigma_dict_list.append({'type': 'ineq', 'fun': lambda x: -x[3*i+2]+10})

    with no default-argument binding, so every lambda reads ``i`` at call time --
    by which point the loop has ended and ``i`` is its last value.  The result is
    that only the *final* line's sigma is bounded, N times over, and every other
    line's sigma is left completely unconstrained (bug B21).

    The fixed behaviour is one bound per line, each on its own sigma.
    """
    constraints = []
    n = len(sigma_rel)
    if _preserve_late_binding_bug:
        i = n - 1  # the value every original lambda actually saw
        for _ in range(n):
            constraints.append({"type": "ineq", "fun": lambda x: x[3 * i + 2]})
            constraints.append({"type": "ineq", "fun": lambda x: -x[3 * i + 2] + 10})
        return constraints
    for i in range(n):
        constraints.append({"type": "ineq", "fun": lambda x, i=i: x[3 * i + 2]})
        constraints.append({"type": "ineq", "fun": lambda x, i=i: -x[3 * i + 2] + 10})
    return constraints


def velocity_constraints(vel_rel, lines, line_dict, axis_step):
    """
    Tie the velocities of lines that share a group in ``vel_rel``.

    Grouped lines must agree to within one channel: ``|v_i - v_0| <= axis_step``. That is a
    two-sided band, so it takes two inequalities per tied line.

    It used to be the single inequality ``v_i - v_0 - axis_step >= 0``, which only bounds
    ``v_i`` from *below* -- a grouped line could sit arbitrarily far to the red of the first one
    and still satisfy it. Fitting the SN4 Halpha complex with ``vel_rel=[1, 1, 1]`` came out at
    -806 / -576 / -221 km/s, three unrelated velocities, rather than one shared value.
    """
    constraints = []

    def relative_velocity(x, ind, line):
        """Velocity of line `ind` in km/s, from its fitted position x[3*ind+1] in cm-1."""
        return SPEED_OF_LIGHT * ((1e7 / x[3 * ind + 1] - line_dict[line]) / line_dict[line])

    for unique_ in np.unique(vel_rel):
        inds_unique = [i for i, e in enumerate(vel_rel) if e == unique_]
        if len(inds_unique) > 1:
            ind_0 = inds_unique[0]
            ind_0_line = lines[ind_0]
            for ind_unique in inds_unique[1:]:
                ind_unique_line = lines[ind_unique]
                # axis_step - (v_i - v_0) >= 0  and  axis_step + (v_i - v_0) >= 0
                for sign in (1.0, -1.0):
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda x,
                            i_=ind_unique,
                            i0_=ind_0,
                            line_=ind_unique_line,
                            line0_=ind_0_line,
                            sign_=sign: (
                                axis_step
                                - sign_ * (relative_velocity(x, i_, line_) - relative_velocity(x, i0_, line0_))
                            ),
                        }
                    )
    return constraints


def nii_doublet_constraint(lines, model_type, sinc_width):
    """Hold NII6548 at one third the flux of NII6583."""
    nii_6548 = np.argwhere(np.array(lines) == "NII6548")[0][0]
    nii_6583 = np.argwhere(np.array(lines) == "NII6583")[0][0]

    if model_type == "sincgauss":

        def func_(x):
            def flux(index):
                return x[3 * index] * (
                    (np.sqrt(2 * np.pi) * x[3 * index + 2]) / sps.erf(x[3 * index + 2] / (np.sqrt(2) * sinc_width))
                )

            return (1 / 3) * flux(nii_6583) - flux(nii_6548)
    else:  # 'gaussian' and 'sinc' share the simple amplitude*sigma form

        def func_(x):
            return (1 / 3) * (x[3 * nii_6583] * x[3 * nii_6583 + 2]) - x[3 * nii_6548] * x[3 * nii_6548 + 2]

    return [{"type": "eq", "fun": func_}]


def distinct_position_constraints(lines):
    """Require the components of a multi-component fit to sit at different positions."""
    constraints = []
    inds = list(range(len(lines)))
    ind_0 = inds[0]
    for ind_unique in inds[1:]:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, ind_unique=ind_unique, ind_0=ind_0: x[3 * ind_unique + 1] + x[3 * ind_0 + 1] + 1,
            }
        )
    return constraints


def amplitude_constraints(lines):
    """Keep line amplitudes and the continuum inside the normalised range."""
    constraints = []
    for ind_unique in range(len(lines)):
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, ind_unique=ind_unique: -x[3 * ind_unique] + 1.1,
            }
        )
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, ind_unique=ind_unique: x[3 * ind_unique] + 1e-8,
            }
        )
    constraints.append({"type": "ineq", "fun": lambda x: -x[-1] + 0.99})
    constraints.append({"type": "ineq", "fun": lambda x: x[-1] + 1e-8})
    return constraints
