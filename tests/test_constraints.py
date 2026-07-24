"""
Tests for the SLSQP constraint builders.

Centred on B21: the sigma bounds were built with lambdas that closed over the
loop variable without binding it, so all of them read the loop's final value at
call time. Only the last line's sigma was ever bounded.
"""

from __future__ import annotations

import numpy as np
import pytest

from LUCI.fitting.constraints import (
    amplitude_constraints,
    distinct_position_constraints,
    nii_doublet_constraint,
    sigma_bounds,
    sigma_group_constraints,
    velocity_constraints,
)

LINES = ["Halpha", "NII6548", "NII6583"]
LINE_DICT = {"Halpha": 656.280, "NII6583": 658.341, "NII6548": 654.803}


def _solution(sigmas):
    """A solution vector [amp, pos, sigma] * n_lines + [continuum]."""
    x = []
    for sigma in sigmas:
        x += [1.0, 15200.0, sigma]
    return np.array(x + [0.1])


# --------------------------------------------------------------------------
# B21
# --------------------------------------------------------------------------

def test_sigma_bounds_constrain_every_line():
    """One positivity bound and one upper bound per line, each on its own sigma."""
    bounds = sigma_bounds([1, 1, 1], _preserve_late_binding_bug=False)
    assert len(bounds) == 6

    # A negative sigma on line 0 must be rejected by at least one constraint.
    x = _solution([-5.0, 3.0, 3.0])
    assert any(c["fun"](x) < 0 for c in bounds), "line 0's negative sigma was not caught"

    # ... and likewise for the middle line.
    x = _solution([3.0, -5.0, 3.0])
    assert any(c["fun"](x) < 0 for c in bounds), "line 1's negative sigma was not caught"


def test_late_binding_bug_left_all_but_the_last_line_unconstrained():
    """
    Pins the original defect so the fix cannot be quietly reverted.

    Every lambda read the loop variable at call time, so all six constraints
    evaluated the *last* line's sigma. A negative sigma anywhere else sailed
    through.
    """
    buggy = sigma_bounds([1, 1, 1], _preserve_late_binding_bug=True)
    x = _solution([-5.0, -5.0, 3.0])  # only the last line is sane
    assert all(c["fun"](x) >= 0 for c in buggy), (
        "expected the buggy bounds to miss the first two lines entirely"
    )
    # The fixed version catches it.
    fixed = sigma_bounds([1, 1, 1], _preserve_late_binding_bug=False)
    assert any(c["fun"](x) < 0 for c in fixed)


def test_sigma_upper_bound_is_ten():
    bounds = sigma_bounds([1], _preserve_late_binding_bug=False)
    assert all(c["fun"](_solution([5.0])) >= 0 for c in bounds)
    assert any(c["fun"](_solution([50.0])) < 0 for c in bounds)


# --------------------------------------------------------------------------
# The other builders
# --------------------------------------------------------------------------

def test_sigma_group_ties_are_satisfied_by_equal_dispersions():
    """Lines in one group must share a velocity dispersion, not a raw sigma."""
    ties = sigma_group_constraints([1, 1, 1])
    assert len(ties) == 2  # first line tied to each of the other two
    # Equal sigma at equal position => equal dispersion => residual 0.
    assert all(c["fun"](_solution([2.0, 2.0, 2.0])) == pytest.approx(0.0) for c in ties)


def test_ungrouped_sigmas_are_not_tied():
    assert sigma_group_constraints([1, 2, 3]) == []


def test_velocity_constraints_tie_grouped_lines():
    cons = velocity_constraints([1, 1, 1], LINES, LINE_DICT, axis_step=2.0)
    assert len(cons) == 2
    assert all(np.isfinite(c["fun"](_solution([2.0, 2.0, 2.0]))) for c in cons)


@pytest.mark.parametrize("model", ["gaussian", "sinc", "sincgauss"])
def test_nii_constraint_is_satisfied_at_the_one_third_ratio(model):
    """NII6548 at one third of NII6583 must give a residual of zero."""
    cons = nii_doublet_constraint(LINES, model, sinc_width=2.5)
    assert len(cons) == 1
    x = _solution([2.0, 2.0, 2.0])
    x[3 * LINES.index("NII6583")] = 3.0  # amplitude
    x[3 * LINES.index("NII6548")] = 1.0  # one third
    assert cons[0]["fun"](x) == pytest.approx(0.0, abs=1e-9)


def test_amplitude_constraints_reject_a_negative_amplitude():
    cons = amplitude_constraints(LINES)
    x = _solution([2.0, 2.0, 2.0])
    x[3 * 1] = -0.5  # line 1 amplitude negative
    assert any(c["fun"](x) < 0 for c in cons)


def test_distinct_position_constraints_cover_every_extra_component():
    assert len(distinct_position_constraints(LINES)) == len(LINES) - 1
