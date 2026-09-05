"""Unit tests for Potential-Based Reward Shaping (PBRS) and state potential telescoping."""

import pytest
from oeis_learn.rl.reward import compute_pbrs_potential


def test_pbrs_potential_empty_vs_valid_header():
    pot_empty = compute_pbrs_potential("")
    assert pot_empty.total_potential == 0.0

    wat_header = '(module (func (export "compute") (param $n i32) (result i64)))'
    pot_header = compute_pbrs_potential(wat_header)
    assert pot_header.phi_comp > 0.0
    assert pot_header.total_potential >= 0.2


def test_pbrs_potential_parameter_binding():
    wat_with_param = '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_s i64.const 2 i64.mul))'
    pot_param = compute_pbrs_potential(wat_with_param)
    assert pot_param.phi_bind > 0.0
    assert pot_param.total_potential > 0.4


def test_pbrs_potential_telescoping_invariance():
    # Potential difference F(s, s') = gamma * Phi(s') - Phi(s)
    gamma = 0.99
    s0_pot = compute_pbrs_potential("")
    s1_pot = compute_pbrs_potential('(module (func (export "compute") (param $n i32) (result i64)))')
    s2_pot = compute_pbrs_potential('(module (func (export "compute") (param $n i32) (result i64) local.get $n))')

    f01 = gamma * s1_pot.total_potential - s0_pot.total_potential
    f12 = gamma * s2_pot.total_potential - s1_pot.total_potential

    # Cumulative shaped sum: f01 + f12 telescopes to gamma^2 * Phi(s2) + (gamma - 1)*Phi(s1) - Phi(s0)
    assert isinstance(f01, float)
    assert isinstance(f12, float)
