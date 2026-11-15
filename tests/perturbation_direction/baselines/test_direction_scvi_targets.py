"""scVI target masking must follow scDEBART's detection rule."""

import sys
from pathlib import Path

import numpy as np
from experiments.perturbation_direction.baselines.scvi_targets import mask_low_detection


def test_genes_detected_in_both_arms_keep_their_value():
    logfc = np.array([[1.5, -2.0]])
    nz1 = np.array([[0.9, 0.5]])
    nz2 = np.array([[0.8, 0.4]])
    out = mask_low_detection(logfc, nz1, nz2, threshold=0.3)
    np.testing.assert_allclose(out, [[1.5, -2.0]])


def test_gene_undetected_in_either_arm_is_zeroed():
    logfc = np.array([[1.5, -2.0, 3.0]])
    nz1 = np.array([[0.9, 0.1, 0.5]])  # arm 1 fails on gene 1
    nz2 = np.array([[0.8, 0.5, 0.2]])  # arm 2 fails on gene 2
    out = mask_low_detection(logfc, nz1, nz2, threshold=0.3)
    np.testing.assert_allclose(out, [[1.5, 0.0, 0.0]])


def test_threshold_is_strict_at_the_boundary():
    # exactly at the threshold is kept, matching `nz < threshold` in their code
    logfc = np.array([[1.0]])
    nz1 = np.array([[0.3]])
    nz2 = np.array([[0.3]])
    assert mask_low_detection(logfc, nz1, nz2, threshold=0.3)[0, 0] == 1.0


def test_input_is_not_mutated():
    logfc = np.array([[1.5]])
    before = logfc.copy()
    mask_low_detection(logfc, np.array([[0.0]]), np.array([[0.0]]), threshold=0.3)
    np.testing.assert_allclose(logfc, before)


def test_mask_is_elementwise_across_perturbations():
    logfc = np.array([[1.0, 2.0], [3.0, 4.0]])
    nz1 = np.array([[0.9, 0.9], [0.1, 0.9]])
    nz2 = np.array([[0.9, 0.9], [0.9, 0.9]])
    out = mask_low_detection(logfc, nz1, nz2, threshold=0.3)
    np.testing.assert_allclose(out, [[1.0, 2.0], [0.0, 4.0]])
