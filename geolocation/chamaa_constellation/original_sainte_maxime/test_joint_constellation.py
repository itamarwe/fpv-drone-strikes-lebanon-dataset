#!/usr/bin/env python3
"""Deterministic, data-free geometry and assignment checks; no reference pin."""
import unittest

import numpy as np

from joint_building_constellation import Problem, rotation


class ConstellationMathTests(unittest.TestCase):
    def test_rotation_is_proper(self):
        rng = np.random.default_rng(2026)
        for yaw, pitch, roll in rng.uniform([-180, -89, -180], [180, 89, 180], (100, 3)):
            r = rotation(yaw, pitch, roll)
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(r), 1., places=13)

    def test_cardinal_camera_axes(self):
        np.testing.assert_allclose(rotation(0, 0, 0), [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-14)
        np.testing.assert_allclose(rotation(90, 0, 0)[2], [1, 0, 0], atol=1e-14)
        self.assertGreater(rotation(0, 10, 0)[2, 2], 0.)

    @staticmethod
    def geometry_problem(mode="fixed"):
        p = Problem.__new__(Problem)
        p.mode, p.focal = mode, 1000.
        p.origin = np.array([100., 200., 0.])
        p.terrain = lambda en: np.full(len(en), 10.)
        p.heights = np.array([10., 10.])
        p.axes = np.array([[[2., 0.], [0., 3.]], [[2., 0.], [0., 3.]]])
        return p

    def test_projection_and_inverse_known_points(self):
        problem = self.geometry_problem()
        params = np.array([2., 3., 13., -4., 2.])
        f, alpha, camera, r = problem.unpack(params)
        points_camera = np.array([[1., -2., 100.], [-5., 3., 200.]])
        problem.xyz = points_camera @ r + camera
        np.testing.assert_allclose((problem.xyz - camera) @ r.T, points_camera, atol=1e-12)
        uv, widths, ids = problem.project(params)
        np.testing.assert_array_equal(ids, [0, 1])
        np.testing.assert_allclose(uv, [[650., 460.], [615., 495.]], atol=1e-12)
        np.testing.assert_allclose(camera + problem.origin, [102., 203., 12.])
        self.assertTrue(np.all(widths > 6))

    def test_roof_offset_and_width(self):
        problem = self.geometry_problem("joint_offset")
        params = np.array([2., 3., 0., 0., 0., np.log(1000.), .5])
        _, _, camera, r = problem.unpack(params)
        problem.xyz = np.array([[0., 0., 100.], [0., 0., -100.]]) @ r + camera
        uv, widths, ids = problem.project(params)
        np.testing.assert_array_equal(ids, [0])  # Behind-camera point rejected.
        np.testing.assert_allclose(uv, [[640., 530.]], atol=1e-12)
        np.testing.assert_allclose(widths, [40.], atol=1e-12)

    @staticmethod
    def assignment_problem(observed, projected):
        p = Problem.__new__(Problem)
        p.uv = np.array(observed, dtype=float).reshape(-1, 2)
        p.wh = np.full((len(observed), 2), 20.)
        p.observed = [{"id": f"observation_{i}"} for i in range(len(observed))]
        p.map = [{"index": i, "id": f"map_{i}"} for i in range(len(projected))]
        p.project = lambda _: (np.array(projected, dtype=float).reshape(-1, 2),
                               np.full(len(projected), 20.), np.arange(len(projected)))
        return p

    def test_exact_permuted_constellation_assignment(self):
        p = self.assignment_problem([[10, 20], [100, 200], [300, 400]], [[300, 400], [10, 20], [100, 200]])
        cost, pairs = p.assignment(None, np.array([0, 1, 2]))
        self.assertAlmostEqual(cost, 0.)
        self.assertEqual([r["map_index"] for r in pairs], [1, 2, 0])
        self.assertEqual(len(set(r["map_id"] for r in pairs)), 3)

    def test_collision_cannot_reuse_map_point(self):
        p = self.assignment_problem([[0, 0], [.1, 0]], [[0, 0]])
        cost, pairs = p.assignment(None, np.array([0, 1]))
        self.assertAlmostEqual(cost, 4.5)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["observed_index"], 0)

    def test_every_detection_can_remain_unmatched(self):
        p = self.assignment_problem([[500, 500], [600, 600], [700, 700]], [[0, 0]])
        cost, pairs = p.assignment(None, np.array([0, 1, 2]))
        self.assertEqual(cost, 9.)
        self.assertEqual(pairs, [])

    def test_no_visible_map_points(self):
        p = self.assignment_problem([[500, 500]], [])
        self.assertEqual(p.assignment(None, np.array([0])), (9., []))
        self.assertEqual(p.assignment(None, np.array([0]), {123}), (9., []))

    def test_heldout_cannot_reuse_training_map_instance(self):
        p = self.assignment_problem([[0, 0]], [[0, 0]])
        cost, pairs = p.assignment(None, np.array([0]), excluded_map_indices={0})
        self.assertEqual(cost, 9.)
        self.assertEqual(pairs, [])

    def test_empty_observation_subset_rejected(self):
        p = self.assignment_problem([[0, 0]], [[0, 0]])
        with self.assertRaises(ValueError):
            p.assignment(None, np.array([], dtype=int))


if __name__ == "__main__":
    unittest.main(verbosity=2)
