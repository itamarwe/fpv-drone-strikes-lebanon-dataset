#!/usr/bin/env python3
"""Geometry regression tests for the Chamaa port (synthetic; no ground truth used).

Mirrors the intent of original_sainte_maxime/test_joint_constellation.py:
conventions, positive depth, one-to-one assignment, unmatched observations, and
reserved-identity exclusion, plus checks for the aerial reparametrisation.
"""
import json
import unittest

import numpy as np

import chamaa_constellation as cc

AREA = json.loads((cc.DATA / "search_areas.json").read_text())["areas"][0]


class Geometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pr = cc.Problem(AREA)
        # Aim at the densest cluster of MAPPED buildings (map-only; the truth is never read).
        xy = cls.pr.xyz[:, :2]
        density = [(np.linalg.norm(xy - q, axis=1) < 250).sum() for q in xy]
        target = xy[int(np.argmax(density))]
        cls.p = np.array([target[0], target[1], 123.0, -40.0, 3.0, np.log(900.0), np.log(450.0)])

    def test_rotation_orthonormal(self):
        for ypr in [(0, -30, 0), (123, -40, 3), (359, -80, -10)]:
            R = cc.rotation(*ypr)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)

    def test_forward_points_down_for_negative_pitch(self):
        self.assertLess(cc.rotation(10, -40, 0)[2, 2], 0)

    def test_ground_point_projects_to_principal_point(self):
        f, c, R, g = self.pr.unpack(self.p)
        v = R @ (g - c)
        self.assertGreater(v[2], 0)
        np.testing.assert_allclose(f * v[:2] / v[2] + [cc.CX, cc.CY], [cc.CX, cc.CY], atol=1e-6)

    def test_camera_height_above_ground_point(self):
        f, c, R, g = self.pr.unpack(self.p)
        self.assertAlmostEqual(c[2] - g[2], np.exp(self.p[6]), places=6)

    def test_projected_points_have_positive_depth(self):
        f, c, R, g = self.pr.unpack(self.p)
        uv, w, ids = self.pr.project(self.p)
        z = ((self.pr.xyz[ids] - c) @ R.T)[:, 2]
        self.assertTrue((z > 0).all())

    def _synthetic(self, drop=0):
        """Replace observations by projections of the map at a known pose."""
        pr = cc.Problem(AREA)
        uv, w, ids = pr.project(self.p)
        inside = (uv[:, 0] > 10) & (uv[:, 0] < cc.IMAGE_W - 10) & (uv[:, 1] > 10) & (uv[:, 1] < cc.IMAGE_H - 20)
        uv, w, ids = uv[inside], w[inside], ids[inside]
        self.assertGreaterEqual(len(ids), 12, "pick a pose that sees buildings")
        pr.uv = uv.copy(); pr.wh = np.c_[w, w * .6]
        pr.observed = [dict(id=f"syn_{i}") for i in range(len(uv))]
        n = len(uv); order = np.random.default_rng(7).permutation(n)
        pr.train = np.sort(order[:round(.7 * n)]); pr.check = np.sort(order[round(.7 * n):])
        pr.obs_feat = np.c_[pr.uv / [8, 6], np.log(pr.wh[:, 0]) / .65]
        return pr, ids

    def test_true_pose_scores_near_zero_and_one_to_one(self):
        pr, ids = self._synthetic()
        cost, pairs = pr.assignment(self.p, pr.train)
        self.assertLess(cost, 0.01)
        self.assertEqual(len({q["map_index"] for q in pairs}), len(pairs))  # one-to-one
        self.assertEqual(len(pairs), len(pr.train))

    def test_wrong_pose_scores_worse(self):
        pr, _ = self._synthetic()
        bad = self.p.copy(); bad[2] += 40; bad[0] += 300
        self.assertGreater(pr.assignment(bad, pr.train)[0], pr.assignment(self.p, pr.train)[0] + 1)

    def test_far_observation_left_unmatched(self):
        pr, _ = self._synthetic()
        pr.uv[pr.train[0]] = [5000, 5000]
        _, pairs = pr.assignment(self.p, pr.train)
        self.assertNotIn(int(pr.train[0]), {q["observed_index"] for q in pairs})

    def test_reserved_identities_excluded(self):
        pr, _ = self._synthetic()
        _, tp = pr.assignment(self.p, pr.train)
        used = {q["map_index"] for q in tp}
        _, cp = pr.assignment(self.p, pr.check, used)
        self.assertFalse(used & {q["map_index"] for q in cp})

    def test_coarse_prefers_true_pose(self):
        pr, _ = self._synthetic()
        bad = self.p.copy(); bad[2] += 60
        self.assertLess(pr.coarse(self.p), pr.coarse(bad))


if __name__ == "__main__":
    unittest.main(verbosity=2)
