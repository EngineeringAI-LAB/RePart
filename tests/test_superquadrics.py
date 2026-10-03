"""Small geometry checks that run without a dataset or GPU."""

import unittest

import numpy as np

from repart.geometry import mps_x_to_primitive, primitive_to_mps_x
from repart.geometry import (
    primitive_assignments,
    primitive_to_surface_mesh,
    primitive_to_surface_points,
)


class SuperquadricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = np.array([0.8, 1.2, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.1, -0.2, 0.3])
        self.primitive = mps_x_to_primitive(self.params)

    def test_mps_parameter_round_trip(self) -> None:
        np.testing.assert_allclose(
            primitive_to_mps_x(self.primitive), self.params, atol=1e-7
        )

    def test_point_assignments(self) -> None:
        second = self.primitive.clone()
        second.t += np.array([1.0, 0.0, 0.0])
        points = np.array([[0.1, -0.2, 0.3], [1.1, -0.2, 0.3]])
        assigned, soft, inside, occupancy = primitive_assignments(
            [self.primitive, second], points
        )
        self.assertEqual(assigned.tolist(), [0, 1])
        self.assertEqual(soft.shape, (2, 2))
        self.assertEqual(inside.shape, (2, 2))
        self.assertEqual(occupancy.shape, (2, 2))
        np.testing.assert_allclose(soft.sum(axis=0), 1.0, atol=1e-6)

    def test_surface_mesh(self) -> None:
        points = primitive_to_surface_points(self.primitive, nu=12, nv=8)
        vertices, faces = primitive_to_surface_mesh(self.primitive, nu=12, nv=8)
        self.assertEqual(points.shape, (96, 3))
        self.assertEqual(vertices.shape, (96, 3))
        self.assertEqual(faces.shape, (168, 3))
        self.assertTrue(np.isfinite(vertices).all())


if __name__ == "__main__":
    unittest.main()
