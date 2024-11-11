import matplotlib.pyplot as plt
import numpy as np
from matplotlib import path

from .geometry import CameraGeometry


class Mesh:
    def __init__(self, vertices, faces, parameters=None, verbose=False):
        self.vertices = vertices
        self.faces = faces
        self.parameters = parameters
        self.num_vertices_face = self.faces.shape[1]

        # Build adjacency matrix of multi-graph induced by faces
        # The (i, j) coordinate is the number of times that edge (i, j) appears in the faces
        # We consider edges as those pairs of vertices that are contiguous in the face
        # (e.g. if face is [1, 2, 3]
        # edges are [1, 2], [2, 3], and [3, 1], where the order does not matter).
        self.adjacency_matrix = np.zeros((len(self), len(self)))
        for t in self.faces:
            for i in range(len(t)):
                idx1, idx2 = t[i], t[(i + 1) % len(t)]
                self.adjacency_matrix[idx1, idx2] += 1
                self.adjacency_matrix[idx2, idx1] += 1
        self.edges = np.array([
            [i, neigh]
            for i in range(len(self))
            for neigh in np.where(self.adjacency_matrix[i, i:] != 0)[0] + i
        ])

        if verbose:
            fig = plt.figure()
            ax = fig.gca(projection="3d")  # type: ignore
            ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2])
            plt.show()

    def get_corner_border_path(self, min_angle_corner=np.pi / 6):
        """
        return:
            corner_vertices: Sorted by convention.
        """
        # Traverse the edge vertices and trace a path [[v1, v2], [v2, v3], ...]
        curr_vertex = first_vertex = 0
        set_visited = set()
        border_path = []
        done = False
        while not done:
            set_visited.add(curr_vertex)
            # Among the vertices in the border that share an edge in the
            # mesh with curr_vertex, remove those that
            # have been already visited
            candidates = set(np.where(self.adjacency_matrix[curr_vertex] == 1)[0]).difference(
                set_visited
            )
            if len(border_path) == 0:
                # In the first vertex of the path, we should have two options
                # (since the path is a cycle)
                assert len(candidates) == 2
                # By convention pick the next vertex as the one with the lowest index
                next_vertex = min(candidates)
            elif len(candidates) == 0:
                # At the end of the path, all vertices have been visited,
                # but we have to come back to the first vertex
                # Make sure the current vertex is connected to it
                assert self.adjacency_matrix[curr_vertex, first_vertex] == 1
                next_vertex = first_vertex
                done = True
            else:
                # In the middle of the path we should only have a candidate
                assert len(candidates) == 1
                next_vertex = candidates.pop()

            border_path.append([curr_vertex, next_vertex])
            curr_vertex = next_vertex
        corner_vertices = []
        if self.parameters is not None:
            for corner in [
                np.array([0, 0]),
                np.array([1, 0]),
                np.array([1, 1]),
                np.array([0, 1]),
            ]:
                candidates = [
                    i for i, a_i in enumerate(self.parameters) if np.allclose(a_i, corner)
                ]
                assert len(candidates) == 1
                corner_vertices.append(candidates[0])
        border_path = np.array(border_path)
        return np.sort(np.array(corner_vertices)), border_path[:, 0], border_path

    @staticmethod
    def _signed_triangle_area(x1, x2, x3):
        return 0.5 * np.linalg.det(np.r_[np.ones((1, 3)), np.c_[x1, x2, x3]])

    def compute_barycentric_coords_image(
        self, camera_geom: CameraGeometry, tracked_points_in_image
    ):
        bary = []
        vertices_in_image = camera_geom.get_image_coords_from_world(self.vertices)
        paths = [path.Path([vertices_in_image[t_i] for t_i in t]) for t in self.faces]
        for i, point in enumerate(tracked_points_in_image):
            idx = np.where([p.contains_points([point]) for p in paths])[0]
            assert len(idx) >= 1
            if len(idx) > 1:
                candidates = [
                    self._compute_barycentric_coords(vertices_in_image, point, idx_i)
                    for idx_i in idx
                ]
                # TODO: Disambiguate with depth
                bary.append(candidates.pop())
            else:
                bary.append(self._compute_barycentric_coords(vertices_in_image, point, idx[0]))
        return bary

    def compute_barycentric_coords_xz(self, tracked_points_in_mesh):
        # This only works for 2d faces (x and z component only, which are indices 0 and 2, resp.)
        bary = []
        paths = [path.Path([self.vertices[t_i, [0, 2]] for t_i in t]) for t in self.faces]
        for i, point in enumerate(tracked_points_in_mesh[:, [0, 2]]):
            idx = np.where([p.contains_points([point]) for p in paths])[0]
            assert len(idx) == 1
            bary.append(self._compute_barycentric_coords(self.vertices[:, [0, 2]], point, idx[0]))
        return bary

    def _compute_barycentric_coords(self, vertex_coords, feature_coords, idx):
        if self.num_vertices_face == 3:
            # Compute barycentric coordinates using edge approach from
            # https://en.wikipedia.org/wiki/Barycentric_coordinate_system
            v1, v2, v3 = self.faces[idx]
            T = np.c_[
                vertex_coords[v1] - vertex_coords[v3],
                vertex_coords[v2] - vertex_coords[v3],
            ]
            lam1, lam2 = np.linalg.inv(T) @ (feature_coords - vertex_coords[v3])
            assert lam1 > 0 and lam2 > 0 and lam1 + lam2 < 1
            lam3 = 1 - lam1 - lam2
            bary = {v1: lam1, v2: lam2, v3: lam3}
        elif self.num_vertices_face == 4:
            # Compute generalized barycentric coordinates
            # (in particular inverse bilinear coordinates) from
            # https://www.mn.uio.no/math/english/people/aca/michaelf/papers/gbc.pdf
            v1, v2, v3, v4 = self.faces[idx]
            corners = vertex_coords[self.faces[idx]]
            A = np.array([
                self._signed_triangle_area(
                    feature_coords,
                    corners[i],
                    corners[(i + 1) % self.num_vertices_face],
                )
                for i in range(self.num_vertices_face)
            ])
            B = np.array([
                self._signed_triangle_area(
                    feature_coords,
                    corners[(i - 1) % self.num_vertices_face],
                    corners[(i + 1) % self.num_vertices_face],
                )
                for i in range(self.num_vertices_face)
            ])
            D = B[0] ** 2 + B[1] ** 2 + 2 * A[0] * A[2] + 2 * A[1] * A[3]
            E = 2 * A - B - np.roll(B, -1) + np.sqrt(D)
            mu, one_minus_lam, one_minus_mu, lam = 2 * A / E
            assert np.isclose(1 - mu, one_minus_mu) and np.isclose(1 - lam, one_minus_lam)
            bary = {
                v1: (1 - lam) * (1 - mu),
                v2: lam * (1 - mu),
                v3: lam * mu,
                v4: (1 - lam) * mu,
            }
        else:
            raise NotImplementedError
        return bary

    def flat_vertices(self, phi_notation=False):
        if phi_notation:
            return self.vertices.T.flatten()
        else:
            return self.vertices.flatten()

    def flat_idx(self, vertex_idx: int | list, phi_notation=False):
        if isinstance(vertex_idx, list):
            return [
                idx
                for vertex_idx_i in vertex_idx
                for idx in self.flat_idx(vertex_idx_i, phi_notation)
            ]
        if phi_notation:
            return [vertex_idx, vertex_idx + len(self), vertex_idx + 2 * len(self)]
        else:
            return [3 * vertex_idx, 3 * vertex_idx + 1, 3 * vertex_idx + 2]

    def update_vertices(self, new_values, phi_notation=False):
        if phi_notation:
            self.vertices = new_values.reshape(3, -1).T
        else:
            self.vertices = new_values.reshape(-1, 3)

    def __len__(self):
        return len(self.vertices)
