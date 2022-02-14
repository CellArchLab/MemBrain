import numpy as np


# from preprocessing import create_DL_data as create_DL_data


class Mesh(object):
    def __init__(self, vertices, triangle_combos):
        self.vertices = vertices
        self.triangles = []
        self.triangle_combos = triangle_combos
        # self.make_triangles(triangle_combos)

    def __len__(self):
        return len(self.triangle_combos)

    def make_triangles(self, triangle_combos):
        for combo in triangle_combos:
            triangle = Triangle(self.vertices[combo[0] - 1], self.vertices[combo[1] - 1], self.vertices[combo[2] - 1])
            self.triangles.append(triangle)

    def get_triangle(self, id):
        combo = self.triangle_combos[int(id)]
        triangle = Triangle(self.vertices[combo[0] - 1], self.vertices[combo[1] - 1], self.vertices[combo[2] - 1])
        return triangle

    def find_closest_triangle(self, position):
        distances = 1000 * np.ones(len(self))
        for i in range(len(self)):
            cur_triangle = self.get_triangle(i)
            distances[i] = np.linalg.norm(cur_triangle.get_center_position() - position)
        return np.argmin(distances)


class Triangle(object):
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.normal = self.compute_normal()

    def compute_normal(self):
        normal = np.cross((self.v2 - self.v1), (self.v3 - self.v1))
        normal = normal / np.linalg.norm(normal)
        return normal

    def get_plane_matrix(self):
        m1 = self.v2 - self.v1
        m1 /= np.linalg.norm(m1)
        m2 = np.cross(self.normal, m1)
        m3 = self.normal

        return np.stack([m1, m2, m3])

    def get_center_position(self):
        return 1.0 / 3 * (self.v1 + self.v2 + self.v3)


def read_obj_file_to_triangles(filename):
    vertex_list = []
    combo_list = []
    with open(filename) as read_obj:
        for row in read_obj:
            if row.startswith('v'):
                coords = row.split(' ')[1:]
                if coords[2].endswith('\n'):
                    coords[2] = coords[2][:-1]
                coords = np.array(coords, dtype=np.float)
                vertex_list.append(coords)
            elif row.startswith('f'):
                combo = row.split(' ')[1:]
                if combo[2].endswith('\n'):
                    combo[2] = combo[2][:-1]
                combo = np.array(combo, dtype=np.int64)
                combo_list.append(combo)
    mesh = Mesh(vertex_list, combo_list)
    return mesh


class GeodesicDistanceComputer(object):
    def __init__(self, mesh, pointStart, pointEnd):
        assert isinstance(mesh, Mesh)
        self.mesh = mesh
        self.pointStart = pointStart
        self.pointEnd = pointEnd
        self.vertexStart = mesh.find_closest_triangle(self.pointStart)
        self.vertexEnd = mesh.find_closest_triangle(self.pointEnd)




# meshes_dir = '/fs/pool/pool-engel/Lorenz/pipeline_spinach/tomograms/Tomo_17/meshes'
# mesh = cur_mesh = read_obj_file_to_triangles(os.path.join(meshes_dir, os.listdir(meshes_dir)[10]))
# position_start = np.array([4531, 9541, 3932])
# position_end = np.array([4971, 8362, 3187])
# test = GeodesicDistanceComputer(mesh, position_start, position_end)





















# in_star = '/fs/pool/pool-engel/Lorenz/pipeline_spinach/rotated_volumes/chlamy.star'
# settings = ParameterSettings(in_star)
# meshes_dir = '/fs/pool/pool-engel/Lorenz/pipeline_spinach/tomograms/Tomo_17/meshes'
#
#
# gt_path_dict = {}
# for tomo_token in settings.tomo_tokens:
#     gt_path_dict[tomo_token] = {}
#     tomo_gt_dir = settings.gt_paths[tomo_token]
#     for mb_token in settings.mb_tokens[tomo_token]:
#         for files in os.listdir(tomo_gt_dir):
#             cur_tomo, cur_mb = data_utils.get_tomo_and_mb_from_file_name(os.path.join(tomo_gt_dir, files), settings)
#             if cur_tomo == tomo_token and cur_mb == mb_token:
#                 gt_path_dict[tomo_token][mb_token] = os.path.join(tomo_gt_dir, files)
#                 break
#
# for tomo in settings.tomo_tokens:
#     for mb in settings.mb_tokens[tomo]:
#         print tomo, mb
#         if mb != 'M8':
#             continue
#         for files in os.listdir(meshes_dir):
#             filename = os.path.join(meshes_dir, files)
#             tomo_token, mb_token = data_utils.get_tomo_and_mb_from_file_name(filename, settings)
#             if (tomo_token == tomo and mb_token == mb):
#                 break
#         cur_mesh = read_obj_file_to_triangles(filename)
#         gt_dict, orientation_dict = create_DL_data.read_GT_data_xml(gt_path_dict[tomo][mb], settings, return_orientation=True, return_ids=True)
#         for k, id in enumerate(gt_dict['PSII'][:, 3]):
#             print gt_dict['PSII'][k, :3] * 14.08
#             reg_mat = cur_mesh.triangles[int(id)].get_plane_matrix()
#             theta = np.deg2rad(orientation_dict['PSII'][k][0])
#             cur_matrix = np.array([[np.cos(theta), -1* np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
#             print np.dot(cur_matrix, reg_mat)







