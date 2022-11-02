import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely import affinity, ops
from shapely.geometry import (LineString, MultiLineString, MultiPolygon, Point,
                              Polygon, box, polygon)

try:
    from ..nuscences_utils.map_api import CNuScenesMapExplorer
except:
    from nuscences_utils.map_api import CNuScenesMapExplorer

import warnings

import networkx as nx
from shapely.strtree import STRtree

warnings.filterwarnings("ignore")


@PIPELINES.register_module(force=True)
class VectorizeLocalMap(object):

    def __init__(self,
                 data_root="/mnt/datasets/nuScenes/",
                 patch_size=(30, 60),
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment','lane'],
                 centerline_class=['lane_connector','lane'],
                 sample_dist=10,
                 num_samples=250,
                 padding=True,
                 max_len=30,
                 normalize=True,
                 fixed_num=50,
                 sample_pts=True,
                 class2label={
                     'ped_crossing': 0,
                     'divider': 1,
                     'contours': 2,
                     'others': -1,
                 }, 
                 **kwargs):
        '''
        Args:
            fixed_num = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = data_root
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.contour_classes = contour_classes
        self.centerline_class = centerline_class


        self.class2label = class2label
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = CNuScenesMapExplorer(self.nusc_maps[loc])

        self.layer2class = {
            'ped_crossing': 'ped_crossing',
            'lane_divider': 'divider',
            'road_divider': 'divider',
            'road_segment': 'contours',
            'lane': 'contours',
        }


        self.process_func = {
            'ped_crossing': self.ped_geoms_to_vectors,
            'divider': self.line_geoms_to_vectors,
            'contours': self.poly_geoms_to_vectors,
            'centerline': self.line_geoms_to_vectors,
        }

        self.colors = {
            # 'ped_crossing': 'blue',
            'ped_crossing': 'royalblue',
            'divider': 'orange',
            'contours': 'green',
            # origin type
            'lane_divider': 'orange',
            'road_divider': 'orange',
            'road_segment': 'green',
            'lane': 'green',
        }

        self.sample_pts = sample_pts

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.max_len = max_len
        self.normalize = normalize
        self.fixed_num = fixed_num
        self.size = np.array([self.patch_size[1], self.patch_size[0]]) + 2


    def retrive_geom(self, patch_params):
        '''
            Get the geometric data.
            Returns: dict
        '''
        patch_box, patch_angle, location = patch_params
        geoms_dict = {}

        layers = \
            self.line_classes + self.ped_crossing_classes + \
            self.contour_classes

        layers = set(layers)
        for layer_name in layers:

            return_token = False
            # retrive the geo
            if layer_name in self.nusc_maps[location].non_geometric_line_layers:
                geoms = self.map_explorer[location]._get_layer_line(
                    patch_box, patch_angle, layer_name)
            elif layer_name in self.nusc_maps[location].lookup_polygon_layers:
                geoms = self.map_explorer[location]._get_layer_polygon(
                    patch_box, patch_angle, layer_name, return_token=return_token)
            else:
                raise ValueError('{} is not a valid layer'.format(layer_name))

            if geoms is None:
                continue

            # change every geoms set to list
            if not isinstance(geoms, list):
                geoms = [geoms, ]

            geoms_dict[layer_name] = geoms

        return geoms_dict

    def union_geoms(self, geoms_dict):

        customized_geoms_dict = {}

        # contour
        roads = geoms_dict['road_segment']
        lanes = geoms_dict['lane']
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])

        customized_geoms_dict['contours'] = ('contours', [union_segments, ])

        # ped
        geoms_dict['ped_crossing'] = self.union_ped(geoms_dict['ped_crossing'])

        for layer_name, custom_class in self.layer2class.items():

            if custom_class == 'contours':
                continue

            customized_geoms_dict[layer_name] = (
                custom_class, geoms_dict[layer_name])

        return customized_geoms_dict

    def union_ped(self, ped_geoms):

        def get_rec_direction(geom):
            rect = geom.minimum_rotated_rectangle
            rect_v_p = np.array(rect.exterior.coords)[:3]
            rect_v = rect_v_p[1:]-rect_v_p[:-1]
            v_len = np.linalg.norm(rect_v, axis=-1)
            longest_v_i = v_len.argmax()

            return rect_v[longest_v_i], v_len[longest_v_i]

        tree = STRtree(ped_geoms)
        index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))

        final_pgeom = []
        remain_idx = [i for i in range(len(ped_geoms))]
        for i, pgeom in enumerate(ped_geoms):

            if i not in remain_idx:
                continue
            # update
            remain_idx.pop(remain_idx.index(i))
            pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
            final_pgeom.append(pgeom)

            for o in tree.query(pgeom):
                o_idx = index_by_id[id(o)]
                if o_idx not in remain_idx:
                    continue

                o_v, o_v_norm = get_rec_direction(o)
                cos = pgeom_v.dot(o_v)/(pgeom_v_norm*o_v_norm)
                if 1 - np.abs(cos) < 0.01:  # theta < 8 degrees.
                    final_pgeom[-1] =\
                        final_pgeom[-1].union(o)
                    # update
                    remain_idx.pop(remain_idx.index(o_idx))

        for i in range(len(final_pgeom)):
            if final_pgeom[i].geom_type != 'MultiPolygon':
                final_pgeom[i] = MultiPolygon([final_pgeom[i]])

        return final_pgeom

    def convert2vec(self, geoms_dict: dict, sample_pts=False, override_veclen: int = None):

        vector_dict = {}
        for layer_name, (customized_class, geoms) in geoms_dict.items():

            line_strings = self.process_func[customized_class](geoms)

            vector_len = self.fixed_num[customized_class]
            if override_veclen is not None:
                vector_len = override_veclen

            vectors = self._geom_to_vectors(
                line_strings, customized_class, vector_len, sample_pts)
            vector_dict.update({layer_name: (customized_class, vectors)})

        return vector_dict

    def _geom_to_vectors(self, line_geom, label, vector_len, sample_pts=False):
        '''
            transfrom the geo type 2 line vectors
        '''
        line_vectors = {'vectors': [], 'length': []}
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for l in line:
                        if sample_pts:
                            v, nl = self._sample_pts_from_line(
                                l, label, vector_len)
                        else:
                            v, nl = self._geoms2pts(l, label, vector_len)
                        line_vectors['vectors'].append(v.astype(np.float))
                        line_vectors['length'].append(nl)
                elif line.geom_type == 'LineString':
                    if sample_pts:
                        v, nl = self._sample_pts_from_line(
                            line, label, vector_len)
                    else:
                        v, nl = self._geoms2pts(line, label, vector_len)
                    line_vectors['vectors'].append(v.astype(np.float))
                    line_vectors['length'].append(nl)
                else:
                    raise NotImplementedError

        return line_vectors

    def poly_geoms_to_vectors(self, polygon_geoms: list):

        results = []
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []

        for geom in polygon_geoms:
            for poly in geom:
                exteriors.append(poly.exterior)
                for inter in poly.interiors:
                    interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            # since the start and end will disjoint
            # after applying the intersection.
            if lines.type != 'LineString':
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if lines.type != 'LineString':
                lines = ops.linemerge(lines)
            results.append(lines)

        return results

    def ped_geoms_to_vectors(self, geoms: list):

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for geom in geoms:
            for ped_poly in geom:
                # rect = ped_poly.minimum_rotated_rectangle
                ext = ped_poly.exterior
                if not ext.is_ccw:
                    ext.coords = list(ext.coords)[::-1]
                lines = ext.intersection(local_patch)

                if lines.type != 'LineString':
                    lines = ops.linemerge(lines)

                # same instance but not connected.
                if lines.type != 'LineString':
                    ls = []
                    for l in lines.geoms:
                        ls.append(np.array(l.coords))

                    lines = np.concatenate(ls, axis=0)
                    lines = LineString(lines)

                results.append(lines)

        return results

    def line_geoms_to_vectors(self, geom):
        # XXX
        return geom

    def _geoms2pts(self, line, label, fixed_point_num):

        # if we still use the fix point
        if fixed_point_num > 0:
            remain_points = fixed_point_num - np.asarray(line.coords).shape[0]
            if remain_points < 0:

                tolerance = 0.4
                while np.asarray(line.coords).shape[0] > fixed_point_num:
                    line = line.simplify(tolerance, preserve_topology=True)
                    tolerance += 0.2

                remain_points = fixed_point_num - \
                    np.asarray(line.coords).shape[0]
                if remain_points > 0:
                    line = self.pad_line_with_interpolated_line(
                        line, remain_points)

            elif remain_points > 0:

                line = self.pad_line_with_interpolated_line(
                    line, remain_points)

            v = line
            if not isinstance(v, np.ndarray):
                v = np.asarray(line.coords)

            valid_len = v.shape[0]

        elif self.padding:  # dynamic points

            if self.max_len < np.asarray(line.coords).shape[0]:

                tolerance = 0.4
                while np.asarray(line.coords).shape[0] > self.max_len:
                    line = line.simplify(tolerance, preserve_topology=True)
                    tolerance += 0.2

            v = np.asarray(line.coords)
            valid_len = v.shape[0]

            pad_len = self.max_len - valid_len
            v = np.pad(v, ((0, pad_len), (0, 0)), 'constant')

        else:
            # dynamic points without padding
            line = line.simplify(0.2, preserve_topology=True)
            v = np.array(line.coords)
            valid_len = len(v)

        if self.normalize:
            v = self.normalize_line(v)

        return v, valid_len

    def pad_line_with_interpolated_line(self, line: LineString, remain_points):
        ''' pad variable line with the interploated points'''

        origin_line = line
        line_length = line.length
        v = np.array(origin_line.coords)
        line_size = v.shape[0]

        interval = np.linalg.norm(v[1:]-v[:-1], axis=-1).cumsum()
        edges = np.hstack((np.array([0]), interval))/line_length

        # padding points
        interpolated_distances = np.linspace(
            0, 1, remain_points+2)[1:-1]  # get rid of start and end
        sampled_points = np.array([list(origin_line.interpolate(distance, normalized=True).coords)
                                   for distance in interpolated_distances]).reshape(-1, 2)

        # merge two line
        insert_idx = np.searchsorted(edges, interpolated_distances) - 1

        last_idx = 0
        new_line = []
        inserted_pos = np.unique(insert_idx)

        for i, idx in enumerate(inserted_pos):
            new_line += [v[last_idx:idx+1], sampled_points[insert_idx == idx]]
            last_idx = idx+1
        # for the remain points
        if last_idx <= line_size-1:
            new_line += [v[last_idx:], ]

        merged_line = np.concatenate(new_line, 0)

        return merged_line

    def _sample_pts_from_line(self, line, label, fixed_point_num):

        if fixed_point_num < 0:
            distances = list(np.arange(self.sample_dist,
                             line.length, self.sample_dist))
            distances = [0, ] + distances + [line.length, ]
            sampled_points = np.array([list(line.interpolate(distance).coords)
                                       for distance in distances]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num

            distances = np.linspace(0, line.length, fixed_point_num)
            sampled_points = np.array([list(line.interpolate(distance).coords)
                                       for distance in distances]).reshape(-1, 2)

        num_valid = len(sampled_points)

        # padding
        if fixed_point_num < 0 and self.padding:

            # fixed distance sampling need padding!
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate(
                    [sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

        if self.normalize:
            sampled_points = self.normalize_line(sampled_points)

        return sampled_points, num_valid

    def normalize_line(self, line):
        '''
            prevent extrime pts such as 0 or 1. 
        '''

        origin = -np.array([self.patch_size[1]/2, self.patch_size[0]/2])
        # for better learning
        line = line - origin
        line = line / self.size

        return line

    def get_global_patch(self, input_dict: dict):
        # transform to global coordination
        location = input_dict['location']
        ego2global_translation = input_dict['ego2global_translation']
        ego2global_rotation = input_dict['ego2global_rotation']
        map_pose = ego2global_translation[:2]
        rotation = Quaternion(ego2global_rotation)
        patch_box = (map_pose[0], map_pose[1],
                     self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        patch_params = (patch_box, patch_angle, location)
        return patch_params

    def vectorization(self, input_dict: dict):

        patch_params = self.get_global_patch(input_dict)

        # Retrive geo
        geoms_dict = self.retrive_geom(patch_params)
        # self.debug_vis(patch_params, geoms_dict=geoms_dict, orgin=False)

        # Optional union the data and convert customized labels
        geoms_dict = self.union_geoms(geoms_dict)
        # self.debug_vis(patch_params, geoms_dict=geoms_dict, origin=False, token=input_dict['token'])

        # Convert Geo 2 vec
        vectors_dict = self.convert2vec(geoms_dict, self.sample_pts)
        # self.debug_vis(patch_params, vectors_dict=vectors_dict,
        #                origin=False, token=input_dict['token'])

        # format the outputs list
        vectors = []
        for k, (custom_class, v) in vectors_dict.items():

            label = self.class2label.get(custom_class, -1)
            # filter out -1
            if label == -1:
                continue

            for vec, l in zip(v['vectors'], v['length']):

                vectors.append((vec, l, label))

        input_dict['vectors'] = vectors

        return input_dict

    def __call__(self, input_dict: dict):

        input_dict = self.vectorization(input_dict)

        return input_dict


def get_start_name(i):
    return str(i)+'_start'


def get_end_name(i):
    return str(i)+'_end'
