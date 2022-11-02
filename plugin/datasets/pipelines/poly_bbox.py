import networkx as nx
from random import random
import mmcv
import numpy as np
import time

from mmdet.datasets.builder import PIPELINES
from shapely.geometry import LineString, Polygon, Point
from shapely.geometry import CAP_STYLE, JOIN_STYLE


def evaluate_lin_curvature(polyline):

    n = polyline.shape[0]
    if n == 2:
        return None

    # len - 2, 3, 2
    tript_pts = np.stack([polyline[:-2], polyline[1:-1], polyline[2:]], axis=1)

    # len - 2
    curvature = cal_curvature(tript_pts.transpose(1, 0, 2))

    denominator = curvature.sum()
    denominator = 1 if denominator == 0 else denominator

    weight = curvature/denominator * ((n-2)/n)
    print('curvature: ', weight)

    return weight


def evaluate_line(polyline, curvature=False):

    edge = np.linalg.norm(polyline[1:] - polyline[:-1], axis=-1)

    start_end_weight = edge[(0, -1), ].copy()
    mid_weight = (edge[:-1] + edge[1:]) * .5

    pts_weight = np.concatenate(
        (start_end_weight[:1], mid_weight, start_end_weight[-1:]))

    denominator = pts_weight.sum()
    denominator = 1 if denominator == 0 else denominator

    pts_weight /= denominator

    if curvature and polyline.shape[0] > 2:
        weight_c = evaluate_lin_curvature(polyline)

        pts_weight_c = pts_weight.copy()
        pts_weight_c[1:-1] = weight_c

        pts_weight = (pts_weight+pts_weight_c)/2

    # add weights for stop index
    pts_weight = np.repeat(pts_weight, 2)/2
    pts_weight = np.pad(pts_weight, ((0, 1)),
                        constant_values=1/(len(polyline)*2))

    return pts_weight


def quantize_verts(
        verts,
        canvas_size=(400, 200, 100),
        coord_dim=3,
):
    """Convert vertices from its original range ([-1,1]) to discrete values in [0, n_bits**2 - 1].
        Args:
            verts: seqlen, 2
    """
    min_range = 0
    max_range = 1
    range_quantize = np.array(canvas_size) - 1  # (0-199) = 200

    verts_ratio = (verts - min_range) / (
        max_range - min_range)
    verts_quantize = verts_ratio * range_quantize[:coord_dim]

    return verts_quantize.astype('int32')


def get_bbox(
        polyline_nd, mode='xyxy', threshold=6, num_points=10, random=False):
    '''
        polyline: seq_len, coord_dim
    '''

    if mode == 'xyxy':
        polyline = LineString(polyline_nd)
        bbox = polyline.bounds
        minx, miny, maxx, maxy = bbox
        W, H = maxx-minx, maxy-miny

        if W < threshold or H < threshold:
            remain = (threshold - min(W, H))/2
            bbox = polyline.buffer(remain).envelope.bounds
            minx, miny, maxx, maxy = bbox

        bbox_np = np.array([[minx, miny], [maxx, maxy]])
        bbox_np = np.clip(bbox_np, 0., 1.)

    elif mode == 'sample':
        polyline = LineString(polyline_nd)
        if random:
            distances = np.random.uniform(
                0-polyline.length*0.2, polyline.length*1.2, (num_points,))
            distances = np.sort(distances)
        else:
            distances = np.linspace(0, polyline.length, num_points)

        sampled_points = np.array([list(polyline.interpolate(distance).coords)
                                   for distance in distances]).reshape(-1, 2)
        bbox_np = sampled_points

    return bbox_np


@PIPELINES.register_module(force=True)
class PolygonizeLocalMapBbox(object):

    def __init__(self,
                 canvas_size=(400, 200),
                 coord_dim=2,
                 num_class=3,
                 mode='xyxy',
                 centerline_mode='xyxy',
                 num_point=10,
                 threshold=6/200,
                 debug=False,
                 test_mode=False,
                 flatten=True,
                 ):

        self.canvas_size = np.array(canvas_size)

        self.mode = mode
        self.centerline_mode = centerline_mode

        self.num_class = num_class

        self.debug = debug

        # for keypoints
        self.num_point = num_point
        self.threshold = threshold

        self.coord_dim = coord_dim

        self.map_stop_idx = 0
        if not flatten:
            self.coord_dim_start_idx = 1
        else:
            self.coord_dim_start_idx = \
                self.num_class + 1  # for eos

        self.flatten = flatten
        self.test_mode = test_mode

    def reorder(self, vectors):
        # TODO
        return vectors

    def format_polyline_map(self, vectors):

        polylines, polyline_masks, polyline_weights = [], [], []

        # quantilize each label's lines individually.
        for vector_data in vectors:

            if len(vector_data) == 3:
                polyline, valid_len, label = vector_data
            elif len(vector_data) == 4:
                polyline, valid_len, label, line_type = vector_data

            # and pad polyline.
            if label == 2:
                polyline_weight = evaluate_line(polyline).reshape(-1)
            else:
                polyline_weight = np.ones_like(polyline).reshape(-1)
                polyline_weight = np.pad(
                    polyline_weight, ((0, 1),), constant_values=1.)
                polyline_weight = polyline_weight/polyline_weight.sum()

            #flatten and quantilized
            fpolyline = quantize_verts(
                polyline, self.canvas_size, self.coord_dim)
            fpolyline = fpolyline.reshape(-1)

            # reindex starting from 1, and add a zero stopping token(EOS),
            fpolyline = \
                np.pad(fpolyline + self.coord_dim_start_idx, ((0, 1),),
                       constant_values=label+1 if self.flatten else 0)
            fpolyline_msk = np.ones(fpolyline.shape, dtype=np.bool)

            polyline_masks.append(fpolyline_msk)
            polyline_weights.append(polyline_weight)
            polylines.append(fpolyline)

        if self.flatten:
            polyline_map = np.concatenate(polylines)
            polyline_map_mask = np.concatenate(polyline_masks)
            polyline_map_weights = np.concatenate(polyline_weights)
            polyline_map = np.pad(polyline_map, ((0, 1),))
            polyline_map_mask = np.pad(
                polyline_map_mask, ((0, 1),), constant_values=1)
            polyline_map_weights = np.pad(
                polyline_map_weights, ((0, 1),),
                constant_values=polyline_map_weights.sum()/len(polylines))
        else:
            polyline_map = polylines
            polyline_map_mask = polyline_masks
            polyline_map_weights = polyline_weights

        return polyline_map, polyline_map_mask, polyline_map_weights

    def format_keypoint(self, vectors, centerline=False, centerline_label=None):

        kps, kp_labels = [], []
        qkps, qkp_masks = [], []
        mode = self.mode if not centerline else 'centerline_keypoint'

        # quantilize each label's lines individually.
        for vector_data in vectors:

            if mode != 'centerline_keypoint':
                if len(vector_data) == 3:
                    polyline, valid_len, label = vector_data
                elif len(vector_data) == 4:
                    polyline, valid_len, label, line_type = vector_data
            else:
                polyline = vector_data
                label = centerline_label

            kp = get_bbox(polyline, mode, self.threshold, self.num_point)
            kps.append(kp)
            kp_labels.append(label)

            gkp = kp
            if not self.test_mode:
                gkp = get_bbox(polyline,
                               mode, self.threshold, self.num_point, random=True)

            # flatten and quantilized
            fkp = quantize_verts(gkp, self.canvas_size, self.coord_dim)
            fkp = fkp.reshape(-1)

            # Reindex starting from 1, and add a class token,
            if self.flatten:
                fkp = \
                    np.pad(fkp + self.coord_dim_start_idx, ((0, 1),),
                           constant_values=label+1 if not self.flatten else 0)
            fkps_msk = np.ones(fkp.shape, dtype=np.bool)

            qkp_masks.append(fkps_msk)
            qkps.append(fkp)

        if self.flatten:
            qkps = np.pad(
                np.concatenate(qkps), ((0, 1),))
            qkp_msks = np.pad(
                np.concatenate(qkp_masks), ((0, 1),), constant_values=1)
        else:
            qkps = np.stack(qkps)
            qkp_msks = np.stack(qkp_masks)

        # format det
        kps = np.stack(kps, axis=0).astype(np.float32)*self.canvas_size
        kp_labels = np.array(kp_labels)
        # restrict the boundary
        kps[..., 0] = np.clip(kps[..., 0], 0.1, self.canvas_size[0]-0.1)
        kps[..., 1] = np.clip(kps[..., 1], 0.1, self.canvas_size[1]-0.1)

        # nbox, boxsize(4)*coord_dim(2)
        kps = kps.reshape(kps.shape[0], -1)
        # unflatten_seq(qkps)

        return kps, kp_labels, qkps, qkp_msks,

    def Polygonization(self, input_dict: dict):
        '''
            Process mesh vertices and faces.
            Returns:
                vertices: Dict[ label, qflines_vectices ]
                flatten_lines: Dict[ label, qflines ]
        '''
        vectors = input_dict.pop('vectors')

        if not len(vectors):
            input_dict['polys'] = []
            return input_dict

        # TODO reorder vectors.
        vectors = self.reorder(vectors)

        polyline_map, polyline_map_mask, polyline_map_weight = \
            self.format_polyline_map(vectors)

        keypoint, keypoint_label, qkeypoint, qkeypoint_mask = \
            self.format_keypoint(vectors)

        # gather
        polys = {
            # for det
            'keypoint': keypoint,
            'det_label': keypoint_label,

            # for gen
            'gen_label': keypoint_label,
            'qkeypoint': qkeypoint,
            'qkeypoint_mask': qkeypoint_mask,

            # nlines(nbox) List[ seq_len*coord_dim ]
            'polylines': polyline_map,  # List[np.array]
            'polyline_masks': polyline_map_mask,  # List[np.array]
            'polyline_weights': polyline_map_weight,
        }

        # Format outputs
        input_dict['polys'] = polys

        return input_dict

    def __call__(self, input_dict: dict):

        input_dict = self.Polygonization(input_dict)
        return input_dict


def unflatten_seq(flat_polylines, class_num=3):
    """Converts from flat face sequence to a list of separate faces."""
    def group(seq):
        g = []
        for el in seq:
            if el == 0:
                yield g
                g = []
            elif 0 < el < class_num+1:
                g.append(el-1)
                yield g
                g = []
            else:
                g.append(el - (class_num+1))
        yield g

    outputs = list(group(flat_polylines))[:-1]
    # Remove empty faces
    return [o for o in outputs if len(o) > 1]


def unflatten_seq_by_others(target_polyline, flat_polylines, class_num=3):
    """Converts from flat face sequence to a list of separate faces."""
    def group(seq, tseq):
        g = []
        for el, tel in zip(seq, tseq):
            if el == 0:
                yield g
                g = []
            elif 0 < el < class_num+1:
                g.append(tel)
                yield g
                g = []
            else:
                g.append(tel)
        yield g

    outputs = list(group(flat_polylines, target_polyline))[:-1]
    # Remove empty faces
    return [o for o in outputs if len(o) > 1]