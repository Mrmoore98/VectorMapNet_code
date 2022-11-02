import numpy as np
from mmdet.datasets.builder import PIPELINES
from shapely.geometry import LineString, box, Polygon
import av2.geometry.interpolate as interp_utils


@PIPELINES.register_module(force=True)
class VectorizeMap(object):
    def __init__(self, coords_dim=3, sample_num=20, sample_dist=-1):
        self.coords_dim = coords_dim
        self.sample_num = sample_num
        self.sample_dist = sample_dist
        assert (sample_dist > 0 and sample_num < 0) or (sample_dist < 0 and sample_num > 0)
        if sample_dist > 0:
            self.sample_fn = self.interp_fixed_dist
        else:
            self.sample_fn = self.interp_fixed_num

    def interp_fixed_num(self, line: LineString, backend='shapely'):
        # TODO: compare two solutions
        # solution 1:
        if backend == 'shapely':
            distances = np.linspace(0, line.length, self.sample_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) 
                for distance in distances]).squeeze()

        # solution 2:
        elif backend == 'argo':
            sampled_points = interp_utils.interp_arc(self.sample_num, np.array(list(line.coords)))

        return sampled_points

    def interp_fixed_dist(self, line: LineString):
        distances = list(np.arange(self.sample_dist, line.length, self.sample_dist))
        # make sure at least two sample points when sample_dist > line.length
        distances = [0,] + distances + [line.length,] 
        
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points
    
    def get_vectorized_lines(self, map_geoms):
        vectors = {}
        for label, geom_list in map_geoms.items():
            vectors[label] = []
            for geom in geom_list:
                if geom.geom_type == 'LineString':
                    line = self.sample_fn(geom)
                    line = line[:, :self.coords_dim]
                    vectors[label].append(line)

                elif geom.geom_type == 'Polygon':
                    # polygon objects will not be vectorized
                    continue
                
                else:
                    raise ValueError('map geoms must be either LineString or Polygon!')
        return vectors
    
    def __call__(self, input_dict):
        map_geoms = input_dict['map_geoms'] # {0: List[ped_crossing: LineString], 1: ...}
        
        '''
        Dict: {label: vector_list(np Array),
            e.g.
            0: [array([[x1, y1], [x2, y2]]), array([[x3, y3], [x4, y4], [x5, y5]])],
            1: ...
        }
        '''
        input_dict['vectors'] = self.get_vectorized_lines(map_geoms)
        return input_dict