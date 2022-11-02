import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Arrow, Rectangle
from matplotlib.patches import Polygon as mPolygon
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.map_expansion.map_api import NuScenesMapExplorer
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from typing import Dict, List, Optional, Tuple, Union


class CNuScenesMapExplorer(NuScenesMapExplorer):
    def __ini__(self, *args, **kwargs):
        super(self, CNuScenesMapExplorer).__init__(*args, **kwargs)

    def render_map_patch(self,
                         box_coords: Tuple[float, float, float, float],
                         layer_names: List[str] = None,
                         alpha: float = 0.5,
                         figsize: Tuple[float, float] = (15, 15),
                         render_egoposes_range: bool = True,
                         render_legend: bool = True,
                         bitmap: Optional[BitMap] = None) -> Tuple[Figure, Axes]:
        """
        Renders a rectangular patch specified by `box_coords`. By default renders all layers.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param layer_names: All the non geometric layers that we want to render.
        :param alpha: The opacity of each layer.
        :param figsize: Size of the whole figure.
        :param render_egoposes_range: Whether to render a rectangle around all ego poses.
        :param render_legend: Whether to render the legend of map layers.
        :param bitmap: Optional BitMap object to render below the other map layers.
        :return: The matplotlib figure and axes of the rendered layers.
        """
        x_min, y_min, x_max, y_max = box_coords.bounds

        if layer_names is None:
            layer_names = self.map_api.non_geometric_layers

        fig = plt.figure(figsize=figsize)

        local_width = x_max - x_min
        local_height = y_max - y_min
        assert local_height > 0, 'Error: Map patch has 0 height!'
        local_aspect_ratio = local_width / local_height

        ax = fig.add_axes([0, 0, 1, 1 / local_aspect_ratio])

        if bitmap is not None:
            bitmap.render(self.map_api.canvas_edge, ax)

        for layer_name in layer_names:
            self._render_layer(ax, layer_name, alpha)

        x_margin = np.minimum(local_width / 4, 50)
        y_margin = np.minimum(local_height / 4, 10)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        if render_egoposes_range:
            bbox = np.array(box_coords.exterior.coords)
            WH = np.linalg.norm(bbox[1:3]-bbox[:2], axis=-1)
            W, iw = max(WH), np.argmax(WH)
            H, ih = min(WH), np.argmin(WH)

            ax.add_patch(mPolygon(np.array(box_coords.exterior.coords), fill=False, linestyle='-.', color='red',
                                  lw=2))
            ax.text(bbox[ih:2+ih, 0].mean(), bbox[ih:2+ih, 1].mean(), "%g m" % H,
                    fontsize=14, weight='bold')
            ax.text(bbox[iw:2+iw, 0].mean(), bbox[iw:2+iw, 1].mean(), "%g m" % W,
                    fontsize=14, weight='bold')

        if render_legend:
            ax.legend(frameon=True, loc='upper right')

        return fig, ax

    def _get_layer_polygon(self,
                           patch_box: Tuple[float, float, float, float],
                           patch_angle: float,
                           layer_name: str,
                           return_token: bool = False) -> Tuple[List[Polygon], Optional[List[str]]]:
        """
         Retrieve the polygons of a particular layer within the specified patch.
         :param patch_box: Patch box defined as [x_center, y_center, height, width].
         :param patch_angle: Patch orientation in degrees.
         :param layer_name: name of map layer to be extracted.
         :return: List of Polygon in a patch box.
         """
        if layer_name not in self.map_api.non_geometric_polygon_layers+['lane_connector']:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_api, layer_name)

        polygon_list = []
        polygon_tokens_list = []
        valid_records = []
        if layer_name == 'drivable_area':
            for record in records:

                for polygon_token in record['polygon_tokens']:

                    polygon = self.map_api.extract_polygon(polygon_token)
                    new_polygon = polygon.intersection(patch)

                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type is 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)
                        polygon_tokens_list.append(polygon_token)
        else:
            for record in records:
                if record['polygon_token'] is None:
                    # import ipdb
                    # ipdb.set_trace()
                    continue
                polygon = self.map_api.extract_polygon(record['polygon_token'])

                # if polygon.intersects(patch) or polygon.within(patch):
                #     if not polygon.is_valid:
                #         print('within: {}, intersect: {}'.format(polygon.within(patch), polygon.intersects(patch)))
                #         print('polygon token {} is_valid: {}'.format(record['polygon_token'], polygon.is_valid))

                # polygon = polygon.buffer(0)

                if polygon.is_valid:
                    # if within or intersect :

                    new_polygon = polygon.intersection(patch)
                    # new_polygon = polygon

                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type is 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)
                        polygon_tokens_list.append(record['polygon_token'])
                        # print('polygon token accepeted:', record['polygon_token'])
                        if layer_name in '(lane,road_block)':
                            record['from_edge_line'] = \
                                to_patch_coord(self.map_api.extract_line(
                                    record['from_edge_line_token']), patch_angle, patch_x, patch_y)
                            record['to_edge_line'] = \
                                to_patch_coord(self.map_api.extract_line(
                                    record['to_edge_line_token']), patch_angle, patch_x, patch_y)

                        if layer_name in ['lane', 'lane_connector']:
                            centerline = self.map_api.discretize_lanes(
                                    record, 0.5)
                            centerline = list(self.map_api.discretize_lanes([record['token']], 0.5).values())[0]
                            centerline = LineString(np.array(centerline)[:,:2])
                            centerline = centerline.intersection(patch)
                            record['centerline'] = \
                                to_patch_coord(centerline, patch_angle, patch_x, patch_y)

                        valid_records.append(record)

        if return_token:
            return polygon_list, polygon_tokens_list, valid_records

        return polygon_list

    def _get_layer_line(self,
                        patch_box: Tuple[float, float, float, float],
                        patch_angle: float,
                        layer_name: str,
                        return_token: bool = False) -> Optional[List[LineString]]:
        """
        Retrieve the lines of a particular layer within the specified patch.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :param layer_name: name of map layer to be converted to binary map mask patch.
        :return: List of LineString in a patch box.
        """
        if layer_name not in self.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name is 'traffic_light':
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.get_patch_coord(patch_box, patch_angle)

        line_list = []
        line_tokens_list = []
        valid_records = []
        records = getattr(self.map_api, layer_name)
        for record in records:
            line = self.map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(
                    new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)
                line_tokens_list.append(record['line_token'])
                valid_records.append(record)

        if return_token:
            return line_list, line_tokens_list, valid_records

        return line_list


def to_patch_coord(new_polygon, patch_angle, patch_x, patch_y):
    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                  origin=(patch_x, patch_y), use_radians=False)
    new_polygon = affinity.affine_transform(new_polygon,
                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
    return new_polygon


# utils
def plot(polygon, buffer=1):
    from descartes.patch import PolygonPath, PolygonPatch
    fig, ax = plt.subplots()
    ax.add_patch(PolygonPatch(polygon.buffer(buffer),
                 fill=False, fc='orange', ec='blue', alpha=0.5))
    xy = np.array(polygon.buffer(buffer).exterior.coords)
    ax.plot(xy[:, 0], xy[:, 1], 'o', color='orange')

    ax.add_patch(PolygonPatch(polygon, fill=False,
                 fc='red', ec='red', alpha=0.5))
    xy = np.array(polygon.exterior.coords)
    ax.plot(xy[:, 0], xy[:, 1], 'o', color='red')
    fig.savefig('test_buffer.png', dpi=200)
