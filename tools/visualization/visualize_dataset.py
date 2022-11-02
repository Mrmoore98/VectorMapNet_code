import os
from typing import Dict
import numpy as np
import mmcv
from IPython import embed
from mmcv import Config
from mmdet3d.datasets import build_dataset
from renderer import Renderer
from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path
from shapely.geometry import Polygon, box, MultiPolygon
from shapely import affinity, ops
import av2.rendering.vector as vector_plotting_utils
import matplotlib.pyplot as plt

CAM_NAMES_AV2 = ['ring_front_center', 'ring_front_right', 'ring_front_left',
    'ring_rear_right','ring_rear_left', 'ring_side_right', 'ring_side_left',
    ]

CAM_NAMES_NUSC = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

def import_plugin(cfg):
    import sys
    sys.path.append(os.path.abspath('.'))    
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            
            def import_path(plugin_dir):
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            plugin_dirs = cfg.plugin_dir
            if not isinstance(plugin_dirs, list):
                plugin_dirs = [plugin_dirs,]
            for plugin_dir in plugin_dirs:
                import_path(plugin_dir)

def get_drivable_areas(data_root, split):
    data_root = os.path.join(data_root, split)
    logs = os.listdir(data_root)
    cities = {}
    for log in logs:
        map_dir = os.path.join(data_root, log, 'map')
        map_json = str(list(Path(map_dir).glob("log_map_archive_*.json"))[0])
        city = map_json.split('____')[-1].split('_')[0]
        avm = ArgoverseStaticMap.from_json(Path(map_json))

        for _, da in avm.vector_drivable_areas.items():
            polygon_xyz = da.xyz[:, :2]
            polygon = Polygon(polygon_xyz)

            if city not in cities.keys():
                cities[city] = []
            cities[city].append(polygon)
    
    return cities

def visualize_whole_city(data_root='/nvme/argoverse2/sensor/'):
    cities_train = get_drivable_areas(data_root, 'train')
    cities_val = get_drivable_areas(data_root, 'val')
    cities_test = get_drivable_areas(data_root, 'test')
    
    val_area1 = box(0, -2000, 1200, 5000)
    val_area2 = box(4700, -2000, 5200, 5000)
    val_area3 = box(7000, -2000, 9000, 5000)
    val_area = MultiPolygon([val_area1, val_area2, val_area3])


    for city in cities_train.keys():
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        for p in cities_train[city]:
            vector_plotting_utils.plot_polygon_patch_mpl(np.array(p.exterior.coords), ax, color='r', alpha=0.4)
        for p in cities_val[city]:
            vector_plotting_utils.plot_polygon_patch_mpl(np.array(p.exterior.coords), ax, color='g', alpha=0.4)
        # for p in cities_test[city]:
        #     vector_plotting_utils.plot_polygon_patch_mpl(np.array(p.exterior.coords), ax, color='g', alpha=0.5)

        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(f"./vis/drivable_areas_{city}.jpg", dpi=500)
    
    city = 'PIT'
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    cnt = 0
    for p in (cities_train[city] + cities_val[city]):
        if p.intersection(val_area).is_empty:
            cnt += 1
            vector_plotting_utils.plot_polygon_patch_mpl(np.array(p.exterior.coords), ax, color='r', alpha=0.4)
        else:
            vector_plotting_utils.plot_polygon_patch_mpl(np.array(p.exterior.coords), ax, color='g', alpha=0.4)
    
    print(f"{cnt}/{len(cities_train[city] + cities_val[city])}")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"./vis/drivable_areas_{city}_resplit.jpg", dpi=500)
    plt.close("all")


def vectors_to_pcd(vectors):
    def _write_obj(points, out_filename):
        """Write points into ``obj`` format for meshlab visualization.

        Args:
            points (np.ndarray): Points in shape (N, dim).
            out_filename (str): Filename to be saved.
        """
        N = points.shape[0]
        fout = open(out_filename, 'w')
        for i in range(N):
            if points.shape[1] == 6:
                c = points[i, 3:].astype(int)
                fout.write(
                    'v %f %f %f %d %d %d\n' %
                    (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

            else:
                fout.write('v %f %f %f\n' %
                        (points[i, 0], points[i, 1], points[i, 2]))
        fout.close()
    
    COLOR_MAPS_RGB = {
        # bgr colors
        0: (0, 0, 255),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (51, 183, 255),
    }
    pcd = []
    for label, v_list in vectors.items():
        for v in v_list:
            for pts in v:
                color = COLOR_MAPS_RGB[label]
                pcd.append([pts[0], pts[1], pts[2], color[0], color[1], color[2]])
    
    pcd = np.array(pcd)
    _write_obj(pcd, 'pcd.obj')

if __name__ == '__main__':
    # visualize_whole_city()
    cfg = Config.fromfile('plugin/configs/debug_nusc.py')
    import_plugin(cfg)

    dataset = build_dataset(cfg.data.val)
    for i in mmcv.track_iter_progress(range(len(dataset))):
        data = dataset[i]
    # data = dataset[41]

    # imgs = data['img']
    # vectors = data['vectors']
    # semantic_mask = data['semantic_mask']
    # intrinsics = data['cam_intrinsics']
    # extrinsics = data['cam_extrinsics']
    
    # cat2id = cfg.cat2id
    # roi_size = cfg.roi_size
    # renderer = Renderer(cat2id, roi_size, CAM_NAMES_NUSC)
    # out_dir = './vis'
    # os.makedirs(out_dir, exist_ok=True)
    # renderer.render_camera_views_from_vectors(vectors, imgs, extrinsics, intrinsics, thickness=3, out_dir=out_dir)
    # renderer.render_bev_from_vectors(vectors, out_dir=out_dir)
    # renderer.render_bev_from_mask(semantic_mask, out_dir=out_dir)