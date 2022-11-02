import os
import os.path as osp
import time

import mmcv
import numpy as np
from IPython import embed
from mmdet.datasets import DATASETS

from .base_dataset import BaseMapDataset
from .evaluation.precision_recall.average_precision_gen import eval_chamfer


@DATASETS.register_module()
class NuscDataset(BaseMapDataset):
    def __init__(self,
                 ann_file,
                 data_root,
                 cat2id,
                 roi_size,
                 modality=dict(
                     use_camera=True,
                     use_lidar=False,
                     use_radar=False,
                     use_map=True,
                     use_external=False,
                 ),
                 pipeline=None,
                 coord_dim=3,
                 interval=1,
                 work_dir=None,
                 eval_cfg: dict = dict(),
                 **kwargs,
                 ):
        super().__init__(
            ann_file,
            modality=modality,
            pipeline=pipeline,
            cat2id=cat2id,
            interval=interval,
        )
        self.roi_size = roi_size
        self.coord_dim = coord_dim
        self.eval_cfg = eval_cfg

        # dummy flag to fit with mmdet
        self.flag = np.zeros(len(self), dtype=np.uint8)
        # self.map_extractor = NuscMapExtractor(data_root, self.roi_size)
        self.work_dir = work_dir

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        print('collecting samples...')
        start_time = time.time()
        ann = mmcv.load(ann_file)
        samples = ann[::self.interval]
        print(
            f'collected {len(samples)} samples in {(time.time() - start_time):.2f}s')
        self.samples = samples

    def get_sample(self, idx):
        '''
        aaa
        '''

        sample = self.samples[idx]
        location = sample['location']

        ego2img_rts = []
        for c in sample['cams'].values():
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)

        input_dict = {
            # for nuscenes, the order is
            # 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            # 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            'sample_idx': sample['token'],
            'location': location,
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            # intrinsics are 3x3 Ks
            'cam_intrinsics': [c['intrinsics'] for c in sample['cams'].values()],
            # extrinsics are 4x4 tranform matrix, **ego2cam**
            'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            # 'map_geoms': map_label2geom, # {0: List[ped_crossing(LineString)], 1: ...}
            'ego2global_translation': sample['e2g_translation'],
            'ego2global_rotation': sample['e2g_rotation'],
        }

        if self.modality['use_lidar']:
            input_dict.update(
                dict(
                    pts_filename=sample['lidar_path'],
                )
            )

        return input_dict

    def format_results(self, results, name, prefix=None, patch_size=(60, 30), origin=(0, 0)):

        meta = self.modality
        submissions = {
            'meta': meta,
            'results': {},
            "groundTruth": {},  # for validation
        }
        patch_size = np.array(patch_size)
        origin = np.array(origin)

        for case in mmcv.track_iter_progress(results):
            '''
                vectorized_line {
                    "pts":               List[<float, 2>]  -- Ordered points to define the vectorized line.
                    "pts_num":           <int>,            -- Number of points in this line.
                    "type":              <0, 1, 2>         -- Type of the line: 0: ped; 1: divider; 2: boundary
                    "confidence_level":  <float>           -- Confidence level for prediction (used by Average Precision)
                }
            '''

            if case is None:
                continue

            vector_lines = []
            for i in range(case['nline']):
                vector = case['lines'][i] * patch_size + origin
                vector_lines.append({
                    'pts': vector,
                    'pts_num': len(case['lines'][i]),
                    'type': case['labels'][i],
                    'confidence_level': case['scores'][i],
                })
                submissions['results'][case['token']] = {}
                submissions['results'][case['token']]['vectors'] = vector_lines

            if 'groundTruth' in case:

                submissions['groundTruth'][case['token']] = {}
                vector_lines = []
                for i in range(case['groundTruth']['nline']):
                    line = case['groundTruth']['lines'][i] * \
                        patch_size + origin

                    vector_lines.append({
                        'pts': line,
                        'pts_num': len(case['groundTruth']['lines'][i]),
                        'type': case['groundTruth']['labels'][i],
                        'confidence_level': 1.,
                    })
                submissions['groundTruth'][case['token']
                                           ]['vectors'] = vector_lines

        # Use pickle format to minimize submission file size.
        print('Done!')
        mmcv.mkdir_or_exist(prefix)
        res_path = os.path.join(prefix, '{}.pkl'.format(name))
        mmcv.dump(submissions, res_path)

        return res_path

    def evaluate(self,
                 results,
                 logger=None,
                 name=None,
                 **kwargs):
        '''
        Args:
            results (list[Tensor]): List of results.
            eval_cfg (Dict): Config of test dataset.
            output_format (str): Model output format, should be either 'raster' or 'vector'.

        Returns:
            dict: Evaluation results.
        '''

        print('len of the results', len(results))
        name = 'results_nuscence' if name is None else name
        result_path = self.format_results(
            results, name, prefix=self.work_dir, patch_size=self.eval_cfg.patch_size, origin=self.eval_cfg.origin)

        self.eval_cfg.evaluation_cfg['result_path'] = result_path
        self.eval_cfg.evaluation_cfg['ann_file'] = self.ann_file

        mean_ap = eval_chamfer(
            self.eval_cfg.evaluation_cfg, update=True, logger=logger)

        result_dict = {
            'mAP': mean_ap,
        }

        print('VectormapNet Evaluation Results:')
        print(result_dict)

        return result_dict
