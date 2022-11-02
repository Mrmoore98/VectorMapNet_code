import mmcv
import numpy as np
import tempfile
import warnings
from os import path as osp
from torch.utils.data import Dataset
from mmdet3d.datasets.utils import extract_result_dict, get_loading_pipeline

from mmdet.datasets import DATASETS

from mmdet3d.datasets.pipelines import Compose
import time


@DATASETS.register_module()
class BaseMapDataset(Dataset):
    """BaseClass for Map Dataset

    This is the base dataset of nuScenes and argoverse 2dataset.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    def __init__(self,
                 ann_file,
                 modality=dict(
                     use_camera=True,
                     use_lidar=False,
                     use_radar=False,
                     use_map=True,
                     use_external=False,
                 ),
                 pipeline=None,
                 cat2id=None,
                 interval=1,):
        super().__init__()
        self.ann_file = ann_file
        self.modality = modality
        
        self.classes = list(cat2id.keys())
        self.num_classes = len(self.classes)
        self.cat2id = cat2id
        self.interval = interval

        # 
        self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None
            
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        print('collecting samples...')
        start_time = time.time()
        samples = mmcv.load(ann_file)[::self.interval]
        print(f'collected {len(samples)} samples in {(time.time() - start_time):.2f}s')
        self.samples = samples

    def get_sample(self, index):
        info = self.samples[index]
        
        input_dict = dict(
            info=info,
        )

        return input_dict

    def prepare_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_sample(index)
        example = self.pipeline(input_dict)
        return example

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results, \
                tmp_dir is the temporal directory created for saving json \
                files when ``jsonfile_prefix`` is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
            out = f'{pklfile_prefix}.pkl'
        mmcv.dump(outputs, out)
        return outputs, tmp_dir

    def evaluate(self,
                 results,
                 logger=None,
                 show=True,
                 **kwargs):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.

        Returns:
            dict: Evaluation results.
        """
        if show:
            print('show results')  # or call another function
        
        ret_dict = {}
        return ret_dict

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.samples)
        
    def _rand_another(self, idx):
        """Randomly get another item.

        Returns:
            int: Another index of item.
        """
        return np.random.choice(self.__len__)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        data = self.prepare_data(idx)
        return data
