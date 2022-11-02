from multiprocessing import Pool
from shapely.geometry import LineString, Polygon
import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
import os
from functools import partial

from .tgfg import tpfp_gen

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def get_cls_results(gen_results, annotations, num_sample=100, class_id=0, fix_interval=False, coord_dim=2):
    """Get det results and gt information of a certain class.

    Args:
        gen_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes
    """
    # if len(gen_results) == 0 or 

    cls_gens, cls_scores = [], []
    for res in gen_results['vectors']:
        if res['type'] == class_id:
            if len(res['pts']) < 2:
                continue
            line = res['pts']
            line = LineString(line)

            if fix_interval:
                distances = list(np.arange(1., line.length, 1.))
                distances = [0,] + distances + [line.length,]
                sampled_points = np.array([list(line.interpolate(distance).coords)
                                        for distance in distances]).reshape(-1, coord_dim)
            else:
                distances = np.linspace(0, line.length, num_sample)
                sampled_points = np.array([list(line.interpolate(distance).coords)
                                            for distance in distances]).reshape(-1, coord_dim)
                
            cls_gens.append(sampled_points)
            cls_scores.append(res['confidence_level'])
    num_res = len(cls_gens)
    if num_res > 0:
        cls_gens = np.stack(cls_gens).reshape(num_res,-1)
        cls_scores = np.array(cls_scores)[:,np.newaxis]
        cls_gens = np.concatenate([cls_gens,cls_scores],axis=-1)
    else:
        cls_gens = np.zeros((0,num_sample*coord_dim+1))

    cls_gts = []
    for ann in annotations['vectors']:
        if ann['type'] == class_id:
            line = ann['pts']
            line = LineString(line)
            distances = np.linspace(0, line.length, num_sample)
            sampled_points = np.array([list(line.interpolate(distance).coords)
                                        for distance in distances]).reshape(-1, coord_dim)
            
            cls_gts.append(sampled_points)
    num_gts = len(cls_gts)
    if num_gts > 0:
        cls_gts = np.stack(cls_gts).reshape(num_gts,-1)
    else:
        cls_gts = np.zeros((0,num_sample*coord_dim))
        
    return cls_gens, cls_gts


def _eval_map(gen_results,
              annotations,
              results_path =None,
              threshold=0.5,
              metric='POR',
              num_classes=3,
              class_name=None,
              logger=None,
              tpfp_fn_name='vec',
              nproc=4,
              update=False,
              coord_dim=2,
              fix_interval=False):
    """Evaluate mAP of a dataset.

    Args:


        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )

        scale_ranges (list[tuple] | None): canvas_size
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_default` is used as default
            unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this
            case). If it is given as a function, then this function is used
            to evaluate tp & fp. Default None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    timer = mmcv.Timer()
    assert len(gen_results) == len(annotations)

    pool = Pool(nproc)  
    cls_gens, cls_gts = {}, {}
    print('Formatting ...')
    if results_path is not None:
        formatting_file = results_path.split('.pkl')[0]
        formatting_file = formatting_file + '_test_gen.pkl'
    else:
        formatting_file = 'test_gen.pkl'

    if not os.path.exists(formatting_file) or update:
        for i, clsname in enumerate(class_name):
            # get gt and det bboxes of this class
            gengts = pool.starmap(
                        partial(get_cls_results, num_sample=100, class_id=i,fix_interval=fix_interval,coord_dim=coord_dim),
                        zip(gen_results, annotations))       

            gen, gts = tuple(zip(*gengts))
            cls_gens[clsname] = gen
            cls_gts[clsname] = gts
        mmcv.dump([cls_gens, cls_gts],formatting_file)
    else:
        cls_gens, cls_gts = mmcv.load(formatting_file)
    print('Data formatting done in {:2f}s!!'.format(float(timer.since_start())))

    eval_results = []
    for i, clsname in enumerate(class_name):
        # get gt and det bboxes of this class
        cls_gen = cls_gens[clsname]
        cls_gt = cls_gts[clsname]
        # choose proper function according to datasets to compute tp and fp
        tpfp_fn = tpfp_gen
        # Trick for serialized
        # only top-level function can be serized
        # somehow use partitial the return function is defined
        # at the top level.
        tpfp_fn = partial(tpfp_fn, threshold=threshold, metric=metric, coord_dim=coord_dim)
        args = []
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_fn,
            zip(cls_gen, cls_gt, *args))
       
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = 0
        for j, bbox in enumerate(cls_gt):
            num_gts += bbox.shape[0]

        # sort all det bboxes by score, also sort tp and fp
        cls_gen = np.vstack(cls_gen)
        num_dets = cls_gen.shape[0]
        sort_inds = np.argsort(-cls_gen[:, -1])
        tp = np.hstack(tp)[sort_inds]
        fp = np.hstack(fp)[sort_inds]

        # calculate recall and precision with tp and fp
        # num_det*num_res
        tp = np.cumsum(tp, axis=0)
        fp = np.cumsum(fp, axis=0)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        # calculate AP
        # if dataset != 'voc07' else '11points'
        mode = 'area'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
        print('cls:{} done in {:2f}s!!'.format(clsname,float(timer.since_last_check())))
    pool.close()
    aps = []
    for cls_result in eval_results:
        if cls_result['num_gts'] > 0:
            aps.append(cls_result['ap'])
    mean_ap = np.array(aps).mean().item() if len(aps) else 0.0

    print_map_summary(
        mean_ap, eval_results, class_name=class_name, logger=logger)

    return mean_ap, eval_results


def eval_map(cfg: dict, update=False, fix_interval=False, logger=None):

    print('results path: {}'.format(cfg['result_path']))
    results_file = mmcv.load(cfg['result_path'])
    gen_res = list(results_file['results'].values())
    anns = list(results_file['groundTruth'].values())

    if 'gen_threshold' in cfg:
        threshold = cfg['gen_threshold']
    else:
        threshold = 0.3

    if 'metric' in cfg:
        metric = cfg['metric']
    else:
        metric = 'POR'

    if 'tpfp_fn_name' in cfg:    
        tpfp_fn_name = cfg['tpfp_fn_name']
    else:
        tpfp_fn_name = 'xxx'

    print('metric:',metric)
    print('threshold:',threshold)
    print('update:',update)
    print('fix_interval:',fix_interval)
    print('class_num:',cfg['class_name'])


    if 'coord_dim' in cfg:    
        coord_dim = cfg['coord_dim']
    else:
        coord_dim = 2
    
    mean_ap, eval_results = _eval_map(
        gen_res,
        anns,
        threshold=threshold,
        metric=metric,
        results_path=cfg['result_path'],
        num_classes=cfg['num_class'],
        class_name=cfg['class_name'],
        tpfp_fn_name=tpfp_fn_name,
        logger=logger,
        nproc=32,
        update=update,
        coord_dim=coord_dim,
        fix_interval=fix_interval)
    
    return mean_ap, eval_results


def print_map_summary(mean_ap,
                      results,
                      class_name=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    label_names = class_name

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)


def eval_chamfer(cfg, update=True, logger=None):

    thresholds=[0.5,1,1.5]
    cfg['metric'] = 'chamfer'

    cls_aps = np.zeros((len(thresholds),cfg['num_class']))
    for i, thr in enumerate(thresholds):
        print('-*'*10+f'thershold:{thr}'+'-*'*10)
        cfg['gen_threshold'] = -thr
        update = update if i ==0 else False
        mAP, cls_ap = eval_map(cfg,update=update,logger=logger)
        
        for j in range(cfg['num_class']):
            cls_aps[i, j] = cls_ap[j]['ap']
        
    for i, name in enumerate(cfg['class_name']):
        print('{}: {}'.format(name, cls_aps.mean(0)[i]))
    
    mAP =  cls_aps.mean(0).mean()
    print('map: {}'.format(mAP))
    
    return mAP

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Evaluate nuScenes local HD Map Construction Results.')
    parser.add_argument('--result-path', type=str,
                        default='')
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--num_class', type=int, default=3)
    parser.add_argument('--class_name', type=str, nargs='+',
                        default=['ped_crossing', 'divider', 'contours'])
    parser.add_argument('--CD_threshold', type=int, default=10)
    parser.add_argument('--coord-dim', type=int, default=2)
    parser.add_argument('--update', action="store_true")
    parser.add_argument('--metric', type=str, default='POR')
    parser.add_argument('--name', type=str, default='sme')

    args = parser.parse_args()

    args = vars(args)

    results_path = {}

    if len(args['result_path']) == 0:
        args['result_path'] = results_path[args['name']]
    args['tpfp_fn_name'] = 'xxx'
    
    if args['metric'] == 'POR':
        args['gen_threshold'] = 0.3
        eval_map(args,update=args['update'])
    elif args['metric'] == 'frechet':
        args['gen_threshold'] = -5
        eval_map(args,update=args['update'])
    elif args['metric'] in ['chamfer', 'chamfer_v2']:
        eval_chamfer(args)

    # eval_map(args,update=args['update'])