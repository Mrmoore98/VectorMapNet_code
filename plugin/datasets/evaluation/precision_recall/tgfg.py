import mmcv
import numpy as np
import similaritymeasures
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from scipy.spatial import distance
from shapely.geometry import CAP_STYLE, JOIN_STYLE, LineString, Polygon
from shapely.strtree import STRtree


def tpfp_gen(gen_lines,
             gt_lines,
             threshold=0.5,
             coord_dim=2,
             metric='POR'):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """

    num_gens = gen_lines.shape[0]
    num_gts = gt_lines.shape[0]

    # tp and fp
    tp = np.zeros((num_gens), dtype=np.float32)
    fp = np.zeros((num_gens), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if num_gts == 0:
        fp[...] = 1
        return tp, fp
    
    if num_gens == 0:
        return tp, fp
    
    gen_scores = gen_lines[:,-1] # n
    # distance matrix: n x m
    matrix = polyline_score(
            gen_lines[:,:-1].reshape(num_gens,-1,coord_dim), 
            gt_lines.reshape(num_gts,-1,coord_dim),linewidth=2.,metric=metric)
    # for each det, the max iou with all gts
    matrix_max = matrix.max(axis=1)
    # for each det, which gt overlaps most with it
    matrix_argmax = matrix.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-gen_scores)

    gt_covered = np.zeros(num_gts, dtype=bool)

    # tp = 0 and fp = 0 means ignore this detected bbox,
    for i in sort_inds:
        if matrix_max[i] >= threshold:
            matched_gt = matrix_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    return tp, fp


def polyline_score(pred_lines, gt_lines, linewidth=1., metric='POR'):
    '''
        each line with 1 meter width
        pred_lines: num_preds, List [npts, 2]
        gt_lines: num_gts, npts, 2
        gt_mask: num_gts, npts, 2
    '''
    positive_threshold = 1.
    num_preds = len(pred_lines)
    num_gts = len(gt_lines)
    line_length = pred_lines.shape[1]

    # gt_lines = gt_lines + np.array((1.,1.))

    pred_lines_shapely = \
        [LineString(i).buffer(linewidth,
            cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                          for i in pred_lines]
    gt_lines_shapely =\
        [LineString(i).buffer(linewidth,
            cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                        for i in gt_lines]

    # construct tree
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(pred_lines_shapely))

    if metric=='POR':
        iou_matrix = np.zeros((num_preds, num_gts),dtype=np.float64)
    elif metric=='frechet':
        iou_matrix = np.full((num_preds, num_gts), -100.)
    elif metric=='chamfer':
        iou_matrix = np.full((num_preds, num_gts), -100.)
    elif metric=='chamfer_v2':
        iou_matrix = np.full((num_preds, num_gts), -100.)

    for i, pline in enumerate(gt_lines_shapely):

        for o in tree.query(pline):
            if o.intersects(pline):
                pred_id = index_by_id[id(o)]

                if metric=='POR':
                    dist_mat = distance.cdist(
                        pred_lines[pred_id], gt_lines[i], 'euclidean')
                    
                    valid_ab = (dist_mat.min(-1) < positive_threshold).sum()
                    valid_ba = (dist_mat.min(-2) < positive_threshold).sum()

                    iou_matrix[pred_id, i] = min(valid_ba,valid_ab) / line_length
                    # iou_matrix[pred_id, i] = ((valid_ba+valid_ab)/2) / line_length
                    # assert iou_matrix[pred_id, i] <= 1. and iou_matrix[pred_id, i] >= 0.
                elif metric=='frechet':
                    fdistance_1 = \
                        -similaritymeasures.frechet_dist(pred_lines[pred_id], gt_lines[i])
                    fdistance_2 = \
                        -similaritymeasures.frechet_dist(pred_lines[pred_id][::-1], gt_lines[i])
                    fdistance = max(fdistance_1,fdistance_2)
                    iou_matrix[pred_id, i] = fdistance

                elif metric=='chamfer':
                    dist_mat = distance.cdist(
                        pred_lines[pred_id], gt_lines[i], 'euclidean')
                    
                    valid_ab = dist_mat.min(-1).sum()
                    valid_ba = dist_mat.min(-2).sum()

                    iou_matrix[pred_id, i] = -(valid_ba+valid_ab)/(2*line_length)
                    # if iou_matrix[pred_id, i] == 0:
                    #     import ipdb; ipdb.set_trace()
                elif metric=='chamfer_v2':
                    dist_mat = distance.cdist(
                        pred_lines[pred_id], gt_lines[i], 'euclidean')
                    
                    valid_ab = dist_mat.min(-1).sum()
                    valid_ba = dist_mat.min(-2).sum()

                    iou_matrix[pred_id, i] = -(valid_ba/pred_lines[pred_id].shape[0]
                                                +valid_ab/gt_lines[i].shape[0])/2

    
    return iou_matrix
