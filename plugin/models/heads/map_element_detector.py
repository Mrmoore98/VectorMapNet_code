import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear
from mmcv.runner import force_fp32
from torch.distributions.categorical import Categorical

from mmdet.core import multi_apply, reduce_mean
from mmdet.models import HEADS
from .detr_bbox import DETRBboxHead
from mmdet.models.utils.transformer import inverse_sigmoid

@HEADS.register_module(force=True)
class MapElementDetector(DETRBboxHead):

    def __init__(self, *args, **kwargs):
        super(MapElementDetector, self).__init__(*args, **kwargs)

    def _init_embedding(self):

        self.label_embed = nn.Embedding(
            self.num_classes, self.embed_dims)

        self.img_coord_embed = nn.Linear(2, self.embed_dims)

        # query_pos_embed & query_embed
        self.query_embedding = nn.Embedding(self.num_query,
                                            self.embed_dims*2)
        
        # for bbox parameter xstart, ystart, xend, yend
        self.bbox_embedding = nn.Embedding( self.bbox_size, 
                                            self.embed_dims*2)

    def _init_branch(self,):
        """Initialize classification branch and regression branch of head."""

        fc_cls = Linear(self.embed_dims*self.bbox_size, self.cls_out_channels)
        # fc_cls = Linear(self.embed_dims, self.cls_out_channels)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.LayerNorm(self.embed_dims))
            reg_branch.append(nn.ReLU())

        if self.discrete_output:
            reg_branch.append(nn.Linear(
                self.embed_dims, max(self.canvas_size), bias=True,))
        else:
            reg_branch.append(nn.Linear(
                self.embed_dims, self.coord_dim, bias=True,))

        reg_branch = nn.Sequential(*reg_branch)
        # add sigmoid or not

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = self.transformer.decoder.num_layers

        if self.iterative:
            fc_cls = _get_clones(fc_cls, num_pred)
            reg_branch = _get_clones(reg_branch, num_pred)
        else:
            reg_branch = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            fc_cls = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])

        self.pre_branches = nn.ModuleDict([
            ('cls', fc_cls),
            ('reg', reg_branch), ])

    def _prepare_context(self, batch, context):
        """Prepare class label and vertex context."""

        global_context_embedding = None
        if self.separate_detect:
            global_context_embedding = self.label_embed(batch['class_label'])

        # Image context
        if self.separate_detect:
            image_embeddings = assign_bev(
                context['bev_embeddings'], batch['batch_idx'])
        else:
            image_embeddings = context['bev_embeddings']

        image_embeddings = self.input_proj(
            image_embeddings)  # only change feature size

        # Pass images through encoder
        device = image_embeddings.device

        # Add 2D coordinate grid embedding
        B, C, H, W = image_embeddings.shape
        Ws = torch.linspace(-1., 1., W)
        Hs = torch.linspace(-1., 1., H)
        image_coords = torch.stack(
            torch.meshgrid(Hs, Ws), dim=-1).to(device)
        image_coord_embeddings = self.img_coord_embed(image_coords)

        image_embeddings += image_coord_embeddings[None].permute(0, 3, 1, 2)

        # Reshape spatial grid to sequence
        sequential_context_embeddings = image_embeddings.reshape(
            B, C, H, W)

        return (global_context_embedding, sequential_context_embeddings)

    def forward(self, batch, context, img_metas=None, multi_scale=False):
        '''
        Args:
            bev_feature (List[Tensor]): shape [B, C, H, W]
                feature in bev view
            img_metas

        Outs:
            preds_dict (Dict):
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_lines_preds (Tensor):
                    [nb_dec, bs, num_query, num_points, 2].
        '''

        (global_context_embedding, sequential_context_embeddings) =\
            self._prepare_context(batch, context)

        x = sequential_context_embeddings
        B, C, H, W = x.shape

        query_embedding = self.query_embedding.weight[None,:,None].repeat(B, 1, self.bbox_size, 1)
        bbox_embed = self.bbox_embedding.weight
        query_embedding = query_embedding + bbox_embed[None,None]
        query_embedding = query_embedding.view(B, -1, C*2)

        img_masks = x.new_zeros((B, H, W))
        pos_embed = self.positional_encoding(img_masks)

        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        hs, init_reference, inter_references = self.transformer(
                    [x,],
                    [img_masks.type(torch.bool)],
                    query_embedding,
                    [pos_embed],
                    reg_branches= self.reg_branches if self.iterative else None,  # noqa:E501
                    cls_branches= None,  # noqa:E501
            )
        outs_dec = hs.permute(0, 2, 1, 3)

        outputs = []
        for i, (query_feat) in enumerate(outs_dec):
            if i == 0:
                reference = init_reference
            else:
                reference = inter_references[i - 1]
            outputs.append(self.get_prediction(i,query_feat,reference))

        return outputs

    def get_prediction(self, level, query_feat, reference):

        bs, num_query, h = query_feat.shape
        query_feat = query_feat.view(bs, -1, self.bbox_size,h)

        ocls = self.pre_branches['cls'][level](query_feat.flatten(-2))
        # ocls = ocls.mean(-2)
        reference = inverse_sigmoid(reference)
        reference = reference.view(bs, -1, self.bbox_size,self.coord_dim)

        tmp = self.pre_branches['reg'][level](query_feat)
        tmp[...,:self.kp_coord_dim] =  tmp[...,:self.kp_coord_dim] + reference[...,:self.kp_coord_dim]
        lines = tmp.sigmoid() # bs, num_query, self.bbox_size,2

        lines = lines * self.canvas_size[:self.coord_dim]
        lines = lines.flatten(-2)

        return dict(
            lines=lines,  # [bs, num_query, bboxsize*2]
            scores=ocls,  # [bs, num_query, num_class]
            embeddings= query_feat, # [bs, num_query, bbox_size, h]
        )
