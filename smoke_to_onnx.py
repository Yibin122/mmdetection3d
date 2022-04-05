import torch
import torch.nn.functional as F

from mmdet3d.apis import init_model
# from mmdet.models.utils.gaussian_target import get_topk_from_heatmap
# from mmdet3d.models.dense_heads.smoke_mono3d_head import SMOKEMono3DHead


class SMOKEONNX(torch.nn.Module):
    def __init__(self, model):
        super(SMOKEONNX, self).__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.bbox_head

    def forward(self, img, topk=100):
        # # FIXME: onnx GroupNorm 结果不一致
        # out1 = self.backbone.base_layer[0](img)
        # out1 = self.backbone.base_layer[1](out1)
        # return out1

        # out[0]: 16, stride = 1
        # out[1]: 32, stride = 2
        # out[2]: 64, stride = 4
        # out[3]: 128, stride = 8
        # out[4]: 256, stride = 16
        # out[5]: 512, stride = 32
        out = self.backbone(img)
        # out[0]: 64, stride = 4
        out = self.neck(out)
        # cls_scores[0]: (1, 3, H/4, W/4)
        # bbox_preds[0]: (1, 8, H/4, W/4)
        cls_scores, bbox_preds = self.head(out)

        # get_local_maximum
        hmax = F.max_pool2d(cls_scores[0], 3, stride=1, padding=1)
        keep = (hmax == cls_scores[0]).float()
        scores = cls_scores[0] * keep                               # (1, 3, H/4, W/4)

        # get_topk_from_heatmap
        batch, _, height, width = scores.size()
        scores = scores.view(batch, -1)
        topk_scores, topk_inds = torch.topk(scores, topk)           # (1, 100), (1, 100)
        topk_clses = topk_inds // (height * width)                  # (1, 100)
        topk_inds = topk_inds % (height * width)
        topk_ys = topk_inds // width                                # (1, 100)
        topk_xs = (topk_inds % width).int().float()                 # (1, 100)
        points = torch.cat([topk_xs.view(-1, 1),
                            topk_ys.view(-1, 1).float()], dim=1)    # (100, 2)

        # transpose_and_gather_feat
        bbox_pred = bbox_preds[0].permute(0, 2, 3, 1).contiguous()  # (1, H/4, W/4, 8)
        bbox_pred = bbox_pred.view(-1, 8)                           # (H*W/16, 8)
        topk_inds = topk_inds.view(-1)                              # (100)
        bbox_pred = bbox_pred[topk_inds, :]                         # (100, 8)
        topk_clses = topk_clses.view(-1)                            # (100)
        topk_scores = topk_scores.view(-1)                          # (100)

        return bbox_pred, points, topk_clses.float(), topk_scores

    def export_onnx(self, onnx_file_path):
        dummy_input = torch.randn(1, 3, 384, 1280, device='cuda:0')
        torch.onnx.export(self, dummy_input, onnx_file_path, opset_version=11)
        print('Saved smoke onnx file {}'.format(onnx_file_path))


if __name__ == '__main__':
    # https://github.com/open-mmlab/mmdetection3d/tree/master/configs/smoke
    config_file = 'configs/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py'
    checkpoint_file = 'checkpoints/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth'
    checkpoint_model = init_model(config_file, checkpoint_file)

    smoke = SMOKEONNX(checkpoint_model)
    smoke.export_onnx('smoke.onnx')
    
