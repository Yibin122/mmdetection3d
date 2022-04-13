import torch

from mmdet3d.apis import init_model
from mmdet3d.models.dense_heads.smoke_mono3d_head import SMOKEMono3DHead


class SMOKEONNX(torch.nn.Module):
    def __init__(self, model):
        super(SMOKEONNX, self).__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.bbox_head

    def forward(self, img, topk=100):
        # out[0]: (1, 16, H, W)
        # out[1]: (1, 32, H/2, W/2)
        # out[2]: (1, 64, H/4, W/4)
        # out[3]: (1, 128, H/8, W/8)
        # out[4]: (1, 256, H/16, W/16)
        # out[5]: (1, 512, H/32, W/32)
        out = self.backbone(img)
        # out[0]: (1, 64, H/4, W/4)
        out = self.neck(out)
        # cls_scores[0]: (1, 3, H/4, W/4)
        # bbox_preds[0]: (1, 8, H/4, W/4)
        cls_scores, bbox_preds = self.head(out)

        # get_local_maximum
        # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/utils/gaussian_target.py#L190
        hmax = torch.nn.functional.max_pool2d(cls_scores[0], 3, stride=1, padding=1)
        keep = (hmax == cls_scores[0]).float()
        scores = cls_scores[0] * keep                               # (1, 3, H/4, W/4)

        # get_topk_from_heatmap
        # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/utils/gaussian_target.py#L207
        batch, _, height, width = scores.size()
        scores = scores.view(batch, -1)
        topk_scores, topk_indices = torch.topk(scores, topk)        # (1, 100), (1, 100)
        # topk_clses = topk_inds // (height * width)                # (1, 100)
        topk_clses = torch.floor(topk_indices / (height * width))   # (1, 100)
        topk_inds = topk_indices % (height * width)
        # topk_ys = topk_inds // width                              # (1, 100)
        topk_ys = torch.floor(topk_inds / width)                    # (1, 100)
        topk_xs = (topk_inds % width).int().float()                 # (1, 100)
        points = torch.cat([topk_xs.view(-1, 1),
                            topk_ys.view(-1, 1).float()], dim=1)    # (100, 2)

        # transpose_and_gather_feat
        # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/utils/gaussian_target.py#L255
        bbox_pred = bbox_preds[0].permute(0, 2, 3, 1).contiguous()  # (1, H/4, W/4, 8)
        bbox_pred = bbox_pred.view(-1, 8)                           # (H*W/16, 8)
        topk_inds = topk_inds.view(-1)                              # (100)
        bbox_pred = bbox_pred[topk_inds, :]                         # (100, 8)
        topk_clses = topk_clses.view(-1)                            # (100)
        topk_scores = topk_scores.view(-1)                          # (100)

        # return bbox_pred, points, topk_clses.float(), topk_scores
        return bbox_pred, topk_scores, topk_indices.float()

    def export_onnx(self, onnx_file_path):
        dummy_input = torch.randn(1, 3, 384, 1280, device='cuda:0')
        torch.onnx.export(self, dummy_input, onnx_file_path, opset_version=11)
        print('Saved SMOKE onnx file {}'.format(onnx_file_path))


if __name__ == '__main__':
    # https://github.com/open-mmlab/mmdetection3d/tree/master/configs/smoke
    config_file = 'configs/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d.py'
    checkpoint_file = 'checkpoints/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth'
    checkpoint_model = init_model(config_file, checkpoint_file)

    smoke = SMOKEONNX(checkpoint_model)
    smoke.export_onnx('smoke.onnx')
    
