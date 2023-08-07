from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from ibug.roi_tanh_warping import roi_tanh_polar_restore, roi_tanh_polar_warp
import ibug.roi_tanh_warping.reference_impl as ref
from .rtnet import rtnet50, rtnet101, FCN
from .resnet import Backbone, DeepLabV3Plus
from torch.nn.functional import softmax
from typing import Optional, Any
import kornia

ENCODER_MAP = {
    'rtnet50': [rtnet50, 2048],  # model_func, in_channels
    'rtnet101': [rtnet101, 2048],
}
DECODER_MAP = {
    'fcn': FCN,
    'deeplabv3plus': DeepLabV3Plus
}

WEIGHT = {
    # 'rtnet50-fcn-11': (Path(__file__).parent / 'rtnet/weights/rtnet50.torch', (0.406, 0.456, 0.485), (0.225, 0.224, 0.229), (512, 512)),
    # 'rtnet101-fcn-11': (Path(__file__).parent / 'rtnet/weights/rtnet101.torch', (0.406, 0.456, 0.485), (0.225, 0.224, 0.229), (512, 512)),
    'rtnet50-fcn-11': (Path(__file__).parent / 'rtnet/weights/rtnet50-fcn-11.torch', 0.5, 0.5, (513, 513)),
    'rtnet50-fcn-14': (Path(__file__).parent / 'rtnet/weights/rtnet50-fcn-14.torch', 0.5, 0.5, (513, 513)),
    'rtnet101-fcn-14': (Path(__file__).parent / 'rtnet/weights/rtnet101-fcn-14.torch', 0.5, 0.5, (513, 513)),
    'resnet50-fcn-14': (Path(__file__).parent / 'resnet/weights/resnet50-fcn-14.torch', 0.5, 0.5, (513, 513)),
    'resnet50-deeplabv3plus-14': (Path(__file__).parent / 'resnet/weights/resnet50-deeplabv3plus-14.torch', 0.5, 0.5, (513, 513)),
}


class SegmentationModel(nn.Module):

    def __init__(self, encoder='rtnet50', decoder='fcn', num_classes=14, export_onnx=False, image_h=480, image_w=640):
        super().__init__()

        if 'rtnet' in encoder:
            encoder_func, in_channels = ENCODER_MAP[encoder.lower()]
            self.encoder = encoder_func()
        else:
            self.encoder = Backbone(encoder)
            in_channels = self.encoder.num_channels
        self.decoder = DECODER_MAP[decoder.lower()](
            in_channels=in_channels, num_classes=num_classes)
        self.low_level = getattr(self.decoder, 'low_level', False)
        self.export_onnx = export_onnx
        self.image_h = image_h
        self.image_w = image_w

    def atanh_approx(self, x):
        numerator = 1.0 + x
        denominator = 1.0 - x
        return 0.5 * torch.log(numerator / denominator)

    def roi_tanh_polar_warp(
        self,
        image: np.ndarray,
        roi: np.ndarray,
        target_width: int,
        target_height: int,
        angular_offset: float = 0.0,
        interpolation: Optional[int] = cv2.INTER_LINEAR,
        border_mode: Optional[int] = cv2.BORDER_CONSTANT,
        border_value: Any = 0,
        keep_aspect_ratio: bool = False
    ) -> np.ndarray:

        roi = torch.unsqueeze(roi, dim=2)

        roi_center = (roi[:, 2:4] + roi[:, :2]) / 2.0
        roi_radii = (roi[:, 2:4] - roi[:, :2]) / torch.pi ** 0.5
        cos_offset, sin_offset = torch.cos(torch.Tensor([angular_offset])), torch.sin(torch.Tensor([angular_offset]))
        normalised_dest_indices = torch.stack(
            torch.meshgrid(
                torch.arange(0.0, 2.0 * torch.pi, 2.0 * torch.pi / target_height)[..., :target_height],
                torch.arange(0.0, 1.0, 1.0 / target_width)[..., :target_width]
            ),
            axis=-1
        )
        normalised_dest_indices = torch.reshape(normalised_dest_indices, shape=[1,target_height,target_width,2])

        radii = normalised_dest_indices[..., 0]
        orientation_x = torch.cos(normalised_dest_indices[..., 1])
        orientation_y = torch.sin(normalised_dest_indices[..., 1])

        if keep_aspect_ratio:
            src_radii = self.atanh_approx(radii) * (roi_radii[:, 0:1, :] * roi_radii[:, 1:2, :] / torch.sqrt(roi_radii[:, 1:2, :] ** 2 * orientation_x ** 2 + roi_radii[:, 0:1, :] ** 2 * orientation_y ** 2))
            src_x_indices = src_radii * orientation_x
            src_y_indices = src_radii * orientation_y

        else:
            src_radii = self.atanh_approx(radii)
            src_x_indices = roi_radii[:, 0:1] * src_radii * orientation_x
            src_y_indices = roi_radii[:, 1:2] * src_radii * orientation_y
        src_x_indices, src_y_indices = (
            roi_center[:, 0:1] + cos_offset * src_x_indices - sin_offset * src_y_indices,
            roi_center[:, 1:2] + cos_offset * src_y_indices + sin_offset * src_x_indices
        )
        x = kornia.geometry.transform.remap(
            image=image,
            map_x=src_x_indices,
            map_y=src_y_indices,
            mode='bilinear',
            padding_mode='zeros',
        )
        return x

    def restore_warp(self, h, w, logits: torch.Tensor, bboxes_tensor):
        logits = softmax(logits, 1)
        logits[:, 0] = 1 - logits[:, 0]  # background class
        logits = roi_tanh_polar_restore(logits, bboxes_tensor, w, h, keep_aspect_ratio=True, export_onnx=self.export_onnx)
        logits[:, 0] = 1 - logits[:, 0]
        predict = torch.argmax(logits, dim=1, keepdim=True)
        return predict

    def forward(self, x, rois):
        x = self.roi_tanh_polar_warp(image=x, roi=rois, target_width=513, target_height=513, keep_aspect_ratio=True)

        input_shape = x.shape[-2:]
        features = self.encoder(x, rois)

        low = features['c1']
        high = features['c4']
        if self.low_level:
            x = self.decoder(high, low)
        else:
            x = self.decoder(high)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        # return x
        return self.restore_warp(self.image_h, self.image_w, x, rois)



class FaceParser(object):
    def __init__(self, device='cuda:0', ckpt=None, encoder='rtnet50', decoder='fcn', num_classes=11, export_onnx=False, image_h=480, image_w=640):
        self.device = device
        model_name = '-'.join([encoder, decoder, str(num_classes)])
        assert model_name in WEIGHT, f'{model_name} is not supported'

        self.model_name = model_name
        pretrained_ckpt, mean, std, sz = WEIGHT[self.model_name]
        self.sz = sz
        self.export_onnx = export_onnx
        self.image_h = image_h
        self.image_w = image_w

        self.model = SegmentationModel(encoder, decoder, num_classes, self.export_onnx, image_h=self.image_h, image_w=self.image_w)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        if ckpt is None:
            ckpt = pretrained_ckpt
        ckpt = torch.load(ckpt, 'cpu')
        ckpt = ckpt.get('state_dict', ckpt)
        self.model.load_state_dict(ckpt, True)
        self.model.eval()
        self.model.to(device)


    @torch.no_grad()
    def predict_img(self, img, bboxes, rgb=False):

        if isinstance(img, str):
            img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            if rgb:
                img = img[:, :, ::-1]
        else:
            raise TypeError
        h, w = img.shape[:2]

        num_faces = len(bboxes)

        # """
        # bboxes.shape
        #     (1, 15)
        # """
        # imgs = [
        #     ref.roi_tanh_polar_warp(
        #         image=img,
        #         roi=b,
        #         target_width=self.sz[0],
        #         target_height=self.sz[1],
        #         keep_aspect_ratio=True
        #     ) for b in bboxes
        # ]
        # # imgs = ref.roi_tanh_polar_warp(
        # #     image=torch.randn([1,3,h,w]),
        # #     roi=torch.Tensor(bboxes),
        # #     target_width=self.sz[0],
        # #     target_height=self.sz[1],
        # #     keep_aspect_ratio=True
        # # )


        # imgs = [self.transform(img) for img in imgs]
        # bboxes_tensor = torch.tensor(bboxes).view(num_faces, -1).to(self.device)

        # # img = img.repeat(num_faces, 1, 1, 1)
        # # img = roi_tanh_polar_warp(
        #     # img, bboxes_tensor, target_height=self.sz[0], target_width=self.sz[1], keep_aspect_ratio=True)
        # # img = self.transform(img).unsqueeze(0).to(self.device)

        # img = torch.stack(imgs).to(self.device)
        """
        img.shape
            torch.Size([1, 3, 513, 513])

        bboxes_tensor.shape
            torch.Size([1, 15])

        logits.shape
            torch.Size([1, 11, 513, 513])

        h
            480
        w
            640

        mask.shape
            (1, 480, 640)
        """
        # logits = self.model(img, bboxes_tensor)
        # mask = self.restore_warp(h, w, logits, bboxes_tensor)

        # img = torch.randn([1,3,h,w])

        # if not self.export_onnx:
        #     mask = self.model(img, bboxes_tensor, image_height=h, image_width=w)

        if self.export_onnx:
            self.model.eval()

            import onnx
            from onnxsim import simplify
            RESOLUTION = [
                # [180,320],
                # [180,416],
                # [180,512],
                # [180,640],
                # [180,800],
                # [240,320],
                # [240,416],
                # [240,512],
                # [240,640],
                # [240,800],
                # [240,960],
                # [288,480],
                # [288,512],
                # [288,640],
                # [288,800],
                # [288,960],
                # [288,1280],
                # [320,320],
                # [360,480],
                # [360,512],
                # [360,640],
                # [360,800],
                # [360,960],
                # [360,1280],
                # [376,1344],
                # [416,416],
                # [480,640],
                # [480,800],
                # [480,960],
                # [480,1280],
                # [512,512],
                # [540,800],
                # [540,960],
                # [540,1280],
                # [640,640],
                # [640,960],
                # [720,1280],
                # [720,2560],
                # [1080,1920],
                [513,513],
            ]
            MODEL = f'{self.model_name.replace("-", "_")}'
            bboxes_tensor = torch.randn([1,4], dtype=torch.float32).cpu()
            for H, W in RESOLUTION:
                img = torch.randn([1,3,H,W]).cpu()
                onnx_file = f"{MODEL}_1x3x{H}x{W}_1x4.onnx"
                torch.onnx.export(
                    self.model,
                    args=(img, bboxes_tensor),
                    f=onnx_file,
                    opset_version=16,
                    input_names=[
                        'input_image',
                        'bboxes_tensor_x1y1x2y2',
                    ],
                    output_names=['output'],
                )
                model_onnx1 = onnx.load(onnx_file)
                model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
                onnx.save(model_onnx1, onnx_file)
                model_onnx2 = onnx.load(onnx_file)
                model_simp, check = simplify(model_onnx2)
                onnx.save(model_simp, onnx_file)
                model_onnx2 = onnx.load(onnx_file)
                model_simp, check = simplify(model_onnx2)
                onnx.save(model_simp, onnx_file)
                model_onnx2 = onnx.load(onnx_file)
                model_simp, check = simplify(model_onnx2)
                onnx.save(model_simp, onnx_file)


                onnx_file = f"{MODEL}_1x3x{H}x{W}_Nx4.onnx"
                torch.onnx.export(
                    self.model,
                    args=(img, bboxes_tensor),
                    f=onnx_file,
                    opset_version=16,
                    input_names=[
                        'input_image',
                        'bboxes_tensor_x1y1x2y2',
                    ],
                    output_names=['output'],
                    dynamic_axes={
                        'bboxes_tensor_x1y1x2y2' : {0: 'N'},
                    }
                )
                model_onnx1 = onnx.load(onnx_file)
                model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
                onnx.save(model_onnx1, onnx_file)
                model_onnx2 = onnx.load(onnx_file)
                model_simp, check = simplify(model_onnx2)
                onnx.save(model_simp, onnx_file)
                model_onnx2 = onnx.load(onnx_file)
                model_simp, check = simplify(model_onnx2)
                onnx.save(model_simp, onnx_file)
                model_onnx2 = onnx.load(onnx_file)
                model_simp, check = simplify(model_onnx2)
                onnx.save(model_simp, onnx_file)


            onnx_file = f"{MODEL}_1x3xHxW_Nx4.onnx"
            torch.onnx.export(
                self.model,
                args=(img, bboxes_tensor),
                f=onnx_file,
                opset_version=16,
                input_names=[
                    'input_image',
                    'bboxes_tensor_x1y1x2y2',
                ],
                output_names=['output'],
                dynamic_axes={
                    'input_image' : {2: 'height', 3: 'width'},
                    'bboxes_tensor_x1y1x2y2' : {0: 'N'},
                    'output' : {0: 'N', 2: 'height', 3: 'width'},
                }
            )
            model_onnx1 = onnx.load(onnx_file)
            model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
            onnx.save(model_onnx1, onnx_file)
            model_onnx2 = onnx.load(onnx_file)
            model_simp, check = simplify(model_onnx2)
            onnx.save(model_simp, onnx_file)
            model_onnx2 = onnx.load(onnx_file)
            model_simp, check = simplify(model_onnx2)
            onnx.save(model_simp, onnx_file)
            model_onnx2 = onnx.load(onnx_file)
            model_simp, check = simplify(model_onnx2)
            onnx.save(model_simp, onnx_file)


            import sys
            sys.exit(0)

        # return mask[:, 0, ...].cpu()

    def restore_warp(self, h, w, logits: torch.Tensor, bboxes_tensor):
        logits = softmax(logits, 1)
        logits[:, 0] = 1 - logits[:, 0]  # background class
        logits = roi_tanh_polar_restore(logits, bboxes_tensor, w, h, keep_aspect_ratio=True
        )
        logits[:, 0] = 1 - logits[:, 0]
        predict = logits.cpu().argmax(1).numpy()
        return predict

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, rois: torch.Tensor):
        features = self.model.encoder(x, rois, return_features=True)
        x = self.model.decoder(features['c4'])
        features['logits'] = x
        return features
