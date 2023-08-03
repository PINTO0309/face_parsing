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

    def __init__(self, encoder='rtnet50', decoder='fcn', num_classes=14, export_onnx=False):
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

    def restore_warp(self, h, w, logits: torch.Tensor, bboxes_tensor):
        logits = softmax(logits, 1)
        logits[:, 0] = 1 - logits[:, 0]  # background class
        logits = roi_tanh_polar_restore(logits, bboxes_tensor, w, h, keep_aspect_ratio=True, export_onnx=self.export_onnx)
        logits[:, 0] = 1 - logits[:, 0]
        predict = torch.argmax(logits, dim=1, keepdim=True)
        return predict

    def forward(self, x, rois, image_height, image_width):
        input_shape = x.shape[-2:]
        features = self.encoder(x, rois)

        low = features['c1']
        high = features['c4']
        if self.low_level:
            x = self.decoder(high, low)
        else:
            x = self.decoder(high)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return self.restore_warp(image_height, image_width, x, rois)


class FaceParser(object):
    def __init__(self, device='cuda:0', ckpt=None, encoder='rtnet50', decoder='fcn', num_classes=11, export_onnx=False):
        self.device = device
        model_name = '-'.join([encoder, decoder, str(num_classes)])
        assert model_name in WEIGHT, f'{model_name} is not supported'

        self.model_name = model_name
        pretrained_ckpt, mean, std, sz = WEIGHT[self.model_name]
        self.sz = sz
        self.export_onnx = export_onnx

        self.model = SegmentationModel(encoder, decoder, num_classes, self.export_onnx)

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


        imgs = [ref.roi_tanh_polar_warp(img, b, *self.sz, keep_aspect_ratio=True) for b in bboxes]
        imgs = [self.transform(img) for img in imgs]
        bboxes_tensor = torch.tensor(bboxes).view(num_faces, -1).to(self.device)

        # img = img.repeat(num_faces, 1, 1, 1)
        # img = roi_tanh_polar_warp(
            # img, bboxes_tensor, target_height=self.sz[0], target_width=self.sz[1], keep_aspect_ratio=True)
        # img = self.transform(img).unsqueeze(0).to(self.device)

        img = torch.stack(imgs).to(self.device)
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
        mask = self.model(img, bboxes_tensor, image_height=h, image_width=w)

        if self.export_onnx:
            self.model.eval()

            import onnx
            from onnxsim import simplify
            RESOLUTION = [
                [self.sz[0],self.sz[1]],

            ]
            MODEL = f'{self.model_name.replace("-", "_")}'
            original_image_height = torch.tensor([h], dtype=torch.int64).cuda()
            original_image_width = torch.tensor([w], dtype=torch.int64).cuda()
            for H, W in RESOLUTION:
                onnx_file = f"{MODEL}_{H}x{W}.onnx"
                torch.onnx.export(
                    self.model,
                    args=(img, bboxes_tensor, original_image_height, original_image_width),
                    f=onnx_file,
                    opset_version=16,
                    input_names=[
                        'input_image',
                        'bboxes_tensor',
                        'original_image_height',
                        'original_image_width',
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

            onnx_file = f"{MODEL}_HxW.onnx"
            # x = torch.randn(1, 3, self.sz[0], self.sz[1]).cuda()
            torch.onnx.export(
                self.model,
                args=(img, bboxes_tensor, original_image_height, original_image_width),
                f=onnx_file,
                opset_version=16,
                input_names=[
                    'input_image',
                    'bboxes_tensor',
                    'original_image_height',
                    'original_image_width',
                ],
                output_names=['output'],
                dynamic_axes={
                    'input_image' : {2: 'height', 3: 'width'},
                    'output' : {1: 'height', 2: 'width'},
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



        return mask[:, 0, ...].cpu()

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
