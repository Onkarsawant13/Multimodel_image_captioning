"""
Object Model v1 — Compatible with BLEU-4 30.89 checkpoint
==========================================================
Uses tiny 3-layer CNN for RoI features (not ResNet-101).
This is the version that produced BLEU-4 30.89.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.ops import roi_align
from ultralytics import YOLO
from PIL import Image
import numpy as np

MAX_OBJECTS   = 16
D_MODEL       = 512
ROI_OUTPUT_SZ = 7
CONF_THRESH   = 0.25
NUM_CLASSES   = 80

class BoxGeometryEncoder(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 128), nn.ReLU(), nn.Linear(128, d_model))
    def forward(self, boxes):
        area = ((boxes[...,2]-boxes[...,0])*(boxes[...,3]-boxes[...,1])).unsqueeze(-1)
        return self.mlp(torch.cat([boxes, area], dim=-1))

class ClassEmbedding(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, d_model=D_MODEL):
        super().__init__()
        self.embedding = nn.Embedding(num_classes+1, d_model, padding_idx=0)
    def forward(self, class_ids):
        return self.embedding(class_ids)

class RoIVisualEncoder(nn.Module):
    def __init__(self, d_model=D_MODEL, roi_size=ROI_OUTPUT_SZ):
        super().__init__()
        self.roi_size = roi_size
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        feat_dim = 256 * roi_size * roi_size
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(feat_dim, d_model), nn.ReLU())

    def forward(self, images, boxes_norm):
        B, N, _ = boxes_norm.shape
        feat_map = self.backbone(images)
        _, _, H_f, W_f = feat_map.shape
        abs_boxes = boxes_norm.clone()
        abs_boxes[..., [0,2]] *= W_f
        abs_boxes[..., [1,3]] *= H_f
        batch_idx = torch.arange(B, device=images.device).unsqueeze(1).expand(B,N).reshape(-1,1).float()
        rois = torch.cat([batch_idx, abs_boxes.view(B*N, 4)], dim=1)
        pooled = roi_align(feat_map, rois, output_size=self.roi_size, spatial_scale=1.0, aligned=True)
        return self.proj(pooled).view(B, N, -1)

class ObjectModel(nn.Module):
    def __init__(self, yolo_weights="yolov8n.pt", d_model=D_MODEL,
                 max_objects=MAX_OBJECTS, freeze_yolo=True, freeze_resnet=False):
        super().__init__()
        self.max_objects = max_objects
        self.d_model     = d_model
        _yolo = YOLO(yolo_weights)
        _yolo.model.eval()
        for param in _yolo.model.parameters():
            param.requires_grad = False
        object.__setattr__(self, "_yolo", _yolo)
        self.box_encoder = BoxGeometryEncoder(d_model)
        self.class_emb   = ClassEmbedding(NUM_CLASSES, d_model)
        self.roi_encoder = RoIVisualEncoder(d_model)
        self.layer_norm  = nn.LayerNorm(d_model)
        self.img_transform = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

    @torch.no_grad()
    def _run_yolo(self, pil_images):
        results = self._yolo(pil_images, conf=CONF_THRESH, verbose=False)
        all_boxes, all_classes, all_confs = [], [], []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                all_boxes.append(torch.zeros(0,4))
                all_classes.append(torch.zeros(0, dtype=torch.long))
                all_confs.append(torch.zeros(0))
                continue
            boxes   = r.boxes.xyxyn.cpu()
            classes = r.boxes.cls.long().cpu() + 1
            confs   = r.boxes.conf.cpu()
            order   = confs.argsort(descending=True)[:self.max_objects]
            all_boxes.append(boxes[order])
            all_classes.append(classes[order])
            all_confs.append(confs[order])
        return all_boxes, all_classes, all_confs

    def _pad_to_max(self, tensors, pad_value=0):
        B = len(tensors)
        device = tensors[0].device if tensors[0].numel() > 0 else torch.device("cpu")
        if tensors[0].dim() == 1:
            out = torch.full((B, self.max_objects), pad_value, dtype=tensors[0].dtype, device=device)
        else:
            out = torch.zeros(B, self.max_objects, tensors[0].shape[-1], dtype=tensors[0].dtype, device=device)
        mask = torch.ones(B, self.max_objects, dtype=torch.bool, device=device)
        for i, t in enumerate(tensors):
            n = min(len(t), self.max_objects)
            if n > 0:
                out[i,:n] = t[:n].to(device)
                mask[i,:n] = False
        return out, mask

    def forward(self, pil_images, device=None):
        if device is None:
            device = next(self.parameters()).device
        B = len(pil_images)
        raw_boxes, raw_classes, _ = self._run_yolo(pil_images)
        raw_boxes   = [b.to(device) for b in raw_boxes]
        raw_classes = [c.to(device) for c in raw_classes]
        boxes_padded,   box_mask   = self._pad_to_max(raw_boxes,   0.0)
        classes_padded, _          = self._pad_to_max(raw_classes, 0)
        obj_mask       = box_mask.to(device)
        boxes_padded   = boxes_padded.to(device)
        classes_padded = classes_padded.to(device)
        img_tensors = torch.stack([self.img_transform(img) for img in pil_images]).to(device)
        self._yolo.model.to(device)
        box_feats   = self.box_encoder(boxes_padded)
        class_feats = self.class_emb(classes_padded)
        roi_feats   = self.roi_encoder(img_tensors, boxes_padded)
        object_features = self.layer_norm(box_feats + class_feats + roi_feats)
        object_features = object_features.masked_fill(obj_mask.unsqueeze(-1), 0.0)
        return {"object_features": object_features, "object_mask": obj_mask,
                "boxes": boxes_padded, "class_ids": classes_padded}