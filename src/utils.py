import torch
from detrex.modeling.backbone import TimmBackbone
import torch.nn as nn
from functools import partial
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from safetensors.torch import load_file
import json
from pathlib import Path
from pycocotools.coco import COCO
import numpy as np


CLASSES = ["", 'Atelectasis', 'Cardiomegaly', "Pleural effusion", "Nodule/Mass", "Infiltration", "Pneumothorax", "Lung Opacity", 
          "Consolidation", "Pulmonary fibrosis", " Pleural thickening", "Aortic enlargement", "ILD", "Calcification", "Other lesion"]

label2color = np.array([[0, 0, 0], [199, 164, 32], [247, 240, 115], [20, 1, 87], [101, 213, 158], 
               [42, 78, 232], [93, 113, 236], [126, 177, 239], [243, 82, 31], [214, 225, 229], 
               [114, 186, 70], [204, 41, 125], [5, 36, 133], [254, 253, 219], [24, 187, 154]])


def interpolate_patch_14to16(state_dict: dict, interpolate_position: True, num_patches=1024) -> dict:
    """
    Adapting ViT-B/14 based RadDINO checkpoint to ViT-B/16 based Detrex by 
    interpolating the relevant embeddings from patch size 14 to 16. 
    More detail about interpolation implemenation at https://github.com/baaivision/EVA/blob/master/EVA-01/seg/interpolate_patch_14to16.py
    
    Args:
        state_dict (dict): The state dictionary containing the model parameters.
    """
    
    def interpolate_patch_embed(state_dict: dict) -> dict:
        """
        Interpolate the patch embeddings from patch size 14 to 16.
        More detail, from ([768, 3, 14, 14]) to ([768, 3, 16, 16]) using bicubic interpolation.
        """
        patch_embed = state_dict.get("patch_embed.proj.weight", None)
        
        if patch_embed is not None:
            
            print(f"Interpolating patch_embed from shape {patch_embed.shape} to (768, 3, 16, 16)")
            
            C_o, C_in, H, W = patch_embed.shape
            
            patch_embed = torch.nn.functional.interpolate(
                patch_embed.float(), 
                size=(16, 16), 
                mode='bicubic', 
                align_corners=False
            )
            
            state_dict["patch_embed.proj.weight"] = patch_embed
        
        return state_dict
    
    def interpolate_pos_embed(state_dict: dict, num_patches=1024) -> dict:
        """
        Interpolate the positional embeddings from patch size 14 to 16.
        More detail, from ([1, 1370, 768]) to ([1, 4097, 768]) using bicubic interpolation.
        """
        pos_embed = state_dict.get("pos_embed", None)
        
        if pos_embed is not None:
            
            print(f"Interpolating pos_embed from shape {pos_embed.shape} to (1, {num_patches + 1}, 768)")
            
            embedding_size = pos_embed.shape[-1]
            # num_patches = 4096
            num_extra_tokens = 1
        
            # orig_size = 37 (as 1370 - 1 = 1369 -> sqrt = 37)
            orig_size = int((pos_embed.shape[-2] - num_extra_tokens) ** 0.5)
            # new_size = 64 (due to sqrt(4096) = 64)
            new_size = int(num_patches ** 0.5) 
            
            if orig_size != new_size:
                print(f"Interpolation position embedding from {orig_size}x{orig_size} to {new_size}x{new_size}")
                
            extra_tokens = pos_embed[:, :num_extra_tokens]
            pos_tokens = pos_embed[:, num_extra_tokens:]
            
            # Reshape -> Permute -> Interpolate -> Permute -> Flatten
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            state_dict['pos_embed'] = new_pos_embed
            
        return state_dict
    
    state_dict = interpolate_patch_embed(state_dict)
    
    if interpolate_position:
        state_dict = interpolate_pos_embed(state_dict, num_patches=num_patches)
    
    return state_dict


def convert_rad_dino_for_detrex(backbone_safetensor_path: str, output_pth: str, unexpected_keys: list = ['mask_token', "norm.bias", "norm.weight"]): 
    """ 
    Convert the RadDINO checkpoint to be compatible with Detrex by removing unexpected keys, 
    interpolating the patch embeddings and modify key names to match Detrex's expected format. 
    The converted checkpoint will be saved in PyTorch format at the specified output path.
    
    Args:
        backbone_safetensor_path (str): The path to the RadDINO backbone checkpoint in safetensor format.
        output_pth (str): The path to save the converted checkpoint in PyTorch format.
        unexpected_keys (list): A list of keys that are present in the RadDINO checkpoint but not expected in the Detrex model. 
        These keys will be removed from the state dictionary before saving
    """
    
    safe_state_dict = load_file(backbone_safetensor_path)
    interpolated_safe_state_dict = interpolate_patch_14to16(safe_state_dict.copy(), interpolate_position=True, num_patches=1024)
    
    for key in unexpected_keys:
        if key in interpolated_safe_state_dict:
            print(f"Removing unexpected key: {key}")
            del interpolated_safe_state_dict[key]
            
    new_state_dict = {}
    for k, v in interpolated_safe_state_dict.items():
            new_state_dict[f"model.{k}"] = v
    
    torch.save({"model": new_state_dict}, output_pth)
    
    return output_pth

def get_train_transform():
    return A.Compose([
        A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(8, 8), p=0.5),
        A.Affine(
            scale=(0.9, 1.1), 
            translate_percent=(-0.05, 0.05), 
            rotate=(-10, 10),  
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            keep_ratio=True
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.OneOf([
            A.MedianBlur(blur_limit=7, p=1.0),
            A.Blur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0),
        ], p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), per_channel=False, p=0.2),
        # A.CoarseDropout(
        #     max_holes=8, max_height=32, max_width=32, 
        #     fill_value=0, p=0.3, mask_fill_value=0
        # ),
        # Below is the statistics during pre-training RadDINO
        A.Normalize(
            mean=[0.5307, 0.5307, 0.5307], 
            std=[0.2583, 0.2583, 0.2583],
            max_pixel_value=255.0, 
            p=1.0
        ),
        ToTensorV2(p=1.0)
    ], 
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=16, min_visibility=0.1))

def get_valid_transform():
    return A.Compose([
        A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(8, 8), p=1.0),
        # Below is the statistics during pre-training RadDINO
        A.Normalize(
            mean=[0.5307, 0.5307, 0.5307], 
            std=[0.2583, 0.2583, 0.2583],
            max_pixel_value=255.0, 
            p=1.0
        ),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def load_raddino(checkpoint_path: str) -> TimmBackbone:
    """
    Load the RadDINO checkpoint from the specified path and return in TimmBackbone format (A detrex
    compatible format). Notice that the checkpoint should have been converted to be compatible with Detrex using the convert_rad_dino_for_detrex function, 
    which includes interpolation of patch embeddings and removal of unexpected keys.
    
    Args:
        checkpoint_path (str): The file path to the RadDINO checkpoint in PyTorch format.
    """
    try:
        net = TimmBackbone(
            model_name="vit_base_patch14_dinov2.lvd142m",
            features_only=True,
            pretrained=False,
            checkpoint_path=checkpoint_path,
            in_channels=3,
            out_indices=(-1,),
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            patch_size=16, 
            dynamic_img_size=True,
            dynamic_img_pad=True,
            drop_path_rate=0.1,
        )
    except Exception as e:
        print(f"Error loading RadDINO checkpoint: {e}")
        raise e
    
    return net

def get_bounding_boxes(image_path: str, annotations_path: str | None):
    """
    Load bounding box annotations from a given image path from a JSON file and 
    return them as a list of tuples.
    
    Args:
        image_path (str): The file path to the input image. This is used to ensure that the bounding boxes correspond to the correct image.
        annotations_path (str): The file path to the JSON annotation file containing bounding box information. 
        The JSON file should have a structure that includes the image filename and corresponding bounding box coordinates.
    
    Returns:
        list[tuple[int, int, int, int]]: A list of bounding boxes, where each bounding box is represented as a tuple of (x_min, y_min, x_max, y_max).
        list[str]: A list of labels corresponding to each bounding box.
        list[tuple[int, int, int]]: A list of colors corresponding to each bounding box, where each color is represented as a tuple of (R, G, B) values.
    """
    if annotations_path is None:
        print("No annotations path provided. Returning empty bounding boxes, labels, and colors.")
        return [], [], []
    
    id2label = {idx:cls for idx, cls in enumerate(CLASSES)}
    
    indexed_dataset = COCO(annotations_path)
    
    img_stem = Path(image_path).stem
    
    img_id = [id for id, img in indexed_dataset.imgs.items() if Path(img["file_name"]).stem == img_stem][0]
    
    anns = indexed_dataset.loadAnns(indexed_dataset.getAnnIds(imgIds=[img_id]))
    
    coco_boxes = [ann['bbox'] for ann in anns]
    
    converted_boxes = [
        (
            int(box[0]), # x_min
            int(box[1]), # y_min
            int(box[0] + box[2]), # x_max = x_min + width
            int(box[1] + box[3]) # y_max = y_min + height
        ) for box in coco_boxes
    ]
    
    labels = [id2label[ann['category_id']] for ann in anns]
    
    colors = [label2color[ann['category_id']] for ann in anns]
    
    return converted_boxes, labels, colors