import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=512):
    """
    Returns train data transforms with keypoint scaling.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
    )

def get_val_transforms(img_size=512):
    """
    Returns validation data transforms with keypoint scaling.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
    )
