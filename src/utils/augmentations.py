import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=224):
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-45, 45), p=0.5),
        A.ElasticTransform(alpha=120, sigma=6, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, image_size // 8),
            hole_width_range=(8, image_size // 8),
            p=0.3
        ),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=224):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size=224, n_augments=10):
    base = [
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Rotate(limit=(90, 90), p=1.0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Rotate(limit=(180, 180), p=1.0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ]),
    ]

    extra = [
        A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.85, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.CLAHE(p=0.5),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
        for _ in range(n_augments - len(base))
    ]

    return base + extra