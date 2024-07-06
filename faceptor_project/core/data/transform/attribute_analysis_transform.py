
from timm.data import create_transform
from torchvision import transforms

def attribute_train_transform(**kwargs):

    default = dict(
        input_size=112,
        scale=(1.0, 1.0),
        ratio=(1.0, 1.0),
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        interpolation='bicubic',
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    default.update(kwargs)

    return create_transform(**default)


def attribute_test_transform(**kwargs):

    default = dict(
        input_size=112,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    default.update(kwargs)

    return transforms.Compose([
                transforms.Resize([default["input_size"], default["input_size"]]),
                transforms.ToTensor(),
                transforms.Normalize(mean=default["mean"], std=default["std"])
            ])