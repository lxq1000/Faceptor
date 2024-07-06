
from torchvision import transforms


def recognition_mxface_transform(**kwargs):
    default = dict(
        input_size=112,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    default.update(kwargs)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([default["input_size"], default["input_size"]]),
        transforms.RandomHorizontalFlip(),
        # transforms.GaussianBlur(51),
        transforms.ToTensor(),
        transforms.Normalize(mean=default["mean"], std=default["mean"])])
    
    return transform