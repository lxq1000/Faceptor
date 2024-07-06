

from blueprint.ml.augmenters import *
from torchvision import transforms


def to_tensor_and_normalize(**kwargs):
     
    default = dict(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    default.update(kwargs)

    return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=default["mean"], std=default["std"])
            ])

def celebam_train_transform(**kwargs):
     default = dict(
          input_size=512
          )
     default.update(kwargs)

     transform = Sequential([
          AttachConstData('align_matrix', np.eye(3, dtype=np.float32)),
          With(tags_str="image -> image", aug=Normalize255()),
          AttachConstData(tag_name="shape", const_data=[default["input_size"], default["input_size"]]),
          With(tags_str="shape, align_matrix -> shape, align_matrix", 
               aug=UpdateRandomTransformMatrix(target_shape=[default["input_size"], default["input_size"]],
                                               shift_sigma=0.01,
                                               rot_sigma=0.314,
                                               scale_sigma=0.1,
                                               shift_normal=False,
                                               ret_shape=True)),
          With(tags_str="align_matrix -> transform_map", 
               aug=GetTransformMap(warped_shape=[default["input_size"], default["input_size"]],
                                   warp_factor=0.0)),
          With(tags_str="image, transform_map -> image", 
               aug=TransformByMap(interpolation="bilinear")),
          With(tags_str="label, transform_map -> label", 
               aug=TransformByMap(interpolation="nearest")),
          With(tags_str="image -> image",
               aug=Sequential([
                   RandomGray(),
                   RandomGamma(),
                   RandomBlur()
                   ])),
          Filter(tags=["image", "label"])
          ])

     return transform


def celebam_test_transform(**kwargs):
     default = dict(
         input_size=512,
         warp_factor=0.0  # celebam: 0.0 lapa: 0.8
         )
     default.update(kwargs)

     transform = Sequential([
          AttachConstData('align_matrix', np.eye(3, dtype=np.float32)),
          With(tags_str="image -> ori_image", aug=Normalize255()),
          With(tags_str="align_matrix -> transform_map", 
               aug=GetTransformMap(warped_shape=[default["input_size"], default["input_size"]],
                                   warp_factor=default["warp_factor"])),
          With(tags_str="ori_image, transform_map -> image", 
               aug=TransformByMap(interpolation="bilinear")),
          Filter(tags=["image", "ori_image", "label", "filename", "align_matrix"])
          ])

     return transform



def celebam_test_post_transform(**kwargs):
     default = dict(
         input_size=512,
         warp_factor=0.0)
     default.update(kwargs)

     transform = Sequential([
          With(tags_str="ori_image -> image_shape",
               aug=GetShape()),
          With(tags_str="align_matrix, image_shape -> transform_map_inv",
               aug=GetInvertedTransformMap(warped_shape=[default["input_size"], default["input_size"]],
                                           warp_factor=default["warp_factor"])),
          With(tags_str="pred_warped_logits, transform_map_inv -> pred",
               aug=Sequential([TransformByMap(interpolation="bilinear"), 
                               ArgMax(axis=-1)])),          
          ])
     
     return transform
     


def lapa_train_transform(**kwargs):
     default = dict(
          input_size=512,
          warp_factor=0.8
          )
     default.update(kwargs)

     transform = Sequential([
          With('landmarks', 'face_align_pts', lambda landmarks:landmarks[[104, 105, 54, 84, 90], :]),
          With('face_align_pts', 'align_matrix', GetFaceAlignMatrix(target_shape=(512, 512))),
          With(tags_str="image -> image", aug=Normalize255()),
          AttachConstData(tag_name="shape", const_data=[default["input_size"], default["input_size"]]),
          With(tags_str="shape, align_matrix -> shape, align_matrix", 
               aug=UpdateRandomTransformMatrix(target_shape=[default["input_size"], default["input_size"]],
                                               shift_sigma=0.01,
                                               rot_sigma=0.314,
                                               scale_sigma=0.1,
                                               shift_normal=False,
                                               ret_shape=True)),
          With(tags_str="align_matrix -> transform_map", 
               aug=GetTransformMap(warped_shape=[default["input_size"], default["input_size"]],
                                   warp_factor=default["warp_factor"])), #  0.8
          With(tags_str="image, transform_map -> image", 
               aug=TransformByMap(interpolation="bilinear")),
          With(tags_str="label, transform_map -> label", 
               aug=TransformByMap(interpolation="nearest")),
          With(tags_str="image -> image",
               aug=Sequential([
                   RandomGray(),
                   RandomGamma(),
                   RandomBlur()
                   ])),
          Filter(tags=["image", "label"])
          ])

     return transform

def lapa_test_transform(**kwargs):
     default = dict(
         input_size=512,
         warp_factor=0.8)
     default.update(kwargs)

     transform = Sequential([
          With('landmarks', 'face_align_pts', lambda landmarks:landmarks[[104, 105, 54, 84, 90], :]),
          With('face_align_pts', 'align_matrix', GetFaceAlignMatrix(target_shape=(512, 512))),
          With(tags_str="image -> ori_image", aug=Normalize255()),
          With(tags_str="align_matrix -> transform_map", 
               aug=GetTransformMap(warped_shape=[default["input_size"], default["input_size"]],
                                   warp_factor=default["warp_factor"])),
          With(tags_str="ori_image, transform_map -> image", 
               aug=TransformByMap(interpolation="bilinear")),
          Filter(tags=["image", "ori_image", "label", "filename", "align_matrix"])
          ])

     return transform

def lapa_test_post_transform(**kwargs):
     default = dict(
         input_size=512, 
         warp_factor=0.8)
     default.update(kwargs)

     transform = Sequential([
          With(tags_str="ori_image -> image_shape",
               aug=GetShape()),
          With(tags_str="align_matrix, image_shape -> transform_map_inv",
               aug=GetInvertedTransformMap(warped_shape=[default["input_size"], default["input_size"]],
                                           warp_factor=default["warp_factor"])),
          With(tags_str="pred_warped_logits, transform_map_inv -> pred",
               aug=Sequential([TransformByMap(interpolation="bilinear"), 
                               ArgMax(axis=-1)])),          
          ])
     
     return transform


def align_train_transform(**kwargs):
     default = dict(
          input_size=512,
          shift_sigma=0.05,
          rot_sigma=0.174,
          scale_sigma=0.1,
          scale_mu=1.0,
          warp_factor=0.0,
          )
     default.update(kwargs)


     transform = Sequential([
          With(('box', None), 'crop_matrix', 
               UpdateCropAndResizeMatrix((512, 512), align_corners=False)),
          With(tags_str="image -> image", aug=Normalize255()),
          AttachConstData(tag_name="shape", const_data=[default["input_size"], default["input_size"]]),
          With(tags_str="shape, crop_matrix -> shape, crop_matrix", 
               aug=UpdateRandomTransformMatrix(target_shape=[default["input_size"], default["input_size"]],
                                               shift_sigma=default["shift_sigma"],
                                               rot_sigma=default["rot_sigma"],
                                               scale_sigma=default["scale_sigma"],
                                               scale_mu=default["scale_mu"],
                                               rot_normal=False,
                                               scale_normal=False,
                                               shift_normal=False,
                                               ret_shape=True)),
          With(tags_str="crop_matrix -> transform_map", 
               aug=GetTransformMap(warped_shape=[default["input_size"], default["input_size"]],
                                   warp_factor=default["warp_factor"])),
          With(tags_str="image, transform_map -> image", 
               aug=TransformByMap(interpolation="bilinear")),
          With(tags_str="landmarks, crop_matrix -> label", 
               aug=TransformPoints2D(warped_shape=[default["input_size"], default["input_size"]],
                                     warp_factor=default["warp_factor"])),
          With(tags_str="image -> image",
               aug=Sequential([RandomOcclusion(),
                               Maybe(prob=0.5, then_branch=NoiseFusion()),
                               RandomGray(),
                               RandomGamma(),
                               RandomBlur()])),
          Filter(tags=["image", "label", "filename"])
          ])

     return transform


def align_test_transform(**kwargs):
     default = dict(
          input_size=512,
          scale_mu=1.0,
          warp_factor=0.0,
          )
     default.update(kwargs)


     transform = Sequential([
          With(('box', None), 'crop_matrix', 
               UpdateCropAndResizeMatrix((512, 512), align_corners=False)),
          With(tags_str="image -> image", aug=Normalize255()),
          AttachConstData(tag_name="shape", const_data=[default["input_size"], default["input_size"]]),
          With(tags_str="shape, crop_matrix -> shape, crop_matrix", 
               aug=UpdateTransformMatrix(target_shape=[default["input_size"], default["input_size"]],
                                               scale_mu=default["scale_mu"],
                                               ret_shape=True)),
          With(tags_str="crop_matrix -> transform_map", 
               aug=GetTransformMap(warped_shape=[default["input_size"], default["input_size"]],
                                   warp_factor=default["warp_factor"])),
          With(tags_str="image, transform_map -> image", 
               aug=TransformByMap(interpolation="bilinear")),
          Filter(tags=["image", "landmarks", "crop_matrix", "box", "filename"])
          ])

     return transform

def align_test_post_transform(**kwargs):
     default = dict(
          input_size=512,
          warp_factor=0.0,
          )
     default.update(kwargs)


     transform = Sequential([
          With(tags_str="pred_warped_landmarks, crop_matrix -> pred_landmarks", 
               aug=TransformPoints2DInverted(warped_shape=[default["input_size"], default["input_size"]],
                                             warp_factor=default["warp_factor"])),
          ])

     return transform