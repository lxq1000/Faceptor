from .recognition_transform import recognition_mxface_transform
from .attribute_analysis_transform import attribute_train_transform, attribute_test_transform
from .dense_prediction_transform import celebam_train_transform, celebam_test_transform, celebam_test_post_transform
from .dense_prediction_transform import lapa_train_transform, lapa_test_transform, lapa_test_post_transform
from .dense_prediction_transform import align_train_transform, align_test_transform, align_test_post_transform


ARCFACE_INIT_MEAN = [0.5, 0.5, 0.5]
ARCFACE_INIT_STD = [0.5, 0.5, 0.5]

FARL_INIT_MEAN = [0.48145466, 0.4578275, 0.40821073] 
FARL_INIT_STD = [0.26862954, 0.26130258, 0.27577711]

def transform_entry(config):
    return globals()[config['type']](**config['kwargs'])