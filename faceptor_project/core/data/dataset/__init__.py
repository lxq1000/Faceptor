from.recog_dataset import MXFaceDataset
from.age_dataset import UTKFaceDataset_V2, MORPH2Dataset_V2
from.biattr_dataset import BiAttrDataset
from.affect_dataset import AffectNetDataset_V2, RAFDBDataset, FERPlusDataset_V2
from.parsing_dataset import CelebAMaskHQDataset, LaPaDataset
from.align_dataset import AFLWDataset, IBUG300WDataset, WFLWDataset, COFWDataset

def dataset_entry(config):
    return globals()[config['type']](**config['kwargs'])
