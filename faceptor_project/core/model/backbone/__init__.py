from .farl_vit import FaRLVisualFeatures

def backbone_entry(config):
    return globals()[config['type']](**config['kwargs'])