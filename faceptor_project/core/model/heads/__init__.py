from .task_specific_heads import TaskSpecificHeadsHolder
from .decoder import DecoderNewHolder

def heads_holder_entry(config):
    return globals()[config['type']](**config['kwargs'])