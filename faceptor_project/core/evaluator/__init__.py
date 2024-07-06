from.recog_evaluator import RecogEvaluator
from.age_evaluator import AgeEvaluator_V2
from.biattr_evaluator import BiAttrEvaluator
from.affect_evaluator import SingferEvaluator
from.parsing_evaluator import ParsingEvaluator
from.align_evaluator import IBUG300WEvaluator, COFWEvaluator, WFLWEvaluator, AFLWEvaluator
from core.utils import printlog

def evaluator_entry(config):
    printlog('Evaluator config[kwargs]',config['kwargs'])
    return globals()[config['type']](**config['kwargs'])
