import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 420
DATA_DIR = 'data'
TRAIN_FILE = 'fold%s_train.json'
VALID_FILE = 'fold%s_valid.json'
TEST_FILE = 'fold%s_test.json'

# Storing all clauses containing sentimental word, based on the ANTUSD lexicon 'opinion_word_simplified.csv'. see https://academiasinicanlplab.github.io
SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses.pkl'


class Config(object):
    def __init__(self):
        self.split = 'split10'

        self.bert_cache_path = '../src/bert-base-chinese'
        self.feat_dim = 768
        self.hidden_size = 200
        self.K = 3
        self.epsilon = 1e-8
        self.max_doc_len = 73
        self.max_token_len = 512
        self.dropout = 0.
        self.layers = 1
        self.epochs = 35
        self.lr = 2e-5
        self.batch_size = 4
        self.gradient_accumulation_steps = 1
        self.l2 = 1e-5
        self.l2_bert = 0.
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8

    def __str__(self):
        return obj_to_string(Config, self)


def obj_to_string(cls, obj):
    """
    简单地实现类似对象打印的方法
    :param cls: 对应的类(如果是继承的类也没有关系，比如A(object), cls参数传object一样适用，如果你不想这样，可以修改第一个if)
    :param obj: 对应类的实例
    :return: 实例对象的to_string
    """
    if not isinstance(obj, cls):
        raise TypeError("obj_to_string func: 'the object is not an instance of the specify class.'")
    to_string = str(cls.__name__) + "("
    items = obj.__dict__
    n = 0
    for k in items:
        if k.startswith("_"):
            continue
        to_string = to_string + str(k) + "=" + str(items[k]) + ","
        n += 1
    if n == 0:
        to_string += str(cls.__name__).lower() + ": 'Instantiated objects have no property values'"
    return to_string.rstrip(",") + ")"
