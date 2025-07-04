# The base code for BERT configuration mapping from hugging face

from transformers import PreTrainedModel, BertModel, BertConfig
# model map for BERT
from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
from transformers.models.bert.modeling_bert import BertEncoder, BertEmbeddings, BertPooler
import torch.nn as nn
from bert_utils import *


BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
 'bert-base-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin',
 'bert-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin'
}


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module): # call when model instantiation that don't have pretrained weights from from_pretrained()
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range) #normal distribution with mean=0 and standard deviation
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_() #bias = 0
            module.weight.data.fill_(1.0) #weight = 1
        if isinstance(module, nn.Linear) and module.bias is not None: #linear layer with bias set to 0 -> prevent initial bias
            module.bias.data.zero_()
