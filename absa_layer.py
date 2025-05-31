# Build downstream task based on bert.py
import torch
import torch.nn as nn
from transformers import BertModel
from seq_utils import *
from bert import BertPreTrainedModel
from torch.nn import CrossEntropyLoss

class TaggerConfig:
    # define hyper parameters
    def __init__(self):
        self.hidden_dropout_prob = 0.1 # drop out = 0.1
        self.hidden_size = 768 # size of BERT output
        self.n_rnn_layers = 1  # number of RNN layers - not used if tagger is non-RNN model
        self.bidirectional = True  # not used if tagger is non-RNN model


class CNN_BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, cnn_kernels=[3, 5, 7], bidirectional=True):
        super(CNN_BiGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.bidirectional = bidirectional

        # CNN layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(input_size, input_size // len(cnn_kernels), k, padding=k // 2)
            for k in cnn_kernels
        ])

        # GRU layer comes after CNN
        self.gru = GRU(
            input_size=input_size,  # Input size remains the same after concatenation
            hidden_size=hidden_size,
            bidirectional=bidirectional
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # CNN processing first
        # Transpose for CNN: (batch_size, seq_len, input_size) -> (batch_size, input_size, seq_len)
        cnn_in = x.transpose(1, 2)

        # Apply multiple CNNs with different kernel sizes
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(cnn_in)  # shape: (batch_size, input_size//len(kernels), seq_len)
            conv_outputs.append(conv_out)

        # Concatenate outputs from different CNNs
        combined = torch.cat(conv_outputs, dim=1)  # shape: (batch_size, input_size, seq_len)

        # Transpose back: (batch_size, input_size, seq_len) -> (batch_size, seq_len, input_size)
        cnn_out = combined.transpose(1, 2)

        # BiGRU processing
        gru_out, _ = self.gru(cnn_out)  # shape: (batch_size, seq_len, hidden_size)

        # Apply layer normalization
        output = self.layer_norm(gru_out)

        return output, None

class GRU(nn.Module):
    # customized GRU with layer normalization
    def __init__(self, input_size, hidden_size, bidirectional=True):
        """
        :param input_size:
        :param hidden_size:
        :param bidirectional:
        """
        super(GRU, self).__init__() # calling parent class -> ensure entire inheritance chain
        self.input_size = input_size
        if bidirectional:
            self.hidden_size = hidden_size // 2
        else:
            self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.Wxrz = nn.Linear(in_features=self.input_size, out_features=2*self.hidden_size, bias=True)
        self.Whrz = nn.Linear(in_features=self.hidden_size, out_features=2*self.hidden_size, bias=True)
        self.Wxn = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=True)
        self.Whn = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=True)
        self.LNx1 = nn.LayerNorm(2*self.hidden_size)
        self.LNh1 = nn.LayerNorm(2*self.hidden_size)
        self.LNx2 = nn.LayerNorm(self.hidden_size)
        self.LNh2 = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        """
        :param x: input tensor, shape: (batch_size, seq_len, input_size)
        :return:
        """
        def recurrence(xt, htm1):
            """
            :param xt: current input
            :param htm1: previous hidden state
            :return:
            """
            gates_rz = torch.sigmoid(self.LNx1(self.Wxrz(xt)) + self.LNh1(self.Whrz(htm1)))
            rt, zt = gates_rz.chunk(2, 1)
            nt = torch.tanh(self.LNx2(self.Wxn(xt))+rt*self.LNh2(self.Whn(htm1)))
            ht = (1.0-zt) * nt + zt * htm1
            return ht

        steps = range(x.size(1))
        bs = x.size(0)
        hidden = self.init_hidden(bs)
        # shape: (seq_len, bsz, input_size)
        input = x.transpose(0, 1)
        output = []
        for t in steps:
            hidden = recurrence(input[t], hidden)
            output.append(hidden)
        # shape: (bsz, seq_len, input_size)
        output = torch.stack(output, 0).transpose(0, 1)

        if self.bidirectional:
            output_b = []
            hidden_b = self.init_hidden(bs)
            for t in steps[::-1]:
                hidden_b = recurrence(input[t], hidden_b)
                output_b.append(hidden_b)
            output_b = output_b[::-1]
            output_b = torch.stack(output_b, 0).transpose(0, 1)
            output = torch.cat([output, output_b], dim=-1)
        return output, None

    def init_hidden(self, bs):
        h_0 = torch.zeros(bs, self.hidden_size).cuda()
        return h_0

class BertABSATagger(BertPreTrainedModel):
    def __init__(self, bert_config):
        """
        :param bert_config: configuration for bert model
        """
        super(BertABSATagger, self).__init__(bert_config) # call the parent BERT Pretrained Model
        self.num_labels = bert_config.num_labels # number of labels
        self.tagger_config = TaggerConfig()
        self.tagger_config.absa_type = bert_config.absa_type.lower()
        if bert_config.tfm_mode == 'finetune':
            # initialized with pre-trained BERT and perform finetuning
            # print("Fine-tuning the pre-trained BERT...")
            self.bert = BertModel(bert_config)
        else:
            raise Exception("Invalid transformer mode %s!!!" % bert_config.tfm_mode)
        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        # fix the parameters in BERT and regard it as feature extractor
        if bert_config.fix_tfm:
            # fix the parameters of the (pre-trained or randomly initialized) transformers during fine-tuning
            for p in self.bert.parameters():
                p.requires_grad = False

        self.tagger = None
        if self.tagger_config.absa_type == 'linear':
            # hidden size at the penultimate layer
            penultimate_hidden_size = bert_config.hidden_size
        else:
            self.tagger_dropout = nn.Dropout(self.tagger_config.hidden_dropout_prob)
            if self.tagger_config.absa_type == 'gru':
                self.tagger = GRU(input_size=bert_config.hidden_size,
                                  hidden_size=self.tagger_config.hidden_size,
                                  bidirectional=self.tagger_config.bidirectional)
            elif self.tagger_config.absa_type == 'tfm':
                # transformer encoder layer
                self.tagger = nn.TransformerEncoderLayer(d_model=bert_config.hidden_size,
                                                         nhead=12,
                                                         dim_feedforward=4*bert_config.hidden_size,
                                                         dropout=0.1)
            elif self.tagger_config.absa_type == 'cnn_bigru':
                # CNN-biGRU
                self.tagger = CNN_BiGRU(input_size=bert_config.hidden_size,
                                                   hidden_size=self.tagger_config.hidden_size,
                                                   bidirectional=self.tagger_config.bidirectional)
            else:
                raise Exception('Unimplemented downstream tagger %s...' % self.tagger_config.absa_type)
            penultimate_hidden_size = self.tagger_config.hidden_size
        self.classifier = nn.Linear(penultimate_hidden_size, bert_config.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz) batchsize - seqlen - hiddensize
        tagger_input = outputs[0] # outputs -> BaseModelOutput object -> index 0 = last_hidden_state
        tagger_input = self.bert_dropout(tagger_input)
        #print("tagger_input.shape:", tagger_input.shape)
        if self.tagger is None or self.tagger_config.absa_type == 'crf':
            # regard classifier as the tagger
            logits = self.classifier(tagger_input)
        else:
            if self.tagger_config.absa_type == 'gru':
                # customized GRU
                classifier_input, _ = self.tagger(tagger_input)
            elif self.tagger_config.absa_type == 'tfm':
                # vanilla self-attention networks or transformer
                # adapt the input format for the transformer or self attention networks
                tagger_input = tagger_input.transpose(0, 1)
                classifier_input = self.tagger(tagger_input)
                classifier_input = classifier_input.transpose(0, 1)
            elif self.tagger_config.absa_type == 'cnn_bigru':
                # CNN_BiGRU
                classifier_input, _ = self.tagger(tagger_input)
            else:
                raise Exception("Unimplemented downstream tagger %s..." % self.tagger_config.absa_type)
            classifier_input = self.tagger_dropout(classifier_input)
            logits = self.classifier(classifier_input) #scores for each token with each label percentage
        outputs = (logits,) + outputs[2:]

        if labels is not None: # loss for ground truth labels - training process
            if self.tagger_config.absa_type != 'crf':
                loss_fct = CrossEntropyLoss() # softmax + neg log
                if attention_mask is not None: # only loss for real tokens and ignore padding tokens
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss] # logit of real tokens
                    active_labels = labels.view(-1)[active_loss] # label
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs
            else:
                log_likelihood = self.tagger(inputs=logits, tags=labels, mask=attention_mask)
                loss = -log_likelihood # negative log likelihood
                outputs = (loss,) + outputs
        return outputs
