# sequence utility functions - conversion between tagging schemas
import torch
import math
import numpy as np

def ot2bieos_ts(ts_tag_sequence):
    """
    ot2bieos function for targeted-sentiment task, ts refers to targeted -sentiment / aspect-based sentiment
    :param ts_tag_sequence: tag sequence for targeted sentiment
    :return:
    """
    n_tags = len(ts_tag_sequence)
    new_ts_sequence = []
    prev_pos = '$$$'
    for i in range(n_tags):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag == 'O' or cur_ts_tag == 'EQ':
            # when meet the EQ label, regard it as O label
            new_ts_sequence.append('O')
            cur_pos = 'O'
        else:
            cur_pos, cur_sentiment = cur_ts_tag.split('-')
            # cur_pos is T
            if cur_pos != prev_pos:
                # prev_pos is O and new_cur_pos can only be B or S
                if i == n_tags - 1:
                    new_ts_sequence.append('S-%s' % cur_sentiment)
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('S-%s' % cur_sentiment)
                    else:
                        new_ts_sequence.append('B-%s' % cur_sentiment)
            else:
                # prev_pos is T and new_cur_pos can only be I or E
                if i == n_tags - 1:
                    new_ts_sequence.append('E-%s' % cur_sentiment)
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('E-%s' % cur_sentiment)
                    else:
                        new_ts_sequence.append('I-%s' % cur_sentiment)
        prev_pos = cur_pos
    return new_ts_sequence


def ot2bieos_ts_batch(ts_tag_seqs):
    """
    batch version of function ot2bieos_ts
    :param ts_tag_seqs:
    :return:
    """
    new_ts_tag_seqs = []
    n_seqs = len(ts_tag_seqs)
    for i in range(n_seqs):
        new_ts_seq = ot2bieos_ts(ts_tag_sequence=ts_tag_seqs[i])
        new_ts_tag_seqs.append(new_ts_seq)
    return new_ts_tag_seqs


def ot2bio_ts(ts_tag_sequence):
    """
    ot2bio function for ts tag sequence
    :param ts_tag_sequence:
    :return:
    """
    new_ts_sequence = []
    n_tag = len(ts_tag_sequence)
    prev_pos = '$$$'
    for i in range(n_tag):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag == 'O':
            new_ts_sequence.append('O')
            cur_pos = 'O'
        else:
            # current tag is subjective tag, i.e., cur_pos is T
            # print(cur_ts_tag)
            cur_pos, cur_sentiment = cur_ts_tag.split('-')
            if cur_pos == prev_pos:
                # prev_pos is T
                new_ts_sequence.append('I-%s' % cur_sentiment)
            else:
                # prev_pos is O
                new_ts_sequence.append('B-%s' % cur_sentiment)
        prev_pos = cur_pos
    return new_ts_sequence


def ot2bio_ts_batch(ts_tag_seqs):
    """
    batch version of function ot2bio_ts
    :param ts_tag_seqs:
    :return:
    """
    new_ts_tag_seqs = []
    n_seqs = len(ts_tag_seqs)
    for i in range(n_seqs):
        new_ts_seq = ot2bio_ts(ts_tag_sequence=ts_tag_seqs[i])
        new_ts_tag_seqs.append(new_ts_seq)
    return new_ts_tag_seqs


def bio2ot_ts(ts_tag_sequence):
    """
    perform bio-->ot for ts tag sequence
    :param ts_tag_sequence:
    :return:
    """
    new_ts_sequence = []
    n_tags = len(ts_tag_sequence)
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        if ts_tag == 'O' or ts_tag == 'EQ':
            new_ts_sequence.append('O')
        else:
            pos, sentiment = ts_tag.split('-')
            new_ts_sequence.append('T-%s' % sentiment)
    return new_ts_sequence


def bio2ot_ts_batch(ts_tag_seqs):
    """
    batch version of function bio2ot_ts
    :param ts_tag_seqs:
    :return:
    """
    new_ts_tag_seqs = []
    n_seqs = len(ts_tag_seqs)
    for i in range(n_seqs):
        new_ts_seq = bio2ot_ts(ts_tag_sequence=ts_tag_seqs[i])
        new_ts_tag_seqs.append(new_ts_seq)
    return new_ts_tag_seqs


def tag2ts(ts_tag_sequence):
    """
    transform ts tag sequence to targeted sentiment
    :param ts_tag_sequence: tag sequence for ts task
    :return:
    """
    n_tags = len(ts_tag_sequence)
    ts_sequence, sentiments = [], []
    beg, end = -1, -1
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        # current position and sentiment
        # tag O and tag EQ will not be counted
        eles = ts_tag.split('-')
        if len(eles) == 2:
            pos, sentiment = eles
        else:
            pos, sentiment = 'O', 'O'
        if sentiment != 'O':
            # current word is a subjective word
            sentiments.append(sentiment)
        if pos == 'S':
            # singleton
            ts_sequence.append((i, i, sentiment))
            sentiments = []
        elif pos == 'B':
            beg = i
            if len(sentiments) > 1:
                # remove the effect of the noisy I-{POS,NEG,NEU}
                sentiments = [sentiments[-1]]
        elif pos == 'E':
            end = i
            # schema1: only the consistent sentiment tags are accepted
            # that is, all of the sentiment tags are the same
            if end > beg > -1 and len(set(sentiments)) == 1:
                ts_sequence.append((beg, end, sentiment))
                sentiments = []
                beg, end = -1, -1
    return ts_sequence




