# Tosin Adewumi

"""
Utility functions
"""
import pandas as pd


def prepare_sequence(seq, to_ix):
    """

    :param seq: sequence to encode
    :param to_ix: dictionary of enumerated vocab for encoding
    :return: return tensor of seq_to_ids
    """
    idxs = [to_ix[w] for w in seq]
    return idxs  # torch.tensor(idxs, dtype=torch.long)


def pad_seq(sequence, tag2idx, max_len, seq_type='tok_ids'):
    padded_seq = []
    if seq_type == 'tok_ids':
        # initialize list with 0s to maximum seq length
        padded_seq.extend([0 for i in range(max_len)])
    elif seq_type == 'tag_ids':
        # initialize tag list with 'O's (NER- O: Other) to maximum seq length
        padded_seq.extend([tag2idx['O'] for i in range(max_len)])
    if len(sequence) > max_len:
        # cut sequence longer than the maximum SEQ_LEN
        padded_seq[:] = sequence[:max_len]
    else:
        # replace parts of default seq with values of original
        padded_seq[:len(sequence)] = sequence
    return padded_seq


def pad_seq_bi(sequence, tag2idx, max_len, seq_type='tok_ids'):
    padded_seq = []
    if seq_type == 'tok_ids':
        padded_seq.extend([0 for i in range(max_len)])              # initialize list with 0s to maximum seq length
    elif seq_type == 'tag_ids':
        padded_seq.extend([tag2idx[0] for i in range(max_len)])
    if len(sequence) > max_len:
        padded_seq[:] = sequence[:max_len]              # cut sequence longer than the maximum SEQ_LEN
    else:
        padded_seq[:len(sequence)] = sequence           # replace parts of default seq with values of original
    return padded_seq


def list_into_chunks(list_name, n_chunks):
    for i in range(0, len(list_name), n_chunks):
        yield list_name[i:i + n_chunks]


def preprocess_clean(data, columns):
    # word_tokens = []
    df_ = pd.DataFrame(columns=columns)
    df_['Class'] = data['Class']
    df_['Sentence'] = data['Sentence'].str.lower()
    df_['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)  # remove emails
    df_['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
    df_['Sentence'] = data['Sentence'].str.replace('[^\w\s]','')               # remove special characters
    df_['Sentence'] = data['Sentence'].replace('\d', '', regex=True)           # remove numbers
    return df_


def seq_to_ix(seq, to_ix):
    """
    :param seq: sequence to encode
    :param to_ix: dictionary of enumerated vocab for encoding
    :return: return tensor of seq_to_ids
    """
    idxs = [to_ix[w] for w in seq]
    return idxs # torch.tensor(idxs, dtype=torch.long)


def encode_sents(all_sents, words_to_ix):
    encoded_sents = list()
    for sent in all_sents:
        encoded_sent = list()
        for word in sent.split():
            if word not in words_to_ix.keys():
                encoded_sent.append(0)                  # put 0 for out of vocab words
            else:
                encoded_sent.append(words_to_ix[word])
        encoded_sents.append(encoded_sent)
    return encoded_sents


class MakeSentence(object):
    """
    Makes sentences of the data of tokens passed to it
    """
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(), s["POS"].values.tolist(), s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except ValueError:
            return None


class MakeSentenceSv(object):
    """
    Makes sentences of the data of tokens passed to it
    """
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist(), s["Type"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except ValueError:
            return None
