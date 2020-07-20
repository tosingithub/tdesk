# Tosin Adewumi

"""
Named Entity Recognition on GMB Dataset
"""

import argparse
import time
import os
import math
import pandas as pd
import numpy as np
from tqdm.auto import trange, tqdm
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from gensim.models import Word2Vec, KeyedVectors, FastText, fasttext
import torch.nn as nn
from io import open
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from ufuncs import *
from sigoptfunc import connfunc


# commandline arguments
parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
# parser.add_argument('--data', type=str, default='data/wikitext-2', help='location of the data corpus')
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--ofile1', type=str, default='output_ner_val_cc.en.300.txt', help='output file')
parser.add_argument('--ofile2', type=str, default='output_ner_test_cc.en.300.txt', help='output file 2')
parser.add_argument('--emsize', type=int, default=300, help='size of word embeddings or input: dmodel')
parser.add_argument('--model', type=str, default='Transformer', \
    help='type of recurrent net (RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--nhid', type=int, default=2048, help='the dimension of the feedforward network model in nn.TransformerEncoder')
parser.add_argument('--nlayers', type=int, default=7, help='number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--max_seq', type=int, default=75, help='maximum sequence length')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=35, help='sequence length for backpropagation through time')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
parser.add_argument('--cuda', default='cuda', action='store_true', help='use CUDA')
# parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='modeltrans_ner_cc.en.300.pt', help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='', help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=3, help='the number of heads - parallel attention heads')
parser.add_argument('--nsamples', type=int, default=5, help='the number of samples in output file')
parser.add_argument('--siginner', type=int, default=20, help='the number of SigOpt inner training loop')
parser.add_argument('--nfullexp', type=int, default=7, help='the number of full experiments for averages')
parser.add_argument('--pretrained', type=str, default='true', help='use pretrained embeddings- false or true')
parser.add_argument('--sigopt', type=str, default='false', help='use SigOpt optimization or not')
args = parser.parse_args()


def batchify(data, bsz):
    #data = TEXT.numericalize([data.examples[0].text]) # used for torchtext wikitext
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def train(data_src):
    """One epoch of a training loop"""
    print("Training...")
    model.train()  # Turn on training mode
    # total_loss = 0.0
    #ntokens = len(corpus.dictionary)    #len(TEXT.vocab.stoi)
    epoch_loss, train_steps, train_loss = 0, 0, 0
    #for i in trange(0, data_src.size(0) - 1, args.bptt):   #tqdm or trange prints progress bar/statistics during training
    for batch in tqdm(data_src):
        # model.reset_history()
        optimizer.zero_grad()
        batch_input_ids, batch_input_masks, batch_labels = batch
        batch_input_ids = batch_input_ids.to(device)
        batch_labels = batch_labels.to(device)
        tag_scores = model(batch_input_ids)
        try:
            # flatten both inputs in the criterion with view(-1)
            loss = criterion(tag_scores.view(-1, ntokens), batch_labels.view(-1))
        except ValueError:
            print("Invalid training sample")
            continue  # next iteration
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_steps += 1
    epoch_loss = train_loss / train_steps
    return epoch_loss


def evaluate(data_src, true_labels):
    """One epoch of evaluation (or validation) loop"""
    print("Evaluation...")
    predictions = []
    counter = 0
    eval_loss, eval_acc, eval_steps = 0, 0, 0
    model.eval()
    for batch in tqdm(data_src):
        pred2 = []
        with torch.no_grad():                                       # no gradient computation during validation
            vbatch_inputs, vbatch_masks, vbatch_tags = batch
            vbatch_inputs = vbatch_inputs.to(device)
            vbatch_tags = vbatch_tags.to(device)
            tag_scores = model(vbatch_inputs)
            tag_scores = tag_scores.view(-1, ntokens)   # flatten
            try:
                loss = criterion(tag_scores, vbatch_tags.view(-1))
            except ValueError:
                print("Invalid validation sample")
                continue                                            # next iteration
            eval_loss += loss.item()
            eval_steps += 1
            # for computing our metrics the next few lines are necessary because we need to make sense
            # of the tag_scores (tensors) returned. In the tensor, the predicted tag is the maximum scoring tag,
            # hence we identify the corresponding index and map them to our tag/label dictionary.
            pred_values, indices = torch.max(tag_scores, dim=-1)    # flatten and return max values & their indices
            indices = indices.to("cpu")                             # copy to cpu before changing to numpy
            ind_num = indices.numpy()                               # change to numpy for easy manipulation
            pred1 = {k: v for k, v in tag_to_ix.items() if v in ind_num}  # dict comprehension: unique predictions
            for v in ind_num:                                       # loop over indices to return predicted keys
                for k in pred1:
                    if pred1[k] == v:
                        pred2.append(k)     # TODO: list comprehension
            pred2 = list(list_into_chunks(pred2, args.max_seq))      # split flattened list into multiple lists of max_seq each
            # for accurate metrics, we need to remove padding
            for i in range(len(pred2)):
                index_no = counter * args.batch_size + i             # ensures loop through every value in list
                max_len_original = len(true_labels[index_no])
                pred2[i] = pred2[i][:max_len_original]          # remove padding from predicitions - to the length of original
            for i in range(len(pred2)):
                predictions.append(pred2[i])
            counter += 1
    val_loss = eval_loss / eval_steps
    val_values = calc_metrics(true_labels, predictions)
    return val_loss, val_values, predictions


def calc_metrics(true_labels, predictions):
    """ returns evaluation in this order: [eval_acc, f1, precision, recall] """
    eval_acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    val_values = [eval_acc, f1, precision, recall]
    return val_values


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'ner_train.csv'))
        self.valid = self.tokenize(os.path.join(path, 'ner_val.csv'))
        self.test = self.tokenize(os.path.join(path, 'ner_test.csv'))

    def tokenize(self, path):
        """ Tokenizes a text file. """
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp_dmodel_emb, nhead, nhid, nlayers, dropout=0.5, pretrained_model=0, sigopt=0):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp_dmodel_emb, dropout)
        if sigopt == 0:      # not using SigOpt hyper-parameter optimization
            encoder_layers = TransformerEncoderLayer(ninp_dmodel_emb, nhead, nhid, dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        else:
            encoder_layers = TransformerEncoderLayer(ninp_dmodel_emb, sigopt['nhead'], nhid, dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, sigopt['nlayers'])
        if pretrained_model == 0:       # not use pre-trained word vectors
            self.encoder = nn.Embedding(ntoken, ninp_dmodel_emb)
        else:
            self.weights = torch.FloatTensor(pretrained_model.wv.vectors)
            self.weights.requires_grad = False          # freeze weights - essential for optimal results
            self.encoder = nn.Embedding.from_pretrained(self.weights)

        self.ninp_dmodel_emb = ninp_dmodel_emb
        self.decoder = nn.Linear(ninp_dmodel_emb, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp_dmodel_emb)
        src = self.pos_encoder(src)             # (64, 75, 300)
        output = self.transformer_encoder(src, self.src_mask)      #  (64, 75, 300) : (64, 75, 300) (64, 64)
        output = self.decoder(output)           # (64, 75, 2894)
        return F.log_softmax(output, dim=-1)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def ids_to_sents(id_tensor, vocab, join=None):
    """ Convert word ids to sentences """
    if isinstance(id_tensor, torch.LongTensor):
        ids = id_tensor.transpose(0, 1).contiguous().view(-1)
    elif isinstance(id_tensor, np.ndarray):
        ids = id_tensor.transpose().reshape(-1)
    batch = [vocab.itos[ind] for ind in ids]    # denumericalize
    if join is None:
        return batch
    else:
        return join.join(batch)


def preprocessdata(sentences, labels):
    data_ids, tags_ids, tags_record, tags_record2, data_record = [], [], [], [], []
    # Shuffle & join everything first, so we don't have to do it later scattering order of masking
    train_data, t_data, train_tags, t_tags = train_test_split(sentences, labels, test_size=0.15, shuffle=True)
    train_data, val_data, train_tags, val_tags = train_test_split(train_data, train_tags, test_size=0.176, shuffle=True)
    sentences = train_data + val_data + t_data
    labels = train_tags + val_tags + t_tags

    for rg in range(len(sentences)):
        data_ids.append(prepare_sequence(sentences[rg].split(), word_to_ix))
        tags_ids.append(prepare_sequence(labels[rg], tag_to_ix))      # TODO: list comprehension
        data_record.append(sentences[rg].split())                     # keep original format for use later
        tags_record.append(labels[rg])                                # keep original format for metrics use

    # Padding before converting to tensors for batching
    data_inputs = [pad_seq(ii, tag_to_ix, max_len=args.max_seq) for ii in data_ids]
    tags = [pad_seq(ti, tag_to_ix, max_len=args.max_seq, seq_type='tag_ids') for ti in tags_ids]

    # masking to ignore padded items by giving floating value above 0.0 (1.0 in this case) to ids in input sequence
    attention_masks = [[float(i>0) for i in ii] for ii in data_inputs]

    # data set split (including masked data split if padding) | Padding (of both features & labels), masking & custom
    # loss are required if batching, which is faster but cumbersome. Otherwise, it's not compulsory but slower loading
    train_data, t_data, train_tags, t_tags = train_test_split(data_inputs, tags, test_size=0.15, shuffle=False)       # for 70:15:15
    temp_tr = train_data              # needed in the mask section
    train_data, val_data, train_tags, val_tags = train_test_split(train_data, train_tags, test_size=0.176, shuffle=False)  # for 70:15:15
    train_masks, test_masks, i1, i2 = train_test_split(attention_masks, data_inputs, test_size=0.15, shuffle=False)
    train_masks, val_masks, i3, i4 = train_test_split(train_masks, temp_tr, test_size=0.176, shuffle=False)
    # Now split the record sets according to the above also
    rec_traindata, rec_tdata, rec_traintags, rec_ttags = train_test_split(data_record, tags_record, test_size=0.15, shuffle=False)
    rec_traindata, rec_valdata, rec_traintags, rec_valtags = train_test_split(rec_traindata, rec_traintags, test_size=0.176, shuffle=False)

    # convert data to tensors
    train_inputs = torch.tensor(train_data)
    train_tags = torch.tensor(train_tags)
    val_inputs = torch.tensor(val_data)
    val_tags = torch.tensor(val_tags)
    test_inputs = torch.tensor(t_data)
    test_tags = torch.tensor(t_tags)
    train_masks = torch.tensor(train_masks)
    val_masks = torch.tensor(val_masks)
    test_masks = torch.tensor(test_masks)

    # pack inputs into tensordataset & dataloader
    train_tensor = TensorDataset(train_inputs, train_masks, train_tags)
    train_sampler = RandomSampler(train_tensor)
    train_dloader = DataLoader(train_tensor, sampler=train_sampler, batch_size=args.batch_size)
    #print(next(iter(train_dloader)))
    val_tensor = TensorDataset(val_inputs, val_masks, val_tags)
    val_sampler = SequentialSampler(val_tensor)
    val_dloader = DataLoader(val_tensor, sampler=val_sampler, batch_size=args.batch_size)
    test_tensor = TensorDataset(test_inputs, test_masks, test_tags)
    test_sampler = SequentialSampler(test_tensor)
    test_dloader = DataLoader(test_tensor, sampler=test_sampler, batch_size=args.batch_size)

    # to be used for metrics
    val_true_labels = rec_valtags
    val_sent_samples = rec_valdata
    test_true_labels = rec_ttags
    test_sent_samples = rec_tdata

    return train_dloader, val_dloader, test_dloader, val_sent_samples, val_true_labels, test_sent_samples, test_true_labels


if __name__ == '__main__':
    torch.manual_seed(args.seed)  # set for reproducibility
    device = torch.device(args.cuda)        # initialize device
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            device = torch.device("cuda" if args.cuda else "cpu")
    else:
        device = torch.device("cpu")
    data = pd.read_csv("data/ner_dataset.csv", encoding="latin1").fillna(method="ffill")
    # gk = data.groupby('Sentence #')
    # print(gk.first())
    get_sent = MakeSentence(data)           # instantiate sentence maker
    sentences = [" ".join(s[0] for s in sent) for sent in get_sent.sentences]   # concat data (originally in tokens) into sentences
    # sentences = sentences[:20]
    labels = [[s[2] for s in sent] for sent in get_sent.sentences]      # construct true labels for each sentence
    tags_vals = list(set(data["Tag"].values))                           # generate set of unique labels
    vocab = list(set(data["Word"].values))                              # generate vocab/unique data vales
    tag_to_ix = {t: i for i, t in enumerate(tags_vals)}                 # dictionary of labels/tags
    word_to_ix = {j: k for k, j in enumerate(vocab)}                    # dictionary of vocab/data
    # # print(tag_to_ix)
    # # tag_cnt = [t for t in data["Tag"] if t == "O"] for checking data distribution balance
    ntokens = len(vocab)
    pretrained_model = 0        # Initiliaze to 0 for status check when called in model
    if args.pretrained == 'true':
        print("Using Pretrained vectors...")
        #pretrained_model = KeyedVectors.load("en_ft_model_w8s1h1.bin")
        pretrained_model = fasttext.load_facebook_vectors("cc.en.300.bin")
        #pretrained_model = KeyedVectors.load_word2vec_format("w2v_ft_model_w8s0h1.bin", binary=True)
        # Using original Facebook library to load
        # import fasttext.util
        # fasttext.util.download_model('en', if_exists='ignore')  # English
        # ft = fasttext.load_model('cc.en.300.bin')
    criterion = nn.CrossEntropyLoss()
    # Initialize key values
    assignments = 0      # for SigOpt
    model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout, pretrained_model, assignments).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #, betas=(0.7, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # Full training & evaluation | repeated at least 5 times in order to take averages & significance test
    for exp in range(1, args.nfullexp + 1):
        print("Start full experiment {}".format(exp))
        best_val_loss = None # float("inf") - for torchtext
        #  if os.path.exists(args.save):
            #  try
                #  os.remove(args.save)
            #  except OSError:
                #  print("File delete error")
        train_dloader, val_dloader, test_dloader, val_sent_samples, val_true_labels, test_sent_samples, test_true_labels = preprocessdata(sentences, labels)

        if args.sigopt == 'true':       # SigOpt experiment for optimization
            conn, sigoptexp = connfunc("transformerencoder_default_embed")
            optimizers = {
                'rmsprop': torch.optim.RMSprop,
                'adam': torch.optim.Adam
                # 'sgd': torch.optim.SGD,
                # 'adw': torch.optim.AdamW,
                # 'asgd': torch.optim.ASGD
            }
            # SigOpt optimization loop
            for _ in range(sigoptexp.observation_budget):
                suggestion = conn.experiments(sigoptexp.id).suggestions().create()
                assignments = suggestion.assignments
                model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout, pretrained_model, assignments).to(device)
                optimizer = optimizers[assignments['optimizer']](model.parameters(), lr=args.lr) # betas=(0.7, 0.99))
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
                F1 = 0.0                    # initialize F1 score for SigOpt evaluation
                for r in range(args.siginner):
                    train_loss = train(train_dloader)
                    val_loss, val_values, predictions = evaluate(val_dloader, val_true_labels)
                    if F1 < val_values[1]:       # compare F1 scores to save highest
                        F1 = val_values[1]
                conn.experiments(sigoptexp.id).observations().create(
                    suggestion=suggestion.id,
                    value=F1                      # F1 score to determine the best
                )
            assignments = conn.experiments(sigoptexp.id).best_assignments().fetch().data[0].assignments   # fetch best
            print(assignments) 
            # SigOpt-tuned best model to fully train with
            model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout, pretrained_model, assignments).to(device)
            optimizer = optimizers[assignments['optimizer']](model.parameters(), lr=args.lr)  #, betas=(0.7, 0.99))
        args.sigopt = "false"           # turn off SigOpt operation
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(train_dloader)
            val_loss, val_values, predictions = evaluate(val_dloader, val_true_labels)
            epoch_time_elapsed = time.time() - epoch_start_time
            print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, train_loss, val_loss))        
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:        # create file but deletes implicitly 1st if already exists
                    torch.save(model.state_dict(), f)    # save best model's learned parameters (based on lowest loss)
                best_val_loss = val_loss
                # best_model = model
            # log to CSV files | Note: val_values = [eval_acc, f1, precision, recall]
            with open(args.ofile1, "a+") as f:  # 12 items
                s = f.write("Epoch: " + str(epoch) + "\n" + "Val Predicted tag samples 5: " + str(predictions[:args.nsamples]) + "\n" + "Val Sentence samples 5: " + str(val_sent_samples[:args.nsamples]) + "\n" + "Val True tag samples 5: " + str(val_true_labels[:args.nsamples]) + "\n" + "Epoch time elapsed: " + str(epoch_time_elapsed) + "\n" + "Training loss: " + str(train_loss) + "\n" + "Validation loss: " + str(val_loss) + "\n" + "Validation accuracy: " + str(val_values[0]) + "\n" + "F1: " + str(val_values[1]) + "\n" + "Precision: " + str(val_values[2]) + "\n" + "Recall: " + str(val_values[3]) + "\n" + "Best loss: " + str(best_val_loss) + "\n")
        # load the optimized, saved model & evaluate the test set
        with open(args.save, 'rb') as f:
            model.load_state_dict(torch.load(f))    # load learned parameters (with capability for retraining)
        test_loss, test_values, predictions = evaluate(test_dloader, test_true_labels)
        with open(args.ofile2, "a+") as f:  # 8 items
            s = f.write("Val Predicted tag samples 5: " + str(predictions[:args.nsamples]) + "\n" + "Val Sentence samples 5: " + str(test_sent_samples[:args.nsamples]) + "\n" + "Val True tag samples 5: " + str(test_true_labels[:args.nsamples]) + "\n" + "Test loss: " + str(test_loss) + "\n" + "Test accuracy: " + str(test_values[0]) + "\n" + "F1: " + str(test_values[1]) + "\n" + "Precision: " + str(test_values[2]) + "\n" + "Recall: " + str(test_values[3]) + "\n")
