# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from random import randint
from common import *
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
import jieba

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx


def sent_to_idx(sent, word_to_idx):
    indices = [word_to_idx[w] if w in word_to_idx else 1 for w in sent]
    return torch.tensor(indices, dtype=torch.long)


def tag_to_idx(tags, tag_to_idx):
    return torch.tensor([tag_to_idx[t] for t in tags])


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    # am = argmax(vec)
    # max_score = torch.index_select(vec, 1, am)
    max_score, _ = torch.max(vec, 1)
    max_score_broadcast = max_score.view(-1, 1).expand(-1, vec.size()[1])

    t1 = torch.exp(vec - max_score_broadcast)
    t2 = torch.sum(t1, dim=1)
    t3 = torch.log(t2)

    # return max_score + \
           # torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    return max_score + t3


class BiLSTM_CRF(nn.Module):

    def __init__(self, filename, hidden_dim, sent_list, tag_list, mode, learning_rate=0.01, batch_size=1, dropout1=0, dropout2=0):
        super(BiLSTM_CRF, self).__init__()
        word_to_idx, weights = read_embeddings(filename)
        self.epoch_n = 100
        self.drop1 = dropout1
        self.drop2 = dropout2
        self.embedding_dim = len(weights[0])
        self.hidden_dim = hidden_dim
        self.word_to_idx = word_to_idx
        self.vocab_size = len(self.word_to_idx)
        self.tag_to_idx, self.idx_to_tag = get_tags(tag_list)
        self.tags_size = len(self.tag_to_idx)
        self.mode = mode
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=True)

        # Dropout layer applied to embeddings
        self.dropoutLayer1 = nn.Dropout(dropout1)

        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tags_size)

        # Dropout layer applied to lstm features
        self.dropoutLayer2 = nn.Dropout(dropout2)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tags_size, self.tags_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.tag_to_idx[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_idx[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()
        # train the data in the end
        # self.fit(sent_list, tag_list)

    def fit(self, sent_list: list, tag_list):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # create figure of losses
        fig, ax = plt.subplots()

        # Read training and test data for validation on each epoch end
        # train_char, train_tag = read_char_tag('data/train.txt')
        # test_char, test_tag = read_char_tag('data/test.txt')

        # Create and open log file
        log_dir = os.path.join(os.getcwd(), 'logs')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        log_name = "gpu_epoch{}_{}_{}_{}_{}_{}_training_log.txt".format(self.epoch_n, self.batch_size, self.hidden_dim,
                                                                        self.learning_rate, self.drop1, self.drop2)
        log_path = os.path.join(log_dir, log_name)
        txt_log = open(log_path, mode='wt', buffering=1)

        total_step = math.ceil(len(sent_list) / self.batch_size)

        # Make sure prepare_sequence from earlier in the LSTM section is loaded
        for epoch in range(self.epoch_n):  # again, normally you would NOT do 300 epochs, it is toy data
            start = time.time()
            s = 0
            losses = []
            for step in range(total_step):
                start_idx = self.batch_size * s
                end_idx = start_idx + self.batch_size

                sentence = sent_list[start_idx: end_idx]
                tags = tag_list[start_idx: end_idx]

                s += 1
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.zero_grad()

                # Step 2. Get our inputs ready for the network, that is,
                # turn them into Tensors of word indices.
                sentence_in = [sent_to_idx(ss, self.word_to_idx) for ss in sentence]
                sent_lens = torch.tensor([len(ss) for ss in sentence]).cuda()
                sentence_in = pad_sequence(sentence_in, batch_first=True).cuda()
                # sent_lens.cuda()

                targets = [torch.tensor([self.tag_to_idx[t] for t in tt], dtype=torch.long) for tt in tags]
                targets = pad_sequence(targets, batch_first=True, padding_value=-1)

                # Step 3. Run our forward pass.
                loss = self.neg_log_likelihood(sentence_in, targets, sent_lens)

                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss.backward()
                optimizer.step()

                # add to losses
                loss_value = loss.item()
                # print('epoch {}, sentence {}, loss={}'.format(epoch+1,s,loss_value))
                losses.append(loss_value)

            duration = time.time() - start
            avg_loss = sum(losses) / s

            # if self.mode == "word":
            #     word_list_train = word_seg(train_char)
            #     chunk_list_train = self.predict(word_list_train)
            #     word_tag_train = chunk2tag(word_list_train, chunk_list_train)
            #     word_score_train = score(word_tag_train, train_tag)
            #     precision_train, recall_train, f1_train = word_score_train
            #
            #     word_list = word_seg(test_char)
            #     chunk_list = self.predict(word_list)
            #     word_tag = chunk2tag(word_list, chunk_list)
            #     word_score = score(word_tag, test_tag)
            #     precision, recall, f1 = word_score
            #
            # elif self.mode == "char":
            #     char_tag_train = self.predict(train_char)
            #     char_score_train = score(char_tag_train, train_tag)
            #     precision_train, recall_train, f1_train = char_score_train
            #
            #     char_tag = self.predict(test_char)
            #     char_score = score(char_tag, test_tag)
            #     precision, recall, f1 = char_score

            # precision_train = 0
            # recall_train = 0
            # f1_train = 0
            # precision = 0
            # recall = 0
            # f1 = 0


            print(
                'epoch {}, time = {}, average training loss = {}'.format(
                    epoch + 1, duration, avg_loss))

            if True:
                precision_train, recall_train, f1_train, precision, recall, f1 = self.evaluate('data/train.txt', 'data/test.txt')

            txt_log.write('\t'.join(
                [str(duration), str(avg_loss), str(precision_train), str(recall_train), str(f1_train), str(precision),
                 str(recall), str(f1)]) + '\n')

        txt_log.close()

        # ax.plot(losses)
        # plt.show()
        # sent_idx=randint(0,len(sent_list))
        # sentence=sent_list[sent_idx]
        # tags=tag_list[sent_idx]
        # sentence_in = sent_to_idx(sentence, self.word_to_idx)
        # targets = torch.tensor([self.tag_to_idx[t] for t in tags], dtype=torch.long)
        # loss=self.neg_log_likelihood(sentence_in,targets)
        # writer.add_scalar('training loss',loss.item(),epoch)

    def predict(self, sent_list: list) -> list:
        self.eval()
        tag_list = []
        with torch.no_grad():
            for sent in sent_list:
                precheck_sent = sent_to_idx(sent, self.word_to_idx).cuda()
                _, tag_indices = self(precheck_sent)
                sent_tag = [self.idx_to_tag[i.item()] for i in tag_indices]
                tag_list.append(sent_tag)
        self.train()
        return tag_list

    def evaluate(self, train_file, test_file):
        start_evaluate_time = time.time()
        train_char, train_tag = read_char_tag(train_file)
        test_char, test_tag = read_char_tag(test_file)

        if self.mode == "word":
            word_list_train = word_seg(train_char)
            chunk_list_train = self.predict(word_list_train)
            word_tag_train = chunk2tag(word_list_train, chunk_list_train)
            word_score_train = score(word_tag_train, train_tag)
            precision_train, recall_train, f1_train = word_score_train

            word_list = word_seg(test_char)
            chunk_list = self.predict(word_list)
            word_tag = chunk2tag(word_list, chunk_list)
            word_score = score(word_tag, test_tag)
            precision, recall, f1 = word_score

        elif self.mode == "char":
            char_tag_train = self.predict(train_char)
            char_score_train = score(char_tag_train, train_tag)
            precision_train, recall_train, f1_train = char_score_train

            char_tag = self.predict(test_char)
            char_score = score(char_tag, test_tag)
            precision, recall, f1 = char_score

        evaluate_time = time.time() - start_evaluate_time

        print("Evaluate time:{}".format(evaluate_time))

        print(
            'training precision = {}, training recall = {}, training f1-score = {}, valiadation precision = {}, '
            'validation recall = {}, '
            'validation f1-score = {}'.format(precision_train, recall_train, f1_train, precision, recall, f1))
        return precision_train, recall_train, f1_train, precision, recall, f1


    def init_hidden(self, batch_size=1):
        return (torch.randn(2, batch_size, self.hidden_dim).cuda(),
                torch.randn(2, batch_size, self.hidden_dim).cuda())

    def _forward_alg(self, feats, lens):
        # feats = feats[:l.item]
        _, idx_sort = torch.sort(lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x_sort = torch.index_select(feats, dim=0, index=idx_sort)
        seq_lens_sort = torch.index_select(lens, dim=0, index=idx_sort)

        masks = [[1] * sl + [0] * (seq_lens_sort[0] - sl) for sl in seq_lens_sort]
        ll = np.sum(masks, axis=0)

        feats_trans = torch.transpose(x_sort, 0, 1)
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((feats.size(0), self.tags_size), -10000.).cuda()
        # START_TAG has all of the score.
        for i in range(feats.size(0)):
            init_alphas[i][self.tag_to_idx[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for l, feat in zip(ll, feats_trans):
            alphas_t = []  # The forward tensors at this timestep
            feat = feat[:l]
            feat = torch.transpose(feat, 0, 1)  # T * b
            for next_tag in range(self.tags_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                f = feat[next_tag]  # b
                f = f.view(-1, 1)  # b * 1
                emit_score = f.expand(-1, self.tags_size)  # b * T
                # emit_score = feat[next_tag].view(1, -1).expand(1, self.tags_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1).expand(f.size(0), -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                valid_forward_var = forward_var[:l]
                trunked_forward_var = forward_var[l:]
                next_tag_var = valid_forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            alphas_t = torch.stack(alphas_t)
            alphas_t = torch.transpose(alphas_t, 0, 1)
            forward_var = torch.cat((alphas_t, trunked_forward_var))
        terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return torch.mean(alpha)

    def _get_lstm_features(self, sentence, lens):
        self.hidden = self.init_hidden(sentence.size(0))
        embeds = self.word_embeds(sentence)
        embeds = self.dropoutLayer1(embeds)

        _, idx_sort = torch.sort(lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x_sort = torch.index_select(embeds, dim=0, index=idx_sort)
        seq_lens_sort = torch.index_select(lens, dim=0, index=idx_sort)

        x_packed = pack_padded_sequence(x_sort, seq_lens_sort.cpu(), batch_first=True)

        lstm_out, self.hidden = self.lstm(x_packed, self.hidden)
        # lstm_out = lstm_out.view(len(sentence), len(sentence[0]), self.hidden_dim * 2)
        lstm_out, length = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = torch.index_select(lstm_out, dim=0, index=idx_unsort)

        lstm_out = self.dropoutLayer2(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def _score_sentence(self, feats, tags, lens):
        # Gives the score of a provided tag sequence
        # feats = feats[:l.item]
        # tags = tags[:l.item]

        scores = []
        start_tags = torch.tensor([self.tag_to_idx[START_TAG]] * feats.size(0), dtype=torch.long).view(-1, 1)
        tags = torch.cat([start_tags, tags], dim=1)

        for batch, tag,  l in zip(feats, tags, lens):
            score = torch.zeros(1).cuda()
            for i in range(l):
                feat = batch[i]
                score = score + \
                        self.transitions[tag[i + 1], tag[i]] + feat[tag[i + 1]]
            score = score + self.transitions[self.tag_to_idx[STOP_TAG], tag[l]]
            scores.append(score)
        return torch.mean(torch.cat(scores))

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tags_size), -10000.).cuda()
        init_vvars[0][self.tag_to_idx[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tags_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_idx[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags, lens):
        feats = self._get_lstm_features(sentence, lens)

        ans = []

        # batched_forward_alg = torch.vmap(self._forward_alg)
        # batched_gold_score = torch.vmap(self._gold_score)

        # forward_score = self.batched_forward_alg(feats, lens)
        # gold_score = self._score_sentence(feats, tags, lens)

        # for l, feat, tag in zip(lens, feats, tags):
        #     feat = feat[:l]
        #     tag = tag[:l]
        #     forward_score = self._forward_alg(feat)
        #     gold_score = self._score_sentence(feat, tag)
        #     ans.append(forward_score - gold_score)
        forward_score = self._forward_alg(feats, lens)
        gold_score = self._score_sentence(feats, tags, lens)

        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence.view(1, -1), torch.tensor([sentence.size(0)]).cuda())
        lstm_feats = torch.squeeze(lstm_feats)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


def get_tags(tag_list: list) -> (dict, dict):
    tag2idx, idx2tag = {START_TAG: 0, STOP_TAG: 1}, {0: START_TAG, 1: STOP_TAG}
    idx = 0
    for sent_tag in tag_list:
        for tag in sent_tag:
            if tag not in tag2idx:
                tag2idx[tag] = idx
                idx2tag[idx] = tag
                idx += 1
    tag2idx[START_TAG] = idx
    idx2tag[idx] = START_TAG
    idx += 1
    tag2idx[STOP_TAG] = idx
    idx2tag[idx] = STOP_TAG
    idx += 1
    return tag2idx, idx2tag


def word_seg(char_list: list) -> list:
    # char_list=[['武','汉','加','油']]
    ans = []
    for sen in char_list:
        sentence = ""
        for char in sen:
            sentence += char
        words = jieba.cut(sentence)
        ans.append(list(words))
    # print(ans)
    return ans
    # return [['武汉','加油']]


def chunk2tag(word_list: list, chunk_list: list) -> list:
    # word_list,chunk_list=[['武汉','加油']],[['LOC','O']]
    ans = []
    for sen, chunk in zip(word_list, chunk_list):
        tag_list = []
        for word, tag in zip(sen, chunk):
            if len(word) == 1:
                if tag == "O":
                    tag_list.append(tag)
                else:
                    tag_list.append("S-" + tag)
            else:
                for i in range(len(word)):
                    if tag == "O":
                        tag_list.append(tag)
                    else:
                        if i == 0:
                            tag_list.append("B-" + tag)
                        elif i == len(word) - 1:
                            tag_list.append("E-" + tag)
                        else:
                            tag_list.append("M-" + tag)
        ans.append(tag_list)

    return ans
    # return [['B-LOC','E-LOC','O','O']]

def char2word(sent_list:list,tag_list:list)->(list,list):
    out_tag = []

    word_lists = word_seg(sent_list)
    for sen, tags in zip(word_lists, tag_list):
        length = 0
        new_tag_list = []
        for word in sen:
            tag = tags[length]
            if tag != "O":
                _, tag = tag.split("-")
            new_tag_list.append(tag)
            length += len(word)
        out_tag.append(new_tag_list)

    return word_lists, out_tag

train_char, train_tag = read_char_tag('data/train.txt')
train_word,train_chunk=char2word(train_char,train_tag)
# __init__(self, filename, hidden_dim, sent_list, tag_list, mode, batch_size=1, dropout1=0, dropout2=0)
test_model = BiLSTM_CRF('data/word_vec.txt', hidden_dim=50, sent_list=train_word, tag_list=train_chunk, mode="word",
                        batch_size=64, learning_rate=0.1, dropout1=0, dropout2=0).cuda()
test_model.fit(train_word, train_chunk)