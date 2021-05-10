def read_embeddings(filename: str) -> (dict, list):
    word2idx = {}
    weights = []
    with open(filename, encoding='UTF-8') as f:
        idx = 2
        for line in f.readlines():
            tokens = line.split()
            word = tokens[0]
            word2idx[word] = idx
            idx += 1
            embedding = [float(w) for w in tokens[1:]]
            weights.append(embedding)
    emb_len = len(weights[0])
    zeros = [0.0 for _ in range(emb_len)]
    weights.insert(0, zeros)
    ones = [1.0 for _ in range(emb_len)]
    weights.insert(1, ones)
    return word2idx, weights
    return word2idx, weights


def read_char_tag(filename: str) -> (list, list):
    char_list, tag_list = [], []
    sent_char_list, sent_tag_list = [], []
    with open(filename, mode='r', encoding='UTF-8') as f:
        for line in f.readlines():
            if line != '\n':
                tokens = line.split()
                sent_char_list.append(tokens[0])
                sent_tag_list.append(tokens[1])
            else:
                char_list.append(sent_char_list)
                tag_list.append(sent_tag_list)
                sent_char_list = []
                sent_tag_list = []
    return char_list, tag_list


def write_char_tag(char_list: list, tag_list: list, filename: str):
    with open(filename, 'w', encoding='UTF-8') as f:
        for sent, sent_tags in zip(char_list, tag_list):
            for ch, tag in zip(sent, sent_tags):
                print(ch + '\t' + tag, file=f)
            print(file=f)
            f.flush()


def score(pred: list, actual: list) -> (float, float, float):
    correct, incorrect = 0, 0
    for pred_sent, actual_sent in zip(pred, actual):
        for pred_tag, actual_tag in zip(pred_sent, actual_sent):
            if pred_tag == actual_tag:
                correct += 1
            else:
                incorrect += 1
    pred_ner_count = count_ner(pred)
    actual_ner_count = count_ner(actual)
    ner_count = 0
    for pred_sent, actual_sent in zip(pred, actual):
        in_ner = False
        for pred_tag, actual_tag in zip(pred_sent, actual_sent):
            if pred_tag[0] == 'S' and pred_tag == actual_tag:
                ner_count += 1
            elif pred_tag[0] == 'B' and pred_tag == actual_tag:
                in_ner = True
            elif pred_tag[0] == 'E' and pred_tag == actual_tag and in_ner:
                ner_count += 1
                in_ner = False
    accuracy = correct / (correct + incorrect) * 100
    precision = ner_count / pred_ner_count if pred_ner_count > 0 else 0.0
    recall = ner_count / actual_ner_count
    F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision * 100, recall * 100, F1 * 100


def count_ner(para: list) -> int:
    count = 0
    for sent in para:
        for tag in sent:
            if tag[0] == 'S' or tag[0] == 'B':
                count += 1
    return count

