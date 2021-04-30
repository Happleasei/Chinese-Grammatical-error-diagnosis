import os
import re
import codecs

from BERT_BILSTM_CRF.bert import tokenization
from BERT_BILSTM_CRF.utils import convert_single_example

tokenizer = tokenization.FullTokenizer(vocab_file='../BERT_BILSTM_CRF/chinese_L-12_H-768_A-12/vocab.txt',
                                       do_lower_case=True)

from BERT_BILSTM_CRF.data_utils import create_dico, create_mapping, zero_digits
from BERT_BILSTM_CRF.data_utils import iob2, iob_iobes, get_seg_features


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num+=1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        # print(list(line))
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word= line.split()
            assert len(word) >= 2, print([word[0]])
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000

    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]

    dico = create_dico(tags)
    dico['[SEP]'] = len(dico) + 1
    dico['[CLS]'] = len(dico) + 2

    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, max_seq_length, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0].strip() for w in s]
        #chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
        #         for w in string]
        char_line = ' '.join(string)   #使用空格把汉字拼起来
        text = tokenization.convert_to_unicode(char_line)

        if train:
            tags = [w[-1] for w in s]
        else:
            tags = ['O' for _ in string]

        labels = ' '.join(tags)     #使用空格把标签拼起来
        labels = tokenization.convert_to_unicode(labels)

        ids, mask, segment_ids, label_ids = convert_single_example(char_line=text,
                                                                   tag_to_id=tag_to_id,
                                                                   max_seq_length=max_seq_length,
                                                                   tokenizer=tokenizer,
                                                                   label_line=labels)
        data.append([string, segment_ids, ids, mask, label_ids])

    return data


def input_from_line(line, max_seq_length, tag_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    string = [w[0].strip() for w in line]
    # chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
    #         for w in string]
    char_line = ' '.join(string)  # 使用空格把汉字拼起来
    text = tokenization.convert_to_unicode(char_line)

    tags = ['O' for _ in string]

    labels = ' '.join(tags)  # 使用空格把标签拼起来
    labels = tokenization.convert_to_unicode(labels)

    ids, mask, segment_ids, label_ids = convert_single_example(char_line=text,
                                                               tag_to_id=tag_to_id,
                                                               max_seq_length=max_seq_length,
                                                               tokenizer=tokenizer,
                                                               label_line=labels)
    import numpy as np
    segment_ids = np.reshape(segment_ids,(1, max_seq_length))
    ids = np.reshape(ids, (1, max_seq_length))
    mask = np.reshape(mask, (1, max_seq_length))
    label_ids = np.reshape(label_ids, (1, max_seq_length))
    return [string, segment_ids, ids, mask, label_ids]

def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)

