import numpy as np
import os

#TODO: add logging

PAD = "#PAD#"
UNK = "#UNK#"

def read_conll(file):
    with open(file, 'r') as f:
        conll_words = []
        conll_pos = []
        sentence_words = []
        sentence_pos = []

        for line in f:
            row = line.strip().split()
            if row:
                word = row[1]
                pos = row[3]

                sentence_words.append(word)
                sentence_pos.append(pos)
            else:
                conll_words.append(sentence_words)
                conll_pos.append(sentence_pos)

                sentence_words = []
                sentence_pos = []

    assert len(conll_pos) == len(conll_words)

    return conll_words, conll_pos


def get_pos_vocab(*args, output):
    pos_vocab = set()

    for file in args:
        with open(file, 'r') as f:
            for line in f:
                row = line.strip().split()
                if row:
                    pos_vocab.add(row[3])

    with open(output, 'w') as out:
        for pos in pos_vocab:
            out.write(pos + '\n')
    print("Size of pos vocab:", len(pos_vocab))


def load_glove(file, dim, save_dir=None, dtype=np.float32):
    # files were created
    if save_dir:
        if os.path.isfile(save_dir + '/vocab.txt') and os.path.isfile(save_dir + '/embedding.npy'):
            print("Glove relating files already created")
            return load_saved_glove(save_dir + '/vocab.txt', save_dir + '/embedding.npy')

    # create files for the first time
    vocab = [PAD, UNK]
    with open(file, 'r') as f:
        for line in f:
            vocab.append(line.strip().split()[0])

    embedding = np.zeros((len(vocab), dim), dtype=dtype)
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            row = [float(x) for x in line.strip().split()[1:]]
            row = np.array(row, dtype=dtype)
            embedding[i + 2] = row

    embedding[0] = np.mean(embedding[2:], axis=0)

    if save_dir:
        with open(save_dir + '/vocab.txt', 'w') as out:
            for word in vocab:
                out.write(word + '\n')

        np.save(save_dir + '/embedding', embedding)

    # TODO: add hdf5 supports
    return vocab, embedding



def load_saved_glove(vocab_file, glove_file):
    with open(vocab_file, 'r') as f:
        vocab = [line.strip() for line in f]
    vocab = {word: i for i, word in enumerate(vocab)}
    embedding = np.load(glove_file)

    return vocab, embedding


def load_pos_vocab(pos_vocab_file):
    with open(pos_vocab_file) as f:
        pos_vocab = [line.strip() for line in f]
    pos_vocab.insert(0, PAD)
    pos_vocab = {pos: i for i, pos in enumerate(pos_vocab)}
    return pos_vocab