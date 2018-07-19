import torch
import torch.optim as optim
import torch.nn as nn

from model import LSTMTagger
from model.batch import Batch
from utils.general_utils import read_conll, get_pos_vocab, load_glove, load_saved_glove, load_pos_vocab
from utils import PadSequence


#---------------------------------------------------
# configuration
#---------------------------------------------------
conll_train = 'dataset/HAMSTER/trainset.conll'
conll_val = 'dataset/HAMSTER/validationset.conll'
conll_test = 'dataset/HAMSTER/testset.conll'
pos_vocab_file = 'dataset/pos_vocab.txt'
vocab_file = 'dataset/vocab.txt'
glove = 'dataset/glove.6B.50d.txt'
saved_glove = 'dataset/embedding.npy'
num_epochs = 10

#----------------------------------------------------
# training
#----------------------------------------------------
get_pos_vocab(conll_train, conll_val, conll_test, output=pos_vocab_file)
vocab, embedding = load_glove(glove, dim=50, save_dir='dataset')
pos_vocab = load_pos_vocab(pos_vocab_file)

# convert to ids
words, pos = read_conll(conll_train)
word_ids = [[vocab.get(word, 1) for word in sentence] for sentence in words]
pos_ids = [[pos_vocab.get(pos) for pos in sentence] for sentence in pos]

test_words, test_pos = read_conll(conll_test)
test_word_ids = [[vocab.get(word, 1) for word in sentence] for sentence in words]
test_pos_ids = [[pos_vocab.get(pos) for pos in sentence] for sentence in pos]

embedding = torch.from_numpy(embedding).float()
model = LSTMTagger(embedding, 50, 100, 2, 51)
optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad==True], lr=0.001)
criterion = nn.NLLLoss()
for epoch in range(num_epochs):
    batch = Batch(word_ids, pos_ids, batch_size=16)
    total_step = len(batch)
    i = 0
    for inputs, labels in batch:
        i += 1
        pad_words_obj = PadSequence(inputs, [len(inputs), 100])
        padded_inputs = torch.Tensor(pad_words_obj.embedding).long()
        padded_inputs_lens = torch.Tensor(pad_words_obj.lengths).long()

        outputs = model(padded_inputs, padded_inputs_lens)

        pad_pos_obj = PadSequence(labels, [len(inputs), outputs.size(2)])   # batch, num_classes, seq_len
        padded_labels = torch.Tensor(pad_pos_obj.embedding).long()

        loss = criterion(outputs, padded_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        with torch.no_grad():
            correct = 0
            total = 0
            test_batch = Batch(test_word_ids,test_pos_ids, batch_size=16)
            for test_inputs, test_labels in test_batch:
                test_pad_words_obj = PadSequence(test_inputs, [len(inputs), 100])
                test_padded_inputs = torch.Tensor(pad_words_obj.embedding).long()
                test_padded_inputs_lens = torch.Tensor(pad_words_obj.lengths).long()

                test_outputs = model(test_padded_inputs, test_padded_inputs_lens)

                test_pad_pos_obj = PadSequence(test_labels, [len(inputs), test_outputs.size(2)])
                test_padded_labels = torch.Tensor(pad_pos_obj.embedding).long()

                _, preds = torch.max(test_outputs.data, 1)
                preds = preds.view(-1)
                test_padded_labels = test_padded_labels.view(-1)

                valid_indices = torch.nonzero(test_padded_labels).squeeze()
                selected_preds = torch.index_select(preds, 0, valid_indices)
                selected_labels = torch.index_select(test_padded_labels, 0, valid_indices)

                total += selected_labels.size(0)
                correct += (selected_preds == selected_labels).sum().item()
            print('Test Accuracy of the model: {} %'.format(100 * correct / total))