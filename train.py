import torch
import torch.optim as optim
import torch.nn as nn

from model import LSTMTagger
from model.batch import Batch
from utils.general_utils import read_conll, get_pos_vocab, load_glove, load_saved_glove, load_pos_vocab
from utils import PadSequence
from sklearn.metrics import f1_score

#---------------------------------------------------
# configuration
#---------------------------------------------------
conll_train = 'dataset/HAMSTER/trainset.conll'
conll_val = 'dataset/HAMSTER/validationset.conll'
conll_test = 'dataset/HAMSTER/testset.conll'
pos_vocab_file = 'dataset/pos_vocab.txt'
vocab_file = 'dataset/vocab.txt'
glove = 'dataset/Embeddings/glove.6B.100d.txt'
saved_glove = 'dataset/embedding.npy'
num_epochs = 50
batch_size = 64
embed_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----------------------------------------------------
# training
#----------------------------------------------------
get_pos_vocab(conll_train, conll_val, conll_test, output=pos_vocab_file)
vocab, embedding = load_glove(glove, dim=embed_dim, save_dir='dataset')
pos_vocab = load_pos_vocab(pos_vocab_file)

# convert to ids
words, pos = read_conll(conll_train)
word_ids = [[vocab.get(word, 1) for word in sentence] for sentence in words]
pos_ids = [[pos_vocab.get(pos) for pos in sentence] for sentence in pos]

test_words, test_pos = read_conll(conll_test)
test_word_ids = [[vocab.get(word, 1) for word in sentence] for sentence in words]
test_pos_ids = [[pos_vocab.get(pos) for pos in sentence] for sentence in pos]

embedding = torch.from_numpy(embedding).float()
model = LSTMTagger(embedding, embed_dim, 100, 2, len(pos_vocab)).to(device)
optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad==True], lr=0.001)
criterion = nn.NLLLoss()
for epoch in range(num_epochs):
    batch = Batch(word_ids, pos_ids, batch_size=batch_size)
    total_step = len(batch)
    i = 0
    for inputs, labels in batch:
        i += 1
        pad_words_obj = PadSequence(inputs, [len(inputs), 100])
        padded_inputs = torch.Tensor(pad_words_obj.embedding).long().to(device)
        padded_inputs_lens = torch.Tensor(pad_words_obj.lengths).long().to(device)

        outputs = model(padded_inputs, padded_inputs_lens)

        pad_pos_obj = PadSequence(labels, [len(inputs), outputs.size(2)])   # batch, num_classes, seq_len
        padded_labels = torch.Tensor(pad_pos_obj.embedding).long().to(device)

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
            y_true = []
            y_preds = []
            test_batch = Batch(test_word_ids,test_pos_ids, batch_size=batch_size)
            for test_inputs, test_labels in test_batch:
                test_pad_words_obj = PadSequence(test_inputs, [len(inputs), 100])
                test_padded_inputs = torch.Tensor(pad_words_obj.embedding).long().to(device)
                test_padded_inputs_lens = torch.Tensor(pad_words_obj.lengths).long().to(device)

                test_outputs = model(test_padded_inputs, test_padded_inputs_lens)

                test_pad_pos_obj = PadSequence(test_labels, [len(inputs), test_outputs.size(2)])
                test_padded_labels = torch.Tensor(pad_pos_obj.embedding).long().to(device)

                _, preds = torch.max(test_outputs.data, 1)
                preds = preds.view(-1)
                test_padded_labels = test_padded_labels.view(-1)

                valid_indices = torch.nonzero(test_padded_labels).squeeze()
                selected_preds = torch.index_select(preds, 0, valid_indices)
                selected_labels = torch.index_select(test_padded_labels, 0, valid_indices)

                total += selected_labels.size(0)
                correct += (selected_preds == selected_labels).sum().item()

                tmp_y_preds = selected_preds.tolist()
                tmp_y_true = selected_labels.tolist()
                y_preds += tmp_y_preds
                y_true += tmp_y_true

            f1 = f1_score(y_true, y_preds, average='weighted')
            print('Test Accuracy of the model: {:.2f} %, f1-score: {:.2f}'.
                  format(100 * correct / total, 100 * f1))