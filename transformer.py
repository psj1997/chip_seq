import numpy as np
import pandas as pd
import torch
from torch import nn
import math
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

batchsize = 64
K = 7
epoches = 100
learning_rate = 0.005

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class Transformer_model(nn.Module):
    def __init__(self, d_model = 512, vocab = 16491, n_head = 8, dim_feedforward=2048, num_layers=6, dropout=0.1):
        super(Transformer_model, self).__init__()

        self.embedding = nn.Embedding(vocab, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, x):
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

class feature_Dataset(Dataset):
    def __init__(self, embedding_lists, label_lists):
        self.embedding_lists = embedding_lists
        self.label_lists = label_lists

    def __getitem__(self, item):
        return self.embedding_lists[item], self.label_lists[item]

    def __len__(self):
        return len(self.embedding_lists)

from sklearn.metrics import mean_squared_error, r2_score
def main(mode = 'train'):
    data = pd.read_csv('data.csv', index_col=None)
    kmer_dict = np.load('kmer_dict.npy', allow_pickle=True).item()

    train_chr = ['chr2L', 'chr2R', 'chr3L']
    val_chr = ['chr4']
    test_chr = ['chr3R']

    train = data[data['seqnames'].isin(train_chr)]
    val = data[data['seqnames'].isin(val_chr)]
    test = data[data['seqnames'].isin(test_chr)]

    features_dict = {'train': None, 'val': None, 'test':None}
    labels_dict = {'train': None, 'val': None, 'test':None}

    for data, name in zip([train, val ,test], ['train', 'val', 'test']):
        print(data.shape)
        features = []
        for seq in data['sequence']:
            temp_feature = []
            for index_k in range(len(seq) - K + 1):
                temp_feature.append(kmer_dict[seq[index_k:index_k + K]])
            features.append(temp_feature)
        features = np.array(features)
        labels = data['scaled_correlation_of_NNv1_Age'].values
        features_dict[name] = features
        if name == 'train':
            min_value = np.min(labels)
            max_value = np.max(labels)
            labels = (labels - min_value) / (max_value - min_value)
            labels_dict[name] = labels
            print(max_value)
            print(min_value)
        else:
            labels_dict[name] = labels

    dataset_dict = {'train':None, 'val':None, 'test':None}

    for name in ['train', 'val', 'test']:
        dataset = feature_Dataset(features_dict[name], labels_dict[name])
        if name in ['val', 'test']:
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batchsize,
                shuffle=False,
                num_workers=0)
        else:
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batchsize,
                shuffle=True,
                num_workers=0)
        dataset_dict[name] = dataloader

    if mode == 'train':
        model = Transformer_model(d_model=512, dim_feedforward=512).to(device)
        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

        for epoch in range(epoches):
            losses = []
            for train_input, train_labels in tqdm(dataset_dict['train']):
                model.train()
                train_input = train_input.to(device)
                train_labels = train_labels.to(device)
                out = model(train_input).squeeze(dim = 1)
                optimizer.zero_grad()
                loss = mse_loss(out.double(), train_labels.double()).to(device)
                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().detach().numpy().tolist())

            print(f'Epoch: {epoch}, average losses: {np.mean(losses)}')
            torch.save(model, f'model_{epoch}.h5')

            with torch.no_grad():
                model.eval()
                train_output_list = []
                train_labels_list = []
                for train_input, train_labels in tqdm(dataset_dict['train']):
                    train_input = train_input.to(device)
                    out = model(train_input)
                    out = out.cpu().detach().numpy().tolist()
                    out = [temp[0] for temp in out]
                    train_output_list.extend(out)
                    train_labels_list.extend(train_labels.numpy().tolist())

    else:
        min_MSE = 1e9
        min_epoch = 0

        for epoch in range(epoches):
            model = torch.load(f'model_{epoch}.h5')
            val_output_list = []
            val_labels_list = []
            for val_input, val_labels in dataset_dict['val']:
                val_input = val_input.to(device)
                out = model(val_input)
                out = out.cpu().detach().numpy().tolist()
                out = [temp[0] for temp in out]
                val_output_list.extend(out)
                val_labels_list.extend(val_labels.numpy().tolist())
            val_output_list = [temp * (max_value - min_value) + min_value for temp in val_output_list]
            if mean_squared_error(val_labels_list, val_output_list) < min_MSE:
                min_MSE = mean_squared_error(val_labels_list, val_output_list)
                min_epoch = epoch

        print(min_epoch)
        print(f'min val MSE: {min_MSE}')

        model = torch.load(f'model_{min_epoch}.h5')

        train_output_list = []
        train_labels_list = []
        for train_input, train_labels in dataset_dict['train']:
            train_input = train_input.to(device)
            out = model(train_input)
            out = out.cpu().detach().numpy().tolist()
            out = [temp[0] for temp in out]
            train_output_list.extend(out)
            train_labels_list.extend(train_labels.numpy().tolist())
        train_output_list = [temp * (max_value - min_value) + min_value for temp in train_output_list]
        print(f'train best MSE: {mean_squared_error(train_labels_list, train_output_list)}')

        test_output_list = []
        test_labels_list = []
        for test_input, test_labels in dataset_dict['test']:
            test_input = test_input.to(device)
            out = model(test_input)
            out = out.cpu().detach().numpy().tolist()
            out = [temp[0] for temp in out]
            test_output_list.extend(out)
            test_labels_list.extend(test_labels.numpy().tolist())
        test_output_list = [temp * (max_value - min_value) + min_value for temp in test_output_list]
        print(f'test best MSE: {mean_squared_error(test_labels_list, test_output_list)}')




if __name__ == '__main__':
    main('test')







