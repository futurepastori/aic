import torch
import torchvision
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    """
    Encoder module.

    A ResNet-101 CovNet without the last activation layer meant for classification,
    replaced with our own-dimensional mappings and a BatchNorm1d for 2d->1d mapping
    of the ResNet 14x14 features.

    :param embed_size: embedding dimensions of our decoder
    """    
    def __init__(self, embed_size, fine_tuning=False):   
        super(EncoderCNN, self).__init__()
        self.feat_size = 14
        self.embed_size = embed_size

        resnet = torchvision.models.resnet101(pretrained=True)

        # Removing last layer as it's intended for classification
        resnet_layers = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*resnet_layers)

        self.pooling = nn.AdaptiveAvgPool2d((self.feat_size, self.feat_size))
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=embed_size)

        # Fine-tuning: disabling grad computing on resnet parameters
        # and enabling it only on the last layers, as we want to keep
        # what's trained for the most primitive ones
        for p in self.resnet.parameters():
            p.requires_grad = False
        
        if fine_tuning == True:
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = True


    def forward(self, feat):
        """
        Forward pass

        :param x: images from COCO, preprocessed
        :return: 14x14 features from the encoder
        """

        feat_resnet = self.resnet(feat)
        feat_resnet = self.pooling(feat_resnet)
        feat_resnet = feat_resnet.permute(0, 2, 3, 1)

        return feat_resnet

class Bahdanau(nn.Module):
    def __init__(self, hidden_size, encoder_size):
        super(Bahdanau, self).__init__()
        self.Wx = nn.Linear(in_features=encoder_size, out_features=encoder_size)
        self.Wh = nn.Linear(in_features=hidden_size, out_features=encoder_size)
        self.V = nn.Linear(in_features=encoder_size, out_features=1)

    def forward(self, features, hidden):
        # features == (batch_size, 196, embedding_dim)
        hidden = hidden.unsqueeze(1) # (batch_size, 1, hidden_size)

        x_wx = self.Wx(features)
        h_wh = self.Wh(hidden)
        lambdas = nn.Tanh()(x_wx + h_wh)
        lambdas = self.V(lambdas) # (batch_size, 196, 1)

        alphas = nn.Softmax(dim=1)(lambdas) #(batch_size, 196, 1)

        z = alphas * features
        z = torch.sum(z, 1) # (batch_size, hidden_size)
        
        return z, alphas
        

class DecoderRNN(nn.Module):
    """
    LSTM-based decoder module.

    A repeatedly called LSTM cell, an Embedding layer for vocabulary-length
    mapping to the dense embeddings for the LSTM, a linear layer for activation
    and a dropout layer because why not.

    :param embed_size: dimension of the dense word embeddings vector
    :param vocab_size: length of our vocab dictionary
    :param hidden_size: dimensions of our cell and hidden states
    """    
    def __init__(self, embed_size, vocab_size, hidden_size, att_size=256, num_pixels=196, encoder_size=2048):   
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.encoder_size = encoder_size
        self.vocab_size = vocab_size

        self.init_h = nn.Linear(encoder_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.f_beta = nn.Linear(hidden_size, encoder_size)
        
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.gru = nn.GRUCell(input_size=(encoder_size + embed_size), hidden_size=hidden_size)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.dropout = nn.Dropout()
        
        self.attention = Bahdanau(hidden_size=hidden_size, encoder_size=encoder_size)

    def init_hidden_state(self, feats):
        feats = torch.mean(feats, dim=1)
        h_t = self.init_h(feats)
        return h_t

    def forward(self, features, captions, lengths):
        """
        Forward pass.

        :param feats: the encoder's output
        :param caps: already sorted and sampled captions
        :param lengths: sorted caption length vector
        :return: predictions, embedded feats + captions and lengths
        """
        tf = True
        batch_size = features.size(0)
        encoder_dim = features.size(-1)

        features = features.view(batch_size, -1, encoder_dim)
        h_t = self.init_hidden_state(features).to(device)

        preds = torch.zeros(batch_size, 50, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, 50, (14*14)).to(device)

        prev_words = torch.zeros((batch_size, 1), dtype=torch.long).to(device)
        words = torch.zeros((batch_size, 50), dtype=torch.long)

        if tf:
            embed = self.embed(captions)
        else:
            embed = self.embed(prev_words)

        decode_lengths = (lengths - 1).tolist()

        for t in range(max(decode_lengths)):
            b = sum(i >= t for i in decode_lengths)
            z, alpha = self.attention(features[:b, :], h_t[:b, :])
            gate = self.sigmoid(self.f_beta(h_t[:b]))
            gated_z = gate * z

            if tf:
                _input = torch.cat((gated_z[:b, :], embed[:b, t]), dim=1)
            else:
                embed = embed.squeeze(1) if embed.dim() == 3 else embed
                _input = torch.cat((gated_z[:b, :], embed[:b, :]), dim=1)

            h_t = self.gru(_input, h_t[:b, :])
            x = self.fc1(self.dropout(h_t))
            _, pred_idx = torch.max(x, 1)
            words[:b, t] = pred_idx
            preds[:b, t, :] = x
            alphas[:b, t, :] = alpha.squeeze(-1)

            if not tf:
                embed = self.embed(x.max(1)[1].reshape(b, 1))

        return preds, alphas, decode_lengths

