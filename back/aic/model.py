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
        resnet_layers = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*resnet_layers)

        #self.pooling = nn.AdaptiveAvgPool2d((self.feat_size, self.feat_size))
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

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
        feat_resnet = feat_resnet.reshape(feat_resnet.size(0), -1)
        feat_linear = self.linear(feat_resnet)
        feat_bn_1d = self.bn(feat_linear)         # (N, H, W, 2048)

        return feat_bn_1d
    

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
    def __init__(self, embed_size, vocab_size, hidden_size):   
        super(DecoderRNN, self).__init__()

        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        input_size = embed_size
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, feats, caps, lengths):
        """
        Forward pass.

        :param feats: the encoder's output
        :param caps: already sorted and sampled captions
        :param lengths: sorted caption length vector
        :return: predictions, embedded feats + captions and lengths
        """        
        batch_size = feats.size(0)
        cap_embed = self.embed(caps)
        embeds = torch.cat((feats.unsqueeze(1), cap_embed), 1)
        embeds_pack = pack_padded_sequence(embeds, lengths, batch_first=True)
        lstm_out, (h_t, c_t) = self.lstm(embeds_pack)
        
        out = self.fc(self.dropout(lstm_out.data))

        return out
        


