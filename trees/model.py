from fastai.collab import *
from fastai.tabular.all import *


class CollabNN(Module):

    def __init__(self, src_sz, y_range, n_act=100):
        self.y_range = y_range
        self.n_bits = src_sz[1]
        self.prediction = None

        self.embedding_layers = Embedding(*src_sz)

        self.layers = nn.Sequential(
            nn.Linear(2*self.n_bits, n_act),
            nn.ReLU(),
            nn.Linear(n_act, 1))


    def forward(self, x):
        out = self.embedding_layers(x[:, 0]), self.embedding_layers(x[:, 1])
        out = torch.sign(torch.cat(out, dim=1))

        pred_out = self.layers(out)
        self.prediction = sigmoid_range(pred_out, *self.y_range)

        return self.prediction

    def get_predictions(self):
      return self.prediction
    
    def save_embeddings(self, x):
        #  embedings are used as -1 & 1 but replace -1 with 0 for easy visualization
        out = self.embedding_layers(x)
        self.node_to_emb = {int(x[i]): ''.join([str(int(j)) for j in torch.relu(torch.sign(out[i]))])for i in range(len(x))}
        
        return self.node_to_emb

