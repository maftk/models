from torch import nn
class Lstm0(nn.Module):
    def __init__(self, isize, hsize, layers, batchs):
        super().__init__()
        self.isize = isize
        self.hsize = hsize
        self.layers = layers
        self.batchs = batchs
        self.lstm = nn.LSTM(input_size=self.isize,
                            hidden_size=self.hsize,
                            num_layers=self.layers,
                            batch_first=True,
                            )
        self.line = nn.Linear(hsize, 1)
        # self.drop = nn.Dropout(0.2)
        # self.sm = nn.Softmax(dim=0)

    def forward(self, x, batch,hsize):
        h0 = torch.zeros(self.layers, batch, hsize)
        c0 = torch.zeros(self.layers, batch, hsize)
        #隠れ層、cellの内部状態は使用しない。
        lstmO, (hn, cn) = self.lstm(x, (h0,c0))
        lineO = self.line(lstmO[:,-1,:].view(x.size(0), -1))
        # return torch.sigmoid(linear_out)
        # StandardScalerを使うためsigmoidで０１にしない
        return lineO
