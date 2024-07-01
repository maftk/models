from torch import nn
import torch
class custom1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            )
        self.line = nn.Linear(hidden_size, 1)
        # self.drop = nn.Dropout(0.2)
        # self.sm = nn.Softmax(dim=0)

    def forward(self, x, bch,hls):
        h0 = torch.zeros(self.num_layers, bch,hls)
        c0 = torch.zeros(self.num_layers, bch,hls)
        #隠れ層、cellの内部状態は使用しない。
        lstmO, (hn, cn) = self.lstm(x, (h0,c0))
        lineO = self.line(lstmO[:,-1,:].view(x.size(0), -1))
        # return torch.sigmoid(linear_out)
        # StandardScalerを使うためsigmoidで０１にしない
        return lineO
