from torch import nn
import torch
class custom1(nn.Module):
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
        self.dense = nn.Linear(hsize, 1)
        # self.drop = nn.Dropout(0.2)
        # self.sm = nn.Softmax(dim=0)

    def forward(self, x, batch,hsize):
        h0 = torch.zeros(self.num_layers, batch, hsize)
        c0 = torch.zeros(self.num_layers, batch, hsize)
        #隠れ層、cellの内部状態は使用しない。
        lstmO, (hn, cn) = self.lstm(x, (h0,c0))
        lineO = self.dense(lstmO[:,-1,:].view(x.size(0), -1))
        # return torch.sigmoid(linear_out)
        # StandardScalerを使うためsigmoidで０１にしない
        return lineO
        
class custom2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
                # Define stacked LSTM layers
        self.dense = nn.Linear(hidden_size, 1)
        self.lstms = nn.ModuleList()
        self.lstms.append(nn.LSTM(input_size, hidden_size,batch_first=True))

        for _ in range(1, num_layers):
            self.lstms.append(nn.LSTM(hidden_size, hidden_size,batch_first=True))


    def forward(self, x_data, bch,hls):
        # h0 = torch.zeros(self.num_layers, bch, hls)
        # c0 = torch.zeros(self.num_layers, bch, hls)
        h0 = torch.zeros(1, bch, hls)
        c0 = torch.zeros(1, bch, hls)
        out = x_data
        for i,lstm in enumerate(self.lstms):
          out, (hn, cn) = lstm(out, (h0, c0))
          # Update h0 and c0 for the next layer

        #隠れ層、cellの内部状態は使用しない。
        # lstm_out, (hn, cn) = self.lstm(x_data, (h0,c0))
        out = self.dense(out[:,-1,:].view(x_data.size(0), -1))
        return torch.sigmoid(out)

class custom3(nn.Module):
    def __init__(self, inp_s, hid_s=4,ln=4,bach_size=32):
        super().__init__()
        # hid_s = [10,100,20,1]
        # hid_s = [20,1]
        # hid_s = [10,5,2,1]
        # hid_s = [10,8,4,1]
        # hid_s = [20,10,2,1]
        # hid_s = [20,5,2,1]
        # hid_s = [31,5,2,1]
        # hid_s = [31,15,7,1]
        # hid_s = [31,10,3,1]
        hid_s = [100,50,10,1]
        # assert len(hid_s) == 2, "Length of hidden_sizes should be 4"

        # Define LSTM layers
        self.lstms = nn.ModuleList()
        # hls = [10, 100, 10, 1]  # Sizes for each LSTM layer's hidden state
        self.lstms.append(nn.LSTM(inp_s, hid_s[0], batch_first=True))
        for s in range(0,len(hid_s)-1):
          self.lstms.append(nn.LSTM(hid_s[s], hid_s[s+1], batch_first=True))
        # Output layers (linear + sigmoid)
        self.linears = nn.ModuleList()
        for l in hid_s:
          self.linears.append(nn.Linear(l, l))
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.2)

    def forward(self, x, bch, hls):
        # h0 = torch.zeros(1, bch, 150)
        # c0 = torch.zeros(1, bch, 150)
        # Forward pass through each LSTM layer
        out = x
        c = 0
        for i, lstm in enumerate(self.lstms):
            out, (hn,cn) = lstm(out)
            # Apply linear layer and sigmoid activation after each LSTM layer
            out = self.linears[i](out)
            c+=1
        # out = torch.sigmoid(out[:,-1,:])
        # return False
        return out[:,-1,:]
class custom4(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            )
        h1 = 256
        h2 = 128
        h3 = 64
        self.fc1 = nn.Linear(hidden_size,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.fc3 = nn.Linear(h2,h3)
        self.fc4 = nn.Linear(h3, 1)

    def forward(self, x_data, bch,hls):
        h0 = torch.zeros(self.num_layers, bch, hls)
        c0 = torch.zeros(self.num_layers, bch, hls)
        #隠れ層、cellの内部状態は使用しない。
        out, (hn, cn) = self.lstm(x_data, (h0,c0))
        # out = self.dropout(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.fc4(out[:,-1,:].view(x_data.size(0), -1))
        return torch.sigmoid(out)
