import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Attention(nn.Module):
    def __init__(self, n_hid):
        super().__init__()
        self.attn = nn.Linear(n_hid * 3, n_hid, bias=False)
        self.v = nn.Linear(n_hid, 1, bias=False)

    def forward(self, h, enc_output):
        """
        h = [batch, n_hid]
        enc_output = [batch, past_len, n_hid * 2]
        """
        past_len = enc_output.shape[1]
        h = h.unsqueeze(1).repeat(1, past_len, 1)
        s = torch.tanh(self.attn(torch.cat((h, enc_output), dim=2)))
        attention = self.v(s).squeeze(2)   # attention = [batch, past_len]
        return F.softmax(attention, dim=1)


class Fusion(nn.Module):
    def __init__(self, num_p, n_hid, n_y):
        super().__init__()
        self.fc_p = nn.Linear(num_p, n_hid)
        self.fc_y = nn.Linear(n_y, n_hid)

    def forward(self, c, p, y0):
        """
        c = [batch, 1, n_hid * 2]
        p = [batch, num_p]
        y0 = [batch, n_y]
        """
        q = F.relu(self.fc_p(p))
        u = F.relu(self.fc_y(y0))   # u = [batch, n_hid]
        gru_input = torch.cat((q.unsqueeze(1), c, u.unsqueeze(1)), dim=2)   # gru_input = [batch, 1, n_hid * 4]
        return gru_input, u


class Encoder(nn.Module):
    def __init__(self, n_x, n_hid, num_layers):
        super().__init__()
        self.fc = nn.Linear(n_x, n_hid)
        self.n_hid = n_hid
        self.num_layers = num_layers
        self.gru = nn.GRU(n_hid, n_hid, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.transform = nn.Linear(n_hid * 2, n_hid)

    def forward(self, x_past):
        """
        x_past = [batch, past_len, n_x]
        """
        gru_input = F.relu(self.fc(x_past))

        enc_output, enc_h = self.gru(gru_input)
        # enc_output = [batch, past_len, n_hid * 2]
        # enc_h = [num_layers * 2, batch, n_hid]

        enc_h = enc_h.transpose(0, 1).contiguous()
        h = torch.tanh(self.transform(enc_h.reshape(-1, self.num_layers, self.n_hid * 2)))
        h = h.transpose(0, 1).contiguous()   # h = [num_layers, batch, n_hid]
        return enc_output, h


class PCDecoder(nn.Module):
    def __init__(self, num_p, n_hid, n_y, num_layers):
        super().__init__()
        self.attention = Attention(n_hid)
        self.fusion = Fusion(num_p, n_hid, n_y)
        self.gru = nn.GRU(n_hid * 4, n_hid, batch_first=True, num_layers=num_layers)
        self.linear = nn.Linear(n_hid * 4, n_y)

    def forward(self, y0, p, h, enc_output):
        """
        y0 = [batch, n_y]
        p = [batch, num_p]
        h = [num_layers, batch, n_hid]
        enc_output = [batch, past_len, n_hid * 2]
        """
        a = self.attention(h[-1, :, :], enc_output).unsqueeze(1)  # a = [batch, 1, past_len]
        c = torch.bmm(a, enc_output)
        gru_input, u = self.fusion(c, p, y0)

        dec_output, dec_h = self.gru(gru_input, h)
        # dec_output = [batch, 1, n_hid]
        # dec_h = [num_layers, batch, n_hid]

        y = self.linear(torch.cat((dec_output.squeeze(1), c.squeeze(1), u), dim=1))   # y = [batch, n_y]
        return y, dec_h, a.squeeze(1)


class PAPF(nn.Module):
    def __init__(self, n_x, num_p, n_y, n_hid, pred_len, num_gru_layers, device):
        super().__init__()
        self.encoder = Encoder(n_x, n_hid, num_gru_layers)
        self.decoder = PCDecoder(num_p, n_hid, n_y, num_gru_layers)
        self.n_x = n_x
        self.n_y = n_y
        self.pred_len = pred_len
        self.device = device

    def forward(self, x_past, p, y_true, theta=0.5):
        """
        x_past = [batch, past_len, n_x(=1)]
        p = [batch, pred_len, num_p]
        y_true = [batch, pred_len, n_y(=1)]
        """
        batch = x_past.shape[0]
        past_len = x_past.shape[1]
        y0_ = torch.zeros(batch, self.n_y).to(self.device)
        y_pred = torch.zeros(batch, self.pred_len, self.n_y).to(self.device)
        attn_weights = torch.zeros(batch, self.pred_len, past_len).to(self.device)

        enc_output, h = self.encoder(x_past)

        y0 = x_past[:, -1, :] if self.n_x == self.n_y else y0_
        for t in range(self.pred_len):
            y, h, a = self.decoder(y0, p[:, t, :], h, enc_output)

            y_pred[:, t, :] = y
            attn_weights[:, t, :] = a

            teacher_forcing = random.random() < theta
            y0 = y_true[:, t, :] if teacher_forcing else y

        return y_pred, attn_weights
