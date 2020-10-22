import torch as t
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.params = params
        self.lstm = nn.LSTM(input_size=300,
                        hidden_size=self.params.discriminator_rnn_size,
                        num_layers=self.params.discriminator_num_layers,
                        batch_first=True,
                        dropout=0.3,
                        bidirectional=True)
        self.fc = nn.Linear(4*self.params.discriminator_rnn_size, 4*self.params.discriminator_rnn_size)
        self.relu = nn.Relu()
        self.out = nn.Linear(4*self.params.discriminator_rnn_size, 1)

    def forward(self, sentences):
        """
        :param sentences: [batch_size, seq_len, embed_size] tensor
        """
        # output, _ = self.lstm(x)
        # (seq_len, batch, num_directions*hidden_size)
        state = None
        [batch_size, seq_len, embed_size] = sentences.size()
        _, [h_state, c_state] = self.lstm(sentences, state)
        h_state = h_state.view(2, 2, batch_size, self.params.discriminator_rnn_size)[-1]
        c_state = c_state.view(2, 2, batch_size, self.params.discriminator_rnn_size)[-1]
        h_state = h_state.permute(1,0,2).contiguous().view(batch_size, -1)
        c_state = c_state.permute(1,0,2).contiguous().view(batch_size, -1)
        final_state = t.cat([h_state, c_state], 1)

        output = self.fc(final_state)
        output = self.relu(output)
        output = self.out(output)
        output = t.squeeze(output, 1)
        output = t.sigmoid(output)

        return output

    def learnable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
