import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mmlatch.attention import Attention
from mmlatch.util import pad_mask


class PadPackedSequence(nn.Module):
    """Some Information about PadPackedSequence"""

    def __init__(self, batch_first=True):
        super(PadPackedSequence, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, lengths):
        max_length = lengths.max().item()
        x, _ = pad_packed_sequence(
            x, batch_first=self.batch_first, total_length=max_length
        )
        return x


class PackSequence(nn.Module):
    def __init__(self, batch_first=True):
        super(PackSequence, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, lengths):
        x = pack_padded_sequence(
            x, lengths, batch_first=self.batch_first, enforce_sorted=False
        )
        lengths = lengths[x.sorted_indices]
        return x, lengths


class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        layers=1,
        bidirectional=False,
        merge_bi="cat",
        dropout=0,
        rnn_type="lstm",
        packed_sequence=True,
        device="cpu",
    ):

        super(RNN, self).__init__()
        self.device = device
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.merge_bi = merge_bi
        self.rnn_type = rnn_type.lower()

        self.out_size = hidden_size

        if bidirectional and merge_bi == "cat":
            self.out_size = 2 * hidden_size

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size,
            hidden_size,
            batch_first=batch_first,
            num_layers=layers,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(dropout)
        self.packed_sequence = packed_sequence

        if packed_sequence:
            self.pack = PackSequence(batch_first=batch_first)
            self.unpack = PadPackedSequence(batch_first=batch_first)

    def _merge_bi(self, forward, backward):
        if self.merge_bi == "sum":
            return forward + backward

        return torch.cat((forward, backward), dim=-1)

    def _select_last_unpadded(self, out, lengths):
        gather_dim = 1 if self.batch_first else 0
        gather_idx = (
            (lengths - 1)  # -1 to convert to indices
            .unsqueeze(1)  # (B) -> (B, 1)
            .expand((-1, self.hidden_size))  # (B, 1) -> (B, H)
            # (B, 1, H) if batch_first else (1, B, H)
            .unsqueeze(gather_dim)
        )
        # Last forward for real length or seq (unpadded tokens)
        last_out = out.gather(gather_dim, gather_idx).squeeze(gather_dim)

        return last_out

    def _final_output(self, out, lengths):
        # Collect last hidden state
        # Code adapted from https://stackoverflow.com/a/50950188

        if not self.bidirectional:
            return self._select_last_unpadded(out, lengths)

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])
        # Last backward corresponds to first token
        last_backward_out = backward[:, 0, :] if self.batch_first else backward[0, ...]
        # Last forward for real length or seq (unpadded tokens)
        last_forward_out = self._select_last_unpadded(forward, lengths)

        return self._merge_bi(last_forward_out, last_backward_out)

    def merge_hidden_bi(self, out):
        if not self.bidirectional:
            return out

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])

        return self._merge_bi(forward, backward)

    def forward(self, x, lengths, initial_hidden=None):
        self.rnn.flatten_parameters()

        if self.packed_sequence:
            lengths = lengths.to("cpu")
            x, lengths = self.pack(x, lengths)
            lengths = lengths.to(self.device)

        if initial_hidden is not None:
            out, hidden = self.rnn(x, initial_hidden)
        else:
            out, hidden = self.rnn(x)

        if self.packed_sequence:
            out = self.unpack(out, lengths)

        out = self.drop(out)
        last_timestep = self._final_output(out, lengths)
        out = self.merge_hidden_bi(out)

        return out, last_timestep, hidden


class AttentiveRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        layers=1,
        bidirectional=False,
        merge_bi="cat",
        dropout=0.1,
        rnn_type="lstm",
        packed_sequence=True,
        attention=False,
        return_hidden=False,
        device="cpu",
    ):
        super(AttentiveRNN, self).__init__()
        self.device = device
        self.rnn = RNN(
            input_size,
            hidden_size,
            batch_first=batch_first,
            layers=layers,
            merge_bi=merge_bi,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=packed_sequence,
            device=device,
        )
        self.out_size = self.rnn.out_size
        self.attention = None
        self.return_hidden = return_hidden

        if attention:
            self.attention = Attention(attention_size=self.out_size, dropout=dropout)

    def forward(self, x, lengths, initial_hidden=None):
        out, last_hidden, _ = self.rnn(x, lengths, initial_hidden=initial_hidden)

        if self.attention is not None:
            out, _ = self.attention(
                out, attention_mask=pad_mask(lengths, device=self.device)
            )

            if not self.return_hidden:
                out = out.sum(1)
        else:
            out = last_hidden

        return out
