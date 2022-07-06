import torch
import torch.nn as nn
import torch.nn.functional as F

from mmlatch.attention import Attention, SymmetricAttention
from mmlatch.rnn import RNN, AttentiveRNN


class FeedbackUnit(nn.Module):
    def __init__(
        self,
        hidden_dim,
        mod1_sz,
        mask_type="learnable_sequence_mask",
        dropout=0.1,
        device="cpu",
    ):
        super(FeedbackUnit, self).__init__()
        self.mask_type = mask_type
        self.mod1_sz = mod1_sz
        self.hidden_dim = hidden_dim

        if mask_type == "learnable_sequence_mask":
            self.mask1 = RNN(hidden_dim, mod1_sz, dropout=dropout, device=device)
            self.mask2 = RNN(hidden_dim, mod1_sz, dropout=dropout, device=device)
        else:
            self.mask1 = nn.Linear(hidden_dim, mod1_sz)
            self.mask2 = nn.Linear(hidden_dim, mod1_sz)

        mask_fn = {
            "learnable_static_mask": self._learnable_static_mask,
            "learnable_sequence_mask": self._learnable_sequence_mask,
        }

        self.get_mask = mask_fn[self.mask_type]

    def _learnable_sequence_mask(self, y, z, lengths=None):
        oy, _, _ = self.mask1(y, lengths)
        oz, _, _ = self.mask2(z, lengths)

        lg = (torch.sigmoid(oy) + torch.sigmoid(oz)) * 0.5

        mask = lg

        return mask

    def _learnable_static_mask(self, y, z, lengths=None):
        y = self.mask1(y)
        z = self.mask2(z)
        mask1 = torch.sigmoid(y)
        mask2 = torch.sigmoid(z)
        mask = (mask1 + mask2) * 0.5

        return mask

    def forward(self, x, y, z, lengths=None):
        mask = self.get_mask(y, z, lengths=lengths)
        mask = F.dropout(mask, p=0.2)
        x_new = x * mask

        return x_new


class Feedback(nn.Module):
    def __init__(
        self,
        hidden_dim,
        mod1_sz,
        mod2_sz,
        mod3_sz,
        mask_type="learnable_sequence_mask",
        dropout=0.1,
        device="cpu",
    ):
        super(Feedback, self).__init__()
        self.f1 = FeedbackUnit(
            hidden_dim,
            mod1_sz,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
        )
        self.f2 = FeedbackUnit(
            hidden_dim,
            mod2_sz,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
        )
        self.f3 = FeedbackUnit(
            hidden_dim,
            mod3_sz,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
        )

    def forward(self, low_x, low_y, low_z, hi_x, hi_y, hi_z, lengths=None):
        x = self.f1(low_x, hi_y, hi_z, lengths=lengths)
        y = self.f2(low_y, hi_x, hi_z, lengths=lengths)
        z = self.f3(low_z, hi_x, hi_y, lengths=lengths)

        return x, y, z


class AttentionFuser(nn.Module):
    def __init__(self, proj_sz=None, return_hidden=True, device="cpu"):
        super(AttentionFuser, self).__init__()
        self.return_hidden = return_hidden
        self.ta = SymmetricAttention(
            attention_size=proj_sz,
            dropout=0.1,
        )

        self.va = SymmetricAttention(
            attention_size=proj_sz,
            dropout=0.1,
        )

        self.tv = SymmetricAttention(
            attention_size=proj_sz,
            dropout=0.1,
        )

        self.tav = Attention(
            attention_size=proj_sz,
            dropout=0.1,
        )

        self.out_size = 7 * proj_sz

    def forward(self, txt, au, vi):
        ta, at = self.ta(txt, au)
        va, av = self.va(vi, au)
        tv, vt = self.tv(txt, vi)

        av = va + av
        tv = vt + tv
        ta = ta + at

        tav, _ = self.tav(txt, queries=av)

        # Sum weighted attention hidden states

        if not self.return_hidden:
            txt = txt.sum(1)
            au = au.sum(1)
            vi = vi.sum(1)
            ta = ta.sum(1)
            av = av.sum(1)
            tv = tv.sum(1)
            tav = tav.sum(1)

        # B x L x 7*D
        fused = torch.cat([txt, au, vi, ta, tv, av, tav], dim=-1)

        return fused


class AttRnnFuser(nn.Module):
    def __init__(self, proj_sz=None, device="cpu", return_hidden=False):
        super(AttRnnFuser, self).__init__()
        self.att_fuser = AttentionFuser(
            proj_sz=proj_sz,
            return_hidden=True,
            device=device,
        )
        self.rnn = AttentiveRNN(
            self.att_fuser.out_size,
            proj_sz,
            bidirectional=True,
            merge_bi="cat",
            attention=True,
            device=device,
            return_hidden=return_hidden,
        )
        self.out_size = self.rnn.out_size

    def forward(self, txt, au, vi, lengths):
        att = self.att_fuser(txt, au, vi)  # B x L x 7 * D
        out = self.rnn(att, lengths)  # B x L x 2 * D

        return out


class AudioEncoder(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super(AudioEncoder, self).__init__()
        self.audio = AttentiveRNN(
            cfg["input_size"],
            cfg["hidden_size"],
            batch_first=True,
            layers=cfg["layers"],
            merge_bi="sum",
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"],
            rnn_type=cfg["rnn_type"],
            packed_sequence=True,
            attention=cfg["attention"],
            device=device,
            return_hidden=cfg["return_hidden"],
        )
        self.out_size = self.audio.out_size

    def forward(self, x, lengths):
        x = self.audio(x, lengths)

        return x


class VisualEncoder(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super(VisualEncoder, self).__init__()
        self.visual = AttentiveRNN(
            cfg["input_size"],
            cfg["hidden_size"],
            batch_first=True,
            layers=cfg["layers"],
            merge_bi="sum",
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"],
            rnn_type=cfg["rnn_type"],
            packed_sequence=True,
            attention=cfg["attention"],
            device=device,
            return_hidden=cfg["return_hidden"],
        )
        self.out_size = self.visual.out_size

    def forward(self, x, lengths):
        x = self.visual(x, lengths)

        return x


class GloveEncoder(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super(GloveEncoder, self).__init__()
        self.text = AttentiveRNN(
            cfg["input_size"],
            cfg["hidden_size"],
            batch_first=True,
            layers=cfg["layers"],
            merge_bi="sum",
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"],
            rnn_type=cfg["rnn_type"],
            packed_sequence=True,
            attention=cfg["attention"],
            device=device,
            return_hidden=cfg["return_hidden"],
        )
        self.out_size = self.text.out_size

    def forward(self, x, lengths):
        x = self.text(x, lengths)

        return x


class AudioVisualTextEncoder(nn.Module):
    def __init__(
        self,
        audio_cfg=None,
        text_cfg=None,
        visual_cfg=None,
        fuse_cfg=None,
        feedback=False,
        device="cpu",
    ):
        super(AudioVisualTextEncoder, self).__init__()
        assert (
            text_cfg["attention"] and audio_cfg["attention"] and visual_cfg["attention"]
        ), "Use attention pls."

        self.feedback = feedback
        text_cfg["return_hidden"] = True
        audio_cfg["return_hidden"] = True
        visual_cfg["return_hidden"] = True

        self.text = GloveEncoder(text_cfg, device=device)
        self.audio = AudioEncoder(audio_cfg, device=device)
        self.visual = VisualEncoder(visual_cfg, device=device)

        self.fuser = AttRnnFuser(
            proj_sz=fuse_cfg["projection_size"],
            device=device,
        )

        self.out_size = self.fuser.out_size

        if feedback:
            self.fm = Feedback(
                fuse_cfg["projection_size"],
                text_cfg["input_size"],
                audio_cfg["input_size"],
                visual_cfg["input_size"],
                mask_type=fuse_cfg["feedback_type"],
                dropout=0.1,
                device=device,
            )

    def _encode(self, txt, au, vi, lengths):
        txt = self.text(txt, lengths)
        au = self.audio(au, lengths)
        vi = self.visual(vi, lengths)

        return txt, au, vi

    def _fuse(self, txt, au, vi, lengths):
        fused = self.fuser(txt, au, vi, lengths)

        return fused

    def forward(self, txt, au, vi, lengths):
        if self.feedback:
            txt1, au1, vi1 = self._encode(txt, au, vi, lengths)
            txt, au, vi = self.fm(txt, au, vi, txt1, au1, vi1, lengths=lengths)

        txt, au, vi = self._encode(txt, au, vi, lengths)
        fused = self._fuse(txt, au, vi, lengths)

        return fused


class AudioVisualTextClassifier(nn.Module):
    def __init__(
        self,
        audio_cfg=None,
        text_cfg=None,
        visual_cfg=None,
        fuse_cfg=None,
        modalities=None,
        num_classes=1,
        feedback=False,
        device="cpu",
    ):
        super(AudioVisualTextClassifier, self).__init__()
        self.modalities = modalities

        assert "text" in modalities, "No text"
        assert "audio" in modalities, "No audio"
        assert "visual" in modalities, "No visual"

        self.encoder = AudioVisualTextEncoder(
            text_cfg=text_cfg,
            audio_cfg=audio_cfg,
            visual_cfg=visual_cfg,
            fuse_cfg=fuse_cfg,
            feedback=feedback,
            device=device,
        )

        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, inputs):
        out = self.encoder(
            inputs["text"], inputs["audio"], inputs["visual"], inputs["lengths"]
        )

        return self.classifier(out)


class UnimodalEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        projection_size,
        layers=1,
        bidirectional=True,
        dropout=0.2,
        encoder_type="lstm",
        attention=True,
        return_hidden=False,
        device="cpu",
    ):
        super(UnimodalEncoder, self).__init__()
        self.encoder = AttentiveRNN(
            input_size,
            projection_size,
            batch_first=True,
            layers=layers,
            merge_bi="sum",
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=encoder_type,
            packed_sequence=True,
            attention=attention,
            device=device,
            return_hidden=return_hidden,
        )
        self.out_size = self.encoder.out_size

    def forward(self, x, lengths):
        return self.encoder(x, lengths)


class AVTEncoder(nn.Module):
    def __init__(
        self,
        text_input_size,
        audio_input_size,
        visual_input_size,
        projection_size,
        text_layers=1,
        audio_layers=1,
        visual_layers=1,
        bidirectional=True,
        dropout=0.2,
        encoder_type="lstm",
        attention=True,
        feedback=False,
        feedback_type="learnable_sequence_mask",
        device="cpu",
    ):
        super(AVTEncoder, self).__init__()
        self.feedback = feedback

        self.text = UnimodalEncoder(
            text_input_size,
            projection_size,
            layers=text_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            encoder_type=encoder_type,
            attention=attention,
            return_hidden=True,
            device=device,
        )

        self.audio = UnimodalEncoder(
            audio_input_size,
            projection_size,
            layers=audio_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            encoder_type=encoder_type,
            attention=attention,
            return_hidden=True,
            device=device,
        )

        self.visual = UnimodalEncoder(
            visual_input_size,
            projection_size,
            layers=visual_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            encoder_type=encoder_type,
            attention=attention,
            return_hidden=True,
            device=device,
        )

        self.fuser = AttRnnFuser(
            proj_sz=projection_size,
            device=device,
        )

        self.out_size = self.fuser.out_size

        if feedback:
            self.fm = Feedback(
                projection_size,
                text_input_size,
                audio_input_size,
                visual_input_size,
                mask_type=feedback_type,
                dropout=0.1,
                device=device,
            )

    def _encode(self, txt, au, vi, lengths):
        txt = self.text(txt, lengths)
        au = self.audio(au, lengths)
        vi = self.visual(vi, lengths)

        return txt, au, vi

    def _fuse(self, txt, au, vi, lengths):
        fused = self.fuser(txt, au, vi, lengths)

        return fused

    def forward(self, txt, au, vi, lengths):
        if self.feedback:
            txt1, au1, vi1 = self._encode(txt, au, vi, lengths)
            txt, au, vi = self.fm(txt, au, vi, txt1, au1, vi1, lengths=lengths)

        txt, au, vi = self._encode(txt, au, vi, lengths)
        fused = self._fuse(txt, au, vi, lengths)

        return fused


class AVTClassifier(nn.Module):
    def __init__(
        self,
        text_input_size,
        audio_input_size,
        visual_input_size,
        projection_size,
        text_layers=1,
        audio_layers=1,
        visual_layers=1,
        bidirectional=True,
        dropout=0.2,
        encoder_type="lstm",
        attention=True,
        feedback=False,
        feedback_type="learnable_sequence_mask",
        device="cpu",
        num_classes=1,
    ):
        super(AVTClassifier, self).__init__()

        self.encoder = AVTEncoder(
            text_input_size,
            audio_input_size,
            visual_input_size,
            projection_size,
            text_layers=text_layers,
            audio_layers=audio_layers,
            visual_layers=visual_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            encoder_type=encoder_type,
            attention=attention,
            feedback=feedback,
            feedback_type=feedback_type,
            device=device,
        )

        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, inputs):
        out = self.encoder(
            inputs["text"], inputs["audio"], inputs["visual"], inputs["lengths"]
        )

        return self.classifier(out)
