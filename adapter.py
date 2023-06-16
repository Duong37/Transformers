import torch
from torch import nn
import self_attention_old
import transformer_block_old as tbo
import transformer_block as tb


class AdapterBlock(nn.Module):
    """
    Transformer block to be inserted into GPT2 stack. Allows conditionals
    to be registered prior to forward.
    """

    def __init__(self, emb, *args, mult=0.0, csize=None, cond=[None], **kwargs):

        super().__init__()
        # self.block = tbo.TransformerBlock(emb, *args, **kwargs)
        self.block = tb.TransformerBlock(emb, *args, **kwargs)
        self.mult = nn.Parameter(torch.tensor([mult]))



        self.cond = cond
        self.cond_out = [None]

        if csize is not None:
            self.to_cond = nn.Sequential(
                nn.Linear(csize, 2 * csize), nn.ReLU(),
                nn.Linear(2 * csize, emb)
            )

            # self.to_cond = nn.Linear(csize, emb)

    #def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
    def forward(self, x, layer_past = None, attention_mask = None, head_mask = None, encoder_hidden_states = None, encoder_attention_mask = None, use_cache = False, output_attentions = False):


        b, l, e = x.size()

        if self.cond is not None and len(self.cond) > 0 and self.cond[0] is not None:
            cond = self.to_cond(self.cond[0])
            assert cond.size() == (b, e), f'{cond.size()} versus {b, e}'

            self.cond_out[0] = cond

            xc = x + cond[:, None, :]
        else:
            xc = x

        r = self.mult * self.block(xc) + x


        # print(r.size())
        return r, None, None

    def clear(self):
        del self.cond_out[0]
        del self.cond_out
        self.cond_out = [None]
