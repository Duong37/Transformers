from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, modeling_utils, GPT2ForSequenceClassification
import torch
from torch import nn
import adapter
import transformer_block_old


class NoParam(nn.Module):
    """
    Wraps a module, stopping parameters from being registered
    """

    def __init__(self, mod):
        super().__init__()
        self.mod = [mod]

    def cuda(self):
        self.mod[0].cuda()

    def forward(self, x, *args, **kwargs):
        return self.mod[0](x, *args, **kwargs)


class gpt2Wrapper(nn.Module):
    def __init__(self, iblocks=3, model_name='gpt2', dropout=0.0, csize=None):
        super().__init__()

        self.labels_ids = {'neg': 0, 'pos': 1}
        self.n_labels = len(self.labels_ids)

        self.model_config = GPT2Config.from_pretrained(model_name, num_labels=self.n_labels)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.model = GPT2LMHeadModel.from_pretrained(model_name, config = self.model_config)
        self.model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name,
                                                                   config=self.model_config)

        for param in self.model.parameters():
            param.requires_grad = False

        emb = self.model.config.n_embd
        self.ctx = self.model.config.n_ctx
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.model.config.eos_token_id

        self.container = [None]

        insert_at = 'start'

        self.iblocks = nn.ModuleList([
            adapter.AdapterBlock(emb, 8, True, 4, dropout, csize=csize,
                          cond=self.container) for _ in range(iblocks)
        ])

        h = self.model.transformer.h  # the main stack of transformer blocks

        if insert_at == 'start':
            for i in range(iblocks - 1, -1, -1):
                print('inserting block at', i)
                block = self.iblocks[i]
                h.insert(i, block)
        elif insert_at == 'middle':
            for i in range(iblocks - 1, -1, -1):
                print('inserting block at', i + 5)
                block = self.iblocks[i]
                h.insert(i + 5, block)
        elif insert_at == 'end':
            for i in range(iblocks - 1, -1, -1):
                print('inserting block at', i + 5)
                block = self.iblocks[i]
                h.insert(i + 10, block)
        elif insert_at == 'everywhere':
            for i in range(iblocks - 1, -1, -1):
                print('inserting block at', i * 6)
                block = self.iblocks[i]
                h.insert(i * 6, block)
        else:
            pass

        # nb = len(self.model.transformer.h) # number of GPT2 blocks
        # print('len nb: ')
        # print(nb)
        # per = nb // iblocks

        # h = self.model.transformer.h  # the main stack of transformer blocks
        # for i in range(iblocks - 1, -1, -1):
        #      print('inserting block at', i * per)
        #      block = self.iblocks[i]
        #      h.insert(i * per, block)
        # h.insert(len(h), self.iblocks[-1])
        #

        print('len h after adding adapters:')
        print(len(h))

        self.register_buffer(name='head_mask', tensor=torch.ones(len(h), self.model.config.n_head))

        self.model = NoParam(self.model)

        # Out own language model head
        # self.headbias = nn.Parameter(torch.zeros(self.tokenizer.vocab_size))  # to token probabilities

    # def forward(self, x, cond=None, layer_past = None, input_ids = None, attention_mask = None, labels = None):
    def forward(self, x, cond=None):
        # b = x.size(0)

        if cond is not None:
            self.container[0] = cond

        x = self.model(x, head_mask=self.head_mask)[0]
        # x =  0.0 * cond.view(b, -1).sum(dim=1) #hack
        # x = x + self.headbias

        return x
