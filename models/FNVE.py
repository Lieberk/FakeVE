from transformers.models.bart.modeling_bart import *
import torch.nn as nn
import torch
from transformers.modeling_outputs import BaseModelOutput


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.eye(768))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1

        output = torch.matmul(adj.float(), hidden.float()) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class FNVE(nn.Module):
    def __init__(self, cfg, tkr):
        super(FNVE, self).__init__()
        self.cfg = cfg
        self.tkr = tkr
        self.text_dim = 768
        self.model_gen = BartForConditionalGeneration.from_pretrained(".\dataset\Pretrain/bart-base")
        self.linear_frames = nn.Sequential(torch.nn.Linear(512, self.text_dim), torch.nn.ReLU(), nn.Dropout(p=0.1))
        self.gc = GraphConvolution(768, 768)

    def forward(self, mode='train', **kwargs):
        target_ids = kwargs['target_ids']
        graph = kwargs['model_adj']

        fea_title = kwargs['title']
        title_mask = kwargs['title_mask']

        audio_transcript = kwargs['audio_transcript']
        audio_transcript_mask = kwargs['audio_transcript_mask']

        frames = kwargs['frames']
        frames_masks = kwargs['frames_masks']
        frames = self.linear_frames(frames)

        concat_feat = torch.cat([fea_title, audio_transcript, frames], dim=1)
        concat_mask = torch.cat([title_mask, audio_transcript_mask, frames_masks], dim=1)

        context_enc_out = self.model_gen.get_encoder()(inputs_embeds=concat_feat, attention_mask=concat_mask)
        context_enc_feat = context_enc_out.last_hidden_state

        x = self.gc(context_enc_feat, graph)
        gen_mask = concat_mask
        gen_feat = context_enc_feat + x

        enc_output = BaseModelOutput(last_hidden_state=gen_feat)

        if mode == 'train':
            gen = self.model_gen(encoder_outputs=enc_output, attention_mask=gen_mask, labels=target_ids)
            return gen

        elif mode == 'eval' or mode == 'gen':
            with torch.no_grad():
                if mode == 'eval':
                    gen = self.model_gen(encoder_outputs=enc_output, attention_mask=gen_mask, labels=target_ids)
                    return gen.loss

                elif mode == 'gen':
                    generation_cfgs = {"max_length": self.cfg.eval.eval_max_len,
                                       "min_length": self.cfg.eval.eval_min_len,
                                       "pad_token_id": self.tkr.pad_token_id,
                                       'eos_token_id': self.tkr.eos_token_id, "num_beams": self.cfg.eval.num_beams,
                                       'top_p': self.cfg.eval.top_p, 'top_k': self.cfg.eval.top_k,
                                       'temperature': self.cfg.eval.temperature, 'do_sample': True,
                                       'repetition_penalty': self.cfg.eval.repetition_penalty,
                                       'no_repeat_ngram_size': self.cfg.eval.no_repeat_ngram_size}

                    gen_result = self.model_gen.generate(encoder_outputs=enc_output,
                                                         attention_mask=gen_mask,
                                                         **generation_cfgs)

                    gen_decoded = self.tkr.batch_decode(gen_result, skip_special_tokens=True)

                    return gen_decoded
                return None

        else:
            raise ValueError('Mode should be among [train, eval, gen].')