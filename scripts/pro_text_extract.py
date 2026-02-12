import os
import pickle
from transformers.models.bart.modeling_bart import *
import json
from transformers import BartTokenizer
import torch

data_path = '.\dataset\FakeVE'
pretrain_path = '.\dataset\Pretrain'

gpu_id = "cuda:0"
device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')

bart_model_path = os.path.join(pretrain_path, 'bart-base')
tkr = BartTokenizer.from_pretrained(bart_model_path)
model_gen = BartForConditionalGeneration.from_pretrained(bart_model_path).to(device)


def text_feature_extraction():
    with open(os.path.join(data_path, 'data.json'), 'r') as f:
        data = json.load(f)

    bert_text_path = os.path.join(data_path, 'preprocess/pre_FNVE/bart_text_feature')
    if not os.path.exists(bert_text_path):
        os.makedirs(bert_text_path)

    max_title_l = 32
    max_audio_l = 64
    max_target_l = 80

    for i, item in enumerate(data):
        video_id = item['video_id']

        title_text = item['title']
        title_tokens = tkr(title_text,
                           is_split_into_words=True,
                           max_length=max_title_l,
                           padding='max_length',
                           truncation=True)
        title_inputid = torch.LongTensor(title_tokens['input_ids']).unsqueeze(0).to(device)
        title_mask = torch.LongTensor(title_tokens['attention_mask']).unsqueeze(0).to(device)
        title_embed = model_gen.get_input_embeddings()(title_inputid).to(device)

        audio_transcript = item['audio_transcript']
        audio_transcript_tokens = tkr(audio_transcript,
                                      is_split_into_words=True,
                                      max_length=max_audio_l,
                                      padding='max_length',
                                      truncation=True)
        audio_transcript_inputid = torch.LongTensor(audio_transcript_tokens['input_ids']).unsqueeze(0).to(device)
        audio_transcript_mask = torch.LongTensor(audio_transcript_tokens['attention_mask']).unsqueeze(0).to(device)
        audio_transcript_embed = model_gen.get_input_embeddings()(audio_transcript_inputid).to(device)

        annotation = item['annotation']
        target_tokens = tkr(annotation,
                            is_split_into_words=True,
                            max_length=max_target_l,
                            padding='max_length',
                            truncation=True)
        target_inputid = torch.LongTensor(target_tokens['input_ids']).unsqueeze(0).to(device)

        data_dict = {'title_embed': title_embed.squeeze(0).detach().cpu().numpy(),
                     'title_mask': title_mask.squeeze(0).detach().cpu().numpy(),
                     'audio_transcript_embed': audio_transcript_embed.squeeze(0).detach().cpu().numpy(),
                     'audio_transcript_mask': audio_transcript_mask.squeeze(0).detach().cpu().numpy(),
                     'target_inputid': target_inputid.squeeze(0).detach().cpu().numpy(),
                     }
        with open(os.path.join(bert_text_path, '{}.pkl'.format(video_id)), 'wb') as f:
            pickle.dump(data_dict, f)
        print('text processing %s' % video_id)

if __name__ == '__main__':
    text_feature_extraction()
