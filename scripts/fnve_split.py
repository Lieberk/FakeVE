import json
import random


def train_split():
    data_train_out = './dataset/FakeVE/data-split/vid_train.txt'
    data_val_out = './dataset/FakeVE/data-split/vid_val.txt'
    data_test_out = './dataset/FakeVE/data-split/vid_test.txt'

    with open('./dataset/FakeVE/data.json', 'r') as f:
        data = json.load(f)

    sent_lengths_fa = {}
    sent_lengths_ft = {}
    sent_lengths_fv = {}
    sent_lengths_fc = {}

    for item in data:
        video_id = item['video_id']
        if item['class'] == 'fa':
            sent_lengths_fa[video_id] = sent_lengths_fa.get(video_id, 0) + 1
        elif item['class'] == 'ft':
            sent_lengths_ft[video_id] = sent_lengths_ft.get(video_id, 0) + 1
        elif item['class'] == 'fv':
            sent_lengths_fv[video_id] = sent_lengths_fv.get(video_id, 0) + 1
        elif item['class'] == 'fc':
            sent_lengths_fc[video_id] = sent_lengths_fc.get(video_id, 0) + 1

    print(len(sent_lengths_fa), len(sent_lengths_ft), len(sent_lengths_fv), len(sent_lengths_fc))
    train_l_fa = int(len(sent_lengths_fa) * 0.8)
    train_l_ft = int(len(sent_lengths_ft) * 0.8)
    train_l_fv = int(len(sent_lengths_fv) * 0.8)
    train_l_fc = int(len(sent_lengths_fc) * 0.8)

    test_l_fa = int(len(sent_lengths_fa) * 0.9)
    test_l_ft = int(len(sent_lengths_ft) * 0.9)
    test_l_fv = int(len(sent_lengths_fv) * 0.9)
    test_l_fc = int(len(sent_lengths_fc) * 0.9)

    train_video_list = list(list(sent_lengths_fa.keys())[:train_l_fa] + list(sent_lengths_ft.keys())[:train_l_ft] \
                       + list(sent_lengths_fv.keys())[:train_l_fv] + list(sent_lengths_fc.keys())[:train_l_fc])
    val_video_list = list(list(sent_lengths_fa.keys())[train_l_fa:test_l_fa] + list(sent_lengths_ft.keys())[train_l_ft:test_l_ft] \
                     + list(sent_lengths_fv.keys())[train_l_fv:test_l_fv] + list(sent_lengths_fc.keys())[train_l_fc:test_l_fc])
    test_vide_list = list(list(sent_lengths_fa.keys())[test_l_fa:] + list(sent_lengths_ft.keys())[test_l_ft:] \
                     + list(sent_lengths_fv.keys())[test_l_fv:] + list(sent_lengths_fc.keys())[test_l_fc:])

    with open(data_train_out, 'a', encoding='utf-8') as f:
        for news_id in train_video_list:
            for item in data:
                if item.get('video_id') == news_id:
                    print(news_id)
                    f.write(str(item['video_id']) + '\n')

    with open(data_val_out, 'a', encoding='utf-8') as f:
        for news_id in val_video_list:
            for item in data:
                if item.get('video_id') == news_id:
                    print(news_id)
                    f.write(str(item['video_id']) + '\n')

    with open(data_test_out, 'a', encoding='utf-8') as f:
        for news_id in test_vide_list:
            for item in data:
                if item.get('video_id') == news_id:
                    print(news_id)
                    f.write(str(item['video_id']) + '\n')

if __name__ == '__main__':
    train_split()
