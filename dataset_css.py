from torch.utils.data import Dataset, DataLoader
import json
import torch
import h5py
import re
import pandas as pd
import json
import os
import pickle
from utils import process_punctuation, preprocess_answers, preprocess_questions, get_answer_ids_from_train

class ViVQADataset(Dataset):
    def __init__(self, df, tokenizer, *image_features_paths, types=None, css_candidates=None, topk=36):

        self.tokenizer = tokenizer
        with open('data/vocab.json', 'r') as f:
            vocab = json.loads(f.read())
        self.vocab_a = vocab['answer']

        if types is not None:
            if not isinstance(types, (list, set, tuple)):
                types = [types]
            df = df[df['type'].isin(types)]

        self.topk = topk

        # IMPORTANT: always reset index so idx matches css_candidates key
        self.dataset = df.reset_index(drop=True)

        # save per-dataset css mapping
        self.css_candidates = css_candidates if css_candidates is not None else {}

        # q and a
        self.questions = [self.question2ids(question) for question in preprocess_questions(self.dataset)]
        self.answers = self.answers2idx(preprocess_answers(self.dataset), self.vocab_a)

        # v
        self.image_features_paths = image_features_paths
        self.visuals_id_to_index = self.create_visuals_id_to_index()
        self.visuals_ids = self.dataset['img_id'].tolist()

        self.feature_names = [os.path.basename(path).split('.')[0] for path in self.image_features_paths]

    def question2ids(self, question, max_len=40):
        tkz = self.tokenizer.encode_plus(
            text=question,
            padding='max_length',
            max_length=max_len,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False
        )
        return tkz

    def answers2idx(self, answers, vocab_a):
        return [vocab_a[answer] for answer in answers]

    def create_visuals_id_to_index(self):
        if not hasattr(self, 'features_files'):
            self.features_files = [h5py.File(image_features_path, 'r') for image_features_path in self.image_features_paths]
        visuals_ids = self.features_files[-1]['ids'][()]
        visuals_id_to_index = {id: i for i, id in enumerate(visuals_ids)}
        return visuals_id_to_index

    def load_image(self, image_id):
        index = self.visuals_id_to_index[image_id]
        data_img = [features_file['features'] for features_file in self.features_files]
        d = {}
        for name, data in zip(self.feature_names, data_img):
            d[name] = torch.from_numpy(data[index].astype('float32'))
        return d

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        image_id = self.visuals_ids[idx]
        visual_features = self.load_image(image_id)

        q_inputs = self.questions[idx]["input_ids"].squeeze(0)
        q_attn_mask = self.questions[idx]["attention_mask"].squeeze(0)

        a = self.answers[idx]
        q_type = self.dataset.iloc[idx]["type"]

        # Select candidate objects
        all_obj_feats = visual_features['VinVL']  # (N, 2048)

        # IMPORTANT: use per-dataset css_candidates keyed by sample idx
        candidates = self.css_candidates.get(idx, list(range(all_obj_feats.shape[0])))

        selected_features = []
        selected_indices = []

        for i in candidates[:self.topk]:
            if i == -1 or i >= all_obj_feats.shape[0]:
                selected_features.append(torch.zeros(all_obj_feats.shape[1], dtype=all_obj_feats.dtype))
                selected_indices.append(-1)
            else:
                selected_features.append(all_obj_feats[i])
                selected_indices.append(int(i))

        if len(selected_features) < self.topk:
            pad = self.topk - len(selected_features)
            selected_features.extend(
                [torch.zeros(all_obj_feats.shape[1], dtype=all_obj_feats.dtype)] * pad
            )
            selected_indices.extend([-1] * pad)

        o_feature = torch.stack(selected_features)

        item = {
            'g_feature': visual_features['blip-2'],
            'o_feature': o_feature,
            'question': q_inputs,
            'padding_mask': 1 - q_attn_mask,
            'labels': torch.tensor(a, dtype=torch.long),
            'type': torch.tensor(int(q_type), dtype=torch.long),

            # optional debug
            'sample_idx': torch.tensor(idx, dtype=torch.long),
            'selected_indices': torch.tensor(selected_indices, dtype=torch.long)
        }

        return item
    
def get_dataset(opt, tokenizer):
    df_train = pd.read_csv(opt.train_path)
    df_test = pd.read_csv(opt.test_path, index_col=0)
    df_val = pd.read_csv(opt.val_path)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    feature_paths = [
        os.path.join(opt.feature_paths, "blip-2.hdf5"), 
        os.path.join(opt.feature_paths, "VinVL.hdf5")
    ]

    train_css_path = os.path.join(opt.css_candidates, "css_candidates_train.pkl")
    val_css_path = os.path.join(opt.css_candidates, "css_candidates_val.pkl")
    test_css_path = os.path.join(opt.css_candidates, "css_candidates_test.pkl")

    with open(train_css_path, "rb") as f:
        css_candidates_train = pickle.load(f)
    with open(val_css_path, "rb") as f:
        css_candidates_val = pickle.load(f)
    with open(test_css_path, "rb") as f:
        css_candidates_test = pickle.load(f)

    train_dataset = ViVQADataset(df_train, tokenizer, *feature_paths, css_candidates=css_candidates_train, topk=opt.top_k)
    val_dataset   = ViVQADataset(df_val, tokenizer, *feature_paths, css_candidates=css_candidates_val, topk=opt.top_k)
    test_dataset  = ViVQADataset(df_test, tokenizer, *feature_paths, css_candidates=css_candidates_test, topk=opt.top_k)

    _, COUNTING_ANSWER_IDS = get_answer_ids_from_train(df_train, train_dataset.vocab_a, 1)
    _, COLOR_ANSWER_IDS = get_answer_ids_from_train(df_train, train_dataset.vocab_a, 2)

    return train_dataset, val_dataset, test_dataset, COUNTING_ANSWER_IDS, COLOR_ANSWER_IDS
