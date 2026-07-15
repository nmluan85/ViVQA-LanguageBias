import torch

class VCSSGenerator:
    def __init__(self, mask_value, mapping):
        self.mask_value = mask_value
        self.mapping = mapping

        self.vi2num = {
            "không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4,
            "năm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10
        }
        self.num2vi = {v: k for k, v in self.vi2num.items()}

    def generate_vcss_top1(self, batch, vocab, types):
        """
        Selector-guided V-CSS:
        remove the top-1 candidate region already ranked by css_candidates.

        Assumption:
        - batch["o_feature"] is ordered by candidate ranking
        - local index 0 is the most relevant selected region
        """
        new_batch = batch.copy()
        o_feature = batch["o_feature"].clone()
        labels = batch["labels"].clone()

        B = o_feature.shape[0]

        for i in range(B):
            q_type = int(types[i].item())

            # skip if top-1 slot is empty / padded
            # since padded regions are all-zero vectors, detect that
            top1_feat = o_feature[i, 0]
            if torch.all(top1_feat == 0):
                continue

            if q_type == self.mapping["counting"]:
                # remove top-1 selected region
                o_feature[i, 0] = self.mask_value

                # adjust count by -1
                ans_idx = labels[i].item()
                ans_idx, num = self.adjust_count_answer(ans_idx, vocab)

                new_num = max(0, num - 1)
                new_word = self.num2vi.get(new_num, None)

                if new_word is not None and new_word in vocab:
                    labels[i] = vocab[new_word]

            elif q_type == self.mapping["color"]:
                # color: remove top-1 selected region, no relabel
                o_feature[i, 0] = self.mask_value

        new_batch["o_feature"] = o_feature
        new_batch["labels"] = labels

        return new_batch
    
    def adjust_count_answer(self, answer_idx, vocab):
        inv_vocab = {v: k for k, v in vocab.items()}
        answer_word = inv_vocab[answer_idx]

        if answer_word not in self.vi2num:
            return answer_idx, 0

        return answer_idx, self.vi2num[answer_word]