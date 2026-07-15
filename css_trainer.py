from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import torch.nn as nn
import torch.nn.functional as F
import torch
from css.vcss import VCSSGenerator
from css.qcss import QCSSGenerator

class CSSTrainer(Trainer):
    def __init__(self, opt, tokenizer, COUNTING_ANSWER_IDS, COLOR_ANSWER_IDS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.css_interval = opt.css_interval
        self.alpha_css = opt.alpha_css
        self.css_start_step =opt.css_start_step

        self.mapping = {
            "counting": 1,
            "color": 2,
        }

        self.CSS_TYPES = {self.mapping[t.lower()] for t in opt.css_types}
        self.USE_VCSS = opt.use_vcss
        self.USE_QCSS = opt.use_qcss

        self.tokenizer = tokenizer

        self.COUNTING_ANSWER_IDS = COUNTING_ANSWER_IDS
        self.COLOR_ANSWER_IDS = COLOR_ANSWER_IDS

        self.vcss = VCSSGenerator(opt.mask_value, self.mapping)
        self.qcss = QCSSGenerator(opt.output_dir)

        

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        types = inputs.pop("type")   # [B]

        # keep raw questions for Q-CSS phrase extraction
        sample_indices = inputs.pop("sample_idx", None) if "sample_idx" in inputs else None
        selected_indices = inputs.pop("selected_indices", None) if "selected_indices" in inputs else None

        # 1) factual/original pass
        outputs = model(**inputs)
        total_loss = outputs.loss

        apply_css = (
            self.state.global_step >= self.css_start_step and
            self.state.global_step % self.css_interval == 0
        )

        if not apply_css:
            return (total_loss, outputs) if return_outputs else total_loss

        # 2) keep only counting/color samples for CSS
        css_mask = torch.tensor(
            [int(t.item()) in self.CSS_TYPES for t in types],
            device=device,
            dtype=torch.bool
        )

        if css_mask.sum() == 0:
            return (total_loss, outputs) if return_outputs else total_loss

        css_types = types[css_mask]

        css_inputs = {}
        for k, v in inputs.items():
            css_inputs[k] = v[css_mask]

        inputs_detached = {
            k: v.detach() if torch.is_tensor(v) else v
            for k, v in css_inputs.items()
        }

        # raw questions aligned to css sub-batch for Q-CSS
        raw_questions_css = [
            self.train_dataset.dataset.iloc[int(i.item())]["question"]
            for i in sample_indices[css_mask]
        ] if sample_indices is not None else [""] * int(css_mask.sum().item())

        # -------------------------
        # V-CSS
        # -------------------------
        if self.USE_VCSS:
            vcss_batch = self.vcss.generate_vcss_top1(
                inputs_detached,
                self.train_dataset.vocab_a,
                css_types
            )

            vcss_labels = vcss_batch["labels"]
            model_inputs_vcss = {k: v for k, v in vcss_batch.items() if k != "labels"}

            vcss_logits = model(**model_inputs_vcss).logits
            log_probs_v = F.log_softmax(vcss_logits, dim=-1)

            vcss_loss = 0.0
            n = len(css_types)

            for i in range(n):
                q_type = int(css_types[i].item())

                if q_type == self.mapping["counting"]:
                    vcss_loss += F.cross_entropy(
                        vcss_logits[i].unsqueeze(0),
                        vcss_labels[i].unsqueeze(0)
                    )

                elif q_type == self.mapping["color"]:
                    target = torch.zeros_like(vcss_logits[i])
                    if len(self.COLOR_ANSWER_IDS) > 0:
                        target[self.COLOR_ANSWER_IDS] = 1.0 / len(self.COLOR_ANSWER_IDS)
                        vcss_loss += F.kl_div(
                            log_probs_v[i].unsqueeze(0),
                            target.unsqueeze(0),
                            reduction='batchmean'
                        )

            total_loss = total_loss + self.alpha_css * (vcss_loss / n)

        # -------------------------
        # Q-CSS
        # -------------------------
        if self.USE_QCSS:
            qcss_batch, masked_any = self.qcss.generate_qcss_type_specific(
                inputs_detached,
                raw_questions_css,
                css_types,
                self.tokenizer
            )

            qcss_logits = model(**qcss_batch).logits
            log_probs_q = F.log_softmax(qcss_logits, dim=-1)

            qcss_loss = 0.0
            valid_count = 0

            for i in range(len(css_types)):
                if not masked_any[i]:
                    continue

                q_type = int(css_types[i].item())
                target = torch.zeros_like(qcss_logits[i])

                if q_type == self.mapping["counting"] and len(self.COUNTING_ANSWER_IDS) > 0:
                    target[self.COUNTING_ANSWER_IDS] = 1.0 / len(self.COUNTING_ANSWER_IDS)
                    qcss_loss += F.kl_div(
                        log_probs_q[i].unsqueeze(0),
                        target.unsqueeze(0),
                        reduction='batchmean'
                    )
                    valid_count += 1

                elif q_type == self.mapping["color"] and len(self.COLOR_ANSWER_IDS) > 0:
                    target[self.COLOR_ANSWER_IDS] = 1.0 / len(self.COLOR_ANSWER_IDS)
                    qcss_loss += F.kl_div(
                        log_probs_q[i].unsqueeze(0),
                        target.unsqueeze(0),
                        reduction='batchmean'
                    )
                    valid_count += 1

            if valid_count > 0:
                total_loss = total_loss + self.alpha_css * (qcss_loss / valid_count)

        return (total_loss, outputs) if return_outputs else total_loss