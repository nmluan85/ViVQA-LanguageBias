from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import torch.nn as nn
import torch.nn.functional as F
import torch
from css.vcss import VCSSGenerator
from css.qcss import QCSSGenerator

def convert_to_object_only_batch(batch):
    """
    Remove global image content during the CSS branch.

    The global token positions are intentionally retained
    because the classifier pools from the first encoder token.
    """
    new_batch = batch.copy()
    new_batch["g_feature"] = torch.zeros_like(batch["g_feature"])
    return new_batch

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
        self.ALTERNATE_VQCSS = opt.alternate_vqcss
        self.USE_OBJECT_ONLY_CSS = opt.use_object_only_css

        # 0 = VCSS, 1 = QCSS
        self.css_mode_toggle = 0

        self.tokenizer = tokenizer

        self.COUNTING_ANSWER_IDS = COUNTING_ANSWER_IDS
        self.COLOR_ANSWER_IDS = COLOR_ANSWER_IDS

        self.vcss = VCSSGenerator(opt.mask_value, self.mapping)
        self.qcss = QCSSGenerator()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
 
        types = inputs.pop("type")
        sample_indices = inputs.pop(
            "sample_idx", None
        )
        inputs.pop("selected_indices", None)

        # 1) factual/original pass
        outputs = model(**inputs)
        total_loss = outputs.loss

        # 2) Decide whether CSS should be applied
        apply_css = (
            model.training
            and self.state.global_step >= self.css_start_step 
            and self.state.global_step % self.css_interval == 0
        )

        if not apply_css:
            return (total_loss, outputs) if return_outputs else total_loss

        # 3) keep only counting/color samples for CSS
        css_mask = torch.tensor(
            [int(t.item()) in self.CSS_TYPES for t in types],
            device=device,
            dtype=torch.bool
        )

        if css_mask.sum() == 0:
            return (total_loss, outputs) if return_outputs else total_loss

        css_types = types[css_mask]

        css_inputs = {
            k: v[css_mask]
            for k, v in inputs.items()
        }

        inputs_detached = {
            k: v.detach() if torch.is_tensor(v) else v
            for k, v in css_inputs.items()
        }

        # 4) Decide CSS branch
        run_vcss = self.USE_VCSS
        run_qcss = self.USE_QCSS

        if (
            self.ALTERNATE_VQCSS
            and self.USE_VCSS
            and self.USE_QCSS
        ):
            if self.css_mode_toggle == 0:
                run_vcss = True
                run_qcss = False
            else:
                run_vcss = False
                run_qcss = True

        # 5) V-CSS
        if run_vcss:
            vcss_batch, vcss_masked_any = (
                self.vcss.generate_vcss_top1(
                    inputs_detached,
                    css_types
                )
            )

            # Labels are not used as factual targets
            # in the new counterfactual loss.
            vcss_batch.pop("labels", None)

            model_inputs_vcss = vcss_batch

            # CSS branch uses object-level visual information only
            if self.USE_OBJECT_ONLY_CSS:
                model_inputs_vcss = convert_to_object_only_batch(
                    model_inputs_vcss
                )

            vcss_logits = model(
                **model_inputs_vcss
            ).logits

            log_probs_v = F.log_softmax(
                vcss_logits,
                dim=-1
            )

            vcss_loss = 0.0
            valid_count = 0

            for i in range(len(css_types)):

                # Skip samples where no visual region
                # was successfully masked
                if not vcss_masked_any[i]:
                    continue

                q_type = int(css_types[i].item())

                target = torch.zeros_like(
                    vcss_logits[i]
                )

                # Counting counterfactual target
                if (
                    q_type == self.mapping["counting"]
                    and len(self.COUNTING_ANSWER_IDS) > 0
                ):
                    target[
                        self.COUNTING_ANSWER_IDS
                    ] = (
                        1.0
                        / len(self.COUNTING_ANSWER_IDS)
                    )

                # Color counterfactual target
                elif (
                    q_type == self.mapping["color"]
                    and len(self.COLOR_ANSWER_IDS) > 0
                ):
                    target[
                        self.COLOR_ANSWER_IDS
                    ] = (
                        1.0
                        / len(self.COLOR_ANSWER_IDS)
                    )

                else:
                    continue

                vcss_loss += F.kl_div(
                    log_probs_v[i].unsqueeze(0),
                    target.unsqueeze(0),
                    reduction="batchmean"
                )

                valid_count += 1

            if valid_count > 0:
                total_loss = (
                    total_loss
                    + self.alpha_css
                    * (vcss_loss / valid_count)
                )

        # 6) Q-CSS
        if run_qcss:

            if sample_indices is None:
                raise RuntimeError(
                    "sample_idx is required for "
                    "Q-CSS critical noun lookup."
                )

            # Retrieve pre-extracted critical nouns
            # corresponding to the CSS sub-batch
            css_sample_indices = (
                sample_indices[css_mask]
                .detach()
                .cpu()
                .tolist()
            )

            critical_nouns_css = [
                self.train_dataset.critical_nouns[idx]
                for idx in css_sample_indices
            ]

            qcss_batch, masked_any = (
                self.qcss.generate_qcss_from_critical_nouns(
                    inputs_detached,
                    critical_nouns_css,
                    self.tokenizer
                )
            )

            model_inputs_qcss = qcss_batch

            # CSS branch uses object-level visual information only
            if self.USE_OBJECT_ONLY_CSS:
                model_inputs_qcss = convert_to_object_only_batch(
                    model_inputs_qcss
                )

            qcss_logits = model(
                **model_inputs_qcss
            ).logits

            log_probs_q = F.log_softmax(
                qcss_logits,
                dim=-1
            )

            qcss_loss = 0.0
            valid_count = 0

            for i in range(len(css_types)):

                # Skip samples whose critical noun
                # could not be masked
                if not masked_any[i]:
                    continue

                q_type = int(css_types[i].item())

                target = torch.zeros_like(
                    qcss_logits[i]
                )

                # Counting counterfactual target
                if (
                    q_type == self.mapping["counting"]
                    and len(self.COUNTING_ANSWER_IDS) > 0
                ):
                    target[
                        self.COUNTING_ANSWER_IDS
                    ] = (
                        1.0
                        / len(self.COUNTING_ANSWER_IDS)
                    )

                # Color counterfactual target
                elif (
                    q_type == self.mapping["color"]
                    and len(self.COLOR_ANSWER_IDS) > 0
                ):
                    target[
                        self.COLOR_ANSWER_IDS
                    ] = (
                        1.0
                        / len(self.COLOR_ANSWER_IDS)
                    )

                else:
                    continue

                qcss_loss += F.kl_div(
                    log_probs_q[i].unsqueeze(0),
                    target.unsqueeze(0),
                    reduction="batchmean"
                )

                valid_count += 1

            if valid_count > 0:
                total_loss = (
                    total_loss
                    + self.alpha_css
                    * (qcss_loss / valid_count)
                )

        # 7) Alternate VCSS / QCSS
        if (
            self.ALTERNATE_VQCSS
            and self.USE_VCSS
            and self.USE_QCSS
        ):
            self.css_mode_toggle = (
                1 - self.css_mode_toggle
            )

        return (
            (total_loss, outputs)
            if return_outputs
            else total_loss
        )