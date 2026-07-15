import py_vncorenlp
import jnius_config
import shutil
import re
import os

class QCSSGenerator:
    def __init__(self, output_dir):

        # jar_file = os.path.join(output_dir, "VnCoreNLP-1.2.jar")
        # model_dir = os.path.join(output_dir, "models")

        # if not os.path.isfile(jar_file):
        #     if os.path.isdir(model_dir):
        #         shutil.rmtree(model_dir)

        #     py_vncorenlp.download_model(save_dir=output_dir)

        # self.segmenter = py_vncorenlp.VnCoreNLP(
        #     annotators=["wseg", "pos", "parse"],
        #     save_dir=output_dir,
        # )
        jar = os.path.abspath(os.path.join(output_dir, "VnCoreNLP-1.2.jar"))

        jnius_config.add_options("-Xmx2g")
        jnius_config.set_classpath(jar)

        from jnius import autoclass

        VnCoreNLP = autoclass("vn.pipeline.VnCoreNLP")

        self.segmenter = VnCoreNLP(["wseg", "pos", "parse"])

        # py_vncorenlp.download_model(save_dir=output_dir)

        # self.segmenter = py_vncorenlp.VnCoreNLP(
        #     annotators=["wseg", "pos", "parse"],
        #     save_dir=output_dir,
        # )

        self.STOPWORDS_AFTER_NP = {
            "là", "gì", "nào", "ra", "sao", "được", "đang", "bị", "trong",
            "trên", "ở", "với", "bên", "phía", "khi", "mà", "và", "hay",
            "có", "của", "theo", "dưới", "giữa", "ngoài", "vào", "màu"
        }

        self.SPECIAL_CASES = { "ngựa_vằn", "cậu", "đứa" }

    def generate_qcss_type_specific(self, batch, raw_questions, types, tokenizer):
        """
        raw_questions: list[str] aligned with the filtered css batch
        types: tensor [B]
        """
        new_batch = batch.copy()

        questions = batch["question"].clone()
        padding_mask = batch["padding_mask"].clone()

        B = questions.shape[0]
        masked_any = []

        for i in range(B):
            q_type = int(types[i].item())
            phrase = self.get_qcss_target_phrase(raw_questions[i], q_type)

            q_new, pm_new, ok = self.mask_phrase_in_question(
                questions[i],
                padding_mask[i],
                phrase,
                tokenizer
            )

            questions[i] = q_new
            padding_mask[i] = pm_new
            masked_any.append(ok)

        new_batch["question"] = questions
        new_batch["padding_mask"] = padding_mask
        return new_batch, masked_any

    def extract_phrase_vncorenlp(self, text: str, max_tokens: int = 6) -> str | None:
        """
        Surface phrase for masking Q-CSS.
        Keeps classifiers if present.
        """
        text = self.clean_tail(text)
        if not text:
            return None
            
        toks = self.vncorenlp_tokens_pos(text, self.segmenter)

        phrase = []
        started = False

        for word, pos in toks:
            w = word.lower()
            if w in self.STOPWORDS_AFTER_NP:
                break

            # keep noun-ish tokens and some modifiers
            if pos in {"N", "Np", "Ny", "Nb", "Nc", "A"} or w in self.SPECIAL_CASES:
                phrase.append(w)
                started = True
                if len(phrase) >= max_tokens:
                    break
            else:
                if started:
                    break

        if not phrase:
            return None
            
        result = " ".join(phrase).strip()
        return result.replace("_", " ")
    
    def vncorenlp_tokens_pos(self, text: str, annotator) -> list[tuple[str, str]]:
        """
        Returns list of (wordForm, posTag)
        """
        out = annotator.annotate_text(text)
        # py_vncorenlp returns dict keyed by sentence index
        sent0 = out[0]
        return [(tok["wordForm"], tok["posTag"]) for tok in sent0]
    
    def clean_tail(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text
    
    def get_qcss_target_phrase(self, question, q_type):
        triggers = {
            1: ["bao nhiêu"],
            2: ["màu sắc của", "màu của"]  # longer first
        }

        q = question.strip()
        q_lower = q.lower()

        if q_type == 2:
            # Exact pattern: "màu (sắc) của ... là gì"
            match = re.search(r"màu(?:\s+sắc)?\s+của\s+(.+?)\s+là gì", q_lower)
            if match:
                return match.group(1).strip()
            return None

        if q_type == 1:
            trigger_found = None
            trigger_index = -1
            for phrase in triggers[1]:
                idx = q_lower.find(phrase)
                if idx != -1:
                    trigger_found = phrase
                    trigger_index = idx
                    break

            if trigger_found is None:
                return None

            after_trigger = q[trigger_index + len(trigger_found):].strip()
            after_trigger = re.split(r"\b(là gì|là|gì|nào|ra sao)\b", after_trigger, maxsplit=1)[0].strip()

            first_np = self.extract_phrase_vncorenlp(after_trigger)
            if first_np:
                return first_np
            
            words = after_trigger.split()
            return " ".join(words[:4]).strip() if words else None

        return None
    
    def mask_phrase_in_question(self, input_ids, padding_mask, phrase, tokenizer, max_len=40):
        """
        input_ids: [L]
        padding_mask: [L] where 1 = padded, 0 = valid
        phrase: string to mask
        """
        if phrase is None or len(phrase.strip()) == 0:
            return input_ids, padding_mask, False

        phrase_tok = tokenizer.encode_plus(
            text=phrase,
            padding=False,
            truncation=True,
            max_length=max_len,
            return_attention_mask=False,
            return_token_type_ids=False
        )["input_ids"]

        # remove BOS/EOS-like special tokens if present
        special_ids = {
            tokenizer.pad_token_id,
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "cls_token_id", None),
            getattr(tokenizer, "sep_token_id", None),
        }
        phrase_tok = [tid for tid in phrase_tok if tid not in special_ids]

        if len(phrase_tok) == 0:
            return input_ids, padding_mask, False

        seq = input_ids.tolist()
        span = self.find_subsequence(seq, phrase_tok)

        if span is None:
            return input_ids, padding_mask, False

        start, end = span
        new_input_ids = input_ids.clone()
        new_padding_mask = padding_mask.clone()

        for pos in range(start, end):
            new_input_ids[pos] = tokenizer.pad_token_id
            new_padding_mask[pos] = 1   # IMPORTANT: masked token becomes padding/inactive

        return new_input_ids, new_padding_mask, True
    
    def find_subsequence(self, sequence, subsequence):
        """
        Find first occurrence of subsequence in sequence.
        Returns (start, end) inclusive-exclusive, or None.
        """
        if len(subsequence) == 0 or len(subsequence) > len(sequence):
            return None

        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i:i+len(subsequence)] == subsequence:
                return (i, i + len(subsequence))
        return None