class QCSSGenerator:

    def mask_phrase_in_question(
        self,
        input_ids,
        padding_mask,
        phrase,
        tokenizer,
        max_len=40
    ):
        """
        Mask a phrase by replacing its tokens with PAD tokens
        and marking those positions as inactive in padding_mask.
        """
        if phrase is None:
            return input_ids, padding_mask, False

        phrase = str(phrase).strip().lower()

        if phrase == "":
            return input_ids, padding_mask, False

        phrase_tok = tokenizer.encode_plus(
            text=phrase,
            padding=False,
            truncation=True,
            max_length=max_len,
            return_attention_mask=False,
            return_token_type_ids=False
        )["input_ids"]

        special_ids = {
            tokenizer.pad_token_id,
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "cls_token_id", None),
            getattr(tokenizer, "sep_token_id", None),
        }

        special_ids.discard(None)

        phrase_tok = [
            tid for tid in phrase_tok
            if tid not in special_ids
        ]

        if len(phrase_tok) == 0:
            return input_ids, padding_mask, False

        seq = input_ids.tolist()

        span = self.find_subsequence(seq, phrase_tok)

        if span is None:
            return input_ids, padding_mask, False

        start, end = span

        new_input_ids = input_ids.clone()
        new_padding_mask = padding_mask.clone()

        new_input_ids[start:end] = tokenizer.pad_token_id
        new_padding_mask[start:end] = 1

        return new_input_ids, new_padding_mask, True

    def generate_qcss_from_critical_nouns(
        self,
        batch,
        critical_nouns,
        tokenizer
    ):
        """
        critical_nouns:
            list[str], aligned with samples in the CSS sub-batch.
        """
        new_batch = batch.copy()

        questions = batch["question"].clone()
        padding_mask = batch["padding_mask"].clone()

        B = questions.shape[0]
        masked_any = []

        for i in range(B):
            phrase = critical_nouns[i]

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

    def find_subsequence(self, sequence, subsequence):
        """
        Find first occurrence of subsequence in sequence.
        Returns (start, end) inclusive-exclusive, or None.
        """
        if len(subsequence) == 0 or len(subsequence) > len(sequence):
            return None

        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i:i + len(subsequence)] == subsequence:
                return (i, i + len(subsequence))

        return None