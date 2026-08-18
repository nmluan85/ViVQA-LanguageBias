import torch

class VCSSGenerator:
    def __init__(self, mask_value, mapping):
        self.mask_value = mask_value
        self.mapping = mapping

    def generate_vcss_top1(self, batch, types):
        """
        V-CSS:
        mask the top-1 critical object region using the visual
        attention mask.

        Returns:
            new_batch
            masked_any: whether a valid region was actually masked
        """

        new_batch = batch.copy()

        o_feature = batch["o_feature"].clone()
        object_padding_mask = batch["object_padding_mask"].clone()

        B = o_feature.shape[0]

        masked_any = []

        for i in range(B):
            q_type = int(types[i].item())

            if q_type not in {self.mapping["counting"], self.mapping["color"]}:
                masked_any.append(False)
                continue

            # top-1 is already padding -> cannot perform V-CSS
            if object_padding_mask[i, 0]:
                masked_any.append(False)
                continue

            # Optional: zero for semantic clarity/debugging
            o_feature[i, 0] = 0.0

            # THIS is the actual removal
            object_padding_mask[i, 0] = True

            masked_any.append(True)

        new_batch["o_feature"] = o_feature
        new_batch["object_padding_mask"] = object_padding_mask

        return new_batch, masked_any