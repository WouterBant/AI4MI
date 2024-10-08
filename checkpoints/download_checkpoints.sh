#!/bin/bash

# 512 is fast samed and 1024 is samed
# 512 corresponds to the original samed implementation
# r stands for the rank of LoRa matrices

# Best weights for the samed like method
# wget --header="Referer: https://huggingface.co/" -P . https://huggingface.co/Wouter01/AI4MI/resolve/main/bestweights//bestweights_samed_512_r6_augment_no_normalize_no.pt
# wget --header="Referer: https://huggingface.co/" -P . https://huggingface.co/Wouter01/AI4MI/resolve/main/bestweights/bestweights_samed_512_r6_augment_yes_normalize_no.pt
# wget --header="Referer: https://huggingface.co/" -P . https://huggingface.co/Wouter01/AI4MI/resolve/main/bestweights/bestweights_samed_512_r6_augment_no_normalize_yes.pt
# wget --header="Referer: https://huggingface.co/" -P . https://huggingface.co/Wouter01/AI4MI/resolve/main/bestweights/bestweights_samed_512_r6_augment_yes_normalize_yes.pt

# wget --header="Referer: https://huggingface.co/" -P . https://huggingface.co/Wouter01/AI4MI/resolve/main/bestweights/bestweights_samed_1024_r4_augment_no_normalize_no.pt
# wget --header="Referer: https://huggingface.co/" -P . https://huggingface.co/Wouter01/AI4MI/resolve/main/bestweights/bestweights_samed_1024_r4_augment_yes_normalize_yes.pt

# wget --header="Referer: https://huggingface.co/" -P . https://huggingface.co/Wouter01/AI4MI/resolve/main/bestweights/bestweights_samed_1024_r6_augment_no_normalize_no.pt
# wget --header="Referer: https://huggingface.co/" -P . https://huggingface.co/Wouter01/AI4MI/resolve/main/bestweights/bestweights_samed_1024_r6_augment_yes_normalize_no.pt
# wget --header="Referer: https://huggingface.co/" -P . https://huggingface.co/Wouter01/AI4MI/resolve/main/bestweights/bestweights_samed_1024_r6_augment_no_normalize_yes.pt
# wget --header="Referer: https://huggingface.co/" -P . https://huggingface.co/Wouter01/AI4MI/resolve/main/bestweights/bestweights_samed_1024_r6_augment_yes_normalize_yes.pt

# wget --header="Referer: https://huggingface.co/" -P . https://huggingface.co/Wouter01/AI4MI/resolve/main/bestweights/bestweights_crf_samed_512_r6_augment_yes_normalize_no.pt

# The segment anything vit_b checkpoint (from meta)
# wget -P src/samed/checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth