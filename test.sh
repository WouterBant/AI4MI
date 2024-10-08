#!/bin/bash

# python src/test.py --from_checkpoint checkpoints/bestweights/bestweights_samed_512_r6_augment_no_normalize_no.pt --r 6 --model samed_fast --batch_size 1
# python src/test.py --from_checkpoint checkpoints/bestweights/bestweights_samed_512_r6_augment_yes_normalize_no.pt --r 6 --model samed_fast --batch_size 1
# python src/test.py --from_checkpoint checkpoints/bestweights/bestweights_samed_512_r6_augment_no_normalize_yes.pt --r 6 --normalize --model samed_fast --batch_size 1
# python src/test.py --from_checkpoint checkpoints/bestweights/bestweights_samed_512_r6_augment_yes_normalize_yes.pt --r 6 --normalize --model samed_fast --batch_size 1

# python src/test.py --from_checkpoint checkpoints/bestweights/bestweights_samed_1024_r4_augment_no_normalize_no.pt --r 4 --normalize --model samed --batch_size 1
# python src/test.py --from_checkpoint checkpoints/bestweights/bestweights_samed_1024_r4_augment_yes_normalize_yes.pt --r 4 --normalize --model samed --batch_size 1
 
# python src/test.py --from_checkpoint checkpoints/bestweights/bestweights_samed_1024_r6_augment_no_normalize_no.pt --r 4 --model samed --batch_size 1
# python src/test.py --from_checkpoint checkpoints/bestweights/bestweights_samed_1024_r6_augment_yes_normalize_no.pt --r 6 --model samed --batch_size 1
# python src/test.py --from_checkpoint checkpoints/bestweights/bestweights_samed_1024_r6_augment_no_normalize_yes.pt --r 4 --normalize --model samed --batch_size 1
# python src/test.py --from_checkpoint checkpoints/bestweights/bestweights_samed_1024_r6_augment_yes_normalize_yes.pt --r 6 --normalize --model samed  --batch_size 1

# python src/test.py --from_checkpoint checkpsoints/bestweights/bestweights_crf_samed_512_r6_augment_yes_normalize_no.pt --r 6 --normalize --model samed_fast --crf --batch_size 1
