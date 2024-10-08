import sys
import torch


def remove_prefix(text, prefix):
    # if text.startswith(prefix):
    #     return text[len(prefix) :]
    prefix = "net._orig_mod."
    if text.startswith(prefix):
        return "net." + text[len(prefix) :]
    return text

# adjusted from https://github.com/pytorch/pytorch/issues/101107
def repair_checkpoint(path):
    in_state_dict = torch.load(path, map_location="cpu")
    print(in_state_dict.keys())
    pairings = [
        (src_key, remove_prefix(src_key, "_orig_mod."))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return  # Do not write checkpoint if no need to repair!
    out_state_dict = {}
    for src_key, dest_key in pairings:
        print(f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]
    torch.save(out_state_dict, "bestweights/"+path)

if __name__ == "__main__":
    paths = sys.argv[1:]
    for path in paths:
        print(path)
        repair_checkpoint(path)
        print("========")