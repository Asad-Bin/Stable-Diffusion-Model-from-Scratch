import torch
import gc

from config.config import *

is_local_test = False

def clear_memory():
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()



def print_gpu_memory(tag):
    if is_local_test == False:
        return
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    print(f"[{tag}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB on {device}")



# Optional: clear VRAM after encoding
# def reduce_vram_usage():
#     vae.to("cpu")
#     import gc
#     gc.collect()
#     torch.cuda.empty_cache()
