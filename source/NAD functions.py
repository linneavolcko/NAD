# %%
import tdt
import matplotlib.pyplot as plt
import trompy as tp
import numpy as np
import pickle

# %%
# Modified from your function. Tidied up a bit and added option for bins and variable pre-event and snip lengths.

def get_snips_baseline(event, fs, signal, pre=10, length=20, bins="none"):
    """
    Gets snips around an event.

    Args:

    """
    start_in_seconds = event - pre
    start_in_sample = int(start_in_seconds * fs)
    length_in_seconds = length
    length_in_sample = int(length_in_seconds * fs)
    snip = signal[start_in_sample: start_in_sample + length_in_sample]
    
    baseline = snip[0 : int(pre * fs)]
    mean = np.mean(baseline)
    std = np.std(baseline)
    
    snip_baseline_corrected = (snip - mean) / std

    if bins != "none":
        rem = len(snip_baseline_corrected) % bins
        if  rem != 0:
            snip_baseline_corrected = snip_baseline_corrected[:-rem]
        snip_reshaped = np.reshape(snip_baseline_corrected, (bins, -1))
        snip_baseline_corrected = np.mean(snip_reshaped)

    return snip_baseline_corrected
