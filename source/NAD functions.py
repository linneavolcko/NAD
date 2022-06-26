# %%
import tdt
import matplotlib.pyplot as plt
import trompy as tp
import numpy as np
import pickle

# %%
# Modified from your function. Tidied up a bit and added option for bins and variable pre-event and snip lengths.

def get_snips_baseline(event, fs, signal, pre=10, length=20, bins=None, baseline_seconds=None):
    """
    Gets snips around an event.

    Args:

    """
    start_in_seconds = event - pre
    start_in_sample = int(start_in_seconds * fs)
    length_in_seconds = length
    length_in_sample = int(length_in_seconds * fs)
    snip = signal[start_in_sample: start_in_sample + length_in_sample]
    
    if baseline_seconds == None:
        baseline_seconds = pre
        
    baseline = snip[0 : int(baseline_seconds * fs)]
    mean = np.mean(baseline)
    std = np.std(baseline)
    
    snip_baseline_corrected = (snip - mean) / std

    if bins != None:
        rem = len(snip_baseline_corrected) % bins
        if  rem != 0:
            snip_baseline_corrected = snip_baseline_corrected[:-rem]
        snip_reshaped = np.reshape(snip_baseline_corrected, (bins, -1))
        snip_baseline_corrected = np.mean(snip_reshaped, axis=1)

    return snip_baseline_corrected

# # %%
# folder = "D:\\Test Data\\photometry\\NAc GRAB photometry\\"

# data = tdt.read_block(folder+"Test-220609-101142")

# # %%
# licks = data.epocs._7RL_.onset
# sipper = data.epocs._7sp_.onset
# blue = data.streams._4657.data
# uv = data.streams._4057.data

# fs = data.streams._4657.fs

# corrected_signal = tp.processdata(blue, uv)
# # %%
# f, ax = plt.subplots(nrows=2, figsize=(4, 5))

# snips = []

# for sip in sipper:
#     snip =get_snips_baseline(sip, fs, corrected_signal, pre=10, length=40, bins=400)
#     snips.append(snip)
#     ax[0].plot(snip, color="grey", alpha = 0.3)
# # %%
# len(snips)
# # %%

# %%
