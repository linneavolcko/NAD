# %%

import numpy as np
import matplotlib.pyplot as plt
import dill
import tdt
import trompy as tp

%run "NAD functions.py"
%run "NAD_fig_fx.py"

# %%
def get_tdt_data(row):
    tank = row[0]
    data = tdt.read_block(datafolder+tank)

    blue_sig = row[12]  
    uv_sig = row[11] 

    blue = getattr(data.streams, blue_sig).data
    uv = getattr(data.streams,uv_sig).data

    fs = getattr(data.streams,uv_sig).fs

    return blue, uv, fs

def get_tdt_epochs(row):
    tank = row[0]
    epoch_data = tdt.read_block(datafolder+tank, evtype=['epocs'])

    sipper_ttl = row[13]
    sipper = getattr(epoch_data.epocs, sipper_ttl).onset 
    
    licks_ttl = row[14]
    try:
        licks = getattr(epoch_data.epocs, licks_ttl).onset
    except AttributeError:
        print("No licks in this session")
        licks = []

    return sipper, licks

def get_snips_baseline(event, fs, signal, pre=10, length=20, bins=None, baseline_seconds=None, do_not_remove_baseline=False):
    """
    Gets snips around an event.

    Args:

    """
    start_in_seconds = event - pre
    start_in_sample = int(start_in_seconds * fs)
    length_in_seconds = length
    length_in_sample = int(length_in_seconds * fs)
    snip = signal[start_in_sample: start_in_sample + length_in_sample]
    
    if do_not_remove_baseline:
        return snip

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


def get_snips(signal, events, pre=10, length=20, bins=None, baseline_seconds=None, do_not_remove_baseline=False):
    snips = []

    for event in events:
        snip =get_snips_baseline(event, fs, signal, pre=pre, length=length, bins=bins, baseline_seconds=baseline_seconds)
        snips.append(snip)

    if len(snips[-1]) != len(snips[0]):
        snips = snips[:-1]
    
    return snips

def get_licks_per_trial(sipper, licks):

    licks_per_trial = []
    for sip in sipper:
        licks_per_trial.append([lick-sip for lick in licks if (lick>sip-10) and (lick<sip+30)])

    latency = []
    for trial in licks_per_trial:
        if len(trial) > 0:
            latency.append([lick for lick in trial if lick>0][0])
        else:
            latency.append(np.nan)

    return licks_per_trial, latency

# %%

rows, header = tp.metafilereader("..//NAD_2_and_3.xls")
datafolder = "D:\\Test Data\\photometry\\NAc GRAB photometry\\"

# %%
## code for troubleshooting functions and running single mice
row = rows[57]
mouse, session, sex, diet = row[2], row[6], row[3], row[4]
blue, uv, fs = get_tdt_data(row)
corrected_signal = tp.processdata(blue, uv, fs=fs)
sipper, licks = get_tdt_epochs(row)
lickdata = tp.lickCalc(licks)
licks_per_trial, latency = get_licks_per_trial(sipper, licks)

# %%
snips_sip = get_snips(corrected_signal, sipper, pre=10, length=40, bins=400, do_not_remove_baseline=True)
snips_sip_with_licks = [snip for snip,lpt in zip(snips_sip, licks_per_trial) if len(lpt)>0]
if len(licks) > 0:
    snips_licks = get_snips(corrected_signal, lickdata["rStart"], pre=10, length=40, bins=400, do_not_remove_baseline=True)
else:
    snips_licks = []

session_data = {"mouse": mouse, "session": session, "sex": sex, "diet": diet,
                "blue": blue, "uv": uv, "fs": fs,
                "corrected_signal": corrected_signal,
                "sipper": sipper, "licks": licks,
                "licks_per_trial": licks_per_trial, "latency": latency,
                "snips_sip": snips_sip,
                "snips_licks": snips_licks,
                "snips_sip_with_licks": snips_sip_with_licks}

all_session_data["{}_{}".format(mouse, session)] = session_data

# # %%
# %run "NAD_fig_fx.py"

# # %%
# make_figs(session_data)
#%%

all_session_data ={}

for row in rows:
    if row[7] == "cues" and row[10] != 0:
        mouse, session, sex, diet = row[2], row[6], row[3], row[4],
        print("Getting data from {}, session {}".format(mouse, session))

        try:
            blue, uv, fs = get_tdt_data(row)
            corrected_signal = tp.processdata(blue, uv, fs=fs)
            sipper, licks = get_tdt_epochs(row)
            lickdata = tp.lickCalc(licks)
            licks_per_trial, latency = get_licks_per_trial(sipper, licks)

            snips_sip = get_snips(corrected_signal, sipper, pre=10, length=40, bins=400, do_not_remove_baseline=True)
            snips_sip_with_licks = [snip for snip,lpt in zip(snips_sip, licks_per_trial) if len(lpt)>0]
            if len(licks) > 0:
                snips_licks = get_snips(corrected_signal, lickdata["rStart"], pre=10, length=40, bins=400, do_not_remove_baseline=True)
            else:
                snips_licks = []

            session_data = {"mouse": mouse, "session": session, "sex": sex, "diet": diet,
                            "blue": blue, "uv": uv, "fs": fs,
                            "corrected_signal": corrected_signal,
                            "sipper": sipper, "licks": licks,
                            "licks_per_trial": licks_per_trial, "latency": latency,
                            "snips_sip": snips_sip,
                            "snips_licks": snips_licks,
                            "snips_sip_with_licks": snips_sip_with_licks}
        except:
            print("Failed with {}, session {}".format(mouse, session))

        all_session_data["{}_{}".format(mouse, session)] = session_data
print("Finished running")

# Failed with NAD22, session 9.0
# 
# %%
import dill
with open(datafolder+"NAD_session_data_no_baseline_removed", 'wb') as pickle_out:
    dill.dump(all_session_data, pickle_out)

# %%
import dill
with open(datafolder+"NAD_session_data_no_baseline_removed_BACKUP", 'rb') as pickle_in:
    all_session_data = dill.load(pickle_in)


# %%
for key, session_data in all_session_data.items():
    make_figs(session_data)
    plt.close()


# %%

f, ax = plt.subplots()
tp.shadedError(ax, snips_sip)

        

# #for all trials 
#         trials = sipper_onset

#         snips = []

#         for trial in trials:
#             snip =get_snips_baseline_8s(trial, fs, correctedSignal)
#             snips.append(snip)

#         snips = np.array(snips)
#         if len(snips[-1]) == len(snips[0]):
#             mean_snips= np.mean(snips, axis=0) 
#         else:
#             snips =snips[0:-1]
#             mean_snips= np.mean(snips, axis=0)
            
# #for trials with licking, aligned to first lick?
#         lickdata = tp.lickCalc(licks_onset)
#         runs = lickdata["rStart"]
#         lick_snips = []

#         for run in runs:
#             snip =get_snips_baseline(run, fs, correctedSignal)
#             lick_snips.append(snip)

#         lick_snips = np.array(lick_snips)
#         if len(lick_snips[-1]) == len(lick_snips[0]):
#             mean_lick_snips= np.mean(lick_snips, axis=0) 
#         else:
#             lick_snips =lick_snips[0:-1]
#             mean_lick_snips= np.mean(lick_snips, axis=0)
            
# #for trials with licking, aligned to sipper? 
#         trials = sipper_onset
#         runs = lickdata["rStart"]
        
#         lick_snips_sipper_out = []
#         no_lick_snips_sipper_out = []
        
        
#         for trial in trials:
#             start_in_seconds = trial
#             start_in_sample = int(start_in_seconds * fs)
#             length_in_seconds = 30
#             length_in_Sample =int(length_in_seconds * fs)
            
#             #time_for_licks_in_trial = correctedSignal[start_in_sample: start_in_sample + length_in_Sample]
            
#             runs_in_trial = [run_ts for run_ts in runs if (run_ts > start_in_seconds) and (run_ts < start_in_seconds + length_in_seconds)]
            
#             if len(runs_in_trial) > 0:
#                 snip = get_snips_baseline(trial, fs, correctedSignal)
#                 lick_snips_sipper_out.append(snip)
#             else:
#                 snip = get_snips_baseline(trial, fs, correctedSignal)
#                 no_lick_snips_sipper_out.append(snip)

#         lick_snips_sipper_out = np.array(lick_snips_sipper_out)
#         if len(lick_snips_sipper_out) > 0:
#             if len(lick_snips_sipper_out[-1]) == len(lick_snips_sipper_out[0]):
#                 mean_lick_snips_sipper_out= np.mean(lick_snips_sipper_out, axis=0) 
#             else:
#                 lick_snips_sipper_out =lick_snips_sipper_out[0:-1]
#                 mean_lick_snips_sipper_out= np.mean(lick_snips_sipper_out, axis=0)
#         else:
#             mean_lick_snips_sipper_out = []

            
#         no_lick_snips_sipper_out = np.array(no_lick_snips_sipper_out)
#         if len(no_lick_snips_sipper_out) > 0:
#             if len(no_lick_snips_sipper_out[-1]) == len(no_lick_snips_sipper_out[0]):
#                 mean_no_lick_snips_sipper_out= np.mean(no_lick_snips_sipper_out, axis=0) 
#             else:
#                 no_lick_snips_sipper_out =no_lick_snips_sipper_out[0:-1]
#                 mean_no_lick_snips_sipper_out= np.mean(no_lick_snips_sipper_out, axis=0)
#         else:
#             no_mean_lick_snips_sipper_out = []

#         key = row[2]+"_s"+str(int(row[6]))

#         snips_dict_cues[key]={}

#         snips_dict_cues[key]["mouse"] = row[2]  
#         snips_dict_cues[key]["sex"] = row[3]
#         snips_dict_cues[key]["diet"] = row[4]
#         snips_dict_cues[key]["session"] = row[6]
#         snips_dict_cues[key]["snips"] = snips 
#         snips_dict_cues[key]["snips_mean"] = mean_snips
#         snips_dict_cues[key]["lick_snips"] = lick_snips 
#         snips_dict_cues[key]["lick_snips_mean"] = mean_lick_snips
#         snips_dict_cues[key]["lick_snips_sipper_out"] = lick_snips_sipper_out
#         snips_dict_cues[key]["lick_snips_sipper_out_mean"] = mean_lick_snips_sipper_out
#         snips_dict_cues[key]["no_lick_snips_sipper_out"] = no_lick_snips_sipper_out
#         snips_dict_cues[key]["no_lick_snips_sipper_out_mean"] = mean_no_lick_snips_sipper_out
#         snips_dict_cues[key]["lick data"] = lickdata
#         snips_dict_cues[key]["sipper"] = sipper_onset
# %%
