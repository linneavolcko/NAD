# %%

import trompy as tp

%run "NAD functions.py"
# %%

rows, header = tp.metafilereader("..//NAD_2_and_3.xls")

# %%
datafolder = "D:\\Test Data\\photometry\\NAc GRAB photometry\\"

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
    licks = getattr(epoch_data.epocs, licks_ttl).onset

    return sipper, licks

def get_snips(signal, events):
    snips = []

    for event in events:
        snip =get_snips_baseline(event, fs, signal, pre=5, length=20, bins=200)
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
#%%

row = rows[53]
mouse, session = row[2], row[6]
blue, uv, fs = get_tdt_data(row)
corrected_signal = tp.processdata(blue, uv, fs=fs)
sipper, licks = get_tdt_epochs(row)
lickdata = tp.lickCalc(licks)
licks_per_trial, latency = get_licks_per_trial(sipper, licks)

snips_sip = get_snips(corrected_signal, sipper)
snips_licks = get_snips(corrected_signal, lickdata["rStart"])

snips_sip_with_licks = [snip for snip,lpt in zip(snips_sip, licks_per_trial) if len(lpt)>0]

session_data = {"mouse": mouse, "session": session,
                "blue": blue, "uv": uv, "fs": fs,
                "corrected_signal": corrected_signal,
                "sipper": sipper, "licks": licks,
                "licks_per_trial": licks_per_trial, "latency": latency,
                "snips_sip": snips_sip,
                "snips_licks": snips_licks,
                "snips_sip_with_licks": snips_sip_with_licks}

# %%

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl

def make_figs(session_data):

    f = plt.figure(figsize=(8.27, 11.69), dpi=100)
    gs1 = gridspec.GridSpec(5, 2)
    gs1.update(left=0.10, right= 0.9, wspace=0.5, hspace = 0.7)
    plt.suptitle("{}: Session {}".format(session_data["mouse"], session_data["session"]))

    session_figs(f, gs1, session_data)
    lick_fig(f, gs1, session_data)
    
    ax5 = f.add_subplot(gs1[1,1])
    make_heatmap(session_data["snips_sip"], ax=ax5, events=session_data["latency"])

    ax6 = f.add_subplot(gs1[2,1])
    tp.shadedError(ax6, session_data["snips_sip"])

    ax8 = f.add_subplot(gs1[3,1])
    tp.shadedError(ax8, session_data["snips_sip_with_licks"])

    f.savefig("..\\output\\{}_{}.pdf".format(session_data["mouse"], session_data["session"]))


def session_figs(f, gs1, session_data):
    
    ax1 = f.add_subplot(gs1[0, 0])
    ax2 = f.add_subplot(gs1[0, 1])

    ax1.plot(session_data["blue"], color="blue")
    ax1.plot(session_data["uv"], color="magenta")

    ax2.plot(session_data["corrected_signal"])

    licks_in_samples = [int(event*fs) for event in session_data["licks"]]
    sipper_in_samples = [int(event*fs) for event in session_data["sipper"]]

    # place markers on top
    ax1.vlines(licks_in_samples, 100, 105)
    ax1.vlines(sipper_in_samples, 105, 110)

def lick_fig(f, gs1, session_data):

    inds = np.argsort(session_data["latency"])
    licks_per_trial_sorted = [session_data["licks_per_trial"][i] for i in inds]

    ax3 = f.add_subplot(gs1[1, 0])
    ax4 = f.add_subplot(gs1[2, 0])

    for idx, licks_in_trial in enumerate(licks_per_trial_sorted):
        ax3.vlines(licks_in_trial, idx, idx+1)

    ax3.set_xlim([-10, 30])
    ax3.set_ylim([0, 40])

    bins=np.arange(-10, 30, 0.1)
    licks_hist = np.histogram(tp.flatten_list(licks_per_trial), bins=bins)
    ax4.plot(licks_hist[0])

def make_heatmap(data, events=None, ax=None, cmap="jet", sort=True, ylabel="Trials"):

    if ax == None:
        f, ax = plt.subplots()

    (ntrials, bins) = np.shape(data)

    xvals = np.linspace(-10,30,bins)
    yvals = np.arange(0, ntrials)
    xx, yy = np.meshgrid(xvals, yvals)

    if sort == True:
        try:
            inds = np.argsort(events)
            data = [data[i] for i in inds]
            events = [events[i] for i in inds]
        except:
            print("Events cannot be sorted")

    mesh = ax.pcolormesh(xx, yy, data, cmap=cmap, shading="auto")

    if events:
        ax.vlines(events, yvals-0.5, yvals+0.5, color='w')

    ax.set_ylabel(ylabel, rotation=90, labelpad=2)

    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax, mesh

make_figs(session_data)

# %%

#%%
for row in rows:
    if row[7] == "cues" and row[10] != 0:

        blue, uv, fs = get_tdt_data(row)
        corrected_signal = tp.processdata(blue, uv, fs=fs)

        sipper, licks = get_tdt_epochs(row)
        lickdata = tp.lickCalc(licks)

        snips_sip = get_snips(corrected_signal, sipper)
        snips_licks = get_snips(corrected_signal, lickdata["rStart"])

        session_data = {"blue": blue, "uv": uv, "fs": fs,
                        "corrected_signal": corrected_signal,
                        "sipper": sipper, "licks": licks,
                        "snips_sip": snips_sip,
                        "snips_licks": snips_licks}
        
        make_figs(session_data)





        

#for all trials 
        trials = sipper_onset

        snips = []

        for trial in trials:
            snip =get_snips_baseline_8s(trial, fs, correctedSignal)
            snips.append(snip)

        snips = np.array(snips)
        if len(snips[-1]) == len(snips[0]):
            mean_snips= np.mean(snips, axis=0) 
        else:
            snips =snips[0:-1]
            mean_snips= np.mean(snips, axis=0)
            
#for trials with licking, aligned to first lick?
        lickdata = tp.lickCalc(licks_onset)
        runs = lickdata["rStart"]
        lick_snips = []

        for run in runs:
            snip =get_snips_baseline(run, fs, correctedSignal)
            lick_snips.append(snip)

        lick_snips = np.array(lick_snips)
        if len(lick_snips[-1]) == len(lick_snips[0]):
            mean_lick_snips= np.mean(lick_snips, axis=0) 
        else:
            lick_snips =lick_snips[0:-1]
            mean_lick_snips= np.mean(lick_snips, axis=0)
            
#for trials with licking, aligned to sipper? 
        trials = sipper_onset
        runs = lickdata["rStart"]
        
        lick_snips_sipper_out = []
        no_lick_snips_sipper_out = []
        
        
        for trial in trials:
            start_in_seconds = trial
            start_in_sample = int(start_in_seconds * fs)
            length_in_seconds = 30
            length_in_Sample =int(length_in_seconds * fs)
            
            #time_for_licks_in_trial = correctedSignal[start_in_sample: start_in_sample + length_in_Sample]
            
            runs_in_trial = [run_ts for run_ts in runs if (run_ts > start_in_seconds) and (run_ts < start_in_seconds + length_in_seconds)]
            
            if len(runs_in_trial) > 0:
                snip = get_snips_baseline(trial, fs, correctedSignal)
                lick_snips_sipper_out.append(snip)
            else:
                snip = get_snips_baseline(trial, fs, correctedSignal)
                no_lick_snips_sipper_out.append(snip)

        lick_snips_sipper_out = np.array(lick_snips_sipper_out)
        if len(lick_snips_sipper_out) > 0:
            if len(lick_snips_sipper_out[-1]) == len(lick_snips_sipper_out[0]):
                mean_lick_snips_sipper_out= np.mean(lick_snips_sipper_out, axis=0) 
            else:
                lick_snips_sipper_out =lick_snips_sipper_out[0:-1]
                mean_lick_snips_sipper_out= np.mean(lick_snips_sipper_out, axis=0)
        else:
            mean_lick_snips_sipper_out = []

            
        no_lick_snips_sipper_out = np.array(no_lick_snips_sipper_out)
        if len(no_lick_snips_sipper_out) > 0:
            if len(no_lick_snips_sipper_out[-1]) == len(no_lick_snips_sipper_out[0]):
                mean_no_lick_snips_sipper_out= np.mean(no_lick_snips_sipper_out, axis=0) 
            else:
                no_lick_snips_sipper_out =no_lick_snips_sipper_out[0:-1]
                mean_no_lick_snips_sipper_out= np.mean(no_lick_snips_sipper_out, axis=0)
        else:
            no_mean_lick_snips_sipper_out = []

        key = row[2]+"_s"+str(int(row[6]))

        snips_dict_cues[key]={}

        snips_dict_cues[key]["mouse"] = row[2]  
        snips_dict_cues[key]["sex"] = row[3]
        snips_dict_cues[key]["diet"] = row[4]
        snips_dict_cues[key]["session"] = row[6]
        snips_dict_cues[key]["snips"] = snips 
        snips_dict_cues[key]["snips_mean"] = mean_snips
        snips_dict_cues[key]["lick_snips"] = lick_snips 
        snips_dict_cues[key]["lick_snips_mean"] = mean_lick_snips
        snips_dict_cues[key]["lick_snips_sipper_out"] = lick_snips_sipper_out
        snips_dict_cues[key]["lick_snips_sipper_out_mean"] = mean_lick_snips_sipper_out
        snips_dict_cues[key]["no_lick_snips_sipper_out"] = no_lick_snips_sipper_out
        snips_dict_cues[key]["no_lick_snips_sipper_out_mean"] = mean_no_lick_snips_sipper_out
        snips_dict_cues[key]["lick data"] = lickdata
        snips_dict_cues[key]["sipper"] = sipper_onset