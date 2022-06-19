# %%
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import trompy as tp

import matplotlib.transforms as transforms

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
    tp.shadedError(ax6, session_data["snips_sip"], linecolor="grey")
    tp.shadedError(ax6, session_data["snips_sip_with_licks"], linecolor="orange")

    ax8 = f.add_subplot(gs1[3,1])
    tp.shadedError(ax8, session_data["snips_licks"], linecolor="blue")

    f.savefig("..\\output\\{}_{}.pdf".format(session_data["mouse"], session_data["session"]))

def session_figs(f, gs1, session_data):
    
    ax1 = f.add_subplot(gs1[0, 0])
    ax2 = f.add_subplot(gs1[0, 1])
    ax3 = f.add_subplot(gs1[1, 0])

    ax1.plot(session_data["blue"], color="blue")
    ax1.plot(session_data["uv"], color="magenta")

    ax2.plot(session_data["corrected_signal"])

    ax3.plot(session_data["corrected_signal"])

    licks_in_samples = [int(event*session_data["fs"]) for event in session_data["licks"]]
    sipper_in_samples = [int(event*session_data["fs"]) for event in session_data["sipper"]]

    # place markers on top
    trans = transforms.blended_transform_factory(ax3.transData, ax3.transAxes)
    ax3.vlines(licks_in_samples, 0.9, 0.95, transform=trans)
    ax3.vlines(sipper_in_samples, 0.95, 1.0, transform=trans)

    ax3.set_xlim([1e6,1.2e6])

def lick_fig(f, gs1, session_data):

    inds = np.argsort(session_data["latency"])
    licks_per_trial_sorted = [session_data["licks_per_trial"][i] for i in inds]

    ax4 = f.add_subplot(gs1[2, 0])
    ax5 = f.add_subplot(gs1[3, 0])

    for idx, licks_in_trial in enumerate(licks_per_trial_sorted):
        ax4.vlines(licks_in_trial, idx, idx+1)

    ax4.set_xlim([-10, 30])
    ax4.set_ylim([0, 40])
    ax4.invert_yaxis()

    # 
    # licks_hist = np.histogram(tp.flatten_list(session_data["licks_per_trial"]), bins=bins)
    # ax5.plot(licks_hist[0])
    
    bins=np.arange(-10, 30, 0.1)
    sns.histplot(data=tp.flatten_list(session_data["licks_per_trial"]),
                kde = True, bins=bins, ax=ax5)

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
