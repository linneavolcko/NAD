# %%

import sys
from matplotlib.markers import MarkerStyle
import numpy as np
import matplotlib.pyplot as plt

import trompy as tp

# from ..\\source.NAD_fig_fx import make_heatmap

# %% Figure settings
%run NAD_fig_fx.py

figsfolder = "C:\\Users\\jmc010\\UiT Office 365\\O365-Pipette 2 - Documents\\People\\Linnea\\Conferences\\2022 SSIB\\poster\\figures\\"

nr_color="grey"   # "#6D6D6D"
nr_errorcolor="#EAEAEA"
pr_color="#00C7CC"  # "#0097A3"
pr_errorcolor="#CCF3F4"

plt.rcParams['font.size'] = '24'
plt.rcParams["lines.linewidth"] = 2


# %%

datafolder = "D:\\Test Data\\photometry\\NAc GRAB photometry\\"

import dill
with open(datafolder+"NAD_session_data_no_baseline_removed", 'rb') as pickle_in:
    all_session_data = dill.load(pickle_in)
# %%
mice = []

for key, session_data in all_session_data.items():
    if session_data["mouse"] not in mice:
        mice.append(session_data["mouse"])

# %%
# mice to exclude
# mice.remove("NAD28") # not much licking or DA release, some responses to licks but few trials
mice.remove("NAD25") # only 2 sessions with 1-2 trials of licking
mice.remove("NAD30") # only 1 session with 1 trial of licking


# %%
avg_data = {}

for mouse in mice:
    avg_data[mouse] = {}
    avg_data[mouse]["diet"] = []
    avg_data[mouse]["all_snips_sip"] = []
    avg_data[mouse]["all_snips_licks"] = []
    avg_data[mouse]["all_snips_sip_with_licks"] = []
    avg_data[mouse]["all_licks_per_trial"] = []

    for key, session_data in all_session_data.items():
        if session_data["mouse"] == mouse:
            if (session_data["diet"] == "NR") or (session_data["diet"] == "PR"):
                avg_data[mouse]["diet"].append(session_data["diet"])
                avg_data[mouse]["all_snips_sip"].append(session_data["snips_sip"])
                avg_data[mouse]["all_snips_licks"].append(session_data["snips_licks"])
                avg_data[mouse]["all_snips_sip_with_licks"].append(session_data["snips_sip_with_licks"])
                avg_data[mouse]["all_licks_per_trial"].append(session_data["licks_per_trial"])

print("Size after adding initial snips:", sys.getsizeof(avg_data))
# %%

for mouse in avg_data.keys():
    d = avg_data[mouse]
    for key, key_cat, key_mean in zip(["all_snips_sip", "all_snips_licks", "all_snips_sip_with_licks"],
                                    ["cat_snips_sip", "cat_snips_licks", "cat_snips_sip_with_licks"],
                                    ["mean_snips_sip", "mean_snips_licks", "mean_snips_sip_with_licks"]):
        d[key_cat] = np.vstack(d[key])
        d[key_mean] = np.mean(d[key_cat], axis=0)

print("Size after adding extra snips:", sys.getsizeof(avg_data))

# %%
import matplotlib.pyplot as plt

f, ax = plt.subplots()
for mouse in avg_data.keys():
    d = avg_data[mouse]
    ax.plot(d["mean_snips_sip"], label=mouse)

ax.axvline(80)
ax.legend()




# %%
d = avg_data["NAD33"]

def calculate_auc(dictionary, snips_key, epoch=[100,150], binsize=0.1):
    """
    Calculates AUC from 1-dimensional data stored as a numpy array in a dictionary.
    Future versions may include more data input options (e.g. 2D data.
    """

    data = dictionary[snips_key]
    time_in_seconds = (epoch[1] - epoch[0]) * binsize
    return np.trapz(data[epoch]) / time_in_seconds

key = "mean_snips_licks"

calculate_auc(d, key, epoch=[60,80])
calculate_auc(d, key, epoch=[80,100])
calculate_auc(d, key, epoch=[100,120])

# %%
key = "mean_snips_sip"

nr_auc_pre, nr_auc_cue, nr_auc_sip = [], [], []
pr_auc_pre, pr_auc_cue, pr_auc_sip = [], [], []

for mouse in avg_data.keys():
    d = avg_data[mouse]
    if d["diet"][0] == "NR":
        print(mouse, d["diet"])
        nr_auc_pre.append(calculate_auc(d, key, epoch=[60,80]))
        nr_auc_cue.append(calculate_auc(d, key, epoch=[80,100]))
        nr_auc_sip.append(calculate_auc(d, key, epoch=[100,120]))
    elif d["diet"][0] == "PR":
        print(mouse, d["diet"])
        pr_auc_pre.append(calculate_auc(d, key, epoch=[60,80]))
        pr_auc_cue.append(calculate_auc(d, key, epoch=[80,100]))
        pr_auc_sip.append(calculate_auc(d, key, epoch=[100,120]))


# %%
plt.rcParams['font.size'] = '24'
plt.rcParams["lines.linewidth"] = 2

f, ax = plt.subplots(figsize=(7, 4), gridspec_kw={"left": 0.25})
_ = tp.barscatter([[nr_auc_pre, nr_auc_cue, nr_auc_sip], [pr_auc_pre, pr_auc_cue, pr_auc_sip]],
                    paired=True,
                    barfacecolor_option="individual",
                    barfacecolor=[nr_color, nr_color, nr_color, pr_color, pr_color, pr_color],
                    scatteredgecolor=["grey"], scattersize=100,
                    linewidth=1.5, sc_kwargs={"alpha": 0.5},
                    ax=ax)

ax.set_ylabel("AUC")
f.savefig(figsfolder+"auc_sip.pdf")
# %%
key = "mean_snips_licks"

nr_auc_prelick, nr_auc_licks = [], []
pr_auc_prelick, pr_auc_licks = [], []

for mouse in avg_data.keys():
    d = avg_data[mouse]
    if d["diet"][0] == "NR":
        print(mouse, d["diet"])
        nr_auc_prelick.append(calculate_auc(d, key, epoch=[50,100]))
        nr_auc_licks.append(calculate_auc(d, key, epoch=[100,150]))
    elif d["diet"][0] == "PR":
        print(mouse, d["diet"])
        pr_auc_prelick.append(calculate_auc(d, key, epoch=[50,100]))
        pr_auc_licks.append(calculate_auc(d, key, epoch=[100,150]))

# %%
f, ax = plt.subplots(figsize=(5.5, 4), gridspec_kw={"left": 0.25})
_ = tp.barscatter([[nr_auc_prelick, nr_auc_licks], [pr_auc_prelick, pr_auc_licks]],
                    paired=True,
                    barfacecolor_option="individual",
                    barfacecolor=[nr_color, nr_color, pr_color, pr_color],
                    scatteredgecolor=["grey"], scattersize=100,
                    linewidth=1.5, sc_kwargs={"alpha": 0.5},
                    ax=ax)
ax.set_ylabel("AUC")
f.savefig(figsfolder+"auc_licks.pdf")

# %%
key = "mean_snips_licks"
nr_trace_licks = []
pr_trace_licks = []

for mouse in avg_data.keys():
    d = avg_data[mouse]
    if d["diet"][0] == "NR":
        nr_trace_licks.append(d[key])
    elif d["diet"][0] == "PR":
        pr_trace_licks.append(d[key])

# %%   

f, ax = plt.subplots(figsize=(7.46, 4))
tp.shadedError(ax, nr_trace_licks, linecolor=nr_color, errorcolor=nr_errorcolor, linewidth=2)
tp.shadedError(ax, pr_trace_licks, linecolor=pr_color, errorcolor=pr_errorcolor, linewidth=2)

ax.axis('off')

y = [y for y in ax.get_yticks() if y>0][:2]
l = y[1] - y[0]

scale_label = '{0:.0f} Z'.format(l)

ax.plot([50,50], [y[0], y[1]], c="k")
ax.text(40, y[0]+(l/2), scale_label, va='center', ha='right')

# Adds x scale bar   
y = ax.get_ylim()[0]
ax.plot([351,400], [y, y], c="k", linewidth=2)
ax.annotate('5 s', xy=(376,y), xycoords='data',
            xytext=(0,-5), textcoords='offset points',
            ha='center',va='top')

f.savefig(figsfolder + "nr_trace_licks.pdf")
# ax.plot(d["mean_snips_sip"], label=mouse)

# ax.axvline(80)
# ax.legend()
# %%
key = "mean_snips_sip"
nr_trace_sip = []
pr_trace_sip = []

for mouse in avg_data.keys():
    d = avg_data[mouse]
    if d["diet"][0] == "NR":
        nr_trace_sip.append(d[key])
    elif d["diet"][0] == "PR":
        pr_trace_sip.append(d[key])
# %%

f, ax = plt.subplots(figsize=(7.46, 4))
tp.shadedError(ax, nr_trace_sip, linecolor=nr_color, errorcolor=nr_errorcolor, linewidth=2)
tp.shadedError(ax, pr_trace_sip, linecolor=pr_color, errorcolor=pr_errorcolor, linewidth=2)

ax.axis('off')

y = [y for y in ax.get_yticks() if y>0][:2]
l = y[1] - y[0]

scale_label = '{0:.1f} Z'.format(l)

ax.plot([50,50], [y[0], y[1]], c="k")
ax.text(40, y[0]+(l/2), scale_label, va='center', ha='right')

# Adds x scale bar   
y = ax.get_ylim()[0]
ax.plot([351,400], [y, y], c="k", linewidth=2)
ax.annotate('5 s', xy=(376,y), xycoords='data',
            xytext=(0,-5), textcoords='offset points',
            ha='center',va='top')

f.savefig(figsfolder + "nr_trace_sip.pdf")

# %%
# make_heatmap(nr_trace_sip)



f, ax = plt.subplots(nrows=2)

data = nr_trace_sip
(ntrials, bins) = np.shape(data)

xvals = np.linspace(-10,30,bins)
yvals = np.arange(0, ntrials)
xx, yy = np.meshgrid(xvals, yvals)

mesh = ax[0].pcolormesh(xx, yy, data, cmap="jet", shading="auto")

# %%
f, ax = plt.subplots(figsize=(8,5), ncols=2, gridspec_kw={"width_ratios": [14,1]})

data = nr_trace_sip + pr_trace_sip
clims = [-0.14, 1.2]
(ntrials, bins) = np.shape(data)

xvals = np.linspace(-10,30,bins)
yvals = np.arange(0, ntrials)
xx, yy = np.meshgrid(xvals, yvals)

mesh = ax[0].pcolormesh(xx, yy, data, cmap="jet", shading="auto")
mesh.set_clim(clims)

ax[0].invert_yaxis()
ax[0].axhline(4.5, color="white", linewidth=3)
ax[0].axvline(-2, color="white", linestyle="--")
ax[0].axvline(0, color="white", linestyle="--")

ax[0].spines['top'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_yticks([])
ax[0].set_xticks([])

xscalebar=True
if xscalebar:
    ax[0].plot([25, 29.9], [ntrials, ntrials], linewidth=2, color='k', clip_on=False)
    ax[0].text(27.5, ntrials+0.2, "5 s", va="top", ha="center", fontsize=24)

cbar = f.colorbar(mesh, cax=ax[1], ticks=[-1, 0, 1])

# cbar_labels = ['{0:.0f}'.format(clims[0]),
#                 '0 Z',
#                 '{0:.0f}'.format(clims[1])]
# cbar.ax.set_yticklabels(cbar_labels)

    # ax.set_ylabel(ylabel, rotation=90, labelpad=2)

    # ax.set_yticks([])
    # ax.set_xticks([])
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

f.savefig(figsfolder+"heatmap_sip.pdf")
# %%



f, ax = plt.subplots(figsize=(8,5), ncols=2, gridspec_kw={"width_ratios": [14,1]})

data = nr_trace_licks + pr_trace_licks
clims = [-0.66, 3.92]
(ntrials, bins) = np.shape(data)

xvals = np.linspace(-10,30,bins)
yvals = np.arange(0, ntrials)
xx, yy = np.meshgrid(xvals, yvals)

mesh = ax[0].pcolormesh(xx, yy, data, cmap="jet", shading="auto")
# mesh.set_clim(clims)

ax[0].invert_yaxis()
ax[0].axhline(4.5, color="white", linewidth=3)
ax[0].axvline(0, color="white", linestyle="--")
ax[0].axvline(5, color="white", linestyle="--")

ax[0].spines['top'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_yticks([])
ax[0].set_xticks([])

xscalebar=True
if xscalebar:
    ax[0].plot([25, 29.9], [ntrials, ntrials], linewidth=2, color='k', clip_on=False)
    ax[0].text(27.5, ntrials+0.2, "5 s", va="top", ha="center", fontsize=24)

cbar = f.colorbar(mesh, cax=ax[1], shrink=0.7)

# ticks=[clims[0], 0, clims[1]]
    

f.savefig(figsfolder+"heatmap_licks.pdf")

# %%
d = avg_data["NAD31"]
f, ax = plt.subplots()
ax.plot(d["mean_snips_sip"])
# %%




# nr_sip_byday, nr_sip_byday_sem = [], []
# pr_sip_byday, pr_sip_byday_sem = [], [] 

# nr_cue_byday, nr_sip_byday, nr_licks_perday = [], [], []
# pr_cue_byday, pr_sip_byday, pr_licks_perday = [], [], []

nr_cue_day1, nr_cue_day2, nr_cue_day3, nr_cue_day4, nr_cue_day5 = [], [], [], [], []
nr_sip_day1, nr_sip_day2, nr_sip_day3, nr_sip_day4, nr_sip_day5 = [], [], [], [], []

nr_licks_day1, nr_licks_day2, nr_licks_day3, nr_licks_day4, nr_licks_day5 = [], [], [], [], []

pr_cue_day1, pr_cue_day2, pr_cue_day3, pr_cue_day4, pr_cue_day5 =  [], [], [], [], []
pr_sip_day1, pr_sip_day2, pr_sip_day3, pr_sip_day4, pr_sip_day5 = [], [], [], [], []

pr_licks_day1, pr_licks_day2, pr_licks_day3, pr_licks_day4, pr_licks_day5 = [], [], [], [], []


for session, cue_nr, sip_nr, licks_nr, cue_pr, sip_pr, licks_pr in zip([str(x) for x in [1.0, 2.0, 3.0, 4.0, 5.0]],
                                    [nr_cue_day1, nr_cue_day2, nr_cue_day3, nr_cue_day4, nr_cue_day5],
                                    [nr_sip_day1, nr_sip_day2, nr_sip_day3, nr_sip_day4, nr_sip_day5],
                                    [nr_licks_day1, nr_licks_day2, nr_licks_day3, nr_licks_day4, nr_licks_day5],
                                    [pr_cue_day1, pr_cue_day2, pr_cue_day3, pr_cue_day4, pr_cue_day5],
                                    [pr_sip_day1, pr_sip_day2, pr_sip_day3, pr_sip_day4, pr_sip_day5],
                                    [pr_licks_day1, pr_licks_day2, pr_licks_day3, pr_licks_day4, pr_licks_day5]):
    for key in all_session_data.keys():
        if session in key:
            d = all_session_data[key]
            d["snips_sip_mean"] =  np.mean(d["snips_sip"], axis=0)
            d["snips_licks_mean"] =  np.mean(d["snips_licks"], axis=0)
            if d["diet"] == "NR":
                cue_nr.append(calculate_auc(d, "snips_sip_mean", epoch=[80,100]))
                sip_nr.append(calculate_auc(d, "snips_sip_mean", epoch=[100,120]))
                licks_nr.append(calculate_auc(d, "snips_licks_mean", epoch=[100,150]))
            elif d["diet"] == "PR":
                cue_pr.append(calculate_auc(d, "snips_sip_mean", epoch=[80,100]))
                sip_pr.append(calculate_auc(d, "snips_sip_mean", epoch=[100,120]))
                licks_pr.append(calculate_auc(d, "snips_licks_mean", epoch=[100,150]))

                print(key, d["diet"])

# %%
def day_comparison_fig(nr_data, pr_data, title="", ylabel="", linewidth=2, ax=[]):

    nr_mean, nr_sem = [], []
    for day in nr_data:
        x, sem = tp.mean_and_sem(day)
        nr_mean.append(x)
        nr_sem.append(sem)

    pr_mean, pr_sem = [], []
    for day in pr_data:
        x, sem = tp.mean_and_sem(day)
        pr_mean.append(x)
        pr_sem.append(sem)

    days = [1,2,3,4,5]

    if ax == []:
        f, ax = plt.subplots()

    plt.rcParams["lines.linewidth"] = linewidth
    ax.errorbar(days, nr_mean, marker="o", markersize=15, yerr=nr_sem, color=nr_color, mec=nr_color, mfc="white", mew=linewidth)
    ax.errorbar(days, pr_mean, marker="o", markersize=15, yerr=pr_sem, color=pr_color, mec=pr_color, mfc="white", mew=linewidth)

    ax.set_xticks(days)
    ax.set_xlabel("session")
    ax.set_ylabel(ylabel)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(title)

    return ax

f, ax = plt.subplots(ncols=3, figsize=(16, 4),
                     gridspec_kw={"left": 0.1, "right": 0.9, "bottom": 0.25, "wspace": 0.35})

_ = day_comparison_fig([nr_cue_day1, nr_cue_day2, nr_cue_day3, nr_cue_day4, nr_cue_day5], [pr_cue_day1, pr_cue_day2, pr_cue_day3, pr_cue_day4, pr_cue_day5], title="cue", ylabel="AUC", ax=ax[0])
ax[0].set_yticks([-.2, -.1, 0, .1])
_ = day_comparison_fig([nr_sip_day1, nr_sip_day2, nr_sip_day3, nr_sip_day4, nr_sip_day5], [pr_sip_day1, pr_sip_day2, pr_sip_day3, pr_sip_day4, pr_sip_day5], title="sipper", ax=ax[1])
ax[1].set_yticks([-.2, -.1, 0, .1])
ax[0].set_ylim(ax[1].get_ylim())

_ = day_comparison_fig([nr_licks_day1, nr_licks_day2, nr_licks_day3, nr_licks_day4, nr_licks_day5], [pr_licks_day1, pr_licks_day2, pr_licks_day3, pr_licks_day4, pr_licks_day5], title="licks", ax=ax[2])
ax[2].set_yticks([0, .2, .4, .6])

f.savefig(figsfolder+"auc_by_day.pdf")

# %%
# for mouse in avg_data.keys():
# d = avg_data[mouse]
d = avg_data["NAD31"]
if d["diet"][0] == "NR":
    for cue, sip, trace in zip([nr_cue_day1, nr_cue_day2, nr_cue_day3, nr_cue_day4, nr_cue_day5],
                                [nr_sip_day1, nr_sip_day2, nr_sip_day3, nr_sip_day4, nr_sip_day5],
                                d["all_snips_sip"]):
        avg_for_day = np.mean(trace, axis=0)
        cue.append()
print(key)
print(np.shape(item["all_snips_sip"]))
    
# %%
NR_licks = []
PR_licks = []

for mouse in avg_data.keys():
    d = avg_data[mouse]
    licks = tp.flatten_list(tp.flatten_list(d["all_licks_per_trial"]))
    if d["diet"][0] == "NR":
        NR_licks.append(licks)
    elif d["diet"][0] == "PR":
        PR_licks.append(licks)

bins=np.arange(-10, 30, 0.1)

f, ax = plt.subplots(figsize=(7.46, 1.5))
sns.kdeplot(data=tp.flatten_list(NR_licks), color=nr_color)
sns.kdeplot(data=tp.flatten_list(PR_licks), color=pr_color)
ax.set_xlim([-10, 30])

ax.axis('off')

f.savefig(figsfolder+"licking_kde.pdf")

# %%
