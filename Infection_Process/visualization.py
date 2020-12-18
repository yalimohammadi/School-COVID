import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




def plot_weekly_community_school_infection(filenames, low_infection_rate=2.2 / 100, total_students=6 * 12 * 25.):
    pc_list = [2 / total_students, 5 / total_students, 10 / total_students]
    intra_cohort_infection_list = [low_infection_rate / 10, low_infection_rate / 5, low_infection_rate]

    p_str = "within school transmission"
    ICI_str = "ICI"
    inboud_str = "daily inbound rate"
    fraction_community_list = [0.001, 0.002, 0.003, 0.004, 0.005]
    new_infected_str = "Total new infected"
    no_test_str = 'no testing'
    weekly_test_str = 'weekly testing'
    biweekly_test_str = 'biweekly testing'
    test_str = "testing strategy"
    new_infected_com_str = "new infected by community"

    data = []
    for filename in filenames:
        temp_data = pickle.load(open(filename, "rb"))
        data.append(temp_data)
    new_data = pd.concat(data)
    new_data["Week"] = new_data.index % 20 + 1

    new_data.columns = [test_str, p_str, ICI_str, inboud_str, new_infected_str, new_infected_com_str, "Week"]

    new_data["new infected in school"] = new_data[new_infected_str] - new_data[new_infected_com_str]

    new_data[p_str] = new_data[p_str].replace([2 / total_students], "low")
    new_data[p_str] = new_data[p_str].replace([5 / total_students], "medium")
    new_data[p_str] = new_data[p_str].replace([10 / total_students], "high")

    new_data[test_str] = new_data[test_str].replace([1.0], weekly_test_str)
    new_data[test_str] = new_data[test_str].replace([.5], biweekly_test_str)
    new_data[test_str] = new_data[test_str].replace([0.0], no_test_str)

    new_data = new_data[new_data[test_str] != "no testing"]
    new_data["school closure community exposure"] = new_data[inboud_str] * (84 / 64) * total_students * 7
    g = sns.FacetGrid(new_data, row=inboud_str, col=p_str, margin_titles=True)

    #     new_data.columns=[test_str, p_str, ICI_str, inboud_str, new_infected_str,"new infected" ,"Week", "new infected in school","sub","school closure community exposure"]

    g.map(sns.lineplot, "Week", "school closure community exposure", palette="pastel", ci=.97, color=".9")
    g.add_legend()
    new_labels = ['school closure community exposure']
    g._legend.set_bbox_to_anchor((1.1, .67))
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)

    g.map(sns.lineplot, "Week", "new infected in school", test_str, palette="bright", ci=.97, color=".9")
    g.add_legend()
    new_labels = ['Biweekly testing - school exposure', 'Weekly testing - school exposure']
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)

    g.map(sns.lineplot, "Week", new_infected_com_str, test_str, palette="pastel", ci=.97, color=".9")
    g.add_legend()
    new_labels = ['Biweekly testing-community exposure', 'Weekly testing - community exposure']
    g._legend.set_bbox_to_anchor((1.1, .57))
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
    g.set(ylabel='new infected')



# uncomment for example
# filenames=[r"/Users/yeganeh/Dec8/new-data/WithMaskWeeklyNewInfected100t.data"]
# plot_weekly_community_school_infection(filenames)


def annotate(data, **kws):
    no_test_str='no testing'
    weekly_test_str='weekly testing'
    biweekly_test_str = 'biweekly testing'
    test_str = "Fraction tested"
    outbreak_str = "Probability of outbreak"

    n = len(data)
    ax = plt.gca()
    tests=[no_test_str,biweekly_test_str,weekly_test_str]
    colors = ["b","orange","g"]
    i=0
    space=0.28
    for t in tests:
        rest_data=data[data[test_str]==t]
        [ax.text(p[0]-0.4+i*space, p[1]*1.1+0.01, p[1], color=colors[i]) for p in zip(ax.get_xticks(), rest_data[outbreak_str])]
        i=i+1