import EoN
import matplotlib.pyplot as plt
from Graph_Generator.single_school_generator import School

def plot_simple_SIR(t,S,I,R,alpha=.1,last=True):
    plt.plot(t, I,alpha=alpha,color="r")
    plt.plot(t,S,alpha=alpha,color="g")
    plt.plot(t,R,alpha=alpha,color="b")
    if last:
        plt.plot(t, I, label="I", alpha=alpha,color="r")
        plt.plot(t, S, label="S", alpha=alpha,color="g")
        plt.plot(t, R, label="R", alpha=alpha,color="b")
        plt.legend()
        plt.xlabel("time")
        plt.show()


def SIR_on_weighted_Graph(G,removal_rate = 1.,transmission_scale=1.,initial_fraction_infected= 0.01,num_sim=10):
    for i in range(num_sim-1):
        t,S,I,R=EoN.fast_SIR(G,gamma=removal_rate, tau=transmission_scale,transmission_weight="weight",rho=initial_fraction_infected)
        plot_simple_SIR(t, S, I, R,last=False)

    t, S, I, R = EoN.fast_SIR(G, gamma=removal_rate, tau=transmission_scale, transmission_weight="weight",
                              rho=initial_fraction_infected)
    plot_simple_SIR(t, S, I, R)
    return t,S,I,R


school = School(name="LA1",  num_grades=4,cohort_sizes=15,num_cohort=10)
t,S,I,R = SIR_on_weighted_Graph(school.network)
# plot_simple_SIR(t,S,I,R)