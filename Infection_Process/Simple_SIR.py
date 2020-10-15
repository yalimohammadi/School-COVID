import Infection_Process.SIR_Basic as EoN
import matplotlib.pyplot as plt
from Graph_Generator.single_school_generator import School
import numpy as np

from Testing_Strategies import Simple_Random

def plot_simple_SIR(t,S,I,T,R,alpha=1,last=True, teststrat="Fully Random"):


    if last:
        plt.plot(t, I, label="Infected: Random within Cohort", alpha=alpha,color="r")
        #plt.plot(t, T, label="Infected Tested: Random within Cohort", alpha=alpha, color="r")
        #plt.plot(t, S, label="S", alpha=alpha,color="g")
        #plt.plot(t, R, label="R", alpha=alpha,color="b")
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("number of people")
        plt.show()
    else:
        plt.plot(t, I, label="Infected: Fully Random", alpha=alpha, color="b")
        #plt.plot(t, T, label="Infected Tested: Fully Random", alpha=alpha, color="b")
        #plt.plot(t, S, alpha=alpha, color="g")
        #plt.plot(t, R, alpha=alpha, color="b")




def SIR_on_weighted_Graph(G,removal_rate = 1.,transmission_scale=1.,initial_fraction_infected= 0.01,num_sim=1):
    for i in range(num_sim):
        t,S,I,T,R=EoN.fast_SIR(G,gamma=removal_rate, tau=transmission_scale,transmission_weight="weight",
                               rho=initial_fraction_infected, all_test_times = np.linspace(0,2,10),test_args=(100,),test_func=Simple_Random.fully_random_test)
        plot_simple_SIR(t, S, I, T, R,last=False)
        #print(R)
        print(R[len(R)-1])

        t, S, I, T, R = EoN.fast_SIR(G, gamma=removal_rate, tau=transmission_scale, transmission_weight="weight",
                                     rho=initial_fraction_infected, all_test_times=np.linspace(0, 2, 10),
                                     test_args=(school, 100,), test_func=Simple_Random.random_from_cohorts)
        plot_simple_SIR(t, S, I, T, R)
        print(R[len(R) - 1])


    return t,S,I,T,R


school = School(name="LA1",  num_grades=4,cohort_sizes=15,num_cohort=10)
print("school size", school.network.number_of_nodes())
t,S,I,T,R = SIR_on_weighted_Graph(school.network)

# plot_simple_SIR(t,S,I,R)
#print(T)