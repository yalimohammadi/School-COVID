import Infection_Process.SIR_Basic as EoN
import matplotlib.pyplot as plt
from Graph_Generator.single_school_generator import School
import numpy as np
import networkx as nx

from Testing_Strategies import Simple_Random

def plot_simple_SIR(t,S,E,I,T,R,alpha=1,last=True, teststrat="Fully Random"):


    if last:
        plt.plot(t, I, label="Infected: Random within Cohort", alpha=alpha,color="r")
        #plt.plot(t, T, label="Infected Tested: Random within Cohort", alpha=alpha, color="r")
        plt.plot(t, S, label="S", alpha=alpha,color="g")
        plt.plot(t, R, label="R", alpha=alpha,color="b")

        plt.legend()
        plt.xlabel("time")
        plt.ylabel("number of people")
        plt.show()
    else:
        plt.plot(t, I, label="Infected: Fully Random", alpha=alpha, color="b")
        #plt.plot(t, T, label="Infected Tested: Fully Random", alpha=alpha, color="b")
        #plt.plot(t, S, alpha=alpha, color="g")
        #plt.plot(t, R, alpha=alpha, color="b")

def print_infected_teachers(status):
    #for t in list(range(0, school.network.number_of_nodes())):
    for t in school.teachers_id:
        print("infected teacher id: ",t)



def SIR_on_weighted_Graph(G,removal_rate = 1.,transmission_scale=1.,initial_fraction_infected= 0.01,num_sim=1) -> object:
    for i in range(num_sim):
        t,S,E,I,T,R,status=EoN.fast_SIR(G,gamma=removal_rate, tau=transmission_scale,transmission_weight="weight",
                               rho=initial_fraction_infected, all_test_times = np.linspace(0,119,120),test_args=(100,),test_func=Simple_Random.fully_random_test,weighted_test=False)
        print_infected_teachers(status)
        #plot_simple_SIR(t, S, E, I, T, R,last=False)
        #print("I= ",I)
        #print("T= ",T)
        print("Full Random strategy: Total number of infected= ", R[len(R)-1])

        t, S, E, I, T, R,status = EoN.fast_SIR(G, gamma=removal_rate, tau=transmission_scale, transmission_weight="weight",
                                     rho=initial_fraction_infected, all_test_times=np.linspace(0, 119, 120),
                                     test_args=(school, 100,), test_func=Simple_Random.random_from_cohorts,weighted_test=False)
        #print("I= ", I)
        #print("T= ", T)
        plot_simple_SIR(t, S, E, I, T, R)
        print("Within Cohort Random strategy: Total number of infected= ", R[len(R) - 1])
        #
        #t, S, E, I, T, R = EoN.fast_SIR(G, gamma=removal_rate, tau=transmission_scale, transmission_weight="weight",
        #                             rho=initial_fraction_infected, all_test_times=np.linspace(0, 119, 120),
        #                             test_args=(school, 100,),weighted_test=True)
        #plot_simple_SIR(t, S, E, I, T, R)
        #print("Weighted testing strategy: Total number of infected= ", R[len(R) - 1])

    return t,S,E,I,T,R




total_students=2000
num_grades=4
num_teachers=60
num_of_students_within_grade=int(total_students/num_grades)
p_c=0.1 # [0.05,0.1,,0.2,0.4]
cg_scale=10 #5 # [5,10]
p_g=p_c*cg_scale
alpha=0.5
high_infection_rate=low_infection_rate=(5/7)*alpha
scale=1/5
intra_cohort_infection_rate=high_infection_rate*scale
#print(intra_cohort_infection_rate)
#intra_grade_infection_rate=needed (1/7) #there is no intra_grade_infection_rate variable, but intra_grade_infection_rate=intra_cohort_infection_rate in the current implementation
teacher_student_infection_rate=student_teacher_infection_rate=high_infection_rate
infection_rate_between_teachers=high_infection_rate*scale #teachers are similar to cohorts, meaning they have a complete graph.


high_risk_probability=0 #(fixed, irrelevant for now)
initial_fraction_infected= 0.001 #initial fraction infected (fix this)
transmission_scale= 1 #transmission rate per edge (fix this)
removal_rate = 1/12 #recovery rate per node (fix this)

#number of days to recover =  time to recovery for non-hospitalized cases (mean: 13.1 days, 95% CI: 8.3, 16.9)
#num_days_between_exposed_infection=Weibull distribution with mean 5.4 days (95% CI: 2.4, 8.3)

for cohort_sizes in [10]: #[10,15,20]
    num_cohort=int(num_of_students_within_grade/cohort_sizes)
    #school = School(name="LA1",  num_grades,cohort_sizes,num_cohort,num_teachers)
    school = School("LA1", num_grades,cohort_sizes,num_cohort,num_teachers,p_c,p_g,high_risk_probability,high_infection_rate,low_infection_rate,intra_cohort_infection_rate,teacher_student_infection_rate,student_teacher_infection_rate,infection_rate_between_teachers)
    print("School Size=", school.network.number_of_nodes())
    #plt.subplot(121)
    #nx.draw(school.network, with_labels=True, font_weight='bold')
    #plt.show()
    t,S,E,I,T,R = SIR_on_weighted_Graph(school.network,removal_rate= removal_rate,transmission_scale=transmission_scale,initial_fraction_infected= initial_fraction_infected,num_sim=1)
    # plt.plot()
# plot_simple_SIR(t,S,I,R)
#print(T)




#S I T

#p

#We want people within same cohort to get infected in rougly 1-3

# Suspectible -- Exposed ----- Infected state-----Tested ------- Recovered

#Suspectible ---- exposed , using contact rates.

#scalar=1

#within same cohort 1-3 exponential (5/7)
#within same grade different cohort 3-7 exp(1/10)
#different grade exp(1/35)



#infected---recover time to recovery for non-hospitalized cases (mean: 13.1 days, 95% CI: 8.3, 16.9)



#exposed---infected Weibull distribution with mean 5.4 days (95% CI: 2.4, 8.3)




#1) Testing strategy: test people from infected cohort with high probability
#change weights with respect to testing


#2) Special testing for teachers. (later, sensitive to parameter values)

#3) Plots for teachers and students separate.

#4) Plots for number of cohorts infected.

#5) Tranmissions to understand T-S or S-T (S-T,T-S)


#COVID spreads first within the cohort and then goes to other cohort.
#quarantine same cohort
#test friends oustide cohort (contact tracing)


#Baseline: FR vs RWC, RWC is better with your additional testing strategy attached.

#Baseline: which is better: frequent tests with less people(40 per day) vs non frequent tests with more people (250 per 5 days)






