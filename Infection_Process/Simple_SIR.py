import Infection_Process.SIR_Basic as EoN
import matplotlib.pyplot as plt
from Graph_Generator.single_school_generator import School
import numpy as np
import scipy as sp
import networkx as nx

from Testing_Strategies import Simple_Random

def plot_simple_SIR(t,S,E,I,T,R,Isolated, alpha=1,last=True, teststrat="Fully Random"):


    if last:
        plt.plot(t, I, label="Infected: Random with Cohort Isolation", alpha=alpha, color="r")
        #plt.plot(t, T, label="Infected Tested: Random within Cohort", alpha=alpha, color="r")
        #plt.plot(t, S, label="S", alpha=alpha,color="g")
        #plt.plot(t, R, label="R", alpha=alpha,color="b")
        #plt.plot(t,Isolated,label="Isolated vertices: Random with Cohort Isolation", alpha=alpha, color="g")


    else:
        plt.plot(t, I, label="Infected:  Random", alpha=alpha, color="b")
        #plt.plot(t, Isolated, label="Isolated vertices:  Random", alpha=alpha, color="orange")

        #plt.plot(t, T, label="Infected Tested: Fully Random", alpha=alpha, color="b")
        #plt.plot(t, S, alpha=alpha, color="g")
        #plt.plot(t, R, alpha=alpha, color="b")

def print_infected_teachers(status):
    #for t in list(range(0, school.network.number_of_nodes())):
    for t in school.teachers_id:
        print("infected teacher id: ",t)



def SIR_on_weighted_Graph(G,removal_rate = 1.,transmission_scale=1.,initial_fraction_infected= 0.01,num_sim=1) -> object:
    final_infected_FR=[]
    final_infected_RWC=[]
    for i in range(num_sim):
        t,S,E,I,T,R,Isolated,status=EoN.fast_SIR(G,gamma=removal_rate, tau=transmission_scale,transmission_weight="weight",
                               rho=initial_fraction_infected, all_test_times = np.linspace(0,300,300),test_args=(400,),test_func=Simple_Random.fully_random_test,
                                                 weighted_test=False,school=school,isolate=True)
        final_infected_FR.append(R[-1])
        #print_infected_teachers(status)
        #print(I)
        #plot_simple_SIR(t, S, E, I, T, R,Isolated,last=False)
        #print("I= ",I)
        # print("T= ",T)
        # print("Full Random strategy: Total number of infected= ", R[len(R)-1])
        # print(Isolated)
        # print("time",t[-1])
        # final_num_infected_with_cohort_isolation_full_random.append(np.trapz(np.array(Isolated), np.array(t)))
        # print(final_num_infected_with_cohort_isolation_full_random)
        t, S, E, I, T, R,Isolated,status = EoN.fast_SIR(G, gamma=removal_rate, tau=transmission_scale, transmission_weight="weight",
                                     rho=initial_fraction_infected, all_test_times=np.linspace(0, 300, 300),
                                     test_args=(school, 400,), test_func=Simple_Random.random_from_cohorts,weighted_test=False,school=school,isolate=True)
        #print("I= ", I)
        # print("T= ", T)
        # print("Within Cohort Random strategy: Total number of infected= ", R[len(R) - 1])
        # final_num_infected_with_cohort_isolation_random_cohort.append(sp.integrate.simps(Isolated, t))
        final_infected_RWC.append(R[-1])
        #print(I)
        #plot_simple_SIR(t, S, E, I, T, R,Isolated, last=True)
        #
        # t,S,E,I,T,R,Isolated,status=EoN.fast_SIR(G,gamma=removal_rate, tau=transmission_scale,transmission_weight="weight",
        #                        rho=initial_fraction_infected, all_test_times = np.linspace(0,300,300),test_args=(300,),test_func=Simple_Random.fully_random_test,
        #                                          weighted_test=False,school=school,isolate=False)
        # print_infected_teachers(status)
        #
        # print("time",t[-1])
        # final_num_infected_full_random.append(np.trapz(np.array(Isolated), np.array(t)))
        # print(final_num_infected_full_random)
        # plot_simple_SIR(t, S, E, I, T, R,Isolated=Isolated,last=False)
        # t,S,E,I,T,R,Isolated,status=EoN.fast_SIR(G,gamma=removal_rate, tau=transmission_scale,transmission_weight="weight",
        #                        rho=initial_fraction_infected, all_test_times = np.linspace(0,300,300),test_args=(school,300,),test_func=Simple_Random.random_from_cohorts,
        #                                          weighted_test=False,school=school,isolate=False)
        #print_infected_teachers(status)
        # final_num_infected_random_cohort.append(R[-1])
        # plot_simple_SIR(t, S, E, I, T, R,Isolated)
        # plot_simple_SIR(t, S, E, I, T, R)
        # print("T= ", T)
        # print("Weighted testing strategy: Total number of infected= ", R[len(R) - 1])

    return np.mean(final_infected_FR), np.mean(final_infected_RWC)




total_students=1680
num_grades=4
num_of_students_within_grade=int(total_students/num_grades)
p_c=0.01 # [0.05,0.1,,0.2,0.4]
cg_scale=1/10 #5 # [5,10]
p_g=p_c*cg_scale
alpha=0.5
high_infection_rate=low_infection_rate=(1/7)*alpha
scale=1/5
intra_cohort_infection_rate=high_infection_rate*scale
#print(intra_cohort_infection_rate)
#intra_grade_infection_rate=needed (1/7) #there is no intra_grade_infection_rate variable, but intra_grade_infection_rate=intra_cohort_infection_rate in the current implementation
teacher_student_infection_rate=student_teacher_infection_rate=high_infection_rate
infection_rate_between_teachers=high_infection_rate*scale #teachers are similar to cohorts, meaning they have a complete graph.


high_risk_probability=0 #(fixed, irrelevant for now)
initial_fraction_infected= 0.001 #initial fraction infected (fix this)
transmission_scale= 1 #transmission rate per edge (fix this)
removal_rate = 1/10#recovery rate per node (fix this)

final_num_infected_full_random=[]
final_num_infected_random_cohort=[]
final_num_infected_with_cohort_isolation_full_random=[]
final_num_infected_with_cohort_isolation_random_cohort=[]
#number of days to recover =  time to recovery for non-hospitalized cases (mean: 13.1 days, 95% CI: 8.3, 16.9)
#num_days_between_exposed_infection=Weibull distribution with mean 5.4 days (95% CI: 2.4, 8.3)



school_sim=4
#cohort_sizes=14
cohort_size_list=[10,12,14]
for cohort_sizes in cohort_size_list: #p_c in [.05]:
    #p_g = p_c * cg_scale
    num_cohort=int(num_of_students_within_grade/cohort_sizes)
    num_teachers = 2*num_cohort
    #school = School(name="LA1",  num_grades,cohort_sizes,num_cohort,num_teachers)
    to_plot1 = []
    to_plot2 = []
    for i in range(school_sim):
        school = School("LA1", num_grades,cohort_sizes,num_cohort,num_teachers,p_c,p_g,high_risk_probability,high_infection_rate,low_infection_rate,intra_cohort_infection_rate,teacher_student_infection_rate,student_teacher_infection_rate,infection_rate_between_teachers)
        print("School Size=", school.network.number_of_nodes())
        #plt.subplot(121)
        #nx.draw(school.network, with_labels=True, font_weight='bold')
        #plt.show()
        avg1, avg2 = SIR_on_weighted_Graph(school.network,removal_rate= removal_rate,transmission_scale=transmission_scale,initial_fraction_infected= initial_fraction_infected,num_sim=5)
        to_plot1.append(avg1/school.network.number_of_nodes())
        to_plot2.append(avg2/ school.network.number_of_nodes())
    # plt.plot()
    final_num_infected_with_cohort_isolation_full_random.append(np.mean(to_plot1))
    final_num_infected_with_cohort_isolation_random_cohort.append(np.mean(to_plot2))

print(final_num_infected_with_cohort_isolation_full_random)
print(final_num_infected_with_cohort_isolation_random_cohort)
plt.plot(cohort_size_list,final_num_infected_with_cohort_isolation_full_random,label="Full Randaom", alpha=1, color= "blue")
plt.plot(cohort_size_list,final_num_infected_with_cohort_isolation_random_cohort,label="Randaom within cohort", alpha=1, color= "red")
plt.show()

# plot_simple_SIR(t,S,I,R)
#print(T)
# print(to_plot)
# to_plot=[0.8024757281553397, 0.9846116504854369, 0.997233009708738, 0.9999514563106796]
# plt.plot([0.01,0.05,0.1,0.2],to_plot,label="")
# plt.legend()
# plt.xlabel("Across Cohort Interaction Probability")
# plt.ylabel("Fraction of Infected Students")
# plt.show()
#
# print(final_num_infected_full_random)
# print(final_num_infected_with_cohort_isolation_full_random)
# print(final_num_infected_random_cohort)
# print(final_num_infected_with_cohort_isolation_random_cohort)
# print(np.mean(final_num_infected_full_random))
# print(np.mean(final_num_infected_with_cohort_isolation_full_random))
# print(np.mean(final_num_infected_random_cohort))
# print(np.mean(final_num_infected_with_cohort_isolation_random_cohort))
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






