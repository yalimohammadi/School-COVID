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




total_students=2000
num_grades=4
num_teachers=60
num_of_students_within_grade=int(total_students/num_grades)
p_c=0.1 # [0.05,0.1,,0.2,0.4]
cg_scale=5 # [5,10]
p_g=p_c/cg_scale
high_infection_rate=low_infection_rate=5/7
intra_cohort_infection_rate=1/7
#intra_grade_infection_rate=needed (1/7) #there is no intra_grade_infection_rate variable, but intra_grade_infection_rate=intra_cohort_infection_rate in the current implementation
teacher_student_infection_rate=student_teacher_infection_rate=1/7
infection_rate_between_teachers=1/7 #teachers are similar to cohorts, meaning they have a complete graph.


high_risk_probability=0 #(fixed, irrelevant for now)
initial_fraction_infected= 0.01 #initial fraction infected (fix this)
transmission_scale= 1 #transmission rate per edge (fix this)
removal_rate= 1 #recovery rate per node (fix this)

#number of days to recover =  time to recovery for non-hospitalized cases (mean: 13.1 days, 95% CI: 8.3, 16.9)
#num_days_between_exposed_infection=Weibull distribution with mean 5.4 days (95% CI: 2.4, 8.3)

for cohort_sizes in [10]: #[10,15,20]
    num_cohort=int(num_of_students_within_grade/cohort_sizes)
    #school = School(name="LA1",  num_grades,cohort_sizes,num_cohort,num_teachers)
    school = School("LA1", num_grades,cohort_sizes,num_cohort,num_teachers,p_c,p_g,high_risk_probability,high_infection_rate,low_infection_rate,intra_cohort_infection_rate,teacher_student_infection_rate,student_teacher_infection_rate,infection_rate_between_teachers)
    print("school size", school.network.number_of_nodes())
    t,S,I,T,R = SIR_on_weighted_Graph(school.network,removal_rate= removal_rate,transmission_scale=transmission_scale,initial_fraction_infected= initial_fraction_infected,num_sim=1)

# plot_simple_SIR(t,S,I,R)
#print(T)