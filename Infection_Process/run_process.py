import Infection_Process.SIR_Basic as EoN
from Graph_Generator.single_school_generator import School
import numpy as np
import scipy as sp
import networkx as nx
import pandas as pd

from Testing_Strategies import Simple_Random




def SIR_on_weighted_Graph(G,school,number_of_tests=0,fraction_infected_at_each_time_step_from_community=0,removal_rate = 1.,transmission_scale=1.,initial_fraction_infected= 0.01,num_sim=1,tmax=60) -> object:
    final_infected_FR=[]
    within_school_final_infected_FR = []
    # final_infected_RWC=[0]
    outbreak= .10*G.number_of_nodes()
    #print(outbreak)
    num_outbreak_FR=0
    # num_outbreak_RWC=0
    for i in range(num_sim):
        t,S,E,I,T,R,Isolated,status,total_infections_from_community=EoN.fast_SIR(G,gamma=removal_rate, tau=transmission_scale,transmission_weight="weight",
                               rho=initial_fraction_infected, all_test_times = np.linspace(0,tmax,tmax+1), fraction_of_infections_from_community_per_day=fraction_infected_at_each_time_step_from_community, test_args=(number_of_tests,),test_func=Simple_Random.fully_random_test,
                                                 weighted_test=False,school=school,isolate=True,tmax=tmax)
        final_infected_FR.append(R[-1]+I[-1]) # Since the process has not ended, we need to add I[-1]
        within_school_final_infected_FR.append(R[-1] + I[-1] - total_infections_from_community) #-total_infections_from_community-initial_fraction_infected*school.network.number_of_nodes()
        if R[-1]>outbreak:
            num_outbreak_FR+=1

    return final_infected_FR, within_school_final_infected_FR, num_outbreak_FR/num_sim



school_sim=1
num_sim= 200
total_students= 6*12*25 #2000
num_grades = 6  # its either 3 or 6
num_of_students_within_grade = int(total_students/num_grades)
cohort_sizes=12
#cohort_size_list=[8,10,12,14,16]
num_cohort=int(num_of_students_within_grade/cohort_sizes)
num_teachers = num_cohort*num_grades

school_size=total_students+num_teachers

print('School Size = ', total_students+num_teachers)









#We are not using these parameters, because I commented out the bus code
capacity_of_bus=25
num_of_cohorts_per_bus=2
bus_interaction_rate=1/10

high_risk_probability=0 #(fixed, irrelevant for now)
transmission_scale= 1 #transmission rate per edge (fix this)
removal_rate = 1/10#recovery rate per node (fix this)

final_num_infected_full_random=[]
final_num_infected_random_cohort=[]
final_num_infected_with_cohort_isolation_full_random= []
final_num_infected_with_cohort_isolation_random_cohort= []
final_num_outbreak_with_cohort_isolation_full_random= []
final_num_outbreak_with_cohort_isolation_random_cohort= []
#number of days to recover =  time to recovery for non-hospitalized cases (mean: 13.1 days, 95% CI: 8.3, 16.9)
#num_days_between_exposed_infection=Weibull distribution with mean 5.4 days (95% CI: 2.4, 8.3)


interaction_list=[1]



#Edge weights
high_infection_rate = 2.2/100.
low_infection_rate= high_infection_rate #Within cohort edge weight
# scale=1/20
# intra_cohort_infection_rate = low_infection_rate*scale   #Across cohort edge weights
#intra_grade_infection_rate=needed (1/7) #there is no intra_grade_infection_rate variable, but intra_grade_infection_rate=intra_cohort_infection_rate in the current implementation


teacher_student_infection_rate = low_infection_rate*1.5
student_teacher_infection_rate = low_infection_rate



#Initial infected and community spread
initial_fraction_infected=0.0001 #initial fraction infected (fix this)



infection_rate_between_teachers=low_infection_rate*0.05 #we will fix this for Nov 13 plot

#SENSITIVITY PARAMETERS

#Edges of the graph: As discussed we will assume a complete graph for now

#p_c will take three different values low, mid, high
pc_list = [2/total_students, 5/total_students, 10/total_students]
cg_scale = 1


intra_cohort_infection_list= [low_infection_rate/10, low_infection_rate/5, low_infection_rate]

testing_fraction_list = [0.,.1,.2,1.]



#per day what fraction of students are infected from the community.
fraction_community_list = [0.0001, 0.0004, 0.001] #4.7/(1000.*7.)
# fraction_community_list =[ 0]
import pickle

for testing_fraction in testing_fraction_list:
    p_str="p_c"
    ICI_str="ICI"
    school_str="infected_from_school"
    total_str="total infected"
    inboud_str="inbound"
    data_infected={p_str:[],ICI_str:[],school_str:[],total_str:[],inboud_str:[]}
    # print("Testing Fraction = ", testing_fraction)
    # fig, ax = plt.subplots(nrows=len(pc_list), ncols=len(intra_cohort_infection_list), sharey=True)
    # fig.suptitle('Fraction of School Tested Per Day: ' + str(testing_fraction), fontsize=16)
    # plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.5, wspace=0.1)
    ax_i = 0

    for p_c in pc_list:
        print("     p_c value = ",p_c)
        p_g = p_c * cg_scale
        ax_j = 0
        for intra_cohort_infection_rate in intra_cohort_infection_list: #cohort_sizes in cohort_size_list: #

            print("Intra cohort infection rate = ", intra_cohort_infection_rate)

            data_violin_plot = []
            for fraction_infected_at_each_time_step_from_community in fraction_community_list:
                print("Fraction Infected from the Community = ",fraction_infected_at_each_time_step_from_community)
                to_plot1 = []
                to_plot3 = []
                for i in range(school_sim):
                    school = School("LA1", num_grades,cohort_sizes,num_cohort,num_teachers,p_c,p_g,high_risk_probability,high_infection_rate,low_infection_rate,intra_cohort_infection_rate,teacher_student_infection_rate,student_teacher_infection_rate,infection_rate_between_teachers,capacity_of_bus=capacity_of_bus,num_of_cohorts_per_bus=num_of_cohorts_per_bus,bus_interaction_rate=bus_interaction_rate)

                    final_infected_FR, within_school_final_infected_FR, prob_outbreak =\
                        SIR_on_weighted_Graph(school.network,school,number_of_tests=int(testing_fraction*school.network.number_of_nodes()), fraction_infected_at_each_time_step_from_community=fraction_infected_at_each_time_step_from_community,removal_rate= removal_rate,
                                                                                           transmission_scale=transmission_scale,initial_fraction_infected= initial_fraction_infected,num_sim=num_sim)



                    data_infected[school_str]+=within_school_final_infected_FR
                    data_infected[total_str]+=final_infected_FR
                    data_infected[p_str]+=[p_c]*num_sim
                    data_infected[ICI_str]+=[intra_cohort_infection_rate]*num_sim
                    data_infected[inboud_str]+=[fraction_infected_at_each_time_step_from_community]*num_sim
        ax_i = ax_i+1
    # define a list of places
    data_to_dump=pd.DataFrame(data_infected)
    print(data_to_dump)
    with open('output/output_violin'+str(testing_fraction)+'.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(data_to_dump, filehandle)


























