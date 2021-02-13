import Infection_Process.SIR_Basic as EoN
from Graph_Generator.single_school_generator import School
import numpy as np
import scipy as sp
import networkx as nx
import pandas as pd
from collections import defaultdict
import time

from Testing_Strategies import Simple_Random


def SIR_on_weighted_Graph(all_test_times, G, school, number_of_tests=0,test_fraction=0, fraction_infected_at_each_time_step_from_community=0,
                          removal_rate=1., transmission_scale=1., initial_fraction_infected=0.01, num_sim=1, tmax=160):
    initial_fraction_infected = 0.0 #changed this on Feb
    final_infected_FR = []
    within_school_final_infected_FR = []
    final_infected_RWC = [0]
    outbreak = .05 * G.number_of_nodes()

    ##########change following
    outbreak= 3.

    num_outbreak_FR30 = 0.0
    num_outbreak_FR60 = 0.0
    num_outbreak_FR90 = 0.0
    num_outbreak_FR120 = 0.0
    num_outbreak_FR150 = 0.0
    total_infected30_list = []
    total_infected60_list = []
    total_infected90_list = []
    total_infected120_list = []
    total_infected150_list = []

    total_positives30_list = []
    total_positives60_list = []
    total_positives90_list = []
    total_positives120_list = []
    total_positives150_list = []

    FN30_list = []
    FN60_list = []
    FN90_list = []
    FN120_list = []
    FN150_list = []

    # total_active_infected30_list = []
    # total_active_infected60_list = []
    # total_active_infected90_list = []
    # total_active_infected120_list = []
    # total_active_infected150_list = []
    weekly_new_infected=[]
    weekly_com_new_infected =[]

    tmin=0

    # will change this code later


    # com_inf_dict is they array, where com_inf_dict[v] holds the infection time of individual v from the community.
    com_inf_dict =  np.random.geometric(fraction_infected_at_each_time_step_from_community, size=G.order())
    # indices = (np.random.uniform(size=G.order()) < fraction_of_infections_from_community_per_day) * 1
    # community_infections = np.where(indices == 1)[0]
    # for u in community_infections:
    #     com_inf_dict = np.random.geometric(fraction_infected_at_each_time_step_from_community, size=G.order())

    avg_time=0
    trans_array=np.zeros((G.number_of_nodes(),G.number_of_nodes()))
    for i in range(num_sim):
        # start_time=time.time()
        # print("Start")
        #convention 0==Monday, 1 ==Tuesday and so on
        t, S, E, I, NI, T, R, Isolated, CI,status, total_infections_from_community = EoN.fast_SIR(G, gamma=removal_rate,
                                                                                           tau=transmission_scale,
                                                                                           transmission_weight="weight",
                                                                                           rho=initial_fraction_infected,
                                                                                           all_test_times=all_test_times,
                                                                                           fraction_of_infections_from_community_per_day=fraction_infected_at_each_time_step_from_community,
                                                                                           test_args=(number_of_tests,),
                                                                                           test_func=Simple_Random.fully_random_test,
                                                                                           weighted_test=False,
                                                                                           school=school, isolate=True,
                                                                                           tmax=tmax, com_inf_dict=com_inf_dict, test_fraction=test_fraction,trans_array=trans_array)
        # total_time=time.time()-start_time
        # avg_time+=total_time
        # print("Iteration time = ",total_time)
        nvalue=0
        nvalueCI=0
        ntime=6
        num_of_weeks=12 #changed this num_weeks to 3 months
        nweeks=0
        k = 0
        while nweeks<num_of_weeks:
            if k < len(t):
                while t[k]<=ntime:
                    nvalue+=NI[k]
                    nvalueCI+=CI[k]
                    k+=1
                    if k >= len(t):
                        break
            ntime+=7
            nweeks+=1
            weekly_new_infected.append(nvalue)
            weekly_com_new_infected.append(nvalueCI)
            nvalue = 0
            nvalueCI=0

        #print(weekly_new_infected)


        final_infected_FR.append(R[-1] + I[-1])  # Since the process has not ended, we need to add I[-1]
        k = 0
        max_infected30 = 0
        max_infected60 = 0
        max_infected90 = 0
        max_infected120 = 0
        max_infected150 = 0

        total_positives30 = 0
        total_positives60 = 0
        total_positives90 = 0
        total_positives120 = 0
        total_positives150 = 0

        max_false_negative30 = 0.0
        max_false_negative60 = 0.0
        max_false_negative90 = 0.0
        max_false_negative120 = 0.0
        max_false_negative150 = 0.0
        for k in range(len(t)):
            if t[k] <= 30:
                max_infected30 = max(I[k], max_infected30)
                if I[k]+R[k]!=0:
                    total_positives30 = T[k]/(I[k]+R[k])  # it will save the last value of T before 30 days
                if E[k] != 0:
                    max_false_negative30 = max(E[k] / (S[k] + E[k]), max_false_negative30)
                else:
                    max_false_negative30 = max(0, max_false_negative30)


            if t[k] <= 60:
                max_infected60 = max(I[k], max_infected60)
                if I[k] + R[k] != 0:
                    total_positives60 = T[k]/(I[k]+R[k])
                if E[k] != 0:
                    max_false_negative60 = max(E[k] / (S[k] + E[k]), max_false_negative60)
                else:
                    max_false_negative60 = max(0, max_false_negative60)

            if t[k] <= 90:
                max_infected90 = max(I[k], max_infected90)
                if I[k] + R[k] != 0:
                    total_positives90 = T[k]/(I[k]+R[k])
                if E[k] != 0:
                    max_false_negative90 = max(E[k] / (S[k] + E[k]), max_false_negative90)
                else:
                    max_false_negative90 = max(0, max_false_negative90)

            if t[k] <= 120:
                max_infected120 = max(I[k], max_infected120)
                if I[k] + R[k] != 0:
                    total_positives120 = T[k]/(I[k]+R[k])
                if E[k] != 0:
                    max_false_negative120 = max(E[k] / (S[k] + E[k]), max_false_negative120)
                else:
                    max_false_negative120 = max(0, max_false_negative120)

            if t[k] <= 150:
                max_infected150 = max(I[k], max_infected150)
                if I[k] + R[k] != 0:
                    total_positives150 = T[k]/(I[k]+R[k])
                if E[k] != 0:
                    max_false_negative150 = max(E[k] / (S[k] + E[k]), max_false_negative150)
                else:
                    max_false_negative150 = max(0, max_false_negative150)

        total_infected30_list.append(max_infected30)
        total_infected60_list.append(max_infected60)
        total_infected90_list.append(max_infected90)
        total_infected120_list.append(max_infected120)
        total_infected150_list.append(max_infected150)

        FN30_list.append(max_false_negative30)
        FN60_list.append(max_false_negative60)
        FN90_list.append(max_false_negative90)
        FN120_list.append(max_false_negative120)
        FN150_list.append(max_false_negative150)

        total_positives30_list.append(total_positives30)
        total_positives60_list.append(total_positives60)
        total_positives90_list.append(total_positives90)
        total_positives120_list.append(total_positives120)
        total_positives150_list.append(total_positives150)

        if max_infected30 > outbreak:
            num_outbreak_FR30 += 1

        if max_infected60 > outbreak:
            num_outbreak_FR60 += 1

        if max_infected90 > outbreak:
            num_outbreak_FR90 += 1

        if max_infected120 > outbreak:
            num_outbreak_FR120 += 1

        if max_infected150 > outbreak:
            num_outbreak_FR150 += 1

    print("Average time of the iteration=",avg_time)

    return weekly_new_infected, weekly_com_new_infected, num_outbreak_FR30 / num_sim, num_outbreak_FR60 / num_sim, num_outbreak_FR90 / num_sim, num_outbreak_FR120 / num_sim, num_outbreak_FR150 / num_sim, total_infected30_list, total_infected60_list, total_infected90_list, total_infected120_list, total_infected150_list, \
        total_positives30_list,  total_positives60_list,  total_positives90_list,  total_positives120_list,  total_positives150_list, FN30_list, FN60_list, FN90_list, FN120_list, FN150_list


school_sim = 1
num_sim = 100
total_students = 6 * 12 * 25  # 2000
num_grades = 6  # its either 3 or 6
num_of_students_within_grade = int(total_students / num_grades)
cohort_sizes = 12
# cohort_size_list=[8,10,12,14,16]
num_cohort = int(num_of_students_within_grade / cohort_sizes)
num_teachers = num_cohort * num_grades
# num_teachers = 0 # same as vaccinating teachers

school_size = total_students + num_teachers

print('School Size = ', total_students + num_teachers)

# We are not using these parameters, because I commented out the bus code
capacity_of_bus = 25
num_of_cohorts_per_bus = 2
bus_interaction_rate = 1 / 10

high_risk_probability = 0  # (fixed, irrelevant for now)
transmission_scale = 1  # transmission rate per edge (fix this)
removal_rate = 1 / 10  # recovery rate per node (fix this)

final_num_infected_full_random = []
final_num_infected_random_cohort = []
final_num_infected_with_cohort_isolation_full_random = []
final_num_infected_with_cohort_isolation_random_cohort = []
final_num_outbreak_with_cohort_isolation_full_random = []
final_num_outbreak_with_cohort_isolation_random_cohort = []
# number of days to recover =  time to recovery for non-hospitalized cases (mean: 13.1 days, 95% CI: 8.3, 16.9)
# num_days_between_exposed_infection=Weibull distribution with mean 5.4 days (95% CI: 2.4, 8.3)


interaction_list = [1]

# Edge weights
high_infection_rate = 2.2 / 100.
low_infection_rate = high_infection_rate  # Within cohort edge weight
# scale=1/20
# intra_cohort_infection_rate = low_infection_rate*scale   #Across cohort edge weights
# intra_grade_infection_rate=needed (1/7) #there is no intra_grade_infection_rate variable, but intra_grade_infection_rate=intra_cohort_infection_rate in the current implementation


teacher_student_infection_rate = low_infection_rate
student_teacher_infection_rate = low_infection_rate

# Initial infected and community spread
initial_fraction_infected = 0.0001  # initial fraction infected (fix this)
###################NOTE I ch
infection_rate_between_teachers = low_infection_rate * 0.01

# SENSITIVITY PARAMETERS

# Edges of the graph: As discussed we will assume a complete graph for now

# p_c will take three different values low, mid, high
pc_list = [2 / total_students, 5 / total_students, 10 / total_students]
cg_scale = 1

intra_cohort_infection_list = [low_infection_rate / 10, low_infection_rate / 5, low_infection_rate]

testing_fraction_list = [0,.5,1,2]#,0.5,1]  # 0, 0.1,

# per day what fraction of students are infected from the community.


fraction_community_list = [0.00016,0.0016,0.008,0.016,0.032]
#, 0.002, 0.003, 0.004, 0.005]  #
# fraction_community_list =[ 0]
import pickle

test_str = "Fraction tested"
p_str = "p_c"
ICI_str = "ICI"
inboud_str = "Inbound"
hue_str = "number of days"
outbreak_str = "Probability of outbreak"
outbreak30 = "30 days"
outbreak60 = "60 days"
outbreak90 = "90 days"
outbreak120 = "120 days"
outbreak150 = "150 days"

FN_str = "False Negative Rates"
Total_Infected_str = "maximum number infected"
New_Infected_str = "new infected"
Com_infected_str = "new infected by community"
positive_tests_str = "positive tests"
week_str = "Week"

data_infected = {test_str: [], p_str: [], ICI_str: [], inboud_str: [], hue_str: [], outbreak_str: []}
full_data_infected = {test_str: [], p_str: [], ICI_str: [], inboud_str: [], hue_str: [],
                      FN_str:[], Total_Infected_str:[], positive_tests_str:[]
                      }
new_data_infected = {test_str: [], p_str: [], ICI_str: [], inboud_str: [],  New_Infected_str:[],Com_infected_str:[]
                      }

diagonal = 0
for testing_fraction in testing_fraction_list:
    test_fraction = 1
    iter = -1
    tmax=160
    all_test_times = []
    # Testing Monday-Friday
    if testing_fraction==0.5:
        for i in range(tmax):
            day_of_week = i % 14
            if day_of_week == 0:
                all_test_times.append(i)
    if testing_fraction==1:
        for i in range(tmax):
            day_of_week = i % 7
            if day_of_week == 0:
                all_test_times.append(i)

    if testing_fraction==2:
        for i in range(tmax):
            day_of_week = i % 7
            if day_of_week == 0 or day_of_week==3:
                all_test_times.append(i)
    print("Testing fraction = ", testing_fraction)
    for p_c in pc_list:
        iter += 1
        print("p_c value = ", p_c)
        p_g = p_c * cg_scale
        for intra_cohort_infection_rate in [intra_cohort_infection_list[iter]]:  # cohort_sizes in cohort_size_list:
            print("ICI Rate= ", intra_cohort_infection_rate)
            for fraction_infected_at_each_time_step_from_community in fraction_community_list:
                print("Fraction Infected from the Community = ", fraction_infected_at_each_time_step_from_community)
                for i in range(school_sim):
                    school = School("LA1", num_grades, cohort_sizes, num_cohort, num_teachers, p_c, p_g,
                                    high_risk_probability, high_infection_rate, low_infection_rate,
                                    intra_cohort_infection_rate, teacher_student_infection_rate,
                                    student_teacher_infection_rate, infection_rate_between_teachers,
                                    capacity_of_bus=capacity_of_bus, num_of_cohorts_per_bus=num_of_cohorts_per_bus,
                                    bus_interaction_rate=bus_interaction_rate)

                    weekly_new_infected_data, weekly_com_new_infected_data, voutbreak30, voutbreak60, voutbreak90, voutbreak120, voutbreak150, vtotal_infected30_list, vtotal_infected60_list, vtotal_infected90_list, vtotal_infected120_list, vtotal_infected150_list ,\
                    total_positives30_list, total_positives60_list, total_positives90_list, total_positives120_list, total_positives150_list, FN30_list, FN60_list, FN90_list, FN120_list, FN150_list= \
                        SIR_on_weighted_Graph(all_test_times, school.network, school,
                                              number_of_tests=int(test_fraction * school.network.number_of_nodes()),
                                              test_fraction=test_fraction,
                                              fraction_infected_at_each_time_step_from_community=fraction_infected_at_each_time_step_from_community,
                                              removal_rate=removal_rate,
                                              transmission_scale=transmission_scale,
                                              initial_fraction_infected=initial_fraction_infected, num_sim=num_sim)

                    data_infected[test_str] += [testing_fraction]*5
                    data_infected[p_str] += [p_c]*5
                    data_infected[ICI_str] += [intra_cohort_infection_rate]*5
                    data_infected[inboud_str] += [fraction_infected_at_each_time_step_from_community]*5
                    data_infected[hue_str] += [outbreak30,outbreak60,outbreak90,outbreak120,outbreak150]
                    data_infected[outbreak_str] += [voutbreak30,voutbreak60,voutbreak90,voutbreak120,voutbreak150]

                    new_data_infected[test_str] += [testing_fraction] * (num_sim * 20)
                    new_data_infected[p_str] += [p_c] * (num_sim * 20)
                    new_data_infected[ICI_str] += [intra_cohort_infection_rate] * (num_sim * 20)
                    new_data_infected[inboud_str] += [fraction_infected_at_each_time_step_from_community] * (num_sim * 20)
                    new_data_infected[New_Infected_str] += weekly_new_infected_data
                    new_data_infected[Com_infected_str] += weekly_com_new_infected_data

                    full_data_infected[test_str] += [testing_fraction] * (num_sim*5)
                    full_data_infected[p_str] += [p_c] * (num_sim*5)
                    full_data_infected[ICI_str] += [intra_cohort_infection_rate] * (num_sim*5)
                    full_data_infected[inboud_str] += [fraction_infected_at_each_time_step_from_community] * (num_sim*5)
                    full_data_infected[hue_str] += [outbreak30]*num_sim +[outbreak60]*num_sim + [outbreak90]*num_sim+ [outbreak120]*num_sim +[outbreak150]*num_sim
                    full_data_infected[FN_str] += FN30_list+ FN60_list+FN90_list+ FN120_list+FN150_list
                    full_data_infected[Total_Infected_str]+= vtotal_infected30_list+ vtotal_infected60_list+ vtotal_infected90_list+ vtotal_infected120_list+ vtotal_infected150_list
                    full_data_infected[positive_tests_str]+= total_positives30_list+total_positives60_list+ total_positives90_list+total_positives120_list+total_positives150_list

    data_to_dump = pd.DataFrame(data_infected)
    full_data_to_dump = pd.DataFrame(full_data_infected)
    new_data_to_dump = pd.DataFrame(new_data_infected)
    print(data_to_dump)
    print(full_data_to_dump)
    with open('withMaskVaccine' + str(int(testing_fraction * 100)) + 't.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(data_to_dump, filehandle)

    with open('withMaskNewInfectedVaccine' + str(int(testing_fraction * 100)) + 't.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(new_data_to_dump, filehandle)

    with open('withMaskFullVaccine' + str(int(testing_fraction * 100)) + 't.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(full_data_to_dump, filehandle)

# data_to_dump=pd.DataFrame(data_infected)
# print(data_to_dump)
# with open('nnnewoutput0t'+'.data', 'wb') as filehandle:
#         # store the data as binary data stream
#         pickle.dump(data_to_dump, filehandle)