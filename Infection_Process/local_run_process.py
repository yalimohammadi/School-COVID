import Infection_Process.SIR_Basic as EoN
from Graph_Generator.single_school_generator import School
import numpy as np
import scipy as sp
import networkx as nx
import pandas as pd
from collections import defaultdict
import time
import os
import pickle

from Testing_Strategies import Simple_Random


def SIR_on_weighted_Graph(all_test_times, G, school, number_of_tests=0,test_fraction=0, fraction_infected_at_each_time_step_from_community=0.0,
                          removal_rate=1., transmission_scale=1., initial_fraction_infected=0.01, num_sim=1, tmax=160):
    # Fullt = []
    # FullS = []
    # FullE = []
    # FullI = []
    # FullNI = []
    # FullT = []
    # FullR = []
    # FullIsolated = []
    # FullCI = []
    # Fullstatus = []
    # Fulltotal_infections_from_community = []

    Full = []


    tmin=0

    # will change this code later

    # com_inf_dict is they array, where com_inf_dict[v] holds the infection time of individual v from the community.
    com_inf_dict =  np.random.geometric(fraction_infected_at_each_time_step_from_community, size=G.order())

    trans_array=np.zeros((G.number_of_nodes(),G.number_of_nodes()))

    avg_time = 0
    for i in range(num_sim):
        Fulliter = []
        start_time=time.time()
        print(i, end =" ")

        #convention 0==Monday, 1 ==Tuesday and so on
        t, S, E, I, NI, T, R, Isolated, CI, status, total_infections_from_community = EoN.fast_SIR(G, gamma=removal_rate,
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

        # Fullt.append(t)
        # FullS.append(S)
        # FullE.append(E)
        # FullI.append(I)
        # FullNI.append(NI)
        # FullT.append(T)
        # FullR.append(R)
        # FullIsolated.append(Isolated)
        # FullCI.append(CI)
        Fulliter.append(t)
        Fulliter.append(S)
        Fulliter.append(E)
        Fulliter.append(I)
        Fulliter.append(NI)
        Fulliter.append(T)
        Fulliter.append(R)
        Fulliter.append(Isolated)
        Fulliter.append(CI)
        Full.append(Fulliter)
        total_time = time.time() - start_time
        avg_time += total_time



    # Full.append(Fullt)
    # Full.append(FullS)
    # Full.append(FullE)
    # Full.append(FullI)
    # Full.append(FullNI)
    # Full.append(FullT)
    # Full.append(FullR)
    # Full.append(FullIsolated)
    # Full.append(FullCI)
    # Full.append(Fullstatus)
    # Full.append(Fulltotal_infections_from_community)
    print("Average time of the iteration=",avg_time/num_sim)

    return Full


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
###################NOTE I changed next line!!!
infection_rate_between_teachers = low_infection_rate * 0.01  # we will fix this for Nov 13 plot

# SENSITIVITY PARAMETERS

# Edges of the graph: As discussed we will assume a complete graph for now

# p_c will take three different values low, mid, high
pc_list = [2 / total_students]#, 5 / total_students, 10 / total_students]
cg_scale = 1

intra_cohort_infection_list = [low_infection_rate / 10]#, low_infection_rate / 5, low_infection_rate]

testing_fraction_list = [0.5]#,0.5,1]  # 0, 0.1,

# per day what fraction of students are infected from the community.
fraction_community_list = [0.001]#, 0.002, 0.003, 0.004, 0.005]  #

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

                    Full = SIR_on_weighted_Graph(all_test_times, school.network, school,
                                              number_of_tests=int(test_fraction * school.network.number_of_nodes()),
                                              test_fraction=test_fraction,
                                              fraction_infected_at_each_time_step_from_community=fraction_infected_at_each_time_step_from_community,
                                              removal_rate=removal_rate,
                                              transmission_scale=transmission_scale,
                                              initial_fraction_infected=initial_fraction_infected, num_sim=num_sim)

                    with open('FullT' + str(int(testing_fraction * 10**6)) + 'P' + str(int(p_c * 10**6)) + 'I' + str(int(intra_cohort_infection_rate*10**6))+ 'C' + str(int(fraction_infected_at_each_time_step_from_community * 10**5)) + '.data', 'wb') as filehandle:
                        pickle.dump(Full, filehandle)



