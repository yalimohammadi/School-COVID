import Infection_Process.SIR_Basic as EoN
from Graph_Generator.single_school_generator import School
import numpy as np
import scipy as sp
import networkx as nx
import pandas as pd


def print_school_data(school,num_cohort):
    total_num_cohort = school.num_cohort * school.num_grades
    print(school.teachers_id)

    for i in range(total_num_cohort):
        print(school.cohorts_list[i])

    for node in school.teachers_id:
        print(school.student_to_cohort[node])



school_sim=1
num_sim= 100
num_grades = 3  # its either 3 or 6
cohort_sizes = 6
total_students = num_grades*cohort_sizes*3 #2000
num_of_students_within_grade = int(total_students/num_grades)
#cohort_size_list=[8,10,12,14,16]
num_cohort=int(num_of_students_within_grade/cohort_sizes)
num_teachers = num_cohort*num_grades

school_size=total_students+num_teachers

print('Number of Students = ' + str(total_students) + ' Number of teachers = ' + str(num_teachers))
print('Cohort Size = ', cohort_sizes)










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
high_infection_rate = 2.86/100.
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
pc_list = [2/total_students]
cg_scale = 1


intra_cohort_infection_list= [low_infection_rate/10]

testing_fraction_list = [1] #0, 0.1, 0.2,



#per day what fraction of students are infected from the community.
fraction_community_list = [0.0001]#
# fraction_community_list =[ 0]
import pickle

test_str = "Fraction tested"
p_str = "p_c"
ICI_str = "ICI"
inboud_str = "Inbound"
outbreak30 = "30 days"
outbreak60 = "60 days"
outbreak90 = "90 days"
outbreak120 = "120 days"
outbreak150 = "150 days"
data_infected = {test_str: [], p_str: [], ICI_str: [], inboud_str: [], outbreak30: [], outbreak60: [], outbreak90: [], outbreak120: [], outbreak150: []}
full_data_infected = {test_str: [], p_str: [], ICI_str: [], inboud_str: [], outbreak30: [], outbreak60: [], outbreak90: [], outbreak120: [], outbreak150: []}

diagonal =0
for testing_fraction in testing_fraction_list:
    iter=-1
    for p_c in pc_list:
        iter+=1
        p_g = p_c * cg_scale
        for intra_cohort_infection_rate in [intra_cohort_infection_list[iter]]: #cohort_sizes in cohort_size_list:
            for fraction_infected_at_each_time_step_from_community in fraction_community_list:
                for i in range(school_sim):
                    school = School("LA1", num_grades,cohort_sizes,num_cohort,num_teachers,p_c,p_g,high_risk_probability,high_infection_rate,low_infection_rate,intra_cohort_infection_rate,teacher_student_infection_rate,student_teacher_infection_rate,infection_rate_between_teachers,capacity_of_bus=capacity_of_bus,num_of_cohorts_per_bus=num_of_cohorts_per_bus,bus_interaction_rate=bus_interaction_rate)
                    print_school_data(school,num_teachers)

