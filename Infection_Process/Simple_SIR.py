import Infection_Process.SIR_Basic as EoN
import matplotlib.pyplot as plt
from Graph_Generator.single_school_generator import School
import numpy as np
import scipy as sp
import networkx as nx

from Testing_Strategies import Simple_Random

def set_axis_style(ax, labels, value):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Intra Cohort Infection Rate: ', value)


def plot_simple_SIR(t,S,E,I,T,R,Isolated, alpha=.1,name="full random", teststrat="Fully Random"):

    if name=="full random":

        plt.plot(t, I, label="I, "+name, alpha=alpha, color="r")
    else:
        plt.plot(t, I, label="I, " + name, alpha=alpha, color="b")
        #plt.plot(t, T, label="Infected Tested: Random within Cohort", alpha=alpha, color="r")
        #plt.plot(t, S, label="S", alpha=alpha,color="g")
        #plt.plot(t, R, label="R", alpha=alpha,color="b")
        #plt.plot(t,Isolated,label="Isolated vertices: Random with Cohort Isolation", alpha=alpha, color="g")


def print_infected_teachers(status):
    #for t in list(range(0, school.network.number_of_nodes())):
    for t in school.teachers_id:
        print("infected teacher id: ",t)



def SIR_on_weighted_Graph(G,school,number_of_tests=0,fraction_infected_at_each_time_step_from_community=0,removal_rate = 1.,transmission_scale=1.,initial_fraction_infected= 0.01,num_sim=1,tmax=30) -> object:
    final_infected_FR=[]
    within_school_final_infected_FR = []
    final_infected_RWC=[0]
    outbreak= .10*G.number_of_nodes()
    #print(outbreak)
    num_outbreak_FR=0
    num_outbreak_RWC=0
    for i in range(num_sim):
        t,S,E,I,T,R,Isolated,status,total_infections_from_community=EoN.fast_SIR(G,gamma=removal_rate, tau=transmission_scale,transmission_weight="weight",
                               rho=initial_fraction_infected, all_test_times = np.linspace(0,tmax,tmax+1), fraction_of_infections_from_community_per_day=fraction_infected_at_each_time_step_from_community, test_args=(number_of_tests,),test_func=Simple_Random.fully_random_test,
                                                 weighted_test=False,school=school,isolate=True,tmax=tmax)
        final_infected_FR.append(R[-1]+I[-1]) # Since the process has not ended, we need to add I[-1]
        within_school_final_infected_FR.append(R[-1] + I[-1]-total_infections_from_community-initial_fraction_infected*school.network.number_of_nodes())
        if R[-1]>outbreak:
            num_outbreak_FR+=1
        # #print_infected_teachers(status)
        # print("Infected", I)
        # plot_simple_SIR(t, S, E, I, T, R,Isolated,name="full random")
        #print("I= ",I)
        # print("T= ",T)
        # print("Full Random strategy: Total number of infected= ", R[len(R)-1])
        # print(Isolated)
        # print("time",t[-1])
        # final_num_infected_with_cohort_isolation_full_random.append(np.trapz(np.array(Isolated), np.array(t)))
        # print(final_num_infected_with_cohort_isolation_full_random)
        # t, S, E, I, T, R,Isolated,status = EoN.fast_SIR(G, gamma=removal_rate, tau=transmission_scale, transmission_weight="weight",
        #                              rho=initial_fraction_infected, all_test_times=np.linspace(0, 500, 500),
        #                              test_args=(school, number_of_tests,), test_func=Simple_Random.random_from_cohorts,weighted_test=False,school=school,isolate=True)

        # final_infected_RWC.append(R[-1])
        # if R[-1]>outbreak:
        #     num_outbreak_RWC+=1
        #print(I)
        # plot_simple_SIR(t, S, E, I, T, R,Isolated, name="Random within cohort")
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
    # print("FR",final_infected_FR)
    # print("RWC", final_infected_RWC)
    return final_infected_FR, within_school_final_infected_FR, final_infected_RWC,num_outbreak_FR/num_sim,num_outbreak_RWC/num_sim #



school_sim=1
num_sim=200
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
cg_scale = 1   # 1/10 #5 # [5,10]
#intra_cohort_infection_rate
intra_cohort_infection_list= [low_infection_rate/10, low_infection_rate/5, low_infection_rate]
#Fraction of Testing
testing_fraction_list = [0, 0.05, 0.1, 0.25, 0.5, 1]
#per day what fraction of students are infected from the community.
fraction_community_list = [0.05/100, 0.1/100, 0.5/100, 1/100] #4.7/(1000.*7.)

for testing_fraction in testing_fraction_list:
    print("Testing Fraction = ", testing_fraction)
    fig, ax = plt.subplots(nrows=len(pc_list), ncols=len(intra_cohort_infection_list), sharey=True)
    fig.suptitle('Fraction of School Tested Per Day: ' + str(testing_fraction), fontsize=16)
    plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.5, wspace=0.1)
    ax_i = 0
    for p_c in pc_list:
        print("     p_c value = ",p_c)
        p_g = p_c * cg_scale
        ax_j = 0
        for intra_cohort_infection_rate in intra_cohort_infection_list: #cohort_sizes in cohort_size_list: #
            print("          Intra cohort infection rate = ", intra_cohort_infection_rate)

            data_violin_plot = []
            for fraction_infected_at_each_time_step_from_community in fraction_community_list:
                print("               Fraction Infected from the Community = ",fraction_infected_at_each_time_step_from_community)
                to_plot1 = []
                to_plot2 = []
                to_plot3 = []
                to_plot4 = []
                for i in range(school_sim):
                    school = School("LA1", num_grades,cohort_sizes,num_cohort,num_teachers,p_c,p_g,high_risk_probability,high_infection_rate,low_infection_rate,intra_cohort_infection_rate,teacher_student_infection_rate,student_teacher_infection_rate,infection_rate_between_teachers,capacity_of_bus=capacity_of_bus,num_of_cohorts_per_bus=num_of_cohorts_per_bus,bus_interaction_rate=bus_interaction_rate)
                    #print("School Size = ", school.network.number_of_nodes())
                    #plt.subplot(121)
                    #nx.draw(school.network, with_labels=True, font_weight='bold')
                    #plt.show()
                    avg1, within_school_avg1, avg2,outbreak1,outbreak2 = SIR_on_weighted_Graph(school.network,school,number_of_tests=int(testing_fraction*school.network.number_of_nodes()), fraction_infected_at_each_time_step_from_community=fraction_infected_at_each_time_step_from_community,removal_rate= removal_rate,
                                                                                           transmission_scale=transmission_scale,initial_fraction_infected= initial_fraction_infected,num_sim=num_sim)


                    to_plot1.append(np.mean(avg1)/school.network.number_of_nodes())
                    to_plot2.append(np.mean(avg2)/ school.network.number_of_nodes())
                    to_plot3.append(outbreak1)
                    to_plot4.append(outbreak2)
                    navg1 = [x/school.network.number_of_nodes() for x in avg1]
                    within_school_navg1 = [x/school.network.number_of_nodes() for x in within_school_avg1]
                    data_violin_plot.append(sorted(within_school_navg1))
                    navg2 = [x/school.network.number_of_nodes() for x in avg2]

                    print("                             Total Infected Full Random = ", avg1)
                    print("                             Total Fraction Infected Full Random = ", navg1)
                    print("                             Total Infected Within School Full Random = ", within_school_avg1)
                    print("                             Total Fraction Infected Within School Full Random = ", within_school_navg1)
                    # print("Total Fraction Infected Random Within Cohort = ", navg2)
                    # print("Total Fraction Infected Random Within Cohort = ", navg2)
                    # print("Fraction Outbreak Full Random = ", outbreak1)
                    # print("Fraction Outbreak Random Within Cohort = ", outbreak2)
                    #print(outbreak1,outbreak2,"fraction outbrek")


                final_num_infected_with_cohort_isolation_full_random.append(np.mean(to_plot1))
                final_num_infected_with_cohort_isolation_random_cohort.append(np.mean(to_plot2))
                final_num_outbreak_with_cohort_isolation_full_random.append(np.mean(to_plot3))
                final_num_outbreak_with_cohort_isolation_random_cohort.append(np.mean(to_plot4))

            #plt.figure(1)

            ax[ax_i,ax_j].set_title('PC Value: ' + str(p_c) + ', ICI Rate: ' + str(np.round(intra_cohort_infection_rate, 3)), color='blue')
            ax[ax_i,ax_j].set_ylabel('Fraction Infected Within 30 days')
            ax[ax_i,ax_j].violinplot(data_violin_plot)
            #labels = ['Community Infection:', 'Community Infection:', 'Fraction Infected From \n Community Per Day:']
            #set_axis_style(ax, labels, intra_cohort_infection_rate)
            ax[ax_i,ax_j].set_xlabel('Community Infections')
            ax_j=ax_j+1

        ax_i = ax_i+1






plt.show()


# print("Final output:")
# print("Total Fraction Infected FR = ", final_num_infected_with_cohort_isolation_full_random)
# print("Total Fraction Infected RWC = ", final_num_infected_with_cohort_isolation_random_cohort)
# print("Fraction Outbreak FR = ", final_num_outbreak_with_cohort_isolation_full_random)
# print("Fraction Outbreak RWC = ", final_num_outbreak_with_cohort_isolation_random_cohort)
# plt.figure(1)
# plt.plot(intra_cohort_infection_list,final_num_infected_with_cohort_isolation_full_random,label="Infected: Full Random Testing", alpha=1, color= "blue")
# # plt.plot(interaction_list,final_num_infected_with_cohort_isolation_random_cohort,label="Infected: Random within Cohort Testing", alpha=1, color= "red")
# # plt.plot(cohort_size_list,final_num_infected_with_cohort_isolation_full_random,label="Infected: Full Random Testing", alpha=1, color= "blue")
# # plt.plot(cohort_size_list,final_num_infected_with_cohort_isolation_random_cohort,label="Infected: Random within Cohort Testing", alpha=1, color= "red")
# plt.legend()
# plt.xlabel("Across cohort infection rate")
# #plt.xlabel("Cohort Size")#push?
# plt.ylabel("Avg Total Fraction of Infected")
# plt.figure(2)
# plt.plot(intra_cohort_infection_list,final_num_outbreak_with_cohort_isolation_full_random,label="Outbreak: Full Random Testing", alpha=1, color= "blue")
# # plt.plot(interaction_list,final_num_outbreak_with_cohort_isolation_random_cohort,label="Outbreak: Random within Cohort Testing", alpha=1, color= "red")
# # plt.plot(cohort_size_list,final_num_outbreak_with_cohort_isolation_full_random,label="Outbreak: Full Random Testing", alpha=1, color= "blue")
# # plt.plot(cohort_size_list,final_num_outbreak_with_cohort_isolation_random_cohort,label="Outbreak: Random within Cohort Testing", alpha=1, color= "red")
# plt.legend()
# #plt.xlabel("Cohort Size")#push?
# plt.xlabel("Across cohort infection rate")
# plt.ylabel("Fraction of Outbreak")
# plt.show()








































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






