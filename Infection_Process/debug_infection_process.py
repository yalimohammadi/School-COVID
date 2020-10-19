
from Graph_Generator.single_school_generator import School
import EoN
import networkx as nx
import matplotlib.pyplot as plt

total_students=1000
num_grades=4
num_teachers=60
num_of_students_within_grade=int(total_students/num_grades)
p_c=1 # [0.05,0.1,,0.2,0.4]
cg_scale=1 #5 # [5,10]
p_g=p_c*cg_scale
alpha=1
high_infection_rate=low_infection_rate=1*alpha
scale=10
teacher_student_infection_rate=student_teacher_infection_rate=high_infection_rate
infection_rate_between_teachers=high_infection_rate*scale #teachers are similar to cohorts, meaning they have a complete graph.


high_risk_probability=0 #(fixed, irrelevant for now)
initial_fraction_infected= 0.001 #initial fraction infected (fix this)
transmission_scale= 1 #transmission rate per edge (fix this)
removal_rate = 1/12 #recovery rate per node (fix this)

cohort_sizes=6
num_cohort = int(num_of_students_within_grade / cohort_sizes)
intra_cohort_infection_rate=high_infection_rate*scale
school = School("LA1", num_grades, cohort_sizes, num_cohort, num_teachers, p_c, p_g, high_risk_probability,
                high_infection_rate, low_infection_rate, intra_cohort_infection_rate, teacher_student_infection_rate,
                student_teacher_infection_rate, infection_rate_between_teachers)

t,S,I,R=EoN.fast_SIR(school.network, gamma=removal_rate, tau=transmission_scale, transmission_weight="weight",
                                     rho=initial_fraction_infected)

plt.plot(t,S)
plt.plot(t,I)
plt.plot(t,R)
plt.show()
print(R[-1])
# print(school.network.edges(0))
# print(school.network.edges(100))
