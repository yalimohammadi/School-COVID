

import networkx as nx
import numpy as np
import random

class Cohort:
    def __init__(self, grade, size,high_risk_probability=.1, high_infection_rate=5/7, low_infection_rate=5/7):
        self.grade = grade
        self.size = size
        self.high_risk_students = []
        self.network = self.generate_class( size,high_risk_probability,high_infection_rate, low_infection_rate)

        self.student_ids= []#generate this after generating the whole network

    def generate_class(self, cohort_size, high_risk_probability =.1, high_infection_rate=5/7, low_infection_rate=5/7):
        G = nx.complete_graph(cohort_size)
        G = G.to_directed()
        G.add_weighted_edges_from(G.edges.data('weight', default=low_infection_rate))
        for v in G.nodes():
            # generate high risk students
            p=random.uniform(0, 1)
            if p < high_risk_probability:
                es=G.edges(v)
                G.update([(e1,e2,{"weight": high_infection_rate} )for (e1,e2) in es])
        return G

    def set_student_ids(self,ids):
        self.student_ids=ids


class Grade:
    def __init__(self, grade_level, num_cohorts,cohort_size,high_risk_probability, high_infection_rate, low_infection_rate):
        self.grade_level = grade_level
        self.classes=  self.generate_grade(num_cohorts,cohort_size,high_risk_probability=high_risk_probability, high_infection_rate=high_infection_rate, low_infection_rate=low_infection_rate)

    def generate_grade(self,num_cohorts,cohort_size,high_risk_probability, high_infection_rate, low_infection_rate):
        list_classes=[]
        for i in range(num_cohorts):
            new_class=Cohort(self.grade_level, cohort_size,high_risk_probability=high_risk_probability, high_infection_rate=high_infection_rate, low_infection_rate=low_infection_rate)
            list_classes.append(new_class)
        return list_classes


class School:
    def __init__(self, name, num_grades,cohort_sizes,num_cohort,num_teachers,p_c=1./7.,p_g=1./35.,high_risk_probability=0.0,high_infection_rate=1.,low_infection_rate=.5,intra_cohort_infection_rate=1.,teacher_student_infection_rate=1.,student_teacher_infection_rate=1.,infection_rate_between_teachers=1.,max_num_students=2000,capacity_of_bus=25,num_of_cohorts_per_bus=2,bus_interaction_rate=1/10):
        self.name = name
        self.size = num_grades*num_cohort*cohort_sizes+num_teachers
        self.num_grades = num_grades
        self.cohort_size=cohort_sizes
        self.num_cohort= num_cohort # num cohort in one grade
        self.list_grades= self.generate_grades(num_grades,num_cohort,cohort_sizes,high_risk_probability, high_infection_rate, low_infection_rate)
        self.cohorts_list=[]

        self.student_to_cohort = dict()
        self.teachers = self.generate_teachers(num_teachers=num_teachers,infection_rate_between_teachers=infection_rate_between_teachers)
        self.teachers_id = []
        self.network = self.generate_school_network(p_c=p_c,p_g=p_g,Cohort_Size=cohort_sizes,Num_Cohorts=num_cohort,Num_Grades= num_grades,intra_cohort_infection_rate=intra_cohort_infection_rate,teacher_student_infection_rate=teacher_student_infection_rate,student_teacher_infection_rate=student_teacher_infection_rate,max_num_students=max_num_students,capacity_of_bus=capacity_of_bus,num_of_cohorts_per_bus=num_of_cohorts_per_bus,bus_interaction_rate=bus_interaction_rate)


    def generate_grades(self,num_grades,num_cohorts,cohort_size,high_risk_probability, high_infection_rate, low_infection_rate):
        list_grades=[]
        for i in range(num_grades):
            new_grade = Grade(i, num_cohorts,cohort_size,high_risk_probability, high_infection_rate, low_infection_rate)
            list_grades.append(new_grade)
        return list_grades

    def generate_teachers(self,num_teachers,infection_rate_between_teachers=0.05):
        # teachers are like a classroom with no high risk student
        teachrs=Cohort(size=num_teachers,grade="teacher",high_risk_probability=0.0, low_infection_rate=infection_rate_between_teachers)
        return teachrs
    def assing_teacher_to_cohort(self,teacher,cohort_ids, teacher_student_infection_rate=.1,student_teacher_infection_rate=.1):

        new_edges=[]
        for s in cohort_ids:
            new_edges.append((teacher,s,teacher_student_infection_rate))
            new_edges.append((s,teacher,student_teacher_infection_rate))

        return new_edges



    def generate_school_network(self, Cohort_Size, Num_Cohorts, Num_Grades,p_c=1/7, p_g=1/35,
                                    intra_cohort_infection_rate=.1, teacher_student_infection_rate=3 / 7,
                                    student_teacher_infection_rate=3 / 7,max_num_students=2000,capacity_of_bus=25,num_of_cohorts_per_bus=2,bus_interaction_rate=1/10):

        # each school has a few grades and each grades has a number of cohorts
        # Num_Cohorts shows the number of cohorts in one grade
        # Cohort_Size is the number of students in each class
        # p_c is the probability that two people in the same grade (but different cohorts) are connected
        # p_G is the probability that two student in different grades are connected

        grade_sizes = [Cohort_Size*Num_Cohorts]*Num_Grades
        probs = np.ones((Num_Grades,Num_Grades))*p_g
        np.fill_diagonal(probs,p_c)

        intra_cohort_network = nx.stochastic_block_model(grade_sizes, probs, seed=0)
        mapping=dict()
        num_teachers=self.teachers.network.number_of_nodes()
        for i in range(intra_cohort_network.number_of_nodes()):
            mapping[i]=i+num_teachers
        nx.relabel_nodes(intra_cohort_network, mapping,copy=False)

        school_network = nx.Graph()
        old = school_network.number_of_nodes()
        school_network=nx.disjoint_union(school_network, self.teachers.network)
        self.teachers_id=list(range(old,school_network.number_of_nodes()))

        for grade in self.list_grades:
            for c in grade.classes:
                old=school_network.number_of_nodes()
                school_network=nx.disjoint_union(school_network, c.network)

                student_ids= list(range(old,school_network.number_of_nodes()))
                self.assign_student_to_cohort(student_ids,cohort_id=len(self.cohorts_list))
                self.cohorts_list.append(student_ids)
                c.set_student_ids(student_ids)



        teacher_edges=[]
        num_cohorts_per_teacher = 1
        total_cohorts=self.num_cohort*self.num_grades
        for t in self.teachers_id:
            for i in range(num_cohorts_per_teacher):
                new_edge=self.assing_teacher_to_cohort(t,self.cohorts_list[(t*num_cohorts_per_teacher+i)%total_cohorts],
                                                       teacher_student_infection_rate,student_teacher_infection_rate)

                self.cohorts_list[(t*num_cohorts_per_teacher+i)%total_cohorts].append(t) # add teacher to cohort when assigning it. we need it for isolation policyt
                self.assign_student_to_cohort([t],(t*num_cohorts_per_teacher+i)%total_cohorts,teacher=True)

                teacher_edges+=new_edge

        # print("t_edges",teacher_edges)
        bus=[]
        school_network.add_weighted_edges_from(teacher_edges)
        school_network.add_weighted_edges_from(intra_cohort_network.edges.data("weight", default=intra_cohort_infection_rate))





        for v in range(max_num_students+num_teachers, school_network.number_of_nodes()):
            school_network.remove_node(v)
        #write code here, including weights according to intra_grade_infection_rate

        # print(school_network.edges.data)

        # #code for school bus starts here
        # b=0
        # num_of_bus=int(np.ceil((Cohort_Size*Num_Cohorts*Num_Grades)/capacity_of_bus))
        # num_of_students_per_bus_per_cohort=min(capacity_of_bus/num_of_cohorts_per_bus,Cohort_Size)
        # cap=np.zeros(int(num_of_bus))
        # bus=np.ones(school_network.number_of_nodes())*-1
        # for i in np.random.permutation(range(total_cohorts)):
        #     iterm=0
        #     for c in np.random.permutation(self.cohorts_list[i]):
        #         if c>=max_num_students+num_teachers:
        #             continue
        #         while cap[b]>=capacity_of_bus:
        #             b=(b+1)%num_of_bus
        #             iterm=0
        #         bus[c]=b
        #         cap[b]=cap[b]+1
        #         iterm = (iterm + 1)%num_of_students_per_bus_per_cohort
        #         if iterm==0:
        #             b=(b+1)%num_of_bus
        # bus_edges=[]
        # for c in range(num_teachers,num_teachers+max_num_students):
        #     for cp in range(num_teachers,num_teachers+max_num_students):
        #         if c==cp:
        #             continue
        #         if bus[c]==bus[cp]:
        #             bus_edges.append((c,cp,bus_interaction_rate))
        #
        # school_network.add_weighted_edges_from(bus_edges)

        return school_network



    def assign_student_to_cohort(self,student_ids,cohort_id,teacher=False):
        for id in student_ids:
            if teacher:
                if id not in self.student_to_cohort.keys():
                    self.student_to_cohort[id]=[]
                if cohort_id not in self.student_to_cohort[id]:
                    self.student_to_cohort[id].append(cohort_id)
            else:
                self.student_to_cohort[id]=cohort_id



