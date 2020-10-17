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
            if p<high_risk_probability:
                es=G.edges(v)
                G.update([(e1,e2,{"weight": high_infection_rate} )for (e1,e2) in es] )
        return G

    def set_student_ids(self,ids):
        self.student_ids=ids



class Grade:
    def __init__(self, grade_level, num_cohorts,cohort_size):
        self.grade_level = grade_level
        self.classes=  self.generate_grade(num_cohorts,cohort_size)

    def generate_grade(self,num_cohorts,cohort_size):
        list_classes=[]
        for i in range(num_cohorts):
            new_class=Cohort(self.grade_level, cohort_size)
            list_classes.append(new_class)
        return list_classes


class School:
    def __init__(self,name, num_grades,cohort_sizes,num_cohort,num_teachers,p_c,p_g,high_risk_probability,high_infection_rate,low_infection_rate,intra_cohort_infection_rate,teacher_student_infection_rate,student_teacher_infection_rate,infection_rate_between_teachers):
        self.name = name
        self.size = num_grades*num_cohort*cohort_sizes
        self.num_grades = num_grades
        self.cohort_size=cohort_sizes
        self.num_cohort= num_cohort # num cohort in one grade
        self.list_grades= self.generate_grades(num_grades,num_cohort,cohort_sizes)
        self.cohorts_list=[]
        self.student_to_cohort = dict()
        self.teachers = self.generate_teachers(num_teachers=num_teachers)
        self.teachers_id = []
        self.network = self.generate_school_network(p_c=p_c,p_g=p_g)



    def generate_grades(self,num_grades,num_cohorts,cohort_size):
        list_grades=[]
        for i in range(num_grades):
            new_grade = Grade(i, num_cohorts,cohort_size)
            list_grades.append(new_grade)
        return list_grades

    def generate_teachers(self,num_teachers,infection_rate_between_teachers=0.05):
        # teachers are like a classroom with no high risk student
        teachrs=Cohort(size=num_teachers,grade="teacher",high_risk_probability=0.0, low_infection_rate=infection_rate_between_teachers)
        return teachrs
    def assing_teacher_to_cohort(self,teacher,cohort_ids, teacher_student_infection_rate=.1,student_teacher_infection_rate=.1):
        new_edges=[]
        for s in cohort_ids:
            new_edges.append((teacher,s,{"weight": teacher_student_infection_rate}))
            new_edges.append((s,teacher,{"weight": student_teacher_infection_rate}))

        self.network.add_weighted_edges_from(new_edges)

    def assign_student_to_cohort(self,student_ids,cohort_id):
        for id in student_ids:
            self.student_to_cohort[id]=cohort_id

    def generate_school_network(self, p_c,p_g,Cohort_Size= 15,Num_Cohorts=10, Num_Grades=4,intra_cohort_infection_rate=.1):
        # each school has a few grades and each grades has a number of cohorts
        # Num_Cohorts shows the number of cohorts in one grade
        # Cohort_Size is the number of students in each class
        # p_c is the probability that two people in the same grade (but different cohorts) are connected
        # p_G is the probability that two student in different grades are connected

        grade_sizes = [Cohort_Size*Num_Cohorts]*Num_Grades
        probs = np.ones((Num_Grades,Num_Grades))*p_g
        np.fill_diagonal(probs,p_c)

        intra_cohort_network = nx.stochastic_block_model(grade_sizes, probs, seed=0)

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



        for t in self.teachers_id:
            if t*3+3<self.num_cohort*self.num_grades:
                # assignments finished
                break
            self.assing_teacher_to_cohort(t,self.cohorts_list[t*3])
            self.assing_teacher_to_cohort(t,self.cohorts_list[t*3+1])
            self.assing_teacher_to_cohort(t,self.cohorts_list[t*3+2])

        school_network.add_weighted_edges_from(intra_cohort_network.edges.data("weight", default=intra_cohort_infection_rate))

        #write code here, including weights according to intra_grade_infection_rate

        # print(school_network.edges.data)
        return school_network


