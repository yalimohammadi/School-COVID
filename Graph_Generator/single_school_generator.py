import networkx as nx
import numpy as np
import random

class Cohort:
    def __init__(self, grade, size):
        self.grade = grade
        self.size = size
        self.high_risk_students = []
        self.network = self.generate_class( size )

    def generate_class(self, cohort_size, high_risk_probability = .1, high_infection_rate = 2., low_infection_rate =.5):
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
    def __init__(self, name, num_grades,cohort_sizes,num_cohort):
        self.name = name
        self.size = num_grades*num_cohort*cohort_sizes
        self.num_grades = num_grades
        self.cohort_size=cohort_sizes
        self.num_cohort= num_cohort # num cohort in one grade
        self.list_grades= self.generate_grades(num_grades,num_cohort,cohort_sizes)
        self.cohorts_list=[]
        self.network = self.generate_school_network(p_c=.1,p_g=.02)


    def generate_grades(self,num_grades,num_cohorts,cohort_size):
        list_grades=[]
        for i in range(num_grades):
            new_grade = Grade(i, num_cohorts,cohort_size)
            list_grades.append(new_grade)
        return list_grades

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
        for grade in self.list_grades:
            for c in grade.classes:
                old=school_network.number_of_nodes()
                school_network=nx.disjoint_union(school_network, c.network)

                self.cohorts_list.append(list(range(old,school_network.number_of_nodes())))


        school_network.add_weighted_edges_from(intra_cohort_network.edges.data("weight", default=intra_cohort_infection_rate))
        # print(school_network.edges.data)
        return school_network


# example:
# school = School(name="LA1",  num_grades=4,cohort_sizes=15,num_cohort=10)
# print(school.network[0])