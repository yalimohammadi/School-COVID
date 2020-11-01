import  random
import math
import numpy as np

def fully_random_test(test_cap,tested, at_school,already_present=[]):
    testable = already_present
    test_cap = int(round(test_cap)) # round to nearest integer

    for v in tested.keys():
        if tested[v]==False and (v not in already_present) and at_school[v]:
            testable.append(v)


    if test_cap>len(testable):
        to_process_test=testable
    else:
        to_process_test = random.sample(testable, test_cap)

    return to_process_test


def find_cohorts_at_school(cohort_list,at_school):
    present_cohorts=[]
    for cohort in cohort_list:
        if at_school[cohort[0]]:
            present_cohorts.append(cohort)
    return present_cohorts

def random_from_cohorts(school,test_cap,tested,at_school,weight=[]):
    # if it is not weighted pass nothing for weight
    to_process_test=[]

    at_school_cohorts=find_cohorts_at_school(school.cohorts_list, at_school)
    total_cohorts=len(at_school_cohorts)+1 #+1 is for teachers

    test_prob=1./total_cohorts
    test_probs=[test_prob]*(total_cohorts)


    teachers_stat = dict((k, tested[k]) for k in school.teachers_id)
    teachers_selected_tests = fully_random_test(test_probs[-1] * test_cap, teachers_stat,at_school)

    to_process_test += teachers_selected_tests

    print("num at school_cohorts", len(at_school_cohorts))
    print("tested cohort", at_school_cohorts[0][0],tested[at_school_cohorts[0][0]],at_school[at_school_cohorts[0][0]]) #why student is both tested and is at school
    for cohort in at_school_cohorts:
        cohort_stat= dict((k, tested[k]) for k in cohort)
        #print(cohort_stat)
        cohort_selected_tests=fully_random_test(test_cap*test_prob, cohort_stat,at_school)

        to_process_test+=cohort_selected_tests

    if len(to_process_test)>test_cap:
        to_process_test=random.sample(to_process_test, test_cap)
    else:
        to_process_test=fully_random_test(test_cap-len(to_process_test), dict((k, tested[k]) for k in range(school.size)), to_process_test)

    #print(len(to_process_test))
    return to_process_test

def calculating_test_weights(school,new_positives,next_weights,second_next_weights,first_coefficient=10,second_coefficient=5):

    total_cohorts= (school.num_cohort*school.num_grades)+1

    for id in new_positives:
        # if p positive students are found in the cohort then add p  weights to its testing
        if id in school.teachers_id:
            next_weights[-1]+=1.*first_coefficient
            second_next_weights[-1]+=1.*second_coefficient
        else:
            cohort=school.student_to_cohort[id]
            next_weights[cohort]+=1.*first_coefficient
            second_next_weights[cohort]+=1.*second_coefficient # we can assign smaller weight for the second test


    return next_weights,second_next_weights



