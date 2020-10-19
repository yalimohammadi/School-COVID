import  random
import math
import numpy as np

def fully_random_test(test_cap,status, already_present=[]):
    testable = already_present
    test_cap = math.ceil(test_cap)
    for v in status.keys():
        if v not in already_present:
            if not(status[v] == "T" and "R"):
                # print(status[v])
                testable.append(v)


    if test_cap>len(testable):
        to_process_test=testable
    else:
        to_process_test = random.sample(testable, test_cap)

    return to_process_test


def random_from_cohorts(school,test_cap,status,weight=[]):
    # if it is not weighted pass nothing for weight
    to_process_test=[]

    total_cohorts= (school.num_cohort*school.num_grades)+1
    if len(weight)>0:
        s=sum(weight)
        test_probs=[w/s for w in weight]
    else:
        test_prob=1./total_cohorts
        test_probs=[test_prob]*(total_cohorts)


    teachers_stat = dict((k, status[k]) for k in school.teachers_id)
    # print(len(teachers_stat))
    # print(test_probs[-1])
    teachers_selected_tests = fully_random_test(test_probs[-1] * test_cap, teachers_stat)
    # print(len(teachers_selected_tests))
    to_process_test += teachers_selected_tests

    for i in range(total_cohorts-1):
        cohort=school.cohorts_list[i]

        cohort_stat= dict((k, status[k]) for k in cohort)
        #print(cohort_stat)
        cohort_selected_tests=fully_random_test(test_probs[i]*test_cap, cohort_stat)

        to_process_test+=cohort_selected_tests

    if len(to_process_test)>test_cap:
        to_process_test=random.sample(to_process_test, test_cap)
    else:
        to_process_test=fully_random_test(test_cap-len(to_process_test), dict((k, status[k]) for k in range(school.size)), to_process_test)

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



