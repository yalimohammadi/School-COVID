import  random


def fully_random_test(test_cap,status):
    testable = []
    for v in status.keys():
        if not(status[v]=="T"):
            testable.append(v)


    if test_cap>len(testable):
        to_process_test=testable
    else:
        to_process_test = random.sample(testable, test_cap)

    return to_process_test


def random_from_cohorts(school,test_cap,status,weight=[]):
    # if it is not weighted pass nothing for weight
    to_process_test=[]

    total_cohorts= (school.num_cohort*school.num_grades)
    if len(weight)>0:
        s=sum(weight)
        test_probs=[w/s for w in weight]
    else:
        test_prob=int(1./total_cohorts)
        test_probs=[test_prob]*(total_cohorts)


    for i in range(total_cohorts):
        cohort=school.cohorts_list[i]

        cohort_stat= dict((k, status[k]) for k in cohort )
        cohort_selected_tests=fully_random_test(test_probs[i]*test_cap, cohort_stat)

        to_process_test+=cohort_selected_tests

    return to_process_test

def calculating_test_weights(school,new_positives,next_weights,second_next_weights,first_coefficient=10,second_coefficient=5):

    total_cohorts= (school.num_cohort*school.num_grades)

    for id in new_positives:
        # if p positive students are found in the cohort then add p  weights to its testing
        cohort=school.student_to_cohort[id]
        next_weights[cohort]+=1.*first_coefficient
        second_next_weights[cohort]+=1.*second_coefficient # we can assign smaller weight for the second test


    return next_weights,second_next_weights



