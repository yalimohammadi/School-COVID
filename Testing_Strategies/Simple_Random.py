import  random
import numbers as np


def fully_random_test(test_cap,status):
    testable = []
    for v in status.keys():
        if not( status[v]=="T"):
            if random.uniform(0,1)<test_cap:
                testable.append(v)


    if test_cap>len(testable):
        to_process_test=testable
    else:
        to_process_test = random.sample(testable, test_cap)

    return to_process_test


def random_from_cohorts(school,test_cap,status):
    to_process_test=[]

    return to_process_test