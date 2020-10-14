import  random
import numbers as np

def create_test_times(tmin,tmax,num_tests):
    return np.linspace(tmin,tmax,num_tests)


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