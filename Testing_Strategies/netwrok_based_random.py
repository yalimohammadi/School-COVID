import random

def test_random_neighbor(school,test_cap,status):
    testable = []
    test_prob=test_cap*1./len(status.keys())
    for v in status.keys():
        if not(status[v]=="T"):
            if random.uniform(0,1)<test_prob:
                #now choose a neighbor of v
                # note that if u appears two times as neighbors of a node, then there is a higher chance they appear
                school



    if test_cap>len(testable):
        to_process_test=testable
    else:
        to_process_test = random.sample(testable, test_cap)

    return to_process_test
