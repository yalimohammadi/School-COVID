import networkx as nx
import random
import heapq
import numpy as np
import EoN
from collections import defaultdict
from scipy.stats import weibull_min


from collections import Counter


#######################
#                     #
#   Auxiliary stuff   #
#                     #
#######################

#testing

def _truncated_exponential_(rate, T):
    r'''returns a number between 0 and T from an
    exponential distribution conditional on the outcome being between 0 and T'''
    t = random.expovariate(rate)
    L = int(t / T)
    return t - L * T


class myQueue(object):
    r'''
    This class is used to store and act on a priority queue of events for
    event-driven simulations.  It is based on heapq.

    Each queue is given a tmax (default is infinity) so that any event at later
    time is ignored.

    This is a priority queue of 4-tuples of the form
                   ``(t, counter, function, function_arguments)``

    The ``'counter'`` is present just to break ties, which generally only occur when
    multiple events are put in place for the initial condition, but could also
    occur in cases where events tend to happen at discrete times.

    note that the function is understood to have its first argument be t, and
    the tuple ``function_arguments`` does not include this first t.

    So function is called as
        ``function(t, *function_arguments)``

    Previously I used a class of events, but sorting using the __lt__ function
    I wrote was significantly slower than simply using tuples.
    '''

    def __init__(self, tmax=float("Inf")):
        self._Q_ = []
        self.tmax = tmax
        self.counter = 0  # tie-breaker for putting things in priority queue

    def add(self, time, function, args=()):
        r'''time is the time of the event.  args are the arguments of the
        function not including the first argument which must be time'''
        if time < self.tmax:
            heapq.heappush(self._Q_, (time, self.counter, function, args))
            self.counter += 1

    def pop_and_run(self):
        r'''Pops the next event off the queue and performs the function'''
        t, counter, function, args = heapq.heappop(self._Q_)
        function(t, *args)

    def __len__(self):
        r'''this will allow us to use commands like ``while Q:`` '''
        return len(self._Q_)

    def current_time(self):
        t = heapq.nsmallest(1, self._Q_)[0][0]
        # print(t)
        return t













#REMOVED SIMULATION CODE (COPY PASTE FROM simulation_code.py file if needed)







### Code starting here does event-driven simulations ###


def _find_trans_and_rec_delays_SIR_(node, sus_neighbors, trans_time_fxn,
                                    rec_time_fxn, trans_time_args=(),
                                    rec_time_args=()):
    rec_delay = rec_time_fxn(node, *rec_time_args)
    trans_delay = {}
    for target in sus_neighbors:
        trans_delay[target] = trans_time_fxn(node, target, *trans_time_args)
    return trans_delay, rec_delay







def _trans_and_rec_time_Markovian_const_trans_(node, sus_neighbors, tau, rec_rate_fxn):
    r'''I introduced this with a goal of making the code run faster.  It looks
    like the fancy way of selecting the infectees and then choosing their
    infection times is slower than just cycling through, finding infection
    times and checking if that time is less than recovery time.  So I've
    commented out the more "sophisticated" approach.
    '''

    duration = random.expovariate(rec_rate_fxn(node))

    trans_prob = 1 - np.exp(-tau * duration)
    number_to_infect = np.random.binomial(len(sus_neighbors), trans_prob)
    # print(len(suscep_neighbors),number_to_infect,trans_prob, tau, duration)
    transmission_recipients = random.sample(sus_neighbors, number_to_infect)
    trans_delay = {}
    for v in transmission_recipients:
        trans_delay[v] = _truncated_exponential_(tau, duration)
    return trans_delay, duration
    #     duration = random.expovariate(rec_rate_fxn(node))
    #     trans_delay = {}
    #
    #
    #     for v in sus_neighbors:
    #         if tau == 0:
    #             trans_delay[v] = float('Inf')
    #         else:
    #             trans_delay[v] = random.expovariate(tau)
    # #        if delay<duration:
    # #            trans_delay[v] = delay
    #     return trans_delay, duration

    ##slow approach 1:
    #    next_delay = random.expovariate(tau)
    #    index, delay = int(next_delay//duration), next_delay%duration
    #    while index<len(sus_neighbors):
    #        trans_delay[sus_neighbors[index]] = delay
    #        next_delay = random.expovariate(tau)
    #        jump, delay = int(next_delay//duration), next_delay%duration
    #        index += jump

    ##slow approach 2:
    # trans_prob = 1-np.exp(-tau*duration)
    # number_to_infect = np.random.binomial(len(sus_neighbors),trans_prob)
    # print(len(suscep_neighbors),number_to_infect,trans_prob, tau, duration)
    # transmission_recipients = random.sample(sus_neighbors,number_to_infect)
    # trans_delay = {}
    # for v in transmission_recipients:
    #    trans_delay[v] = _truncated_exponential_(tau, duration)
    return trans_delay, duration





def fast_SIR(G, tau, gamma, initial_infecteds=None, initial_recovereds=None,
             rho=None, tmin=0, tmax=float('Inf'), transmission_weight=None,
             recovery_weight=None, return_full_data=False, sim_kwargs=None,
             all_test_times=[], fraction_of_infections_from_community_per_day=0.0, test_args=None,test_func=None,weighted_test=True,school=None, isolate=False):
    r'''
    fast SIR simulation for exponentially distributed infection and
    recovery times

    From figure A.3 of Kiss, Miller, & Simon.  Please cite the
    book if using this algorithm.




    :Arguments:

    **G** networkx Graph
        The underlying network

    **tau** number
        transmission rate per edge

    **gamma** number
        recovery rate per node

    **initial_infecteds** node or iterable of nodes
        if a single node, then this node is initially infected

        if an iterable, then whole set is initially infected

        if None, then choose randomly based on rho.

        If rho is also None, a random single node is chosen.

        If both initial_infecteds and rho are assigned, then there
        is an error.

    **initial_recovereds** iterable of nodes (default None)
        this whole collection is made recovered.
        Currently there is no test for consistency with initial_infecteds.
        Understood that everyone who isn't infected or recovered initially
        is initially susceptible.

    **rho** number
        initial fraction infected. number is int(round(G.order()*rho))

    **tmin** number (default 0)
        starting time

    **tmax** number  (default Infinity)
        maximum time after which simulation will stop.
        the default of running to infinity is okay for SIR,
        but not for SIS.

    **transmission_weight**    string  (default None)
        the label for a weight given to the edges.
        transmission rate is
        G.adj[i][j][transmission_weight]*tau

    **recovery_weight**   string (default None))
        a label for a weight given to the nodes to scale their
        recovery rates
        gamma_i = G.nodes[i][recovery_weight]*gamma

    **return_full_data**   boolean (default False)
        Tells whether a Simulation_Investigation object should be returned.

    **sim_kwargs** keyword arguments
        Any keyword arguments to be sent to the Simulation_Investigation object
        Only relevant if ``return_full_data=True``


    :Returns:

    **times, S,E, I, T, R** numpy arrays

    Or if ``return_full_data is True``

    **full_data**  Simulation_Investigation object
            from this we can extract the status history of all nodes.
            We can also plot the network at given times
            and create animations using class methods.

    :SAMPLE USE:

    ::


        import networkx as nx
        import EoN
        import matplotlib.pyplot as plt

        G = nx.configuration_model([1,5,10]*100000)
        initial_size = 10000
        gamma = 1.
        tau = 0.3
        t, S,E, I, T, R = EoN.fast_SIR(G, tau, gamma,
                                    initial_infecteds = range(initial_size))

        plt.plot(t, I)
    '''
    # tested in test_SIR_dynamics

    if transmission_weight is not None or tau * gamma == 0:
        trans_rate_fxn, rec_rate_fxn = EoN._get_rate_functions_(G, tau, gamma,
                                                                transmission_weight,
                                                                recovery_weight)

        def trans_time_fxn(source, target, trans_rate_fxn):
            rate = trans_rate_fxn(source, target)
            if rate > 0:
                return random.expovariate(rate)
            else:
                return float('Inf')

        def rec_time_fxn(node, rec_rate_fxn):
            rate = rec_rate_fxn(node)
            if rate > 0:
                return (1/rate)*weibull_min.rvs(7, size=1)[0]#random.expovariate(rate)
            else:
                return float('Inf')

        trans_time_args = (trans_rate_fxn,)
        rec_time_args = (rec_rate_fxn,)
        return fast_nonMarkov_SIR(G, trans_time_fxn=trans_time_fxn,
                                  rec_time_fxn=rec_time_fxn,
                                  trans_time_args=trans_time_args,
                                  rec_time_args=rec_time_args,
                                  initial_infecteds=initial_infecteds,
                                  initial_recovereds=initial_recovereds,
                                  rho=rho, tmin=tmin, tmax=tmax,
                                  return_full_data=return_full_data,
                                  sim_kwargs=sim_kwargs, all_test_times=all_test_times,
                                  fraction_of_infections_from_community_per_day=fraction_of_infections_from_community_per_day,
                                  test_args=test_args,test_func=test_func,weighted_test=weighted_test,
                                  school=school,isolate=isolate)


    else:
        # the transmission rate is tau for all edges.  We can use this
        # to speed up the code.

        # get rec_rate_fxn (recovery rate may be variable)
        trans_rate_fxn, rec_rate_fxn = EoN._get_rate_functions_(G, tau, gamma,
                                                                transmission_weight,
                                                                recovery_weight)

        return fast_nonMarkov_SIR(G,
                                  trans_and_rec_time_fxn=_trans_and_rec_time_Markovian_const_trans_,
                                  trans_and_rec_time_args=(tau, rec_rate_fxn),
                                  initial_infecteds=initial_infecteds,
                                  initial_recovereds=initial_recovereds,
                                  rho=rho, tmin=tmin, tmax=tmax,
                                  return_full_data=return_full_data,
                                  sim_kwargs=sim_kwargs)





def fast_nonMarkov_SIR(G, trans_time_fxn=None,
                       rec_time_fxn=None,
                       trans_and_rec_time_fxn=None,
                       trans_time_args=(),
                       rec_time_args=(),
                       trans_and_rec_time_args=(),
                       initial_infecteds=None,
                       initial_recovereds=None,
                       rho=None, tmin=0, tmax=float('Inf'),
                       return_full_data=False, sim_kwargs=None,
                       all_test_times=[], fraction_of_infections_from_community_per_day=0.0, test_args=(), test_func=None,
                       weighted_test=True, school=None,isolate=False):
    r'''
    A modification of the algorithm in figure A.3 of Kiss, Miller, &
    Simon to allow for user-defined rules governing time of
    transmission.

    Please cite the book if using this algorithm.

    This is useful if the transmission rule is non-Markovian in time, or
    for more elaborate models.

    Allows the user to define functions (details below) to determine
    the rules of transmission times and recovery times.  There are two ways to do
    this.  The user can define a function that calculates the recovery time
    and another function that calculates the transmission time.  If recovery is after
    transmission, then transmission occurs.  We do this if the time to transmission
    is independent of the time to recovery.

    Alternately, the user may want to model a situation where time to transmission
    and time to recovery are not independent.  Then the user can define a single
    function (details below) that would determine both recovery and transmission times.


    :Arguments:

    **G** Networkx Graph

    **trans_time_fxn** a user-defined function
        returns the delay until transmission for an edge.  May depend
        on various arguments and need not be Markovian.

        Returns float

        Will be called using the form

        ``trans_delay = trans_time_fxn(source_node, target_node, *trans_time_args)``
            Here trans_time_args is a tuple of the additional
            arguments the functions needs.

        the source_node is the infected node
        the target_node is the node that may receive transmission
        rec_delay is the duration of source_node's infection, calculated
        by rec_time_fxn.

    **rec_time_fxn** a user-defined function
        returns the delay until recovery for a node.  May depend on various
        arguments and need not be Markovian.

        Returns float.

        Called using the form

        ``rec_delay = rec_time_fxn(node, *rec_time_args)``
            Here rec_time_args is a uple of additional arguments
            the function needs.

    **trans_and_rec_time_fxn** a user-defined function
        returns both a dict giving delay until transmissions for all edges
        from source to susceptible neighbors and a float giving delay until
        recovery of the source.

        Can only be used **INSTEAD OF** ``trans_time_fxn`` AND ``rec_time_fxn``.

        Gives an **ERROR** if these are also defined

        Called using the form
        ``trans_delay_dict, rec_delay = trans_and_rec_time_fxn(
                                           node, susceptible_neighbors,
                                           *trans_and_rec_time_args)``
        here trans_delay_dict is a dict whose keys are those neighbors
        who receive a transmission and rec_delay is a float.

    **trans_time_args** tuple
        see trans_time_fxn

    **rec_time_args** tuple
        see rec_time_fxn

    **trans_and_rec_time_args** tuple
        see trans_and_rec_time_fxn

    **initial_infecteds** node or iterable of nodes
        if a single node, then this node is initially infected

        if an iterable, then whole set is initially infected

        if None, then choose randomly based on rho.  If rho is also
        None, a random single node is chosen.

        If both initial_infecteds and rho are assigned, then there
        is an error.

    **initial_recovereds** iterable of nodes (default None)
        this whole collection is made recovered.

        Currently there is no test for consistency with initial_infecteds.

        Understood that everyone who isn't infected or recovered initially
        is initially susceptible.

    **rho** number
        initial fraction infected. number is int(round(G.order()*rho))

    **tmin** number (default 0)
        starting time

    **tmax** number (default infinity)
        final time

    **return_full_data** boolean (default False)
        Tells whether a Simulation_Investigation object should be returned.

    **sim_kwargs** keyword arguments
        Any keyword arguments to be sent to the Simulation_Investigation object
        Only relevant if ``return_full_data=True``


    :Returns:

    **times, S,E, I, P, R** numpy arrays

    Or if ``return_full_data is True``

    **full_data**  Simulation_Investigation object
        from this we can extract the status history of all nodes
        We can also plot the network at given times
        and even create animations using class methods.


    :SAMPLE USE:

    ::


        import EoN
        import networkx as nx
        import matplotlib.pyplot as plt
        import random

        N=1000000
        G = nx.fast_gnp_random_graph(N, 5/(N-1.))



        #set up the code to handle constant transmission rate
        #with fixed recovery time.
        def trans_time_fxn(source, target, rate):
            return random.expovariate(rate)

        def rec_time_fxn(node,D):
            return D

        D = 5
        tau = 0.3
        initial_inf_count = 100

        t, S,E, I, P, R = EoN.fast_nonMarkov_SIR(G,
                                trans_time_fxn=trans_time_fxn,
                                rec_time_fxn=rec_time_fxn,
                                trans_time_args=(tau,),
                                rec_time_args=(D,),
                                initial_infecteds = range(initial_inf_count))

        # note the comma after ``tau`` and ``D``.  This is needed for python
        # to recognize these are tuples

        # initial condition has first 100 nodes in G infected.

    '''
    if rho and initial_infecteds:
        raise EoN.EoNError("cannot define both initial_infecteds and rho")
    if rho and initial_recovereds:
        raise EoN.EoNError("cannot define both initial_recovereds and rho")

    if (trans_time_fxn and not rec_time_fxn) or (rec_time_fxn and not trans_time_fxn):
        raise EoN.EoNError("must define both trans_time_fxn and rec_time_fxn or neither")
    elif trans_and_rec_time_fxn and trans_time_fxn:
        raise EoN.EoNError("cannot define trans_and_rec_time_fxn at the same time as trans_time_fxn and rec_time_fxn")
    elif not trans_and_rec_time_fxn and not trans_time_fxn:
        raise EoN.EoNError("if not defining trans_and_rec_time_fxn, must define trans_time_fxn and rec_time_fxn")

    if not trans_and_rec_time_fxn:  # we define the joint function.
        trans_and_rec_time_fxn = _find_trans_and_rec_delays_SIR_
        trans_and_rec_time_args = (trans_time_fxn, rec_time_fxn, trans_time_args, rec_time_args)

    # now we define the initial setup.
    status = defaultdict(lambda: 'S')  # node status defaults to 'S'
    at_school = defaultdict(lambda: 'True')
    tested = defaultdict(lambda: 'False') #shows if a node has been ever tested positive
    for v in G.nodes():
        status[v]='S'
        at_school[v]=True
        tested[v]=False

    rec_time = defaultdict(lambda: tmin - 1)  # node recovery time defaults to -1
    if initial_recovereds is not None:
        for node in initial_recovereds:
            status[node] = 'R'
            rec_time[node] = tmin - 1  # default value for these.  Ensures that the recovered nodes appear with a time
    pred_inf_time = defaultdict(lambda: float('Inf'))
    # infection time defaults to \infty  --- this could be set to tmax,
    # probably with a slight improvement to performance.

    Q = myQueue(tmax)

    if initial_infecteds is None:  # create initial infecteds list if not given
        if rho is None:
            initial_number = 1
        else:
            initial_number = max(int(round(G.order() * rho)),1)
        initial_infecteds = random.sample(G.nodes(), initial_number)
    elif G.has_node(initial_infecteds):
        initial_infecteds = [initial_infecteds]
    # else it is assumed to be a list of nodes.

    times, S, E, I, P, R, Isolated = ([tmin], [G.order()],[0], [0], [0], [0],[0])
    transmissions = []

    status[-1]="I" # -1 is dummy source
    for u in initial_infecteds:
        times.append(tmin+1)
        S.append(S[-1] - 1)  # no change to number susceptible
        I.append(I[-1])  # one less infected
        E.append(E[-1] + 1)  #
        P.append(P[-1])  # no change to number infected tested
        R.append(R[-1])  # one more recovered
        Isolated.append(Isolated[-1])
        #print(S)
        status[u] = 'E'
        inf_time = 2  # weibull distribution
        #pred_inf_time[u] = tmin + inf_time
        Q.add(tmin+inf_time, _process_trans_SIR_, args=(G, -1, u, times, S, E, I, P, R, Isolated, Q,
                                               status, at_school, rec_time,
                                               pred_inf_time, transmissions,
                                               trans_and_rec_time_fxn,
                                               trans_and_rec_time_args
                                               )
              )
        pred_inf_time[u] = tmin + inf_time
    #print("initial infection is done")
    # Note that when finally infected, pred_inf_time is correct
    # and rec_time is correct.
    # So if return_full_data is true, these are correct

    all_test_times=[i+tmin+1 for i in all_test_times] # calibrate with min simulation time
    cur_test_time = tmax + 10
    if len(all_test_times) > 0:
        cur_test_time = all_test_times.pop(0)



    if weighted_test:
        school = test_args[0]
        test_cap = test_args[1]
        total_num_cohorts=school.num_grades*school.num_cohort+1
        curr_weight = [1 for i in range(total_num_cohorts)]
        curr_weight[-1]=10 #last index is for teachers
        next_weight = [1 for i in range(total_num_cohorts)]
        next_weight[-1]=10#last index is for teachers

    community_spread_time =tmin+2

    transmision_from_community = 0
    while Q:  # all the work is done in this while loop.
        cur_time = Q.current_time()
        # percentage_infected=(I[-1]+ R[-1])/G.order()
        # if percentage_infected>0.05:
        #     break
        #COMMUNITY SPREAD CODE

        if community_spread_time < cur_time and community_spread_time <= cur_test_time:

            # print("")
            # community_infections = random.sample(G.nodes(), int(round(G.order() * fraction_of_infections_from_community_per_day)))
            # if the fraction of outside is less than 1/size of school, this will be zero
            # change it to binomial maybr?
            indices=(np.random.uniform(size=G.order()) < fraction_of_infections_from_community_per_day) * 1
            community_infections= np.where(indices==1)[0]

            for u in community_infections:

                if status[u]=='S':
                    transmision_from_community+=1
                    times.append(community_spread_time)
                    S.append(S[-1] - 1)  # no change to number susceptible
                    I.append(I[-1])  # one less infected
                    E.append(E[-1] + 1)  #
                    P.append(P[-1])  # no change to number infected tested
                    R.append(R[-1])  # one more recovered
                    Isolated.append(Isolated[-1])
                    # print(S)
                    status[u] = 'E'
                    inf_time = get_infection_time()#5.4 * weibull_min.rvs(5, size=1)[0]  # weibull distribution
                    # pred_inf_time[u] = tmin + inf_time
                    Q.add(community_spread_time + inf_time, _process_trans_SIR_, args=(G, -1, u, times, S, E, I, P, R, Isolated, Q,
                                                                  status, at_school, rec_time,
                                                                  pred_inf_time, transmissions,
                                                                  trans_and_rec_time_fxn,
                                                                  trans_and_rec_time_args
                                                                  )
                      )

            community_spread_time=community_spread_time+1
    # COMMUNITY SPREAD CODE ENDS HERE

        if cur_test_time < cur_time and cur_test_time < community_spread_time:
            if weighted_test:
                curr_weight, next_weight, positive_nodes= testing_strategy_with_weights(cur_test_time,
                                                                               times, S,E, I, P, R, Isolated, status, tested, school, test_cap, curr_weight,next_weight)
                print("next_weight",next_weight)
                print("curr_weight",curr_weight)
            else:
                positive_nodes = testing_strategy(cur_test_time, times, S, E, I, P, R, Isolated, status,tested,test_args,test_func,at_school) # call process_test_SIR on all indivduals who are in state S and I and they are tested at that particular moment
            if isolate:
                new_isolated=find_isolated_cohorts(positive_nodes,school,at_school)
                isolate_set_of_nodes(cur_test_time,times,S,E,I,P,R,Isolated,Q,new_isolated, at_school)
            else:
                isolate_set_of_nodes(cur_test_time,times,S,E,I,P,R,Isolated,Q,positive_nodes,at_school)
            if len(all_test_times)>0:
                cur_test_time = all_test_times.pop(0)
            else:
                cur_test_time = tmax + 10
        else:
            Q.pop_and_run()

    # print("total transmission from community",transmision_from_community, "in time", times[-1])



    # the initial infections were treated as ordinary infection events at
    # time 0.
    # So each initial infection added an entry at time 0 to lists.
    # We'd like to get rid these excess events.
    times = times[len(initial_infecteds):]
    S = S[len(initial_infecteds):]
    I = I[len(initial_infecteds):]
    E = E[len(initial_infecteds):]
    R = R[len(initial_infecteds):]
    P = P[len(initial_infecteds):] # set of nodes that we are aware that are positive
    Isolated = Isolated[len(initial_infecteds):]
    if not return_full_data:
        return np.array(times), np.array(S), np.array(E), np.array(I), \
               np.array(P), np.array(R), np.array(Isolated),status,transmision_from_community
    else:
        # strip pred_inf_time and rec_time down to just the values for nodes
        # that became infected
        # could use iteritems for Python 2, by   try ... except AttributeError
        infection_times = {node: time for (node, time) in
                           pred_inf_time.items() if status[node] != 'S'}
        recovery_times = {node: time for (node, time) in
                          rec_time.items() if status[node] == 'R'}

        node_history = _transform_to_node_history_(infection_times, recovery_times,
                                                   tmin, SIR=True)
        if sim_kwargs is None:
            sim_kwargs = {}
        return EoN.Simulation_Investigation(G, node_history, transmissions,
                                            possible_statuses=['S', 'E', 'I', 'P', 'R'],
                                            **sim_kwargs)







#######OUR CODE STARTS HERE


def get_infection_time():
    return 5.4 * weibull_min.rvs(5, size=1)[0]

def isolate_set_of_nodes(time, times, S, E, I, T, R, Isolated, Q, set_of_nodes,at_school,isolation_time=14.0):
    for node in set_of_nodes:
        _isolate_a_node(time, times, S, E, I, T, R, Isolated, Q, node, at_school,isolation_time=isolation_time)

def _isolate_a_node(time, times, S, E, I, P, R, Isolated, Q, node, at_school, isolation_time=14.):
    at_school[node]=False
    times.append(time)
    S.append(S[-1])  # no change to number susceptible
    I.append(I[-1])  # no change to number infected
    E.append(E[-1])  #
    R.append(R[-1])  # no change to number recovered
    P.append(P[-1])  # one more infected tested
    Isolated.append(Isolated[-1]+1)

    Q.add(time + isolation_time, _unisolate_a_node,
          args=(times, S, E, I, P, R, Isolated, node, at_school)
          )

def debug(to_print,message):
    print("debugs")

    print(message)
    for i in to_print:
        print(i)

def _unisolate_a_node(time, times, S, E, I, P, R, Isolated, node, at_school):

    at_school[node]=True

    times.append(time)
    S.append(S[-1])  # no change to number susceptible
    I.append(I[-1])  # no change to number infected
    E.append(E[-1])  #
    R.append(R[-1])  # no change to number recovered
    P.append(P[-1])  # one more infected tested

    Isolated.append(Isolated[-1]-1)
    #debug([[Isolated[-1],Isolated[-2]],node, at_school[node]] ,"updated isolation")


def testing_strategy(time, times, S, E, I, P, R, Isolated, status, tested, test_args, test_func,at_school):

    to_test=test_func(*test_args, tested,at_school)########Verify this
    # print(len(to_test))
    new_positive=0
    positive_ids=[]
    for node in to_test:
        if status[node] == 'I':
            status[node] = 'T'
            new_positive+=1
            positive_ids.append(node)
            tested[node]=True
    # if len(set(to_test))<200:
    #     print(test_func)
    #     debug([len(set(to_test))],"num of tests")

    times.append(time)
    S.append(S[-1])  # no change to number susceptible
    I.append(I[-1])  # no change to number infected
    E.append(E[-1])  #
    R.append(R[-1])  # no change to number recovered
    P.append(P[-1] + new_positive)  # one more infected tested
    #debug([positive_ids,status,tested],"in testing strategy, new positives (first line). status (second line) , all already tested positive students (third line)")
    Isolated.append(Isolated[-1])
    return positive_ids

from Testing_Strategies.Simple_Random import calculating_test_weights
from Testing_Strategies.Simple_Random import random_from_cohorts



def testing_strategy_with_weights(time, times, S, E, I, P, R, Isolated, status, tested, school, test_cap, curr_weight, next_weight):

    total_num_cohorts=school.num_grades*school.num_cohort+1

    second_next_weight = [1 for i in range(total_num_cohorts)]
    # last index is for teachers
    second_next_weight[-1]=10.


    to_test = random_from_cohorts(school,test_cap,status,curr_weight)

    new_positive=0
    new_positive_ids=[]
    for node in to_test:
        if status[node] == 'I':
            status[node] = 'T'
            new_positive+=1
            new_positive_ids.append(node)
            tested[node]=True
    print("new_positives",new_positive)
    times.append(time)
    S.append(S[-1])  # no change to number susceptible
    E.append(E[-1])  #
    I.append(I[-1])  # no change to number infected
    R.append(R[-1])  # no change to number recovered
    P.append(P[-1] + new_positive)  # one more infected tested

    Isolated.append(Isolated[-1])

    next_weight,second_next_weight=calculating_test_weights(school, new_positive_ids, next_weight, second_next_weight)

    return next_weight,second_next_weight,new_positive_ids


def find_isolated_cohorts(positives,school,at_school,threshold=1):
    total_num_cohort=school.num_cohort*school.num_grades
    positive_per_cohort=np.zeros(total_num_cohort)
    positive_teachers=0
    for node in positives:
        if node==-1:
            continue
        if node in school.teachers_id:
            positive_teachers+=1
        else:
            positive_per_cohort[school.student_to_cohort[node]]+=1

    to_isolate=[]
    for i in range(total_num_cohort):
        ps=positive_per_cohort[i]
        if ps>=threshold:
            for student in school.cohorts_list[i]:
                #if at_school[student]:
                to_isolate.append(student)
            # to_isolate+=school.cohorts_list[i]
    # print("number of positive teachers",positive_teachers)


    return to_isolate





def _process_trans_SIR_(time, G, source, target, times, S, E, I, P, R, Isolated, Q, status, at_school,
                        rec_time, pred_inf_time, transmissions,
                        trans_and_rec_time_fxn,
                        trans_and_rec_time_args=()):
    r'''
    From figure A.4 of Kiss, Miller, & Simon.  Please cite the book if
    using this algorithm.

    :Arguments:

    time : number
        time of transmission
**G**  networkx Graph
    node : node
        node receiving transmission.
    times : list
        list of times at which events have happened
    S,E, I, T, R : lists
        lists of numbers of nodes of each status at each time
    Q : myQueue
        the queue of events
    status : dict
        dictionary giving status of each node
    rec_time : dict
        dictionary giving recovery time of each node
    pred_inf_time : dict
        dictionary giving predicted infeciton time of nodes
    trans_and_rec_time_fxn : function
        trans_and_rec_time_fxn(node, susceptible_neighbors, *trans_and_rec_time_args)
        returns tuple consisting of
           dict of delays until transmission from node to neighbors and
           float having delay until recovery of node
        An example of how to use this appears in the code fast_SIR where
        depending on whether inputs are weighted, it constructs different
        versions of this function and then calls fast_nonMarkov_SIR.
    trans_and_rec_time_args : tuple (default empty)
        see trans_and_rec_time_fxn

    :Returns:

    nothing returned

    :MODIFIES:

    status : updates status of newly infected node
    rec_time : adds recovery time for node
    times : appends time of event
    S : appends new S (reduced by 1 from last)
    I : appends new I (increased by 1)
    R : appends new R (same as last)
    Q : adds recovery and transmission events for newly infected node.
    pred_inf_time : updated for nodes that will receive transmission

    '''

    if status[target] == 'E': #and status[source] == 'I':  # nothing happens if already infected/tested.
        status[target] = 'I' #status[target] = 'I'
        times.append(time)
        transmissions.append((time, source, target))
        S.append(S[-1])  # one less susceptible
        I.append(I[-1] + 1)  # one more infected
        E.append(E[-1] - 1)  #
        P.append(P[-1])  # no change to infected tested
        R.append(R[-1])  # no change to recovered

        Isolated.append(Isolated[-1])

        suscep_neighbors = [v for v in G.neighbors(target) if status[v] == 'S']

        trans_delay, rec_delay = trans_and_rec_time_fxn(target, suscep_neighbors,
                                                        *trans_and_rec_time_args)
        # print("trans_delay, rec_delay", trans_delay, rec_delay)

        rec_time[target] = time + rec_delay
        if rec_time[target] <= Q.tmax:
            Q.add(rec_time[target], _process_rec_SIR_,
                  args=(target, times, S, E, I, P, R, Isolated, status))
        for v in trans_delay:
            inf_time = time + trans_delay[v]
            if inf_time <= rec_time[target] and inf_time < pred_inf_time[v] and inf_time <= Q.tmax:
                Q.add(inf_time, _process_exp_SIR_,
                      args=(G, target, v, times, S, E, I, P, R, Isolated, Q,
                            status, at_school, rec_time, pred_inf_time,
                            transmissions, trans_and_rec_time_fxn,
                            trans_and_rec_time_args
                            )
                      )




def _process_exp_SIR_(time, G, source, target, times, S, E, I, P, R, Isolated, Q, status, at_school,
                      rec_time, pred_inf_time, transmissions,
                      trans_and_rec_time_fxn,
                      trans_and_rec_time_args=()):
    r'''From figure A.3 of Kiss, Miller, & Simon.  Please cite the
    book if using this algorithm.

    :Arguments:

        event : event
            has details on node and time
        times : list
            list of times at which events have happened
        S,E, I, T, R : lists
            lists of numbers of nodes of each status at each time
        status : dict
            dictionary giving status of each node


    :Returns:
        :
        Nothing

    MODIFIES
    ----------
    status : updates status of newly recovered node
    times : appends time of event
    S : appends new S (same as last)
    I : appends new I (decreased by 1)
    R : appends new R (increased by 1)
    '''
    if status[source] == 'I' and status[target] == 'S' and at_school[source] and at_school[target]:
        times.append(time)
        S.append(S[-1]-1)  # no change to number susceptible
        I.append(I[-1])  # one less infected
        E.append(E[-1]+1)  #
        P.append(P[-1])  # no change to number infected tested  #
        R.append(R[-1])  # one more recovered
        Isolated.append(Isolated[-1])
        status[target] = 'E'
        inf_time = get_infection_time()#5.4 * weibull_min.rvs(5, size=1)[0] #weibull distribution
        Q.add(time+inf_time, _process_trans_SIR_,
          args=(G, source, target, times, S, E, I, P, R, Isolated, Q, status, at_school,
                rec_time, pred_inf_time, transmissions,
                trans_and_rec_time_fxn,
                trans_and_rec_time_args)
          )
        pred_inf_time[target] = time+inf_time



def _process_rec_SIR_(time, node, times, S, E, I, P, R, Isolated, status):
    r'''From figure A.3 of Kiss, Miller, & Simon.  Please cite the
    book if using this algorithm.

    :Arguments:

        event : event
            has details on node and time
        times : list
            list of times at which events have happened
        S,E, I, T, R : lists
            lists of numbers of nodes of each status at each time
        status : dict
            dictionary giving status of each node


    :Returns:
        :
        Nothing

    MODIFIES
    ----------
    status : updates status of newly recovered node
    times : appends time of event
    S : appends new S (same as last)
    I : appends new I (decreased by 1)
    R : appends new R (increased by 1)
    '''
    #if status[node] == 'I':
    times.append(time)
    S.append(S[-1])  # no change to number susceptible
    I.append(I[-1] - 1)  # one less infected
    E.append(E[-1])  #
    P.append(P[-1])  # no change to number infected tested

    R.append(R[-1] + 1)  # one more recovered
    Isolated.append(Isolated[-1])
    status[node] = 'R'








