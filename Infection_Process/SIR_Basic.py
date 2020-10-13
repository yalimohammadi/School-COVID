import networkx as nx
import random
import heapq
import numpy as np
import EoN
from collections import defaultdict
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


# from sortedcontainers import SortedList
#
#
# class _MySortedList_(object):
#     r'''
#     This code is intended as a possible replacement for the ListDict that I've
#     been using.
#
#     I'm leaving it in place, but so far for my tests it is slower than ListDict.
#     It might do better in cases where the probabilities of events have a wider
#     range (in which case the rejection sampling in ListDict would be slow).
#
#     Since the cases I've tried may have fairly similar probabilities, the rejection
#     sampling seems to outperform it.  If rejection sampling often resulted in rejections, this
#     data type might perform better.
#      '''
#
#
#     def __init__(self, weighted = False):
#         if weighted is True:
#             self.weighted = True
#             self.SL = SortedList()
#             self.itemcounter = 0
#             #
#             #the SortedList self.SL will hold tuples that look like
#             #(weight, itemcounter, item).
#             #itemcounter is included to break any ties and make sure that
#             #the comparison algorithm never looks at item (since I don't know
#             #whether comparison is even defined for them.
#             #
#             #The entries will be ordered by weight first, and if there is
#             #a tie, then by itemcounter (which increments whenever something
#             #is added)
#             #
#
#             self.item2weight = {}
#             #
#             #self.item2weight is a dict where self.item2weight[item] is a tuple
#             # of the form (weight, itemcounter), which will allow us to quickly
#             #find the item in the list if we ever need it.
#             #it also allows for a quick test to see if item has been put in.
#             self._total_weight = 0
#         else:
#             self.weighted = False #need to build up this case.  Should basically be a listdict
#             raise EoN.EoNError("you should be using a ListDict instead")
#
#     def insert(self, item, weight = None):
#         r'''
#         If not present, then inserts the thing (with weight if appropriate)
#         if already there, replaces the weight unless weight is 0
#
#         If weight is 0, then it removes the item and doesn't replace.
#
#         WARNING:
#             if already present, then replaces weight.  To increment weight
#             use update
#         '''
#
#         if item in self.item2weight:
#             datum = (*self.item2weight.pop(item), item)
#             self.SL.remove(datum)
#         if weight > 0:
#             self.update(item, weight_increment=weight)
#
#     def update(self, item, weight_increment = None):
#         r'''
#         If not present, then inserts the thing (with weight if appropriate)
#         if already there, increments weight
#
#         WARNING:
#             if already present, just increments weight.  To overwrite weight
#             use insert.
#         '''
#         if weight_increment is not None: #will break if passing a weight to unweighted case
#             self._total_weight += weight_increment
#             if item in self.item2weight:
#                 datum = (*self.item2weight.pop(item), item)
#                 self.SL.remove(datum)
#                 new_weight = datum[0]+weight_increment
#             else:
#                 new_weight = weight_increment
#             if new_weight >0:
#                 self.itemcounter += 1
#                 self.SL.add((new_weight, self.itemcounter, item))
#                 self.item2weight[item] = (new_weight, self.itemcounter)
#         elif self.weighted:
#             raise Exception('if weighted, must assign weight_increment')
#
#
#     def __len__(self):
#         return len(self.items)
#
#     def __contains__(self, item):
#         return item in self.item2weight
#
#     def remove(self, item):
#         datum = (*self.item2weight.pop(item), item)
#         self.SL.remove(datum)
#         self._total_weight -= datum[0]
#
#     def choose_random(self):
#         #presumes weighted and sorted in increasing order.  So start from end
#         #and work back.
#         current_index = len(self.SL)-1
#         largest_r = 0
#         q=0
#         p=1
#         while current_index >= 0:
#             datum = self.SL[current_index]
#             W = datum[0]
#             r = (q+p*random.random())**(1./W)
#             if r> largest_r:
#                 largest_r = r
#                 returnindex = current_index
#
#             q = largest_r**W
#             p = 1-q
#             current_index -= np.random.geometric(p)
#
#         return self.SL[returnindex][2]
#
#
#
#     def random_removal(self):
#         r'''uses other class methods to choose an entry by weight and then remove it'''
#         choice = self.choose_random()
#         self.remove(choice)
#         return choice
#
#     def total_weight(self):
#         if self.weighted:
#             return self._total_weight
#         else:
#             return len(self)
#
#     def update_total_weight(self):
#         self._total_weight = sum(self.weight[item] for item in self.items)


class _ListDict_(object):
    r'''
    The Gillespie algorithm will involve a step that samples a random element
    from a set based on its weight.  This is awkward in Python.

    So I'm introducing a new class based on a stack overflow answer by
    Amber (http://stackoverflow.com/users/148870/amber)
    for a question by
    tba (http://stackoverflow.com/users/46521/tba)
    found at
    http://stackoverflow.com/a/15993515/2966723

    This will allow me to select a random element uniformly, and then use
    rejection sampling to make sure it's been selected with the appropriate
    weight.


    I believe a faster data structure can be created with a (binary) tree.
    We add an object with a weight to the tree.  The nodes track their weights
    and the sum of the weights below it.  So choosing a random object (by weight)
    means that we choose a random number between 0 and weight_sum.  Then
    if it's less than the first node's weight, we choose that.  Otherwise,
    we see if the remaining bit is less than the total under the first child.
    If so, go there, otherwise, it's the other child.  Then iterate.  Adding
    a node would probably involve placing higher weight nodes higher in
    the tree.  Currently I don't have a fast enough implementation of this
    for my purposes.  So for now I'm sticking to the mixture of lists &
    dictionaries.

    I believe this structure I'm describing is similar to a "partial sum tree"
    or a "Fenwick tree", but they seem subtly different from this.
    '''

    def __init__(self, weighted=False):
        self.item_to_position = {}
        self.items = []

        self.weighted = weighted
        if self.weighted:
            self.weight = defaultdict(int)  # presume all weights positive
            self.max_weight = 0
            self._total_weight = 0
            self.max_weight_count = 0

    def __len__(self):
        return len(self.items)

    def __contains__(self, item):
        return item in self.item_to_position

    def _update_max_weight(self):
        C = Counter(self.weight.values())  # may be a faster way to do this, we only need to count the max.
        self.max_weight = max(C.keys())
        self.max_weight_count = C[self.max_weight]

    def insert(self, item, weight=None):
        r'''
        If not present, then inserts the thing (with weight if appropriate)
        if already there, replaces the weight unless weight is 0

        If weight is 0, then it removes the item and doesn't replace.

        WARNING:
            replaces weight if already present, does not increment weight.


        '''
        if self.__contains__(item):
            self.remove(item)
        if weight != 0:
            self.update(item, weight_increment=weight)

    def update(self, item, weight_increment=None):
        r'''
        If not present, then inserts the thing (with weight if appropriate)
        if already there, increments weight

        WARNING:
            increments weight if already present, cannot overwrite weight.
        '''
        if weight_increment is not None:  # will break if passing a weight to unweighted case
            if weight_increment > 0 or self.weight[item] != self.max_weight:
                self.weight[item] = self.weight[item] + weight_increment
                self._total_weight += weight_increment
                if self.weight[item] > self.max_weight:
                    self.max_weight_count = 1
                    self.max_weight = self.weight[item]
                elif self.weight[item] == self.max_weight:
                    self.max_weight_count += 1
            else:  # it's a negative increment and was at max
                self.max_weight_count -= 1
                self.weight[item] = self.weight[item] + weight_increment
                self._total_weight += weight_increment
                self.max_weight_count -= 1
                if self.max_weight_count == 0:
                    self._update_max_weight
        elif self.weighted:
            raise Exception('if weighted, must assign weight_increment')

        if item in self:  # we've already got it, do nothing else
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items) - 1

    def remove(self, choice):
        position = self.item_to_position.pop(choice)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

        if self.weighted:
            weight = self.weight.pop(choice)
            self._total_weight -= weight
            if weight == self.max_weight:
                # if we find ourselves in this case often
                # it may be better just to let max_weight be the
                # largest weight *ever* encountered, even if all remaining weights are less
                #
                self.max_weight_count -= 1
                if self.max_weight_count == 0 and len(self) > 0:
                    self._update_max_weight()

    def choose_random(self):
        # r'''chooses a random node.  If there is a weight, it will use rejection
        # sampling to choose a random node until it succeeds'''
        if self.weighted:
            while True:
                choice = random.choice(self.items)
                if random.random() < self.weight[choice] / self.max_weight:
                    break
            # r = random.random()*self.total_weight
            # for item in self.items:
            #     r-= self.weight[item]
            #     if r<0:
            #         break
            return choice

        else:
            return random.choice(self.items)

    def random_removal(self):
        r'''uses other class methods to choose and then remove a random node'''
        choice = self.choose_random()
        self.remove(choice)
        return choice

    def total_weight(self):
        if self.weighted:
            return self._total_weight
        else:
            return len(self)

    def update_total_weight(self):
        self._total_weight = sum(self.weight[item] for item in self.items)


def _transform_to_node_history_(infection_times, recovery_times, tmin, SIR=True):
    r'''The original (v0.96 and earlier) returned infection_times and recovery_times.
    The new version returns node_history instead. This code transforms
    the former to the latter.

    It is only used for the continuous time cases.
    '''
    if SIR:
        node_history = defaultdict(lambda: ([tmin], ['S']))
        for node, time in infection_times.items():
            if time == tmin:
                node_history[node] = ([], [])
            node_history[node][0].append(time)
            node_history[node][1].append('I')
        for node, time in recovery_times.items():
            if time == tmin:
                node_history[node] = ([], [])
            node_history[node][0].append(time)
            node_history[node][1].append('R')
    else:
        node_history = defaultdict(lambda: ([tmin], ['S']))
        for node, Itimes in infection_times.items():
            Rtimes = recovery_times[node]
            while Itimes:
                time = Itimes.pop(0)
                if time == tmin:
                    node_history[node] = ([], [])
                node_history[node][0].append(time)
                node_history[node][1].append('I')
                if Rtimes:
                    time = Rtimes.pop(0)
                    node_history[node][0].append(time)
                    node_history[node][1].append('S')

    return node_history


##########################
#                        #
#    SIMULATION CODE     #
#                        #
##########################

'''
    The code in the region below is used for stochastic simulation of 
    epidemics on networks
'''


def _simple_test_transmission_(u, v, p):
    r'''
    A simple test for whether u transmits to v assuming constant probability p

    From figure 6.8 of Kiss, Miller, & Simon.  Please cite the book if
    using this test_transmission function for basic_discrete_SIR.

    This handles the simple case where transmission occurs with
    probability p.

    :Arguments:

        u (node)
            the infected node
        v : node
            the susceptible node
        p : number between 0 and 1
            the transmission probability

    :Returns:



            True if u will infect v (given opportunity)
            False otherwise
    '''

    return random.random() < p


 


def discrete_SIR(G, test_transmission=_simple_test_transmission_, args=(),
                 initial_infecteds=None, initial_recovereds=None,
                 rho=None, tmin=0, tmax=float('Inf'),
                 return_full_data=False, sim_kwargs=None):
    # tested in test_discrete_SIR
    r'''
    Simulates an SIR epidemic on G in discrete time, allowing user-specified transmission rules

    From figure 6.8 of Kiss, Miller, & Simon.  Please cite the book
    if using this algorithm.

    Return details of epidemic curve from a discrete time simulation.

    It assumes that individuals are infected for exactly one unit of
    time and then recover with immunity.

    This is defined to handle a user-defined function
    ``test_transmission(node1,node2,*args)``
    which determines whether transmission occurs.

    So elaborate rules can be created as desired by the user.

    By default it uses
    ``_simple_test_transmission_``
    in which case args should be entered as (p,)

    :Arguments:

    **G** NetworkX Graph (or some other structure which quacks like a
                           NetworkX Graph)
        The network on which the epidemic will be simulated.

    **test_transmission** function(u,v,*args)
        (see below for args definition)
        A function that determines whether u transmits to v.
        It returns True if transmission happens and False otherwise.
        The default will return True with probability p, where args=(p,)

        This function can be user-defined.
        It is called like:
        test_transmission(u,v,*args)
        Note that if args is not entered, then args=(), and this call is
        equivalent to
        test_transmission(u,v)

    **args** a list or tuple
        The arguments of test_transmission coming after the nodes.  If
        simply having transmission with probability p it should be
        entered as
        args=(p,)

        [note the comma is needed to tell Python that this is really a
        tuple]

    **initial_infecteds** node or iterable of nodes (default None)
        if a single node, then this node is initially infected
        if an iterable, then whole set is initially infected
        if None, then choose randomly based on rho.  If rho is also
        None, a random single node is chosen.
        If both initial_infecteds and rho are assigned, then there
        is an error.

    **initial_recovereds** as for initial_infecteds, but initially
            recovered nodes.

    **rho** number  (default is None)
        initial fraction infected. initial number infected
        is int(round(G.order()*rho)).

        The default results in a single randomly chosen initial infection.

    **tmin** start time

    **tmax** stop time (default Infinity).

    **return_full_data** boolean (default False)
        Tells whether a Simulation_Investigation object should be returned.

    **sim_kwargs** keyword arguments
        Any keyword arguments to be sent to the Simulation_Investigation object
        Only relevant if ``return_full_data=True``


    :Returns:


    **t, S, I, R** numpy arrays

    Or ``if return_full_data is True`` returns

    **full_data**  Simulation_Investigation object
        from this we can extract the status history of all nodes
        We can also plot the network at given times
        and even create animations using class methods.

    :SAMPLE USE:

    ::

        import networkx as nx
        import EoN
        import matplotlib.pyplot as plt
        G = nx.fast_gnp_random_graph(1000,0.002)
        t, S, I, R = EoN.discrete_SIR(G, args = (0.6,),
                                            initial_infecteds=range(20))
        plt.plot(t,I)


    Because this sample uses the defaults, it is equivalent to a call to
    basic_discrete_SIR
    '''
    if rho is not None and initial_infecteds is not None:
        raise EoN.EoNError("cannot define both initial_infecteds and rho")

    if initial_infecteds is None:  # create initial infecteds list if not given
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(G.order() * rho))
        initial_infecteds = random.sample(G.nodes(), initial_number)
    elif G.has_node(initial_infecteds):
        initial_infecteds = [initial_infecteds]
    # else it is assumed to be a list of nodes.

    if return_full_data:
        node_history = defaultdict(lambda: ([tmin], ['S']))
        transmissions = []
        for node in initial_infecteds:
            node_history[node] = ([tmin], ['I'])
            transmissions.append((tmin - 1, None, node))
        if initial_recovereds is not None:
            for node in initial_recovereds:
                node_history[node] = ([tmin], ['R'])

    N = G.order()
    t = [tmin]
    S = [N - len(initial_infecteds)]
    I = [len(initial_infecteds)]
    R = [0]

    susceptible = defaultdict(lambda: True)
    # above line is equivalent to u.susceptible=True for all nodes.

    for u in initial_infecteds:
        susceptible[u] = False
    if initial_recovereds is not None:
        for u in initial_recovereds:
            susceptible[u] = False

    infecteds = set(initial_infecteds)

    while infecteds and t[-1] < tmax:
        new_infecteds = set()

        infector = {}  # used for returning full data.  a waste of time otherwise
        for u in infecteds:
            for v in G.neighbors(u):
                if susceptible[v] and test_transmission(u, v, *args):
                    new_infecteds.add(v)
                    susceptible[v] = False
                    infector[v] = [u]
                elif return_full_data and v in new_infecteds and test_transmission(u, v, *args):
                    # if ``v`` already infected on this round, consider if it is
                    # multiply infected this round.
                    infector[v].append(u)

        if return_full_data:
            for v in infector.keys():
                transmissions.append((t[-1], random.choice(infector[v]), v))
            next_time = t[-1] + 1
            if next_time <= tmax:
                for u in infecteds:
                    node_history[u][0].append(next_time)
                    node_history[u][1].append('R')
                for v in new_infecteds:
                    node_history[v][0].append(next_time)
                    node_history[v][1].append('I')

        infecteds = new_infecteds

        R.append(R[-1] + I[-1])
        I.append(len(infecteds))
        S.append(S[-1] - I[-1])
        t.append(t[-1] + 1)
    if not return_full_data:
        return np.array(t), np.array(S), np.array(I), \
               np.array(R)
    else:
        if sim_kwargs is None:
            sim_kwargs = {}
        return EoN.Simulation_Investigation(G, node_history, transmissions,
                                            possible_statuses=['S', 'I', 'R'],
                                            **sim_kwargs)





def basic_discrete_SIR(G, p, initial_infecteds=None,
                       initial_recovereds=None, rho=None,
                       tmin=0, tmax=float('Inf'),
                       return_full_data=False, sim_kwargs=None):
    # tested in test_basic_discrete_SIR
    r'''
    Performs simple discrete SIR simulation assuming constant transmission
    probability p.

    From figure 6.8 of Kiss, Miller, & Simon.  Please cite the book if
    using this algorithm.

    Does a simulation of the simple case of all nodes transmitting
    with probability p independently to each neighbor and then
    recovering.

    :Arguments:

    **G**    networkx Graph
            The network the disease will transmit through.

    **p** number
            transmission probability

    **initial_infecteds**  node or iterable of nodes (default None)
            if a single node, then this node is initially infected
            if an iterable, then whole set is initially infected
            if None, then choose randomly based on rho.  If rho is also
            None, a random single node is chosen.
            If both initial_infecteds and rho are assigned, then there
            is an error.

    **initial_recovereds**  as for initial_infecteds, but for initially
            recovered nodes.

    **rho**  number  (default None)
            initial fraction infected. number initially infected
            is int(round(G.order()*rho))

            The default results in a single randomly chosen initial infection.

    **tmin**  float  (default 0)
        start time

    **tmax**  float  (default infinity)
        stop time (if not extinct first).

    **return_full_data**  boolean (default False)
        Tells whether a Simulation_Investigation object should be returned.

    **sim_kwargs** keyword arguments
        Any keyword arguments to be sent to the Simulation_Investigation object
        Only relevant if ``return_full_data=True``

    :Returns:

    if return_full_data is False returns

        **t, S, I, R**    numpy arrays

        these numpy arrays give all the times observed and the number
        in each state at each time.

    Or ``if return_full_data is True`` returns
        **full_data**  Simulation_Investigation object

        from this we can extract the status history of all nodes
        We can also plot the network at given times
        and even create animations using class methods.


    :SAMPLE USE:

    ::


        import networkx as nx
        import EoN
        import matplotlib.pyplot as plt
        G = nx.fast_gnp_random_graph(1000,0.002)
        t, S, I, R = EoN.basic_discrete_SIR(G, 0.6)
        plt.plot(t,S)


        #This sample may be boring if the randomly chosen initial infection
        #doesn't trigger an epidemic.

'''

    return discrete_SIR(G, _simple_test_transmission_, (p,),
                        initial_infecteds, initial_recovereds,
                        rho, tmin, tmax, return_full_data, sim_kwargs=sim_kwargs)









def percolate_network(G, p):
    # tested indirectly in test_basic_discrete_SIR

    r'''
    Performs percolation on a network G with each edge persisting with
    probability p

    From figure 6.10 of Kiss, Miller, & Simon.  Please cite the book
    if using this algorithm.

    Performs bond percolation on the network G, keeping edges with
    probability p

    :Arguments:

    **G**    networkx Graph
        The contact network
    **p** number between 0 and 1
        the probability of keeping edge

    :Returns:

    **H**   NetworkX Graph
        A network with same nodes as G, but with each edge retained
        independently with probability p.

    :SAMPLE USE:

    ::


        import networkx as nx
        import EoN
        import matplotlib.pyplot as plt
        G = nx.fast_gnp_random_graph(1000,0.002)
        H = EoN.percolate_network(G, 0.6)

        #H is now a graph with about 60% of the edges of G
'''

    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for edge in G.edges():
        if random.random() < p:
            H.add_edge(*edge)
    return H


def _edge_exists_(u, v, H):
    r'''
    Tests if directed edge ``u``, ``v`` exists in graph ``H``.

    From figure 6.10 of Kiss, Miller, & Simon.  Please cite the book
    if using this algorithm.

    Tests whether ``H`` has an edge from ``u`` to ``v``.

    :Arguments:

    **u** node
    **v** node
    **H** networkx graph

    :Returns:

    **True**  if ``H`` has the edge
    **False**  if ``H`` does not have the edge
    '''
    return H.has_edge(u, v)





def percolation_based_discrete_SIR(G, p,
                                   initial_infecteds=None,
                                   initial_recovereds=None,
                                   rho=None, tmin=0,
                                   tmax=float('Inf'),
                                   return_full_data=False, sim_kwargs=None):
    # tested in test_basic_discrete_SIR
    r'''
    perfoms a simple SIR epidemic but using percolation as the underlying
    method.

    From figure 6.10 of Kiss, Miller, & Simon.  Please cite the book
    if using this algorithm.


    The simple case of all nodes transmitting with probability p
    independently to each neighbor and then recovering, but using a
    percolation-based approach.

    :Warning:

    You probably **shouldn't use this algorithm**.

    See ``basic_discrete_SIR`` which produces equivalent outputs.

    That algorithm will be faster than this one.

    The value of this function is that by performing many simulations we
    can see that the outputs of the two are equivalent.

    This algorithm leads to a better understanding of the theory, but
    it's not computationally efficient.


    :Arguments:

    **G**    networkx Graph
            The network the disease will transmit through.
    **p** number
            transmission probability

    **initial_infecteds**  node or iterable of nodes (default None)
            if a single node, then this node is initially infected
            if an iterable, then whole set is initially infected
            if None, then choose randomly based on rho.  If rho is also
            None, a random single node is chosen.
            If both initial_infecteds and rho are assigned, then there
            is an error.


    **initial_recovereds**  as for initial_infecteds, but initially
            recovered nodes.

    **rho**  number
            initial fraction infected. number is int(round(G.order()*rho))

    **tmin**  start time

    **tmax**  stop time (if not extinct first).  Default step is 1.

    **return_full_data**  boolean (default False)
            Tells whether a Simulation_Investigation object should be returned.

    **sim_kwargs** keyword arguments
        Any keyword arguments to be sent to the Simulation_Investigation object
        Only relevant if ``return_full_data=True``

    :Returns:

    **t, S, I, R** numpy arrays

    OR if ``return_full_data is True``:

    **full_data**    Simulation_Investigation object
        from this we can extract the status history of all nodes
        We can also plot the network at given times
        and even create animations using class methods.

    :SAMPLE USE:

    ::

        import networkx as nx
        import EoN
        import matplotlib.pyplot as plt
        G = nx.fast_gnp_random_graph(1000,0.002)
        t, S, I, R = EoN.percolation_based_discrete_SIR(G, p)
        plt.plot(t,S)

    This is equivalent to basic_discrete_epidemic (but many simulations
    may be needed before it's clear, since these are stochastic)

'''

    H = percolate_network(G, p)
    return discrete_SIR(H, test_transmission=H.has_edge,
                        initial_infecteds=initial_infecteds,
                        initial_recovereds=initial_recovereds,
                        rho=rho, tmin=tmin, tmax=tmax,
                        return_full_data=return_full_data,
                        sim_kwargs=sim_kwargs)





def estimate_SIR_prob_size(G, p):
    # tested in test_estimate_SIR_prob_size
    r'''
    Uses percolation to estimate the probability and size of epidemics
    assuming constant transmission probability p

    From figure 6.12 of Kiss, Miller, & Simon.  Please cite the
    book if using this algorithm.

    Provies an estimate of epidemic probability and size assuming a
    fixed transmission probability p.

    The estimate is found by performing bond percolation and then
    finding the largest connected component in the remaining network.

    This assumes that there is a single giant component above threshold.

    It will not be an appropriate measure if the network is made up of
    several densely connected components with very weak connections
    between these components.

    :Arguments:

    **G**    networkx Graph
            The network the disease will transmit through.
    **p** number
            transmission probability

    :Returns:

    **PE, AR**   both floats between 0 and 1
        estimates of the probability and proportion
        infected (attack rate) in epidemics
        (the two are equal, but each given for consistency with
        ``estimate_directed_SIR_prob_size``)


    :SAMPLE USE:

    ::

        import networkx as nx
        import EoN

        G = nx.fast_gnp_random_graph(1000,0.002)
        PE, AR = EoN.estimate_SIR_prob_size(G, 0.6)

    '''
    H = percolate_network(G, p)
    size = max((len(CC) for CC in nx.connected_components(H)))
    returnval = float(size) / G.order()
    return returnval, returnval





def directed_percolate_network(G, tau, gamma, weights=True):
    # indirectly tested in test_estimate_SIR_prob_size
    r'''
    performs directed percolation, assuming that transmission and recovery
    are Markovian


    From figure 6.13 of Kiss, Miller, & Simon.  Please cite the
    book if using this algorithm.

    This performs directed percolation corresponding to an SIR epidemic
    assuming that transmission is at rate tau and recovery at rate
    gamma

    :See Also:

    ``nonMarkov_directed_percolate_network`` which allows for duration and
        time to infect to come from other distributions.

    ``nonMarkov_directed_percolate_network`` which allows for more complex
        rules

    :Arguments:

    **G**    networkx Graph
        The network the disease will transmit through.
    **tau**   positive float
        transmission rate
    **gamma**   positive float
        recovery rate
    **weights**   boolean    (default True)
        if True, then includes information on time to recovery
        and delay to transmission.  If False, just the directed graph.

    :Returns:
        :
    **H**   networkx DiGraph  (directed graph)
        a u->v edge exists in H if u would transmit to v if ever
        infected.

        The edge has a time attribute (time_to_infect) which gives the
        delay from infection of u until transmission occurs.

        Each node u has a time attribute (duration) which gives the
        duration of its infectious period.

    :SAMPLE USE:


    ::

        import networkx as nx
        import EoN

        G = nx.fast_gnp_random_graph(1000,0.002)
        H = EoN.directed_percolate_network(G, 2, 1)

    '''

    # simply calls directed_percolate_network_with_timing, using markovian rules.

    def trans_time_fxn(u, v, tau):
        if tau > 0:
            return random.expovariate(tau)
        else:
            return float('Inf')

    trans_time_args = (tau,)

    def rec_time_fxn(u, gamma):
        if gamma > 0:
            return random.expovariate(gamma)
        else:
            return float('Inf')

    rec_time_args = (gamma,)

    return nonMarkov_directed_percolate_network_with_timing(G, trans_time_fxn,
                                                            rec_time_fxn,
                                                            trans_time_args,
                                                            rec_time_args,
                                                            weights=weights)


def _out_component_(G, source):
    '''
    rather than following the pseudocode in figure 6.15 of
        Kiss, Miller & Simon,
    this uses a built-in Networkx command.

    finds the set of nodes (including source) which are reachable from
    nodes in source.

    :Arguments:

    **G**  networkx Graph
        The network the disease will transmit through.
    **source** either a node or an iterable of nodes (set, list, tuple)
        The nodes from which the infections start.  We assume no node
        will ever have a name that is an iterable of other node names.
        It will run, but may not use source user expects.


    :Returns:

    **reachable_nodes** set
        the set of nodes reachable from source (including source).

    :Warning:
    if the graph G has nodes like 1, 2, 3, and (1,2,3), then a
    source of (1,2,3) is potentially ambiguous.  This algorithm will interpret
    the source as the single node (1,2,3)


    '''
    if G.has_node(source):
        source_nodes = {source}
    else:
        source_nodes = set(source)

    reachable_nodes = set().union(source_nodes)

    for node in source_nodes:
        reachable_nodes = reachable_nodes.union(
            set(nx.descendants(G, node)))

    return reachable_nodes


def _in_component_(G, target):
    r'''
    creates the _in_component_ by basically reversing _out_component_.

    :Arguments:

    **G**  networkx Graph
        The network the disease will transmit through.

    **target** a target node (or iterable of target nodes)
        The node whose infection we are interested in.

        In principle target could be an iterable, but in this case we
        would be finding those possible sources whose infection leads to
        infection of at least one target, not all.

    :Warning:

    if the graph G has nodes like 1, 2, 3, and (1,2,3), then a
    target of (1,2,3) is potentially ambiguous.  It will interpret
    the target as the single node (1,2,3)

    :Returns:

    **source_nodes** (set)
        the set of nodes (including target) from which target is
        reachable


    '''
    if G.has_node(target):
        # potential bug if for example G has  nodes like 1, 2, 3, and also
        # (1,2,3).  Then a target of (1,2,3) is ambiguous.
        target_nodes = {target}
    else:
        target_nodes = set(target)

    source_nodes = set().union(target_nodes)

    for node in target_nodes:
        source_nodes = source_nodes.union(set(nx.ancestors(G, node)))

    return source_nodes





def get_infected_nodes(G, tau, gamma, initial_infecteds=None,
                       initial_recovereds=None):
    r'''
    Finds all eventually infected nodes in an SIR simulation, through a
    percolation approach

    From figure 6.15 of Kiss, Miller, & Simon.  Please cite the book if
    using this algorithm

    Assumes that
    the intial infecteds are as given and transmission occurs with rate
    tau and recovery with rate gamma.


    Note that the output of this algorithm is stochastic.

    This code has similar run-time whether an epidemic occurs or not.
    There are much faster ways to implement an algorithm giving the same
    output, for example by actually running one of the epidemic simulation tools.

    :Warning:

    why are you using this command? If it's to better understand the
    relationship between percolation and SIR epidemics, that's fine.
    But this command IS NOT an efficient way to calculate anything.  Don't do
    it like this.  Use one of the other algorithms.  Try ``fast_SIR``,
    for example.

    :Arguments:

    **G**    networkx Graph
        The network the disease will transmit through.
    **tau**   positive float
        transmission rate
    **gamma**   positive float
        recovery rate
    **initial_infecteds**  node or iterable of nodes
        if a single node, then this node is initially infected
        if an iterable, then whole set is initially infected
        if None, then a randomly chosen node is initially infected.
    **initial_recovereds** node or iterable of nodes
        if a single node, then this node is initially recovered
        if an iterable, then whole set is initially recovered

    :Returns:

    **infected_nodes** set
        the set of nodes infected eventually in a simulation.

    :SAMPLE USE:

    ::

        import networkx as nx
        import EoN

        G = nx.fast_gnp_random_graph(1000,0.002)

        finalR = EoN.get_infected_nodes(G, 2, 1, initial_infecteds=[0, 5])

        #finds the nodes infected if 0 and 5 are the initial nodes infected
        #and tau=2, gamma=1
    '''
    if initial_recovereds is None:
        initial_recovereds = set()
    elif G.has_node(initial_recovereds):
        initial_recovereds = set([initial_recovereds])
    else:
        initial_recovereds = set(initial_recovereds)
    if initial_infecteds is None:
        while True:
            node = random.choice(G.nodes())
            if node not in initial_recovereds:
                break
        initial_infecteds = set([node])
    elif G.has_node(initial_infecteds):
        initial_infecteds = set([initial_infecteds])
    else:
        initial_infecteds = set(initial_infecteds)
    if initial_infecteds.intersection(initial_recovereds):
        raise EoN.EoNError("initial infecteds and initial recovereds overlap")
    H = directed_percolate_network(G, tau, gamma)
    for node in initial_recovereds:
        H.remove_node(node)
    infected_nodes = _out_component_(H, initial_infecteds)
    return infected_nodes





def estimate_directed_SIR_prob_size(G, tau, gamma):
    # tested in test_estimate_SIR_prob_size
    '''
    Predicts probability and attack rate assuming continuous-time Markovian SIR disease on network G

    From figure 6.17 of Kiss, Miller, & Simon.  Please cite the book if
    using this algorithm



    :See Also:

    ``estimate_nonMarkov_SIR_prob_size`` which handles nonMarkovian versions

    :Arguments:

    **G**    networkx Graph
        The network the disease will transmit through.
    **tau**   positive float
        transmission rate
    **gamma**   positive float
        recovery rate

    :Returns:

    **PE, AR**  numbers (between 0 and 1)
        Estimates of epidemic probability and attack rate found by
        performing directed percolation, finding largest strongly
        connected component and finding its in/out components.

    :SAMPLE USE:

    ::


        import networkx as nx
        import EoN

        G = nx.fast_gnp_random_graph(1000,0.003)
        PE, AR = EoN.estimate_directed_SIR_prob_size(G, 2, 1)

    '''

    H = directed_percolate_network(G, tau, gamma)
    return estimate_SIR_prob_size_from_dir_perc(H)





def estimate_SIR_prob_size_from_dir_perc(H):
    # indirectly tested in test_estimate_SIR_prob_size
    r'''
    Estimates probability and size of SIR epidemics for an input network after directed percolation

    From figure 6.17 of Kiss, Miller, & Simon.  Please cite the book if
    using this algorithm

    :Arguments:

    **H**  directed graph
        The outcome of directed percolation on the contact network G

    :Returns:

    **PE, AR**  numbers
        Estimates of epidemic probability and attack rate found by
        finding largest strongly connected component and finding in/out
        components.

    :SAMPLE USE:


    ::

        import networkx as nx
        import EoN

        G = nx.fast_gnp_random_graph(1000,0.003)
        H = some_user_defined_operation_to_do_percolation(G, argument)
        PE, AR = EoN.estimate_SIR_prob_size_from_dir_perc(H)

    '''

    Hscc = max((nx.strongly_connected_components(H)), key=len)
    u = list(Hscc)[0]  # random.choice(Hscc)
    inC = _in_component_(H, u)  # includes both H_{IN} and H_{SCC}
    outC = _out_component_(H, u)  # includes both H_{OUT} and H_{SCC}
    N = float(H.order())
    PE = len(inC) / N
    AR = len(outC) / N
    return PE, AR





def estimate_nonMarkov_SIR_prob_size_with_timing(G,
                                                 trans_time_fxn,
                                                 rec_time_fxn,
                                                 trans_time_args=(),
                                                 rec_time_args=()):
    '''
    estimates probability and size for user-input transmission and recovery time functions.

    :Arguments:

    **G** Networkx Graph
        the input graph
    **trans_time_fxn** function
        trans_time_fxn(u, v, *trans_time_args)
        returns the delay from u's infection to transmission to v.
    **rec_time_fxn** function
        rec_time_fxn(u, *rec_time_args)
        returns the delay from u's infection to its recovery.
    **trans_time_args** tuple
        any additional arguments required by trans_time_fxn.  For example
        weights of nodes.
    **rec_time_args** tuple
        any additional arguments required by rec_time_fxn

    :Returns:

    **PE, AR**  numbers (between 0 and 1)
        Estimates of epidemic probability and attack rate found by
        finding largest strongly connected component and finding in/out
        components.

    :SAMPLE USE:


    ::

        #mimicking the standard version with transmission rate tau
        #and recovery rate gamma
        #equivalent to
        #PE, AR = EoN.estimate_SIR_prob_size(G, tau, gamma)

        import networkx as nx
        import EoN
        import random
        from collections import defaultdict

        G=nx.fast_gnp_random_graph(1000,0.002)

        tau = 2

        gamma = 1

        def trans_time_fxn(u,v, tau):

            return random.expovariate(tau)

        def rec_time_fxn(u, gamma):

            return random.expovariate(gamma)

        PE, AR = EoN.estimate_nonMarkov_SIR_prob_size(G,
                                                    trans_time_fxn=trans_time_fxn,
                                                    rec_time_fxn = rec_time_fxn,
                                                    trans_time_args = (tau,),
                                                    rec_time_args = (gamma,)
                                                    )

    '''

    H = nonMarkov_directed_percolate_network_with_timing(G,
                                                         trans_time_fxn,
                                                         rec_time_fxn,
                                                         trans_time_args,
                                                         rec_time_args)
    return estimate_SIR_prob_size_from_dir_perc(H)





def estimate_nonMarkov_SIR_prob_size(G, xi, zeta, transmission):
    '''
    Predicts epidemic probability and size using nonMarkov_directed_percolate_network.

    This is not directly described in Kiss, Miller, & Simon, but is based on (fig 6.18).

    :Warning:
    You probably DON'T REALLY WANT TO USE THIS.
    Check if estimate_nonMarkov_prob_size_with_timing fits your needs better.

    :Arguments:

    **G** (networkx Graph)
            The input graph

    **xi** dict
        xi[u] gives all necessary information to determine what u's
        infectiousness is.
    **zeta** dict
        zeta[v] gives everything needed about v's susceptibility

    **transmission** user-defined function
        transmission(xi[u], zeta[v]) determines whether u transmits to
        v.  Returns True or False depending on whether the transmission would
        happen

    :Returns:

    **PE, AR**  numbers (between 0 and 1)
        Estimates of epidemic probability and attack rate found by
        finding largest strongly connected component and finding in/out
        components.

    :SAMPLE USE:


    ::

        #mimicking the standard version with transmission rate tau
        #and recovery rate gamma

        import networkx as nx
        import EoN
        import random
        from collections import defaultdict

        G=nx.fast_gnp_random_graph(1000,0.002)
        tau = 2
        gamma = 1

        xi = {node:random.expovariate(gamma) for node in G.nodes()}
        #xi[node] is duration of infection of node.

        zeta = defaultdict(lambda : tau) #every node has zeta=tau, so same
                                        #transmission rate

        def my_transmission(infection_duration, trans_rate):
            #infect if duration is longer than time to infection.
            if infection_duration > random.expovariate(trans_rate):
                return True
            else:
                return False

        PE, AR = EoN.estimate_nonMarkov_SIR_prob_size(G, xi, zeta,
                                                        my_transmission)


    '''

    H = nonMarkov_directed_percolate_network(G, xi, zeta, transmission)
    return estimate_SIR_prob_size_from_dir_perc(H)





def nonMarkov_directed_percolate_network_with_timing(G,
                                                     trans_time_fxn,
                                                     rec_time_fxn,
                                                     trans_time_args=(),
                                                     rec_time_args=(),
                                                     weights=True):
    r'''
    Performs directed percolation on G for user-specified transmission time
    and recovery time distributions.

    A generalization of figure 6.13 of Kiss, Miller & Simon




    :See Also:

    **directed_percolate_network**
    if it's just a constant transmission and recovery rate.

    **nonMarkov_directed_percolate_network**
    if your rule for creating the percolated network cannot be expressed
    as simply calculating durations and delays until transmission.


    :Arguments:

    The arguments are very much like in fast_nonMarkov_SIR

    **G**  Networkx Graph
        the input graph
    **trans_time_fxn** user-defined function
        returns the delay from u's infection to transmission to v.

        delay=trans_time_fxn(u, v, *trans_time_args)

    **rec_time_fxn** user-defined function
        returns the duration of from u's infection (delay to its recovery).

        duration = rec_time_fxn(u, *rec_time_args)


        Note:
            if delay == duration, we assume infection happens.

    **trans_time_args** tuple
        any additional arguments required by trans_time_fxn.  For example
        weights of nodes.
    **rec_time_args** tuple
        any additional arguments required by rec_time_fxn
    **weights** boolean
        if true, then return directed network with the delay and duration as
        weights.

    :Returns:

    **H** directed graph.

    The returned graph is built by assigning each node an infection duration
    and then each edge (in each direction) a delay until transmission.
    If   delay<duration  it adds directed edge to ``H``.

    if weights is True, then ``H`` contains the duration and delays
    as weights.  Else it's just a directed graph.

    '''
    H = nx.DiGraph()
    if weights:
        for u in G.nodes():
            duration = rec_time_fxn(u, *rec_time_args)
            H.add_node(u, duration=duration)
            for v in G.neighbors(u):
                delay = trans_time_fxn(u, v, *trans_time_args)
                if delay <= duration:
                    H.add_edge(u, v, delay_to_infection=delay)
    else:
        for u in G.nodes():
            duration = rec_time_fxn(u, *rec_time_args)
            H.add_node(u)
            for v in G.neighbors(u):
                delay = trans_time_fxn(u, v, *trans_time_args)
                if delay <= duration:
                    H.add_edge(u, v)
    return H





def nonMarkov_directed_percolate_network(G, xi, zeta, transmission):
    r'''
    performs directed percolation on a network following user-specified rules.

    From figure 6.18 of Kiss, Miller, & Simon.
    Please cite the book if using this algorithm.

    This algorithm is particularly intended for a case where the duration and
    delays from infection to transmission are somehow related to one another.

    :Warning:
    You probably **shouldn't use this**.
    Check if nonMarkov_directed_percolate_with_timing fits your needs better.

    :See Also:

    ``nonMarkov_directed_percolate_network_with_timing``

        if your rule for creating the percolated network is based on calculating
        a recovery time for each node and then calculating a separate
        transmission time for the edges this will be better.

    ``directed_percolate_network``

        if it's just a constant transmission and recovery rate.



    :Arguments:

    **G** networkx Graph
        The input graph

    **xi** dict
        xi[u] gives all necessary information to determine what us
        infectiousness is.
    **zeta** dict
        zeta[v] gives everything needed about vs susceptibility

    **transmission** user-defined function
        transmission(xi[u], zeta[v]) determines whether u transmits to v.

        returns True if transmission happens and False if it does not

    :Returns:

    **H** networkx DiGraph (directed graph)
        Edge (u,v) exists in H if disease will transmit given the opportunity.

    :SAMPLE USE:

    for now, I'm being lazy.
    Look at the sample for estimate_nonMarkov_SIR_prob_size to infer it.

'''
    H = nx.DiGraph()
    for u in G.nodes():
        H.add_node(u)
        for v in G.neighbors(u):
            if transmission(xi[u], zeta[v]):
                H.add_edge(u, v)
    return H


### Code starting here does event-driven simulations ###


def _find_trans_and_rec_delays_SIR_(node, sus_neighbors, trans_time_fxn,
                                    rec_time_fxn, trans_time_args=(),
                                    rec_time_args=()):
    rec_delay = rec_time_fxn(node, *rec_time_args)
    trans_delay = {}
    for target in sus_neighbors:
        trans_delay[target] = trans_time_fxn(node, target, *trans_time_args)
    return trans_delay, rec_delay


def _process_trans_SIR_(time, G, source, target, times, S, I, R, Q, status,
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
    S, I, R : lists
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

    if status[target] == 'S':  # nothing happens if already infected.
        status[target] = 'I'
        times.append(time)
        transmissions.append((time, source, target))
        S.append(S[-1] - 1)  # one less susceptible
        I.append(I[-1] + 1)  # one more infected
        R.append(R[-1])  # no change to recovered

        suscep_neighbors = [v for v in G.neighbors(target) if status[v] == 'S']

        trans_delay, rec_delay = trans_and_rec_time_fxn(target, suscep_neighbors,
                                                        *trans_and_rec_time_args)

        rec_time[target] = time + rec_delay
        if rec_time[target] <= Q.tmax:
            Q.add(rec_time[target], _process_rec_SIR_,
                  args=(target, times, S, I, R, status))
        for v in trans_delay:
            inf_time = time + trans_delay[v]
            if inf_time <= rec_time[target] and inf_time < pred_inf_time[v] and inf_time <= Q.tmax:
                Q.add(inf_time, _process_trans_SIR_,
                      args=(G, target, v, times, S, I, R, Q,
                            status, rec_time, pred_inf_time,
                            transmissions, trans_and_rec_time_fxn,
                            trans_and_rec_time_args
                            )
                      )
                pred_inf_time[v] = inf_time


def _process_rec_SIR_(time, node, times, S, I, R, status):
    r'''From figure A.3 of Kiss, Miller, & Simon.  Please cite the
    book if using this algorithm.

    :Arguments:

        event : event
            has details on node and time
        times : list
            list of times at which events have happened
        S, I, R : lists
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
    times.append(time)
    S.append(S[-1])  # no change to number susceptible
    I.append(I[-1] - 1)  # one less infected
    R.append(R[-1] + 1)  # one more recovered
    status[node] = 'R'


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
             recovery_weight=None, return_full_data=False, sim_kwargs=None):
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

    **times, S, I, R** numpy arrays

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
        t, S, I, R = EoN.fast_SIR(G, tau, gamma,
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
                return random.expovariate(rate)
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
                                  sim_kwargs=sim_kwargs)
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
                       return_full_data=False, sim_kwargs=None):
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

    **times, S, I, R** numpy arrays

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

        t, S, I, R = EoN.fast_nonMarkov_SIR(G,
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
            initial_number = int(round(G.order() * rho))
        initial_infecteds = random.sample(G.nodes(), initial_number)
    elif G.has_node(initial_infecteds):
        initial_infecteds = [initial_infecteds]
    # else it is assumed to be a list of nodes.

    times, S, I, R = ([tmin], [G.order()], [0], [0])
    transmissions = []

    for u in initial_infecteds:
        pred_inf_time[u] = tmin
        Q.add(tmin, _process_trans_SIR_, args=(G, None, u, times, S, I, R, Q,
                                               status, rec_time,
                                               pred_inf_time, transmissions,
                                               trans_and_rec_time_fxn,
                                               trans_and_rec_time_args
                                               )
              )

    # Note that when finally infected, pred_inf_time is correct
    # and rec_time is correct.
    # So if return_full_data is true, these are correct

    while Q:  # all the work is done in this while loop.
        Q.pop_and_run()

    # the initial infections were treated as ordinary infection events at
    # time 0.
    # So each initial infection added an entry at time 0 to lists.
    # We'd like to get rid these excess events.
    times = times[len(initial_infecteds):]
    S = S[len(initial_infecteds):]
    I = I[len(initial_infecteds):]
    R = R[len(initial_infecteds):]

    if not return_full_data:
        return np.array(times), np.array(S), np.array(I), \
               np.array(R)
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
                                            possible_statuses=['S', 'I', 'R'],
                                            **sim_kwargs)











