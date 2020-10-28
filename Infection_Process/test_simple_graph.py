import Infection_Process.SIR_Basic as EoN
import matplotlib.pyplot as plt
from Graph_Generator.single_school_generator import School
import numpy as np
import scipy as sp
import networkx as nx

from Testing_Strategies import Simple_Random


school=School("LA_simple",1,10,3,4)
G=school.network
removal_rate=1.
transmission_scale=1.
initial_fraction_infected=0.07


t, S, E, I, T, R, Isolated, status = EoN.fast_SIR(G, gamma=removal_rate, tau=transmission_scale,
                                                  transmission_weight="weight",
                                                  rho=initial_fraction_infected,
                                                  all_test_times=np.linspace(0, 300, 500), test_args=(400,),
                                                  test_func=Simple_Random.fully_random_test,
                                                  weighted_test=False, school=school, isolate=False)
