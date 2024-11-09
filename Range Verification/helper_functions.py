import onnx
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from onnx import numpy_helper
from z3 import *
import random
import numpy as np
import tensorflow as tf
import nengo
import nengo_dl
from tensorflow.keras import layers, models, datasets
import warnings
import gurobipy as gp
from gurobipy import GRB

def extract_model_params_tf(nn_model):
    weights_list = []
    biases_list = []
    
    weights_biases = nn_model.get_weights()


    for i in range(0, len(weights_biases), 2):
        weights_list.append(weights_biases[i])
        biases_list.append(weights_biases[i + 1])
        
    weights = [arr.T for arr in weights_list]
    biases = [arr.T for arr in biases_list]
    
    return weights, biases

def set_random_weights(model):
    current_weights = model.get_weights()
    new_weights = [np.random.rand(*w.shape) for w in current_weights]
    model.set_weights(new_weights)
    
    return model


def summon_gurobi(dec, eqn, log):
    all_enc = dec + eqn
    file_path = "Gurobi_encodings_3L_ACC_sim_random.txt"
    with open(file_path, "w") as file:
        for value in all_enc:
            file.write(str(value) + "\n")
#     import time
#     start = time.time()
    model=gp.Model("Encodings")

#     model.Params.MIPGap = 0.000001
    model.Params.LogToConsole = log
    model.setParam('TimeLimit', 8*60*60)
    model.Params.SolutionLimit = 1

    try:
        f = open(file_path,"r")
        try:
            for l in f:
                exec(l)
        finally:
            f.close()
    except IOError:
        pass
#     print('Time taken:', end-start)
    model.optimize()
    
    return model
    

def find_max_output_gurobi(w,b,layer,no,time_steps, cond, input_bounds):##################################################################
    global equations, declare
    equations=[]
    declare=[]
    inputs=w[0].shape[1]
#     solver=Solver()
#     solver.reset()
#     solver.set(timeout=240000000)

    # Declarations
    for time in range(1,time_steps+1):
        for num in range(1,inputs+1):
            declare.append(f"A0_{num}_{time} = model.addVar(name='A0_{num}_{time}')")
            equations.append(f"model.addConstr(A0_{num}_{time}>={input_bounds[num-1][0]})")
            equations.append(f"model.addConstr(A0_{num}_{time}<={input_bounds[num-1][1]})")

    for i in range(1,layer):
        for j in range(1,len(w[i-1])+1):
            declare.append(f"P{i}_{j}_0 = model.addVar(name='P{i}_{j}_0')")

    declare.append(f"P{layer}_{no}_0 = model.addVar(name='P{layer}_{no}_0')")

    for time in range(1,time_steps+1):
        for i in range(1,layer):
            for j in range(1,len(w[i-1])+1):
                declare.append(f"X{i}_{j}_{time} = model.addVar(name='X{i}_{j}_{time}')")
                declare.append(f"P{i}_{j}_{time} = model.addVar(lb=-GRB.INFINITY, name='P{i}_{j}_{time}')")
                declare.append(f"S{i}_{j}_{time} = model.addVar(lb=-GRB.INFINITY, name='S{i}_{j}_{time}')")
                declare.append(f"q{i}_{j}_{time} = model.addVar(vtype=gp.GRB.BINARY, name='q{i}_{j}_{time}')")
                declare.append(f"A{i}_{j}_{time} = model.addVar(vtype=gp.GRB.INTEGER, name='A{i}_{j}_{time}')")


    for time in range(1,time_steps+1):
        declare.append(f"X{layer}_{no}_{time} = model.addVar(name='X{layer}_{no}_{time}')")
        declare.append(f"P{layer}_{no}_{time} = model.addVar(lb=-GRB.INFINITY, name='P{layer}_{no}_{time}')")
        declare.append(f"S{layer}_{no}_{time} = model.addVar(lb=-GRB.INFINITY, name='S{layer}_{no}_{time}')")
        declare.append(f"q{layer}_{no}_{time} = model.addVar(vtype=gp.GRB.BINARY, name='q{layer}_{no}_{time}')")
        declare.append(f"A{layer}_{no}_{time} = model.addVar(vtype=gp.GRB.INTEGER, name='A{layer}_{no}_{time}')")
    
    # Encodings
    for i in range(1,layer):
        for j in range(1,len(w[i-1])+1):
            equations.append(f"model.addConstr(P{i}_{j}_0== 0)")
    equations.append(f"model.addConstr(P{layer}_{no}_0== 0)")

    thresh = 1
    lamb = 1
    M = 99999999
    epsilon = 0.00001

    for time in range(1,time_steps+1):
        for i in range(1,layer):
            for j in range(1,len(w[i-1])+1):
                equations.append(f"model.addConstr(S{i}_{j}_{time} + P{i}_{j}_{time-1} + {M}* q{i}_{j}_{time} >= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(S{i}_{j}_{time} + P{i}_{j}_{time-1} <= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(X{i}_{j}_{time} >= 0)")
                equations.append(f"model.addConstr(X{i}_{j}_{time} <= {M}*(1-q{i}_{j}_{time}))")
                equations.append(f"model.addConstr(A{i}_{j}_{time} <= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(A{i}_{j}_{time} + 1 >= X{i}_{j}_{time} + {epsilon})")
                equations.append(f"model.addConstr(P{i}_{j}_{time} == P{i}_{j}_{time-1} + S{i}_{j}_{time} - A{i}_{j}_{time})")

                equation = f'S{i}_{j}_{time} == ('
                for k in range(len(w[i-1][0])):
                    if(k!=0):
                        equation += f' + '
                    equation+=f'({w[i-1][j-1][k]:.4f} * A{i-1}_{k+1}_{time})'
                equations.append(f"model.addConstr({equation}) + {b[i-1][j-1]})")

    for time in range(1,time_steps+1):
        equations.append(f"model.addConstr(P{layer}_{no}_{time} == P{layer}_{no}_{time-1} + S{layer}_{no}_{time})")
        equation = f'S{layer}_{no}_{time} == ('
        for k in range(len(w[layer-1][0])):
            if(k!=0):
                equation += f' + '
            equation+=f'(({w[layer-1][no-1][k]:.4f}) * A{layer-1}_{k+1}_{time})'
        equations.append(f"model.addConstr({equation})+ {b[layer-1][no-1]})")

    cond=f''
#     for time in range(1,time_steps+1):
#         if(time!=1):
#             cond+=f'+'
#         cond+=f'A{layer}_{no}_{time}'
#     equations.append(f'model.setObjective({cond}, gp.GRB.MAXIMIZE)')

    equations.append(f'model.setObjective(P{layer}_{no}_{time_steps}, gp.GRB.MAXIMIZE)')
    
    
#     equations.append(f"model.addConstr({cond}>={output_range[1]})")
#     equations.append(f'model.setObjective(0, gp.GRB.MAXIMIZE)')

#     equations.append(f'solver.add(Or({cond}<{output_range[0]},{cond}>{output_range[1]}))')
    return equations, declare


def can_go_above(w,b,layer,no,time_steps, cond, input_bounds):
    global equations, declare
    equations=[]
    declare=[]
    inputs=w[0].shape[1]
#     solver=Solver()
#     solver.reset()
#     solver.set(timeout=240000000)

    # Declarations
    for time in range(1,time_steps+1):
        for num in range(1,inputs+1):
            declare.append(f"A0_{num}_{time} = model.addVar(name='A0_{num}_{time}')")
            equations.append(f"model.addConstr(A0_{num}_{time}>={input_bounds[num-1][0]})")
            equations.append(f"model.addConstr(A0_{num}_{time}<={input_bounds[num-1][1]})")

    for i in range(1,layer):
        for j in range(1,len(w[i-1])+1):
            declare.append(f"P{i}_{j}_0 = model.addVar(name='P{i}_{j}_0')")

    declare.append(f"P{layer}_{no}_0 = model.addVar(name='P{layer}_{no}_0')")

    for time in range(1,time_steps+1):
        for i in range(1,layer):
            for j in range(1,len(w[i-1])+1):
                declare.append(f"X{i}_{j}_{time} = model.addVar(name='X{i}_{j}_{time}')")
                declare.append(f"P{i}_{j}_{time} = model.addVar(lb=-GRB.INFINITY, name='P{i}_{j}_{time}')")
                declare.append(f"S{i}_{j}_{time} = model.addVar(lb=-GRB.INFINITY, name='S{i}_{j}_{time}')")
                declare.append(f"q{i}_{j}_{time} = model.addVar(vtype=gp.GRB.BINARY, name='q{i}_{j}_{time}')")
                declare.append(f"A{i}_{j}_{time} = model.addVar(vtype=gp.GRB.INTEGER, name='A{i}_{j}_{time}')")


    for time in range(1,time_steps+1):
        declare.append(f"X{layer}_{no}_{time} = model.addVar(name='X{layer}_{no}_{time}')")
        declare.append(f"P{layer}_{no}_{time} = model.addVar(lb=-GRB.INFINITY, name='P{layer}_{no}_{time}')")
        declare.append(f"S{layer}_{no}_{time} = model.addVar(lb=-GRB.INFINITY, name='S{layer}_{no}_{time}')")
        declare.append(f"q{layer}_{no}_{time} = model.addVar(vtype=gp.GRB.BINARY, name='q{layer}_{no}_{time}')")
        declare.append(f"A{layer}_{no}_{time} = model.addVar(vtype=gp.GRB.INTEGER, name='A{layer}_{no}_{time}')")
    
    # Encodings
    for i in range(1,layer):
        for j in range(1,len(w[i-1])+1):
            equations.append(f"model.addConstr(P{i}_{j}_0== 0)")
    equations.append(f"model.addConstr(P{layer}_{no}_0== 0)")

    thresh = 1
    lamb = 1
    M = 99999999
    epsilon = 0.00001

    for time in range(1,time_steps+1):
        for i in range(1,layer):
            for j in range(1,len(w[i-1])+1):
                equations.append(f"model.addConstr(S{i}_{j}_{time} + P{i}_{j}_{time-1} + {M}* q{i}_{j}_{time} >= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(S{i}_{j}_{time} + P{i}_{j}_{time-1} <= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(X{i}_{j}_{time} >= 0)")
                equations.append(f"model.addConstr(X{i}_{j}_{time} <= {M}*(1-q{i}_{j}_{time}))")
                equations.append(f"model.addConstr(A{i}_{j}_{time} <= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(A{i}_{j}_{time} + 1 >= X{i}_{j}_{time} + {epsilon})")
                equations.append(f"model.addConstr(P{i}_{j}_{time} == P{i}_{j}_{time-1} + S{i}_{j}_{time} - A{i}_{j}_{time})")
                equation = f'S{i}_{j}_{time} == ('
                for k in range(len(w[i-1][0])):
                    if(k!=0):
                        equation += f' + '
                    equation+=f'({w[i-1][j-1][k]:.4f} * A{i-1}_{k+1}_{time})'
                equations.append(f"model.addConstr({equation}) + {b[i-1][j-1]})")

    for time in range(1,time_steps+1):
        equations.append(f"model.addConstr(P{layer}_{no}_{time} == P{layer}_{no}_{time-1} + S{layer}_{no}_{time})")

        equation = f'S{layer}_{no}_{time} == ('
        for k in range(len(w[layer-1][0])):
            if(k!=0):
                equation += f' + '
            equation+=f'(({w[layer-1][no-1][k]:.4f}) * A{layer-1}_{k+1}_{time})'
        equations.append(f"model.addConstr({equation})+ {b[layer-1][no-1]})")
    
    equations.append(f"model.addConstr(P{layer}_{no}_{time_steps}>={cond[0]*time_steps})")
    equations.append(f'model.setObjective(0, gp.GRB.MAXIMIZE)')

    return equations, declare


def can_go_below(w,b,layer,no,time_steps, cond, input_bounds):
    global equations, declare
    equations=[]
    declare=[]
    inputs=w[0].shape[1]
#     solver=Solver()
#     solver.reset()
#     solver.set(timeout=240000000)

    # Declarations
    for time in range(1,time_steps+1):
        for num in range(1,inputs+1):
            declare.append(f"A0_{num}_{time} = model.addVar(name='A0_{num}_{time}')")
            equations.append(f"model.addConstr(A0_{num}_{time}>={input_bounds[num-1][0]})")
            equations.append(f"model.addConstr(A0_{num}_{time}<={input_bounds[num-1][1]})")

    for i in range(1,layer):
        for j in range(1,len(w[i-1])+1):
            declare.append(f"P{i}_{j}_0 = model.addVar(name='P{i}_{j}_0')")

    declare.append(f"P{layer}_{no}_0 = model.addVar(name='P{layer}_{no}_0')")

    for time in range(1,time_steps+1):
        for i in range(1,layer):
            for j in range(1,len(w[i-1])+1):
                declare.append(f"X{i}_{j}_{time} = model.addVar(name='X{i}_{j}_{time}')")
                declare.append(f"P{i}_{j}_{time} = model.addVar(lb=-GRB.INFINITY, name='P{i}_{j}_{time}')")
                declare.append(f"S{i}_{j}_{time} = model.addVar(lb=-GRB.INFINITY, name='S{i}_{j}_{time}')")
                declare.append(f"q{i}_{j}_{time} = model.addVar(vtype=gp.GRB.BINARY, name='q{i}_{j}_{time}')")
                declare.append(f"A{i}_{j}_{time} = model.addVar(vtype=gp.GRB.INTEGER, name='A{i}_{j}_{time}')")


    for time in range(1,time_steps+1):
        declare.append(f"X{layer}_{no}_{time} = model.addVar(name='X{layer}_{no}_{time}')")
        declare.append(f"P{layer}_{no}_{time} = model.addVar(lb=-GRB.INFINITY, name='P{layer}_{no}_{time}')")
        declare.append(f"S{layer}_{no}_{time} = model.addVar(lb=-GRB.INFINITY, name='S{layer}_{no}_{time}')")
        declare.append(f"q{layer}_{no}_{time} = model.addVar(vtype=gp.GRB.BINARY, name='q{layer}_{no}_{time}')")
        declare.append(f"A{layer}_{no}_{time} = model.addVar(vtype=gp.GRB.INTEGER, name='A{layer}_{no}_{time}')")
    

        


    # Encodings
    for i in range(1,layer):
        for j in range(1,len(w[i-1])+1):
            equations.append(f"model.addConstr(P{i}_{j}_0== 0)")
    equations.append(f"model.addConstr(P{layer}_{no}_0== 0)")

    thresh = 1
    lamb = 1
    M = 99999999
    epsilon = 0.00001

    for time in range(1,time_steps+1):
        for i in range(1,layer):
            for j in range(1,len(w[i-1])+1):
                equations.append(f"model.addConstr(S{i}_{j}_{time} + P{i}_{j}_{time-1} + {M}* q{i}_{j}_{time} >= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(S{i}_{j}_{time} + P{i}_{j}_{time-1} <= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(X{i}_{j}_{time} >= 0)")
                equations.append(f"model.addConstr(X{i}_{j}_{time} <= {M}*(1-q{i}_{j}_{time}))")
                equations.append(f"model.addConstr(A{i}_{j}_{time} <= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(A{i}_{j}_{time} + 1 >= X{i}_{j}_{time} + {epsilon})")
                equations.append(f"model.addConstr(P{i}_{j}_{time} == P{i}_{j}_{time-1} + S{i}_{j}_{time} - A{i}_{j}_{time})")
                equation = f'S{i}_{j}_{time} == ('
                for k in range(len(w[i-1][0])):
                    if(k!=0):
                        equation += f' + '
                    equation+=f'({w[i-1][j-1][k]:.4f} * A{i-1}_{k+1}_{time})'
                equations.append(f"model.addConstr({equation}) + {b[i-1][j-1]})")

    for time in range(1,time_steps+1):
        equations.append(f"model.addConstr(P{layer}_{no}_{time} == P{layer}_{no}_{time-1} + S{layer}_{no}_{time})")

        equation = f'S{layer}_{no}_{time} == ('
        for k in range(len(w[layer-1][0])):
            if(k!=0):
                equation += f' + '
            equation+=f'(({w[layer-1][no-1][k]:.4f}) * A{layer-1}_{k+1}_{time})'
        equations.append(f"model.addConstr({equation})+ {b[layer-1][no-1]})")
    
    equations.append(f"model.addConstr(P{layer}_{no}_{time_steps}<={cond[0]*time_steps})")
    equations.append(f'model.setObjective(0, gp.GRB.MAXIMIZE)')

    return equations, declare



def is_within_bounds(w,b,layer,no,time_steps, cond, input_bounds):
    global equations, declare
    equations=[]
    declare=[]
    inputs=w[0].shape[1]
#     solver=Solver()
#     solver.reset()
#     solver.set(timeout=240000000)

    # Declarations
    for time in range(1,time_steps+1):
        for num in range(1,inputs+1):
            declare.append(f"A0_{num}_{time} = model.addVar(name='A0_{num}_{time}')")
            equations.append(f"model.addConstr(A0_{num}_{time}>={input_bounds[num-1][0]})")
            equations.append(f"model.addConstr(A0_{num}_{time}<={input_bounds[num-1][1]})")

    for i in range(1,layer):
        for j in range(1,len(w[i-1])+1):
            declare.append(f"P{i}_{j}_0 = model.addVar(name='P{i}_{j}_0')")

    declare.append(f"P{layer}_{no}_0 = model.addVar(name='P{layer}_{no}_0')")

    for time in range(1,time_steps+1):
        for i in range(1,layer):
            for j in range(1,len(w[i-1])+1):
                declare.append(f"X{i}_{j}_{time} = model.addVar(name='X{i}_{j}_{time}')")
                declare.append(f"P{i}_{j}_{time} = model.addVar(lb=-GRB.INFINITY, name='P{i}_{j}_{time}')")
                declare.append(f"S{i}_{j}_{time} = model.addVar(lb=-GRB.INFINITY, name='S{i}_{j}_{time}')")
                declare.append(f"q{i}_{j}_{time} = model.addVar(vtype=gp.GRB.BINARY, name='q{i}_{j}_{time}')")
                declare.append(f"A{i}_{j}_{time} = model.addVar(vtype=gp.GRB.INTEGER, name='A{i}_{j}_{time}')")


    for time in range(1,time_steps+1):
        declare.append(f"X{layer}_{no}_{time} = model.addVar(name='X{layer}_{no}_{time}')")
        declare.append(f"P{layer}_{no}_{time} = model.addVar(lb=-GRB.INFINITY, name='P{layer}_{no}_{time}')")
        declare.append(f"S{layer}_{no}_{time} = model.addVar(lb=-GRB.INFINITY, name='S{layer}_{no}_{time}')")
        declare.append(f"q{layer}_{no}_{time} = model.addVar(vtype=gp.GRB.BINARY, name='q{layer}_{no}_{time}')")
        declare.append(f"A{layer}_{no}_{time} = model.addVar(vtype=gp.GRB.INTEGER, name='A{layer}_{no}_{time}')")
    

        


    # Encodings
    for i in range(1,layer):
        for j in range(1,len(w[i-1])+1):
            equations.append(f"model.addConstr(P{i}_{j}_0== 0)")
    equations.append(f"model.addConstr(P{layer}_{no}_0== 0)")

    thresh = 1
    lamb = 1
    M = 99999999
    epsilon = 0.00001

    for time in range(1,time_steps+1):
        for i in range(1,layer):
            for j in range(1,len(w[i-1])+1):
                equations.append(f"model.addConstr(S{i}_{j}_{time} + P{i}_{j}_{time-1} + {M}* q{i}_{j}_{time} >= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(S{i}_{j}_{time} + P{i}_{j}_{time-1} <= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(X{i}_{j}_{time} >= 0)")
                equations.append(f"model.addConstr(X{i}_{j}_{time} <= {M}*(1-q{i}_{j}_{time}))")
                equations.append(f"model.addConstr(A{i}_{j}_{time} <= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(A{i}_{j}_{time} + 1 >= X{i}_{j}_{time} + {epsilon})")
                equations.append(f"model.addConstr(P{i}_{j}_{time} == P{i}_{j}_{time-1} + S{i}_{j}_{time} - A{i}_{j}_{time})")
                equation = f'S{i}_{j}_{time} == ('
                for k in range(len(w[i-1][0])):
                    if(k!=0):
                        equation += f' + '
                    equation+=f'({w[i-1][j-1][k]:.4f} * A{i-1}_{k+1}_{time})'
                equations.append(f"model.addConstr({equation}) + {b[i-1][j-1]})")

    for time in range(1,time_steps+1):
        equations.append(f"model.addConstr(P{layer}_{no}_{time} == P{layer}_{no}_{time-1} + S{layer}_{no}_{time})")

        equation = f'S{layer}_{no}_{time} == ('
        for k in range(len(w[layer-1][0])):
            if(k!=0):
                equation += f' + '
            equation+=f'(({w[layer-1][no-1][k]:.4f}) * A{layer-1}_{k+1}_{time})'
        equations.append(f"model.addConstr({equation})+ {b[layer-1][no-1]})")

    declare.append(f'lb=model.addVar(vtype=gp.GRB.BINARY, name="lb")')
    declare.append(f'ub=model.addVar(vtype=gp.GRB.BINARY, name="ub")')
    equations.append(f'model.addConstr({cond[0]*time_steps} - P{layer}_{no}_{time_steps}  <= {M} * lb)')
    equations.append(f'model.addConstr(P{layer}_{no}_{time_steps} - {cond[0]*time_steps} <= {M} * (1-lb))')
    
    equations.append(f'model.addConstr(P{layer}_{no}_{time_steps} - {cond[1]*time_steps} <= {M} * ub)')
    equations.append(f'model.addConstr({cond[1]*time_steps} - P{layer}_{no}_{time_steps} <= {M} * (1-ub))')

    equations.append(f'model.addConstr(lb + ub >= 1)')

    
#     equations.append(f"model.addConstr(P{layer}_{no}_{time_steps}<={cond[0]*{time_steps})")



    equations.append(f'model.setObjective(0, gp.GRB.MAXIMIZE)')

#     equations.append(f'solver.add(Or({cond}<{output_range[0]},{cond}>{output_range[1]}))')
    return equations, declare


def simulate_with_gurobi(w,b,layer, no,time_steps, input_bounds):
    global equations, declare
    equations=[]
    declare=[]
    inputs=w[0].shape[1]

#     solver.set(timeout=240000000)

    # Declarations
    for time in range(1,time_steps+1):
        for num in range(1,inputs+1):
            declare.append(f"A0_{num}_{time} = model.addVar(lb=-GRB.INFINITY, name='A0_{num}_{time}')")
            equations.append(f"model.addConstr(A0_{num}_{time}=={input_bounds[0][time-1][num-1]})")
#             equations.append(f"model.addConstr(A0_{num}_{time}<={input_bounds[num-1][1]})")

    for i in range(1,layer):
        for j in range(1,len(w[i-1])+1):
            declare.append(f"P{i}_{j}_0 = model.addVar(name='P{i}_{j}_0')")

    declare.append(f"P{layer}_{no}_0 = model.addVar(name='P{layer}_{no}_0')")

    for time in range(1,time_steps+1):
        for i in range(1,layer):
            for j in range(1,len(w[i-1])+1):
                declare.append(f"X{i}_{j}_{time} = model.addVar(name='X{i}_{j}_{time}')")
                declare.append(f"P{i}_{j}_{time} = model.addVar(lb=-GRB.INFINITY, name='P{i}_{j}_{time}')")
                declare.append(f"S{i}_{j}_{time} = model.addVar(lb=-GRB.INFINITY, name='S{i}_{j}_{time}')")
                declare.append(f"q{i}_{j}_{time} = model.addVar(vtype=gp.GRB.BINARY, name='q{i}_{j}_{time}')")
                declare.append(f"A{i}_{j}_{time} = model.addVar(vtype=gp.GRB.INTEGER, name='A{i}_{j}_{time}')")


    for time in range(1,time_steps+1):
        declare.append(f"X{layer}_{no}_{time} = model.addVar(name='X{layer}_{no}_{time}')")
        declare.append(f"P{layer}_{no}_{time} = model.addVar(lb=-GRB.INFINITY, name='P{layer}_{no}_{time}')")
        declare.append(f"S{layer}_{no}_{time} = model.addVar(lb=-GRB.INFINITY, name='S{layer}_{no}_{time}')")
        declare.append(f"q{layer}_{no}_{time} = model.addVar(vtype=gp.GRB.BINARY, name='q{layer}_{no}_{time}')")
        declare.append(f"A{layer}_{no}_{time} = model.addVar(vtype=gp.GRB.INTEGER, name='A{layer}_{no}_{time}')")
    
    # Encodings
    for i in range(1,layer):
        for j in range(1,len(w[i-1])+1):
            equations.append(f"model.addConstr(P{i}_{j}_0== 0)")
    equations.append(f"model.addConstr(P{layer}_{no}_0== 0)")

    thresh = 1
    lamb = 1
    M = 9999999
    epsilon = 0.00001

    for time in range(1,time_steps+1):
        for i in range(1,layer):
            for j in range(1,len(w[i-1])+1):
                equations.append(f"model.addConstr(S{i}_{j}_{time} + P{i}_{j}_{time-1} + {M}* q{i}_{j}_{time} >= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(S{i}_{j}_{time} + P{i}_{j}_{time-1} <= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(X{i}_{j}_{time} >= 0)")
                equations.append(f"model.addConstr(X{i}_{j}_{time} <= {M}*(1-q{i}_{j}_{time}))")
                equations.append(f"model.addConstr(A{i}_{j}_{time} <= X{i}_{j}_{time})")
                equations.append(f"model.addConstr(A{i}_{j}_{time} + 1 >= X{i}_{j}_{time} + {epsilon})")
                equations.append(f"model.addConstr(P{i}_{j}_{time} == P{i}_{j}_{time-1} + S{i}_{j}_{time} - A{i}_{j}_{time})")

                equation = f'S{i}_{j}_{time} == ('
                for k in range(len(w[i-1][0])):
                    if(k!=0):
                        equation += f' + '
                    equation+=f'({w[i-1][j-1][k]} * A{i-1}_{k+1}_{time})'
                equations.append(f"model.addConstr({equation}) + {b[i-1][j-1]})")

    for time in range(1,time_steps+1):
        equations.append(f"model.addConstr(P{layer}_{no}_{time} == P{layer}_{no}_{time-1} + S{layer}_{no}_{time})")

        equation = f'S{layer}_{no}_{time} == ('
        for k in range(len(w[layer-1][0])):
            if(k!=0):
                equation += f' + '
            equation+=f'(({w[layer-1][no-1][k]}) * A{layer-1}_{k+1}_{time})'
        equations.append(f"model.addConstr({equation})+ {b[layer-1][no-1]})")

#     cond=f''
#     for time in range(1,time_steps+1):
#         if(time!=1):
#             cond+=f'+'
#         cond+=f'A{layer}_{no}_{time}'
#     equations.append(f'model.setObjective({cond}, gp.GRB.MAXIMIZE)')

#     equations.append(f'model.setObjective(P{layer}_{no}_{time_steps}, gp.GRB.MAXIMIZE)')
    
    
#     equations.append(f"model.addConstr(P{layer}_{no}_{time_steps}>={cond[1]*{time_steps})")
    equations.append(f'model.setObjective(0, gp.GRB.MAXIMIZE)')

# #     equations.append(f'solver.add(Or({cond}<{output_range[0]},{cond}>{output_range[1]}))')
    return equations, declare




