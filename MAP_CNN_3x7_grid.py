import numpy as np
from time import time
import random
import matplotlib.pyplot as plt
import csv
import math
import copy
from numpy.core.numeric import NaN
import pandas as pd
import pickle
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()


data_dir = './GWA/AltamontCA/'
file = 'custom_wind-speed_100m.xyz'

random_seed=137
random.seed(random_seed)

fpn = open(data_dir + file,'r')
lines = fpn.readlines()

u = np.array([line.split()[2] for line in lines])
u = u.astype(float)

# construct x and y
dx, dy = 200, 350
nx, ny = 3, 7
n= nx*ny # *Size of grid
nx0, ny0 = 96, 102
wind_nx, wind_ny= 96, 102 #NOTE: I added this part for now to sample a subsection of the windmap to match the turbine_mask size -Manasi
x = np.linspace(0, nx0*dx, nx0)
y = np.linspace(0, ny0*dy, ny0)
print('x len', len(x))
print('y len', len(y))
#u = np.reshape(u, (nx, ny), order='F')
u = np.reshape(u, (wind_nx, wind_ny), order='F') #NOTE: and this -Manasi
u0 = u  #store the original wind map
# sampling subsection of u
tmp_x= 9   #random.choice(range(wind_nx-nx)) #NOTE: and these -Manasi
tmp_y= 23   #random.choice(range(wind_ny-ny))
print("tmp_x: ", tmp_x)
print("tmp_y: ", tmp_y)
u= u[tmp_x:tmp_x+nx, tmp_y:tmp_y+ny] #NOTE: and this -Manasi
#print("u size: ", u.shape)
x_sub = x[tmp_x:tmp_x+nx]
y_sub = y[tmp_y:tmp_y+ny]
#print('x=', x, 'y=', y)
#print('x sub=', x_sub, 'y sub=', y_sub)

# plot the wind map
yv, xv = np.meshgrid(y_sub, x_sub)
plt.figure()
plt.contourf(xv, yv, u,cmap = 'Spectral_r')
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
cbar = plt.colorbar()
cbar.set_label('u (m/s)')
#plt.show()

"""-------------------------------------------------***Parameters to change***------------------------------------"""
MAX_TURBINES= 5
print("n: ", n)
print("MAX_TURBINES: ", MAX_TURBINES)
#_S_= math.comb(n, MAX_TURBINES)
dir = './3x7grid/'
print('------dir=-------', dir)
generate_all_states= False
gen_NN_model= False
"""----------------------------------------------------------------------------------------------------------------"""

# to convert tuple locations to characters and vice-versa
grid_to_char= dict()
char_to_grid= dict()
actions= []
k= 'a'
for i in range(nx):
    for j in range(ny):
        grid_to_char[(i, j)]= k
        char_to_grid[k]= (i, j)
        actions.append(k)
        k = chr(ord(k) + 1)
STOP_ACTION= k
actions.append(k)

print("Actions: ", actions)
action_to_corr_number= dict()
corr_number_to_action= dict()
for i in range(len(actions)):
    action_to_corr_number[actions[i]]= i
    corr_number_to_action[i]= actions[i]

# to find all states
def generate_all_states_from_scratch():
    state_locs= actions[:n]
    print("State_locs: ", state_locs)
    states= []
    count= 0
    all_states_to_index= dict()
    all_indices_to_states= dict()

    tmp_state= ''
    states.append(tmp_state)
    all_states_to_index[tmp_state]= count 
    all_indices_to_states[count]= tmp_state
    count += 1
    for i in range(n): # loop 1
        tmp_state_i= state_locs[i]
        """print("i: ", i)
        print("tmp state: ", tmp_state)"""
        states.append(tmp_state_i)
        all_states_to_index[tmp_state_i]= count 
        all_indices_to_states[count]= tmp_state_i
        count += 1

        j= i+1
        while j < n: # loop 2
            tmp_state_j = tmp_state_i + state_locs[j]
            """if len(tmp_state) != len(set(tmp_state)):
                print("ERRRRRRRRORRRRRRRRRRRRR: ", tmp_state)
            print("j: ", j)
            print("tmp state: ", tmp_state_j)"""
            states.append(tmp_state_j)
            all_states_to_index[tmp_state_j]= count
            all_indices_to_states[count]= tmp_state_j
            count += 1

            k= j+1
            while k < n: # loop 3
                tmp_state_k = tmp_state_j + state_locs[k]
                """if len(tmp_state) != len(set(tmp_state)):
                    print("ERRRRRRRRORRRRRRRRRRRRR: ", tmp_state)
                print("k: ", k)
                print("tmp state: ", tmp_state_k)"""
                states.append(tmp_state_k)
                all_states_to_index[tmp_state_k]= count
                all_indices_to_states[count]= tmp_state_k
                count += 1

                l= k+1
                while l < n: # loop 4 i, j, k, l, m, o
                    tmp_state_l = tmp_state_k + state_locs[l]
                    """if len(tmp_state) != len(set(tmp_state)):
                        print("ERRRRRRRRORRRRRRRRRRRRR: ", tmp_state)
                    print("l: ", l)
                    print("tmp state: ", tmp_state_l)"""
                    states.append(tmp_state_l)
                    all_states_to_index[tmp_state_l]= count
                    all_indices_to_states[count]= tmp_state_l
                    count += 1

                    m= l+1
                    while m < n: # loop 5
                        #print(count)
                        tmp_state_m = tmp_state_l + state_locs[m]
                        """if len(tmp_state) != len(set(tmp_state)):
                            print("ERRRRRRRRORRRRRRRRRRRRR: ", tmp_state)          
                        print("m: ", m)
                        print("tmp state: ", tmp_state_m)"""
                        states.append(tmp_state_m)
                        all_states_to_index[tmp_state_m]= count
                        all_indices_to_states[count]= tmp_state_m
                        count += 1
                        
                        m += 1
                        #break
                    l += 1
                    #break
                k += 1
                #break
            j += 1
            #break
        #i += 1
        #break

    _S_= len(states)

    f = open(dir+'states_full_list', 'wb')
    pickle.dump(states, f)
    f.close()

    f = open(dir+'all_states_to_index', 'wb')
    pickle.dump(all_states_to_index, f)
    f.close()

    f = open(dir+'all_indices_to_states', 'wb')
    pickle.dump(all_indices_to_states, f)
    f.close()
    return _S_, states, all_states_to_index, all_indices_to_states

def read_in_all_states():
    f = open(dir+'states_full_list', 'rb')
    states_full_list= pickle.load(f)
    f.close()
    _S_= len(states_full_list)

    f = open(dir+'all_states_to_index', 'rb')
    all_states_to_index= pickle.load(f)
    f.close()

    f = open(dir+'all_indices_to_states', 'rb')
    all_indices_to_states= pickle.load(f)
    f.close()
    return _S_, states_full_list, all_states_to_index, all_indices_to_states

if generate_all_states:
    _S_, states_full_list, all_states_to_index, all_indices_to_states= generate_all_states_from_scratch()
else:
    _S_, states_full_list, all_states_to_index, all_indices_to_states= read_in_all_states()




class MAP_class():
    def __init__(self, x, y, u, nx, ny, num_D, std_factor):

        self.raw_x = x
        self.raw_y = y
        self.u = u # raw wind data, DO NOT ACCESS, use get_current_wind() instead!

        #### wake model params
        self.D = 125 # wind turbine size
        self.num_D = num_D #the farthest distance in number of turbine diameters that is a turbine wake can reach
        self.std_factor = std_factor
        ####

        self.nx = nx # size of sampled grid
        self.ny = ny
        self.worldsize = (nx, ny)
        self.turbine_mask = np.zeros(self.worldsize)

        self.world = None
        self.x = None
        self.y = None
        # sample_world will set self.world, self.x and self.y
        self.WORLD_0 = self.sample_world().copy() # the initial wind map. do not update


    def sample_world(self, dialation=0, random_seed=137):
        # create an nxn array of the wind data
        # dialation is yet to be implemented
        random.seed(random_seed)
        nx, ny = self.get_world_shape()
        x = self.raw_x
        y = self.raw_y

        sample_x = random.randint(0,len(x)-nx)
        sample_y = random.randint(0, len(y) - ny)
        print("world location, {}, {}".format(sample_x,sample_y))
        self.world = self.u[sample_x:sample_x+nx, sample_y:sample_y+ny]
        self.x = x[sample_x:sample_x+nx]
        self.y = y[sample_y:sample_y+ny]
        return self.u[sample_x:sample_x+nx, sample_y:sample_y+ny]

    def add_wake(self, wake_matrix):
        # require a nxn matrix that approximate the wake effect
        self.world -= wake_matrix
   
    def get_current_wind(self):
        return self.world

    def get_init_wind(self):
        return self.WORLD_0

    def get_turbine_location(self):
        return self.turbine_mask

    def add_turbine(self, loc):
        x, y = loc
        if self.has_turbine(x,y) is False:
            self.turbine_mask[x,y] =1
            return True
        else:
            return False

    def remove_turbine(self, loc):
        x, y = loc
        if self.has_turbine(x, y):
            self.turbine_mask[x,y] =0
            return True
        else:
            return False

    def has_turbine(self, x, y):
        if self.turbine_mask[x,y] ==0:
            return False
        else:
            return True

    def get_world_shape(self):
        return self.worldsize

    def reset_map(self):
        self.world = self.WORLD_0.copy()
        self.turbine_mask = np.zeros(self.worldsize)



def compute_wake(MAP, new_loc):
    '''
    @ parameters:
      MAP: global wind information with previous wind info before adding the new turbine
      new_loc: (x,y) index of the new turbine
    @ return:
      wake_mat: the wake (velocity deficit) matrix with a negative value at the new turbine location, and zeros elsewhere
    '''
      
    k_wake = 0.075  # wake decay constant for onshore wind
    if (not np.any(MAP.turbine_mask)):  # there is no turbine yet
        return np.zeros((MAP.nx, MAP.ny))
    
    old_locs = np.argwhere(MAP.get_turbine_location())

    wake_mat = np.zeros((MAP.nx, MAP.ny))
    u_wake = []
    wind_map = MAP.get_current_wind()
    
    for loc in old_locs:
        dist = np.linalg.norm(
            np.array([MAP.x[new_loc[0]], MAP.y[new_loc[1]]]) - np.array([MAP.x[loc[0]], MAP.y[loc[1]]]))
        # the turbine of influence should be within 5D a distance of the new turbine, and should be placed to the left of it
        if dist < MAP.num_D * MAP.D and loc[0] < new_loc[0]:
            mu = wind_map[loc[0], loc[1]] * (1 - MAP.D / (MAP.D + 2 * k_wake * dist) ** 2)
            sigma = MAP.std_factor * mu  # PZ: shrunk sigma to guarantee the sampled wake value is the same sign as wind_map[loc[0], loc[1]]
            try:
                u_wake.append(np.random.normal(mu, sigma))
            except ValueError as e:
                print(e)
        if dist < MAP.num_D * MAP.D and loc[0] > new_loc[0]: # modify the wind speed in the wake region of the NEW turbine
            mu = wind_map[new_loc[0], new_loc[1]] * (1 - MAP.D / (MAP.D + 2 * k_wake * dist) ** 2)
            sigma = MAP.std_factor * mu
            u_wake_on_old = np.random.normal(mu, sigma)
            wake_mat[loc[0], loc[1]] = u_wake_on_old if np.abs(u_wake_on_old) < np.abs(wind_map[loc[0], loc[1]]) \
                                       else wind_map[loc[0], loc[1]]
            
            
    if u_wake:  #NOTE: I changed this to make sure it compiles, let me know if it's incorrect -Manasi       
        wake_candidate = np.max(np.abs(u_wake))
        candidate_ind = np.argmax(np.abs(u_wake))
        wake_mat[new_loc[0], new_loc[1]] = wake_candidate*np.sign(u_wake[candidate_ind]) \
                                           if wake_candidate < np.abs(wind_map[new_loc[0], new_loc[1]]) else wind_map[new_loc[0], new_loc[1]]
    return wake_mat


def total_power(MAP):
    # calculate total power generated by the existing turbines
    turbine_locs = np.argwhere(MAP.get_turbine_location())
    wind_map = MAP.get_current_wind()
    total_p = 0
    if turbine_locs.size > 0:
        for loc in turbine_locs:
            u = wind_map[loc[0], loc[1]]
            total_p+= power_generated(u)
        return total_p
    else:
        # empty world with no turbine
        return 0


def add_turbine_and_compute_reward(MAP, new_loc):
    power_before = total_power(MAP)
    MAP.add_wake(compute_wake(MAP, new_loc))  # update the wind map by adding wake
    MAP.add_turbine(new_loc)
    reward = total_power(MAP) - power_before
    if reward==0 and MAP.get_turbine_location().sum()==1:
        alert = 1
    return reward - 2500


def power_generated(u):
    '''
    u: wind speed at the new turbine location (after wake is deducted)
    '''
    D = 125
    rho = 1.225  # kg/m^3, air density
    return 1/2*np.pi/4*D**2*rho*u**3  # power generated at the turbine with u


# Generating dataset

def grid_to_flattened_state(MAP):
    flattened_rep= []
    list_args= np.argwhere(MAP.turbine_mask==1)
    for i in range(len(list_args)):
        flattened_rep.append(str(grid_to_char[(list_args[i][0], list_args[i][1])]))
    return "".join(flattened_rep)


def flattened_state_to_grid(flattened_rep, MAP):
    flattened_rep= [int(x) for x in flattened_rep]
    grid= np.zeros((MAP.nx, MAP.ny))
    for i in flattened_rep:
        tmp_pos= char_to_grid[i]
        grid[tmp_pos[0], tmp_pos[1]]= 1
    return grid


def generate_random_exploration_data(x, y, u, nx, ny, num_D, std_factor, filename= 'dataset', limit=None):
    # count is used to show how many samples we generated so far
    MAP = MAP_class(x, y, u, nx, ny, num_D, std_factor)
    prob_stop= 0.0
    prob_stop_limit= 0.001
    prob_step_size= 50
    VERY_NEG_REWARD= -1000000 # for placing turbine where there is already a turbine
    count = 0

    prob_stop_increment= (prob_stop_limit-prob_stop)/prob_step_size
    action_probs= [(1.0-prob_stop)/n]*n # initial prob distribution
    action_probs += [prob_stop]
    random.seed(random_seed)
    fields= ['s', 'a', 'r', 'sp']
    
    with open(filename+'.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)

    all_visited_states= set()
    #turbines_added= 0

    while(limit is None or count<=limit):
        print(count)
        count += 1
        
        action = random.choices(actions, action_probs)[0]

        # last action to stop adding turbines
        if action == actions[-1]: 
            print("Wind Turbine adding has stopped!")
            reward= 0
            with open(filename+'.csv', 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([current_state, action, reward, current_state])
            prob_stop = 0
            MAP.reset_map()
            continue
        
        # to slowly build up probability of stopping
        if prob_stop < prob_stop_limit:
            prob_stop += prob_stop_increment
            action_probs= [(1.0-prob_stop)/n]*n # initial prob distribution
            action_probs += [prob_stop]

        # corresponding location to update for a specific action
        current_state= grid_to_flattened_state(MAP)
        all_visited_states.add(current_state)
        new_x, new_y= char_to_grid[action]
        if MAP.has_turbine(new_x, new_y):
            reward= VERY_NEG_REWARD
        else:
            reward= add_turbine_and_compute_reward(MAP, (new_x, new_y))
            #turbines_added += 1
        new_state= grid_to_flattened_state(MAP)
  
        # write dataset entry, we need current flattened state, chosen action, resulting reward (power) and next state
        with open(filename+'.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([current_state, action, reward, new_state])

        # if grid has been fully filled, we can remove all turbines (clear it) to explore states anew
        if len(new_state)== MAX_TURBINES:
            MAP.reset_map()

    return count


# Running Q-learning
class QLearning:
  def __init__(self, γ, Q, α):
    self.γ = γ
    self.Q = Q
    self.α = α

def read_in_df(filename):
    temp_df= pd.read_csv(filename+'.csv', dtype=str, keep_default_na=False)
    #temp_df['a'] = temp_df['a'].astype(int)
    temp_df['r'] = temp_df['r'].astype(float)
    return temp_df

def add_state_index_column_to_df(df):
    i= 0
    s_indices= [all_states_to_index[x] for x in df['s']]
    sp_indices= [all_states_to_index[x] for x in df['sp']]
    df.insert(loc=1, column='s_index', value=s_indices)
    df.insert(loc=5, column='sp_index', value=sp_indices)
    return df

"""---------------------------------***CNN STUFF***--------------------------------------"""
def visited_states(df):
    tmp_visited= set(list(df['s'])+list(df['sp']))
    unvisited= set(states_full_list) - tmp_visited
    visited= sorted(list(tmp_visited))
    unvisited= sorted(list(unvisited))
    if len(visited) < _S_:
        print("Not all states explored in dataset!")
    #print("States explored: ", len(visited))
    _S_, states_full_list, all_states_to_index, all_indices_to_states
    return visited, unvisited

def indexed_Q(Q, indices_list):
    return Q[indices_list]

def get_dataset(Q, indices_list):
    X= []
    for s in indices_list:
        for a in range(_𝒜_):
            X.append([s, a])

    X= np.array(X)
    #X= X.reshape(len(X), 1)
    y= Q.flatten()
    y= y.reshape(len(y), 1)
    
    scalarX.fit(X)
    scalarY.fit(y)
    X = scalarX.transform(X)
    y = scalarY.transform(y)
    return X, y


# code for next 2 functions is taken from https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
def get_model(n_inputs, n_outputs): # get the model    
    model = Sequential()
    model.add(Dense(4, input_dim=n_inputs, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model


def train_NN_model(X, y): # evaluate a model using repeated k-fold cross-validation
    n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
    cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=1)
	# enumerate folds
    for train_ix, test_ix in cv.split(X):
		# prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
		# define model
        model = get_model(n_inputs, n_outputs)

		# fit model
        t3= time()
        model.fit(X_train, y_train, verbose=0, epochs=100)
        print("time to fit for 1 k cross: ", (time()-t3)/60)

		# evaluate model on test set
        mse = model.evaluate(X_test, y_test, verbose=0)
		# store result
        print('>%.3f' % mse)
    return model


def predict_Qvalues(s, a, NN_model):
    X= np.array([[s, a]])
    X = scalarX.transform(X)
    y= NN_model.predict(X)
    y = scalarY.inverse_transform(y)
    return y[0][0]

def update_Qvalues(s, NN_model, model):
    for a in range(_𝒜_):
        model.Q[s, a]= predict_Qvalues(s, a, NN_model)
    return model


def generate_NN_model(Q_model):
    # Note: visited=train, unvisited= test
    #print("model.Q 1 size: ", Q_model.Q.shape)
    visited_indices= [all_states_to_index[x] for x in visited]
    X_visited, y_visited= get_dataset(indexed_Q(Q_model.Q, visited_indices), visited_indices)
    if gen_NN_model:
        NN_model= train_NN_model(X_visited, y_visited)
        NN_model.save(dir+'NN_model_3x7')
    else:
        NN_model= load_model(dir+'NN_model_3x7')
    return NN_model

def simulate(df, model, h):
    for j in range(h):
        # for visited
        for i in df.index:
            model= update(model, df['s_index'][i], action_to_corr_number[df['a'][i]], df['r'][i], df['sp_index'][i])
        
    # for unvisited 
    if len(unvisited) != 0:
        NN_model= generate_NN_model(model)
        print("PRINTING UNVISITED NODES: ")
        for i in range(len(unvisited)):
            print(i)
            s= all_states_to_index[unvisited[i]]
            model= update_Qvalues(s, NN_model, model)
        
    return extract_policy(model)
"""---------------------------------------------------------------------------------------"""


def update(model, s, a, r, sp):
    γ, Q, α = model.γ, model.Q, model.α
    Q[s,a] += α*(r + γ*max(Q[sp,:]) - Q[s,a])
    return model

def update_random(model, s, a, r, sp):
    γ, Q, α = model.γ, model.Q, model.α
    Q[s,a] += α*(r + γ*np.mean(Q[sp,:]) - Q[s,a])
    return model


def extract_policy(model):
    Q= model.Q
    U_π= np.max(Q, axis=1)
    π= np.argmax(Q, axis=1)
    π[U_π <= 0] = 9  # if utility is less then action should be stop
    return U_π, π


"""def simulate(df, model, h):
    for j in range(h):
        # for visited
        for i in df.index:
            model= update(model, df['s_index'][i], action_to_corr_number[df['a'][i]], df['r'][i], df['sp_index'][i])
    return extract_policy(model)"""


def write_to_file(U_π, π, filename):
    np.savetxt(filename+".utility", U_π)
    np.savetxt(filename+".policy", π, fmt='%i')

def write_policy_with_states(U_π, π, filename, state_index_to_flat_rep):
    with open(filename+".policy", 'w+') as f:
        f.write('state,state_index,greedy_action,utility\n')
        for i in range(U_π.size):
            f.write('{},{},{},{}\n'.format(state_index_to_flat_rep[i], i, π[i], U_π[i]))


def run_Q_learning(filename, model, h):
    df= read_in_df(filename)
    #flat_rep_to_state_index, state_index_to_flat_rep= flat_rep_and_state_index(df)
    df= add_state_index_column_to_df(df)
    U_π, π= simulate(df, model, h)
    write_to_file(U_π, π, filename)
    write_policy_with_states(U_π, π, filename+"_with_state", all_indices_to_states)


def get_random_policy_utility(filename, model, h):
    df = read_in_df(filename)
    #flat_rep_to_state_index, state_index_to_flat_rep = flat_rep_and_state_index(df)
    df = add_state_index_column_to_df(df)
    for j in range(h):
        # for visited
        for i in df.index:
            model = update_random(model, df['s_index'][i], action_to_corr_number[df['a'][i]], df['r'][i], df['sp_index'][i])
    Q = model.Q
    U_π = np.mean(Q, axis=1)
    np.savetxt(filename+"_random" + ".utility", U_π)


def extract_sequence_from_policy_file(filename): # returns a sequence of actions (tuple locations to add turbines)
    # output list of actions
    df= pd.read_csv(filename, dtype=str, keep_default_na=False)
    df['state_index'] = df['state_index'].astype(int)
    df['greedy_action'] = df['greedy_action'].astype(int)
    df['utility'] = df['utility'].astype(float)

    # initial
    state= df.iloc[0][0]
    greedy_action= corr_number_to_action[df.iloc[0][2]]
    actions= [char_to_grid[greedy_action]]

    while(True):
        state= "".join(sorted(state + str(greedy_action)))
        state_index= all_states_to_index[state]
        temp_df= df.loc[(df['state_index'] == state_index)]
        print("temp_df.iloc[0, 2]: ", temp_df.iloc[0, 2])
        greedy_action= corr_number_to_action[temp_df.iloc[0, 2]]
        if len(state) == MAX_TURBINES:
            break
        if greedy_action in state:
            print("Attempting to add to location of existing turbine-- error!")
            break
        if greedy_action == STOP_ACTION:
            break
        actions.append(char_to_grid[greedy_action])
    return actions

def compute_wind_of_final_layout(turbine_locs, u, nx, ny, x, y, num_D, std_factor):
    '''
    @param: turbine_locs: the action_sequence returned by extract_sequence_from_policy_file
            u: the sampled SUB-grid to run algorithms on
            nx, ny, x, y: the number of points and coordinates of the SUBGRID
            num_D: the farthest distance in number of turbine diameters that is a turbine wake can reach
    @return: the wind map with wake effects of all turbines added
    '''
    wind_map = u
    locs = np.array(action_sequence)
    k_wake = 0.075  # wake decay constant for onshore wind
    D = 125

    wake_mat = np.zeros((nx, ny))

    # for each grid point, iterate through all turbines
    for i in range(nx):
        for j in range(ny):
            u_wake = []
            for loc in turbine_locs:
                if loc[0] == len(x)-1 or loc[0] >= i:
                    continue
                dist = np.linalg.norm(np.array([x[loc[0]], y[loc[1]] ]) - np.array([x[i], y[j]]) )
                # the turbine of influence should be within 5D a distance of the new turbine, and should be placed to the left of it
                print('i=',i,'j=',j,'loc=',loc,'dist=',dist)
                if dist < num_D * D:
                    mu = wind_map[loc[0], loc[1]] * (1 - D / (D + 2 * k_wake * dist) ** 2)
                    sigma = std_factor * mu  # PZ: shrunk sigma to guarantee the sampled wake value is the same sign as wind_map[loc[0], loc[1]]
                    u_wake.append(np.random.normal(mu, sigma))

            if u_wake:  #NOTE: I changed this to make sure it compiles, let me know if it's incorrect -Manasi       
                wake_candidate = np.max(np.abs(u_wake))
                candidate_ind = np.argmax(np.abs(u_wake))
                wake_mat[i, j] = wake_candidate*np.sign(u_wake[candidate_ind]) \
                                  if wake_candidate < np.abs(wind_map[i, j]) else wind_map[i, j]
    return wind_map - wake_mat  

def compute_total_power(wind_map, xi_turb, yi_turb):
    '''
    @ param: the final wind map and turbine indices in the grid
    @ return: the total power   
    '''            
    D = 125
    rho = 1.225  # kg/m^3, air density
    return 1/2*np.pi/4*D**2*rho*np.sum(wind_map[xi_turb, yi_turb]**3)

def plot_turbine_layout(action_sequence, u, nx, ny, x, y, num_D, std_factor, fig_name):
    '''
    @ param: action_sequence returned by extract_sequence_from_policy_file
             tmp_x, tmp_y: the chosen (x, y) index to sample the wind map
             nx, ny: the number of points in x and y in the sampled wind map   
             x, y: the x and y coordinates of the SUBGRID  
             num_D: the farthest distance in number of turbine diameters that is a turbine wake can reach    
             std_factor: the factor multiplied by mu to get the std of the wake 
    @ plot the layout on a grid
    '''
    wind_map = compute_wind_of_final_layout(action_sequence, u, nx, ny, x, y, num_D, std_factor)
    locs = np.array(action_sequence)
    xi_turb = locs[:, 0]
    yi_turb = locs[:, 1]

    # get and plot the total power
    # fpn = open(filename + '.utility', 'r')
    # lines = fpn.readlines()
    total_power = compute_total_power(wind_map, xi_turb, yi_turb)
    print('*** total power=', total_power)
    print('wind map', wind_map)

    yv, xv = np.meshgrid(y, x)
    labels = np.arange(len(xi_turb)) + 1

    ax = plt.subplot()
    plt.contourf(xv, yv, wind_map, cmap = 'Spectral_r', alpha=0.7)

    plt.plot(x[xi_turb], y[yi_turb], 'bo', markersize = 20)
    plt.grid(True)

    
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('y (m)', fontsize=16)
    plt.xticks(x)
    plt.yticks(y)
    #plt.xticks(np.concatenate(( [min(x)+1], x, [max(x)+1])))
    #plt.yticks(np.concatenate(( [min(y)+1], y, [max(y)+1])))
    plt.xlim(np.min(x)-100, np.max(x)+100)
    plt.ylim(np.min(y)-100, np.max(y)+100)

    #cbar = plt.colorbar()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # label the turbines by the sequence they are added
    for i, label in enumerate(labels):
        ax.annotate(str(label), (x[xi_turb[i]], y[yi_turb[i]]), xytext = (x[xi_turb[i]]-8, y[yi_turb[i]]-10),\
                     color='w', fontsize=15)
    #cbar.set_label('u (m/s)')
    plt.savefig(fig_name)
    plt.show()

# Main
count = 0
## parameters to modify ###
filename= dir + 'dataset'
generate_data = False
run_learning = True
run_random = True
num_D = 5
std_factor = 0.3
###########################

if generate_data:
    generate_random_exploration_data(x, y, u0, nx, ny, num_D, std_factor, filename, limit=100000)
else:
    df= read_in_df(filename)
    visited, unvisited= visited_states(df)
    print("Num_visited: ", len(visited))
    print("Num unvisited: ", len(unvisited))
    #flat_rep_to_state_index, state_index_to_flat_rep= flat_rep_and_state_index(df)

    if run_learning:
        _𝒜_= n+1
        Q= np.zeros((_S_, _𝒜_))
        γ= 1
        α= 0.01
        Q_model= QLearning(γ, Q, α)

        #run Q-learning
        #filename= "dataset"
        h= 10

        t1= time()
        if run_random:
            get_random_policy_utility(filename, Q_model, h)
        else:
            run_Q_learning(filename, Q_model, h)
        t2= time()
        print("Total time (s): ", (t2-t1))

    # Extract policy
    policy_file= filename + '_with_state.policy'
    print("Extracting sequence of locations to add turbine at...")
    action_sequence= extract_sequence_from_policy_file(policy_file)
    for a in action_sequence:
        print(a)

    fig_name = dir + 'final_layout.png'
    plot_turbine_layout(action_sequence, u,nx, ny, x_sub, y_sub, num_D, std_factor, fig_name)