import numpy as np
import scipy.optimize as op
import gym

def Sigmoid(z):
    return 1/(1 + np.exp(-z));

def Gradient(theta,x,y):
    m,n = x.shape; 
    x = np.c_[np.ones(m),x]
    m,n = x.shape;    
    theta = theta.reshape((n,1));
    y = y.reshape((m,1))
    sigmoid_x_theta = Sigmoid(x.dot(theta));
    grad = ((x.T).dot(sigmoid_x_theta-y))/m;
    return grad.flatten();

def CostFunc(theta,x,y):
    m,n = x.shape; 
    x = np.c_[np.ones(m),x]
    m,n = x.shape;
    theta = theta.reshape((n,1));
    y = y.reshape((m,1));
    term1 = np.log(Sigmoid(x.dot(theta)));
    term2 = np.log(1-Sigmoid(x.dot(theta)));
    term1 = term1.reshape((m,1))
    term2 = term2.reshape((m,1))
    term = y * term1 + (1 - y) * term2;
    J = -((np.sum(term))/m);
    return J;
def action_pred(x,theta):
    
    store  = x[3]*theta[4]+ x[2]*theta[3]+ x[1]*theta[2]+ x[0]*theta[1]+ theta[0]
    val = Sigmoid(store)
    if val > 0.5:
        return 1;
    else:
        return 0;
#setting cartpole environment
env = gym.make('CartPole-v0')
Y = [1]
env.render()
n = 5
observation = env.reset()
optimal_theta = np.array([6.41304837, -18.21855429, 140.87736902, -20.26560133,  -1.98519816]);

for i in range(1,100):
    state = env.reset()
    Y = [1]
    #print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("********opt theta****")
    print(optimal_theta)

    for j in range(2,50):

        #print(optimal_theta)    
        env.render()
        t = action_pred(observation,optimal_theta)
        observation_new, reward, done, info = env.step(t)
        #store every observation and corresponding expected output
        state = np.vstack([state, observation_new])
        
        if observation[2] < 0:
            if observation_new[2] - observation[2] < 0:
                action_pre = 1 - t
                Y = np.vstack([Y,action_pre])
            else:
                Y = np.vstack([Y,t])    
        if observation[2] > 0:
            if observation_new[2] - observation[2] > 0:
                action_pre = 1 - t
                Y = np.vstack([Y,action_pre])
            else :
                Y = np.vstack([Y,t])
        
        if done == True:
            #print("@@@@@@@......@@@@@@@@")
            env.reset()
            #break

    print("!!!!")
    print(state)            
    print(CostFunc(optimal_theta,state,Y))        
        #print("******obs**********")
        #print(observation)
        #observation = observation_new
        #print("******reward**********")
        #print(reward)
        #print("********action_pred************")
        #print(action_pred(observation,optimal_theta))
        #print("**********...Y...***********")
        #print( Y[j-1])
        
            #observation1 = env.reset()
    #print("optimising theta")        
    #Result = op.minimize(fun = CostFunc, 
    #                             x0 = optimal_theta, 
    #                             args = (state, Y),
    #                             method = 'TNC',
    #                             jac = Gradient);
    #print("in traning loop of domain theory with iteration no. " + str(i))
    res = op.fmin_bfgs(CostFunc, optimal_theta, fprime= Gradient, args=(state,Y))
    optimal_theta = res;
    #Result = op.minimize(fun = CostFunc, 
    #                                     x0 = optimal_theta, 
    #                                     args = (state, Y),
    #                                     method = 'TNC',
    #                                     jac = Gradient);
    print("in traning loop for domain  with iteratio no. " + str(i))
    #optimal_theta = Result.x;
    #observation = env.reset()

    #print("**********state*********")
    #print(state)
    #print("**********Y*********")
    #print(Y)
    
    #print("##########################")
    #print(observation)
while True:
    env.render()
    print("if no analysis trainin hurray bot is balanced")
    observation, reward, done, info = env.step(action_pred(observation,optimal_theta))
    if done == True:     
        for i in range(1,100):
            state = env.reset()
            Y = [1]
            print("********opt theta****")
            print(optimal_theta)
            for j in range(2,50):
                env.render()
                t = action_pred(observation,optimal_theta)
                observation_new, reward, done, info = env.step(t)
                #store every observation and corresponding expected output
                state = np.vstack([state, observation_new])
                
                if observation[2] < 0:
                    if observation_new[2] - observation[2] < 0:
                        action_pre = 1 - t
                        Y = np.vstack([Y,action_pre])
                    else:
                        Y = np.vstack([Y,t])    
                if observation[2] > 0:
                    if observation_new[2] - observation[2] > 0:
                        action_pre = 1 - t
                        Y = np.vstack([Y,action_pre])
                    else :
                        Y = np.vstack([Y,t])
                if done == True:
                    #print("@@@@@@@..done..@@@@@@@@")
                    env.reset()
            res = op.fmin_bfgs(CostFunc, optimal_theta, fprime= Gradient, args=(state,Y))
            optimal_theta = res;
            print("in traning loop for analy.   with iteratio no. " + str(i))
