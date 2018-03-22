
import numpy as np

import gym
from gym import wrappers



#The game will be split into 40x40 grid.  Each intersection on the grid will represent a possible state you can be in, or State[X][Y].
#You start the game at state[20][0] X = 20 means middle of the track in regards to tracklength, Y = 0 means the lowest point of the track in regards to height of the track.
#We need a sufficient amount of states to represent all possible positions the robot can possible find itself in. 40 is just a good number in this case.
#More states = more training rounds to fill in data about those states.
n_states = 40

#Training episodes
iter_max = 1000

#How much time we are willing to give our robot to finish the training round in each episode.
#It seems to be around 5 seconds long.  Might be measured in game frames instead, I am not sure.
t_max = 10000

#initial learning rate
initial_lr = 1.0 # Learning rate

#assuming learning rate decreases per round, to a minimum of 0.003
min_lr = 0.003

#A value from 0 to 1(0 = I want only immediate reward,0.5 = I value both immediate and future rewards equally , 1 = I want only future awards)
#If you put 0 as gamma for example, the robot may only forward with the game car and not backwards, because he wants instant rewards!
#He won't consider that moving backwards first might give him momentum later go to forward more.
gamma = 1.0

#How the program should do something random, think discovery mode!
#If robot knows pressing button A will give him 100 points and button B gives 0 points
#he will press button A 98% of the time, and button B 2% of the time.
#This is because maybe pressing button B has a 10% chance of giving 3000 points, but the robot hasn't pressed button B enough times to find out.
eps = 0.02

#runs animation of the game after we are done training our robot.
def run_episode(env, policy=None, render=True):
    #Reset position of car on the track.
    obs = env.reset()
    #Reset all the rewards we have gotten.
    total_reward = 0
    #How much time/frames did this take us?
    step_idx = 0
    for _ in range(t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a,b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

#Calculates what state our car is currently in.
def obs_to_state(env, obs):
    """ Maps an observation to state """
    #env.observation_low.space is a built in function
    #in Open Gym AI, it gives us information about what's happening in the gym.
    #In the case of this game it is giving us the coordinate of the botton left corner of our game screen environment).
    env_low = env.observation_space.low
    #env.observation_space.high gives us the coordinate of the top right corner of our game screen (environment).
    env_high = env.observation_space.high

    #We will now split the game screen up into 40 slices, and each slice is an increment.
    env_dx = (env_high - env_low) / n_states

    #Returns states of our function.
    #(Gets the distance traveled by the car, subtract it by the maximum distance) now we divide it by the length of an increment to see how many increments we have traveled.
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])

    return a, b

if __name__ == '__main__':
    #Name of the game in the  Gym Open AI import
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    #Makes it so the random number generator gives the same numbers each time.
    env.seed(0)
    np.random.seed(0)

    print ('----- using Q Learning -----')
    #We have 40X40 possible positions on the track.  In all those possible positions, we have 3 actions we can give to the car: Move Forward, Move Backwards or No Command
    q_table = np.zeros((n_states, n_states, 3))

    for i in range(iter_max):
        #resets the game parameters, we are going to give it 1000 training rounds while constantly giving it random commands.
        obs = env.reset()
        total_reward = 0
        # eta: learning rate is decreased at each step
        eta = max(min_lr, initial_lr * (0.85 ** (i//100)))

        for j in range(t_max):
            #calls the obs_to_state function passing the game as the environment, and reset environment function as the second..??
            a, b = obs_to_state(env, obs)

            #Generates a random number between 0-1, if that number is lower than eps, then the robot will take a random action and not the optimal action the current state.
            if np.random.uniform(0, 1) < eps:
                action = np.random.choice(env.action_space.n)

            #Otherwise we just choose the best action in the state and perform that.
            else:
                #get an array of actions and the points granted to the robot depending on which action he takes.
                logits = q_table[a][b]
                #scales the numbers down
                logits_exp = np.exp(logits)
                probs = logits_exp / np.sum(logits_exp)
                action = np.random.choice(env.action_space.n, p=probs)

            #gets some values from the game depending on what action you took.
            #obs = position in game (used to tell what state we are in)
            #reward = your reward for that given action.
            #done = Did you lose the game yet?
            obs, reward, done, _ = env.step(action)

            #adds your reward to the total reward for the training round.
            #This value is important as it will be used to calculate actual points at the end of the game.
            #If you took had 200 points and took 100 actions to finish the, then your reward will be 200/100 = 2 points.
            #This is a way to tell the robot --If you want more points, finish the game with the least amount of moves!--
            total_reward += reward

            #Calls a function to get the index of the state we are now in.
            a_, b_ = obs_to_state(env, obs)

            #Update the value in our Q table with our action.  The if you perform the same action in the same state many times and constantly recieve rewards,
            #then the robot will be more likely to take that action in the future.
            #Visit http://mnemstudio.org/path-finding-q-learning-tutorial.htm if your not at all familiar with it.
            q_table[a][b][action] = q_table[a][b][action] + eta * (reward + gamma *  np.max(q_table[a_][b_]) - q_table[a][b][action])
            if done:
                break
        #At every 100 training rounds, calculate our average reward (How fast the robot is able to make it up the ramp)
        if i % 100 == 0:
            print('Iteration #%d -- Total reward = %d.' %(i+1, total_reward))

    solution_policy = np.argmax(q_table, axis=2)
    solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    # Animate it
    run_episode(env, solution_policy, True)
