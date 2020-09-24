import gym
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import four_room_env

class SMDP():

    def __init__(self, env, eps_no, intra_option_flag):       

        self.env = env
        self.gamma = 0.9
        self.alpha = 0.1
        self.eps_no = eps_no
        self.epsilon = 0.1
        self.q_table = np.zeros((env.option_size, env.height, env.width))        
        self.intra_option_flag = intra_option_flag


    def choose_option(self):
        options_possible = [0, 1, 2, 3]
        is_room, room_hallway_index = self.env.get_room_hallway()
                                            # Adding options possible in current state room        
        if is_room:
            options_possible.append(room_hallway_index * 2 + len(self.env.moves))
            options_possible.append(room_hallway_index * 2 + len(self.env.moves) + 1)
        else:
            options_possible.append(room_hallway_index * 2 + 1 + len(self.env.moves))
            options_possible.append((room_hallway_index * 2 + 2 + len(self.env.moves)) % self.env.option_size)
                
        if np.random.uniform(0,1) < self.epsilon:       # Exploration
            return options_possible[np.random.randint(0, len(options_possible))]
        else:                                           # Exploitation
            best_options = []
            q_max = np.max(self.q_table[options_possible, self.env.S[0], self.env.S[1]])
            for k in options_possible:
                if self.q_table[k, self.env.S[0], self.env.S[1]] == q_max:
                    best_options.append(k)
            return best_options[np.random.randint(0, len(best_options))]


    def option_step(self, option_taken):
        count = 0
        total_r = 0
        option_reward = 0
        discount = 1
        not_hallway, room_hallway_index = self.env.get_room_hallway()
                                            # target hallway of the option taken
        target_room = (int)((option_taken - 4) / 2)
        target_hallway = self.env.hallways[target_room][(option_taken - 4) % 2]
        target_pre_hallway = self.env.pre_hallways[target_room][(option_taken - 4) % 2]
        
        while not (self.env.done or self.env.S == target_hallway):
            prev_S = self.env.S            
            r = self.option_take_action(target_hallway, target_pre_hallway)
            count += 1
            is_room, current_room = self.env.get_room_hallway() 
            
            if self.intra_option_flag:
                total_r += r
                if self.env.S == target_hallway:                # Target-state update for intra-option q-learning
                    self.q_table[option_taken, prev_S[0], prev_S[1]] += self.alpha * (r + self.gamma * max(self.q_table[:, self.env.S[0], self.env.S[1]]) - self.q_table[option_taken, prev_S[0], prev_S[1]])
                    break
                elif is_room and current_room != target_room:
                    self.q_table[option_taken, prev_S[0], prev_S[1]] += self.alpha * (r + self.gamma * max(self.q_table[:, self.env.S[0], self.env.S[1]]) - self.q_table[option_taken, prev_S[0], prev_S[1]])
                    break
                else:                                           # Non-target state update
                    self.q_table[option_taken, prev_S[0], prev_S[1]] += self.alpha * (r + self.gamma * self.q_table[option_taken, self.env.S[0], self.env.S[1]] - self.q_table[option_taken, prev_S[0], prev_S[1]])
            else:
                total_r += r
                option_reward += r * discount
                discount *= self.gamma                
                       
            if is_room and current_room != target_room:            # if option starts at a hallway but doesn't go into the right room                    
                break
        return option_reward, total_r, discount, count


    def option_take_action(self, target_hallway, target_pre_hallway):
        
        is_room, room_hallway_index = self.env.get_room_hallway()
        if not is_room:
            if target_hallway == self.env.hallways[(room_hallway_index + 1) % len(self.env.hallways)][0]:
                action = len(self.env.moves) - 1 - room_hallway_index
            else:
                action = (len(self.env.moves) - 1 - room_hallway_index + 2) % len(self.env.moves)
        
        x = self.env.S[0] - target_pre_hallway[0]       # +ve -> target is above, -ve -> target is below
        y = self.env.S[1] - target_pre_hallway[1]       # +ve -> target is on left, -ve -> target is on right

        if x == 0 and y == 0:                           # Reached pre-hallway
            x = self.env.S[0] - target_hallway[0]
            y = self.env.S[1] - target_hallway[1]
        
        if x < 0:
                         # action -> DOWN
            action = 2
        elif x > 0:
                         # action -> UP
            action = 0
        elif y < 0:
                         # action -> RIGHT
            action = 3
        elif y > 0:
                         # action -> LEFT
            action = 1
        
        return self.env.step(action) 

    
    def smdp_q_learning(self):
        steps = np.zeros(self.eps_no)
        rewards = np.zeros(self.eps_no)
        action_count = np.zeros(self.eps_no)

        for ep in range(self.eps_no):            
            self.env.reset()
            
            while not self.env.done:
                prev_S = self.env.S
                option_taken = self.choose_option()
                
                if option_taken < len(self.env.moves):      # Primitive actions
                    r = self.env.step(option_taken)
                    steps[ep] += 1
                    action_count[ep] += 1
                    self.q_table[option_taken, prev_S[0], prev_S[1]] += self.alpha * (r + self.gamma * max(self.q_table[:, self.env.S[0], self.env.S[1]]) - self.q_table[option_taken, prev_S[0], prev_S[1]])                    
                else:                                       # Multi-step options
                    option_r, r, discount, option_step_count = self.option_step(option_taken)
                    steps[ep] += option_step_count
                    action_count[ep] += 1
                    if not self.intra_option_flag:                    
                        self.q_table[option_taken, prev_S[0], prev_S[1]] += self.alpha * (option_r + discount * max(self.q_table[:, self.env.S[0], self.env.S[1]]) - self.q_table[option_taken, prev_S[0], prev_S[1]])                

                rewards[ep] += r
        
        return steps, rewards, action_count
    

if __name__ == '__main__':
        
    runs = 1
    eps_no = 10000
    avg_steps = np.zeros(eps_no)
    avg_rewards = np.zeros(eps_no)
    avg_actions = np.zeros(eps_no)
    
    for j in range(runs):
        if j % 10 == 0:
            print(j)
        env = gym.make('four_room-v0')  
        env.reset_type = 3                          # 0 -> start in room 1 at random, 3 -> room 4 center
        env.goal = [8,8]                            # change goal position if required
        agent = SMDP(env, eps_no, False)               # True -> intra-option        
        x, y, z = agent.smdp_q_learning()
        avg_steps += x
        avg_rewards += y
        avg_actions += z

    #De-comment to shows graphs for average steps
    
    plt.figure()
    plt.plot(avg_steps / runs)
    plt.title('Avg Steps vs Episode across 100 runs with Intra-option learning' if agent.intra_option_flag else 'Avg Steps vs Episode across 100 runs without Intra-option learning')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    #plt.ylim([0, 200])
    plt.savefig('Steps_intra_'+ str(agent.intra_option_flag))
    plt.show()
    
    plt.figure()
    plt.plot(avg_rewards / runs)
    plt.title('Avg Reward vs Episode across 100 runs with Intra-option learning' if agent.intra_option_flag else 'Avg Reward vs Episode across 100 runs without Intra-option learning')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    #plt.ylim([-1, 1])
    plt.savefig('Reward_intra_'+ str(agent.intra_option_flag))
    plt.show()
    
    plt.figure()
    plt.plot(avg_actions / runs)
    plt.title('Avg Actions taken vs Episode across 100 runs with Intra-option learning' if agent.intra_option_flag else 'Avg Actions taken vs Episode across 100 runs without Intra-option learning')
    plt.xlabel('Episodes')
    plt.ylabel('Number of Actions/Options taken')
    #plt.ylim([0, 100])
    plt.savefig('Actions_intra_'+ str(agent.intra_option_flag))
    plt.show()

    
    eps_label = str((int)(agent.epsilon * 100))
     
    agent.q_table[:,env.S[0],env.S[1]] = 1
    option_names = ['UP', 'LEFT', 'DOWN', 'RIGHT', 'OP1: Rm 1, Target=[2,5]',
                    'OP2: Rm 1, Target=[5,1]', 'OP3: Rm 2, Target=[6,8]',
                    'OP4: Rm 2, Target=[2,5]', 'OP5: Rm 3, Target=[9,5]',
                    'OP6: Rm 3, Target=[6,8]', 'OP7: Rm 4, Target=[5,1]',
                    'OP8: Rm 4, Target=[9,5]']
    
    for j in range(env.option_size):
        plt.figure()        
        sns.heatmap(agent.q_table[j,:,:])
        plt.title('Q-values for '+ option_names[j] +', Start=Rm'+ str(env.reset_type + 1) +', Goal='+ str(env.goal))
        plt.savefig(str(agent.intra_option_flag) +'_qvalues_'+ str(j) + '_rm' + str(env.reset_type + 1) +'_gl'+ str(env.goal[0]) +'_eps'+ eps_label)
    
    plt.figure()
    sns.heatmap(np.max(agent.q_table, axis = 0))
    plt.title('State Value function, Start=Rm'+ str(env.reset_type + 1) +' Goal='+ str(env.goal))
    plt.savefig(str(agent.intra_option_flag) +'_Value_fig_rm' + str(env.reset_type + 1) +'_gl'+ str(env.goal[0]) +'_eps'+ eps_label)
    
    
    
