import torch 
import torch.multiprocessing as mp
from cs285.infrastructure import utils

class Worker(mp.Process): 
    # env is a gym environment
    def __init__(self, ga_net, gv_net, gopt, agent, env, batch_size, max_ep_len, iters): 
        super(Worker, self).__init__()

        self.ga_net = ga_net
        self.gv_net = gv_net 
        self.gopt = gopt
        self.env = env
        self.iters = iters

        # create a copy of the agent's actor + critic
        self.agent = agent

        self.batch_size = batch_size
        self.max_ep_len = max_ep_len

    def run(self): 
        for itr in range(self.iters): 
            print(f"\n********** Iteration {itr} ************")
            # sample rollout
            trajs, envsteps_this_batch = utils.sample_trajectories(self.env, self.agent, self.batch_size, self.max_ep_len)
            trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

            self.gopt.zero_grad()

            # collect local gradients
            self.agent.update(trajs_dict["observation"], trajs_dict["action"], trajs_dict["reward"], trajs_dict["terminal"], nostep = True)
            
            # push and pull
            self.agent.syncGradients(self.ga_net, self.gv_net)
            self.gopt.step()
            self.agent.reload(self.ga_net, self.gv_net)



    
    
    
