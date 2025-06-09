"""
Runs behavior cloning and DAgger for homework 1

Functions to edit:
    1. run_training_loop
"""

import pickle
import os
import time
import gym

import numpy as np
import torch

from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.policies.MLP_policy import MLPPolicySL
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below

MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]


def run_training_loop(params):
    """
    Runs training with the specified parameters
    (behavior cloning or dagger)

    Args:
        params: experiment parameters
    """

    #############
    ## INIT
    #############

    # Get params, create logger, create TF session
    logger = Logger(params['logdir'])

    # Set random seeds
    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    ptu.init_gpu(
        use_gpu=not params['no_gpu'],
        gpu_id=params['which_gpu']
    )

    # Set logger attributes
    log_video = True
    log_metrics = True

    #############
    ## ENV
    #############

    # Make the gym environment
    env = gym.make(params['env_name'], render_mode=None)
    env.reset(seed=seed)

    # Maximum length for episodes
    params['ep_len'] = params['ep_len'] or env.spec.max_episode_steps
    MAX_VIDEO_LEN = params['ep_len']

    assert isinstance(env.action_space, gym.spaces.Box), "Environment must be continuous"
    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if 'model' in dir(env):
        fps = 1/env.model.opt.timestep
    else:
        fps = env.env.metadata['render_fps']

    #############
    ## AGENT
    #############

    # TODO: Implement missing functions in this class.
    actor = MLPPolicySL(
        ac_dim,
        ob_dim,
        params['n_layers'],
        params['size'],
        learning_rate=params['learning_rate'],
    )

    # replay buffer
    replay_buffer = ReplayBuffer(params['max_replay_buffer_size'])

    #######################
    ## LOAD EXPERT POLICY
    #######################

    print('Loading expert policy from...', params['expert_policy_file'])
    expert_policy = LoadedGaussianPolicy(params['expert_policy_file'])
    expert_policy.to(ptu.device)
    print('Done restoring expert policy...')

    #######################
    ## TRAINING LOOP
    #######################

    # init vars at beginning of training
    total_envsteps = 0
    start_time = time.time()

    allLogs = [] 

    for itr in range(params['n_iter']):
        print("\n\n********** Iteration %i ************"%itr)

        # decide if videos should be rendered/logged at this iteration
        log_video = ((itr % params['video_log_freq'] == 0) and (params['video_log_freq'] != -1))
        # decide if metrics should be logged
        log_metrics = (itr % params['scalar_log_freq'] == 0)

        print("\nCollecting data to be used for training...")
        if itr == 0:
            # BC training from expert data.
            paths = pickle.load(open(params['expert_data'], 'rb'))
            envsteps_this_batch = 0
        else:
            # DAGGER training from sampled data relabeled by expert
            assert params['do_dagger']
            # TODO: collect `params['batch_size']` transitions (state, reward, action)
            # HINT: use utils.sample_trajectories
            # TODO: implement missing parts of utils.sample_trajectory
            paths, envsteps_this_batch = utils.sample_trajectories(env, actor, params['batch_size'], params['ep_len'])

            # relabel the collected obs with actions from a provided expert policy
            if params['do_dagger']:
                print("\nRelabelling collected observations with labels from an expert policy...")

                # TODO: relabel collected obsevations (from our policy) with labels from expert policy
                # HINT: query the policy (using the get_action function) with paths[i]["observation"]
                # and replace paths[i]["action"] with these expert labels

                for i in range(len(paths)):
                    paths[i]["action"] = expert_policy.get_action(paths[i]["observation"])

        total_envsteps += envsteps_this_batch
        # add collected data to replay buffer
        replay_buffer.add_rollouts(paths)

        # train agent (using sampled data from replay buffer)
        print('\nTraining agent using sampled data from replay buffer...')
        training_logs = []
        for _ in range(params['num_agent_train_steps_per_iter']):

          # TODO: sample some data from replay_buffer
          # HINT1: how much data = params['train_batch_size']
          # HINT2: use np.random.permutation to sample random indices
          # HINT3: return corresponding data points from each array (i.e., not different indices from each array)
          # for imitation learning, we only need observations and actions.  

          indices = np.random.permutation(len(replay_buffer.obs))[:params['train_batch_size']]
          ob_batch, ac_batch = ptu.from_numpy(replay_buffer.obs[indices]), ptu.from_numpy(replay_buffer.acs[indices])

    
          # use the sampled data to train an agent
          train_log = actor.update(ob_batch, ac_batch)
          training_logs.append(train_log)


        # log/save
        # print('\nBeginning logging procedure...')
        '''
        if log_video:
            # save eval rollouts as videos in tensorboard event file
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(
                env, actor, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            # save videos
            if eval_video_paths is not None:
                logger.log_paths_as_videos(
                    eval_video_paths, itr,
                    fps=fps,
                    max_videos_to_save=MAX_NVIDEO,
                    video_title='eval_rollouts')
        '''


        if log_metrics:
            # save eval metrics
            print("\nCollecting data for eval...")

            # if you want to visualize
            # env = gym.make(params["env_name"], render_mode="human")

            # how do we do vs how does our expert do ()
            eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
                env, actor, params['eval_batch_size'], params['ep_len'])

            # paths of expert VS our paths
            logs = utils.compute_metrics(paths, eval_paths)
            # compute additional metrics
            logs.update(training_logs[-1]) # Only use the last log for now
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs["Train_AverageReturn"]

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            logger.flush()

        if params['save_params']:
            print('\nSaving agent params')
            actor.save('{}/policy_itr_{}.pt'.format(params['logdir'], itr))

        allLogs.append(logs)

    # return logs for every iteration
    return allLogs
    
def loadName(params, name): 
    params['expert_data'] = 'cs285/expert_data/expert_data_' + name + '-v4.pkl'
    params['expert_policy_file'] = 'cs285/policies/experts/' + name + '.pkl'
    params['env_name'] = name + '-v4'
    params['exp_name'] = name + "0"

    
def getLogDir(params): 
    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if params['do_dagger']:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        params['logdir_prefix'] = 'q2_'
        assert params['n_iter']>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        params['logdir_prefix']= 'q1_'
        assert params['n_iter']==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    # directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = params['logdir_prefix'] + params['exp_name'] + '_' + params['env_name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    print("Directory for logging: " + params['logdir'])
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

def part1(params): 
    eval_means, eval_sds, expert_means, expert_sds = [], [], [], []

    names = ["Ant", "HalfCheetah", "Hopper", "Walker2d"]
    for name in names: 
        loadName(params, name)
        getLogDir(params)

        logs = run_training_loop(params)[0]

        eval_means.append(round(logs["Eval_AverageReturn"], 2))
        eval_sds.append(round(logs["Eval_StdReturn"], 2))
        expert_means.append(round(logs["Train_AverageReturn"], 2))
        expert_sds.append(round(logs["Train_StdReturn"], 2))

    pd.DataFrame({
        'Expert_AvgReturn': expert_means, 
        'Expert_AvgStdReturn': expert_sds, 
        'Eval_AvgReturn': eval_means, 
        'Eval_AvgStdReturn': eval_sds,
        'Tasks': names
    }).to_csv('cs285/tables/AllTasks.csv')

    # modify lr in hopper + see what happens
    lrs = [3e-2, 3e-3, 3e-4, 3e-5, 3e-6, 3e-7]
    returns = [] 
    for lr in lrs: 
        loadName(params, name)
        getLogDir(params)
        params['learning_rate'] = lr 
        logs = run_training_loop(params)[0]

        returns.append(round(logs['Eval_AverageReturn'], 2))
    
    df = pd.DataFrame({
        'learning-rate': lrs, 
        'EvalAvgReturn': returns
    })
    df.to_csv('cs285/tables/HopperLR.csv')

    fig = plt.figure()
    sns.lineplot(data=df, x='learning-rate', y='EvalAvgReturn')
    plt.title('Learning Rate vs Avg Return (Hopper Task)', fontsize=30)
    # plt.show()
    plt.savefig('cs285/visuals/lrVsReturn.png')
    plt.close()

# Ant and Hopper Task
def part2(params): 
    params['do_dagger'] = True
    params['n_iter'] = 10

    names = ["Ant", "Hopper"]
    returns = []
    for name in names: 
        loadName(params, name)
        getLogDir(params)

        allLogs = run_training_loop(params)

        # get return per iteration
        for log in allLogs: 
            returns.append(round(log['Eval_AverageReturn'], 2))

    plt.figure()
    # DAGGER with Ant
    sns.lineplot(x=list(range(params['n_iter'])), y=returns[:params['n_iter']], color='red')
    # DAGGER with Hopper 
    sns.lineplot(x=list(range(params['n_iter'])), y=returns[params['n_iter']:], color='blue')

    # expert policy + BC policy
    df = pd.read_csv('cs285/tables/AllTasks.csv')
    ant = df[df['Tasks'] == 'Ant'].iloc[0]
    hopper = df[df['Tasks'] == 'Hopper'].iloc[0]

    # draw horizontal lines

    # ant
    plt.axhline(y=ant['Expert_AvgReturn'], color='r', linestyle='--')
    plt.axhline(y=ant['Eval_AvgReturn'], color='r', linestyle=":")

    # hopper 
    plt.axhline(y=hopper['Expert_AvgReturn'], color='b', linestyle='--')
    plt.axhline(y=hopper['Eval_AvgReturn'], color='b', linestyle=":")

    plt.savefig('cs285/visuals/dagger.png')
    plt.close()

def run(createTable=False): 
    import argparse
    parser = argparse.ArgumentParser()
    required = (not createTable)

    parser.add_argument('--expert_policy_file', '-epf', type=str, required=required)  # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=required) #relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=required)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=required)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=1000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    ###################
    ### RUN TRAINING
    ###################
    
    if(createTable):
        temp = params.copy()
        # without dagger
        part1(params)
        # with dagger
        part2(temp)
    else:
        getLogDir(params)
        run_training_loop(params)


def main():
    run()
    # if you want to replicate experiments use this line instead
    # run(createTable=True)


if __name__ == "__main__":
    main()
