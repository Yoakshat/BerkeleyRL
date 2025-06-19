from tensorboard.backend.event_processing import event_accumulator
import glob
import numpy as np

import seaborn as sns 
import matplotlib.pyplot as plt

logdir = "data"

def accumulate(target: str):
    look_for = logdir + "/*" + target + "*/events.out.tfevents.*"
    print("Looking for: " + look_for)
    event_files = glob.glob(look_for, recursive=True)
    print("Found: " + str(len(event_files)))

    assert len(event_files) > 0

    avg_over_seeds = np.array([])

    for e in event_files: 
        ea = event_accumulator.EventAccumulator(e)
        ea.Reload()  # Loads events

        scalars = ea.Scalars("Eval_AverageReturn")
        scalars = np.array([s.value for s in scalars])
        # step is just length of values
        if(avg_over_seeds.shape[0] == 0): 
            avg_over_seeds = scalars
        else: 
            avg_over_seeds += scalars
        
    avg_over_seeds /= len(event_files)
    return avg_over_seeds


def plot(targets_list, plot_name): 
    # list of numpy arrays
    lines = [accumulate(target) for target in targets_list]
    plt.figure()

    for l, t in zip(lines, targets_list): 
        sns.lineplot(x=list(range(len(l))), y=l, label=t)

    plt.legend()
    plt.savefig("cs285/graphs/" + plot_name)

plot(["pendulum_default", "pendulum_only_na"], "pendulumCompare.png")




