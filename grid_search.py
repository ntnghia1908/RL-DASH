from utils import *
from singlepath_gym import SinglepathEnvGym
import os, random, numpy as np, torch


def set_global_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


fcc_train = np.asarray(pickle.load(open("./bw/fcc_train100kb.pickle", "rb")))
fcc_train = np.repeat(fcc_train, 10, axis=1)
lte_train = np.asarray(pickle.load(open("./bw/LTE_train100kb.pickle", "rb")))
train = np.concatenate((fcc_train, lte_train), axis=0)

fcc_test = np.asarray(pickle.load(open("./bw/fcc_test100kb.pickle", "rb")))
fcc_test = np.repeat(fcc_test, 10, axis=1)
lte_test = np.asarray(pickle.load(open("./bw/LTE_test100kb.pickle", "rb")))


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


TRIALS = 3000
bests = []
EPS = 200
epoch_time = AverageMeter()

best_actions = np.zeros((200, 60))

for i in range(EPS):
    best = 0
    for j in range(TRIALS):
        start_time = time.time()
        env = SinglepathEnvGym(
            log_qoe=True, bitrate_list=np.array([fcc_test[i]]), train=False
        )
        actions = [0]
        state = env.reset()

        done = False
        while not done:
            action = env.action_space.sample()
            actions.append(action)
            state, reward, done, info = env.step(action)
        if env.total_reward > best:
            best = env.total_reward
            best_actions[i] = np.array(actions)

        epoch_time.update(time.time() - start_time)
        if j % 200 == 0:
            print(f"Episode: {i+1}, best so far: {best}")
            need_hour, need_mins, need_secs = convert_secs2time(
                epoch_time.avg * (EPS * TRIALS - (i + 1) * j)
            )
            need_time = "Estimated Time Left: {:02d}:{:02d}:{:02d}\n\n".format(
                need_hour, need_mins, need_secs
            )
            print(need_time)
    bests.append(best)
print(best_actions)
print(bests)
print(np.mean(bests))

np.save("best_fcc", np.array(bests))
np.save("best_actions_fcc", np.array(best_actions))
