import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from collections import Counter
from abc import ABC, abstractmethod
from scipy.stats import norm, beta

@dataclass
class BanditEnvironment:
    '''
    Class to siimulate bandit environment with changes in structural reward over time.
    len(arm_reward) defines the number of arms.
    len(structural_reward) defines
    '''
    arm_rewards: List[float]
    structural_reward: List[float]
    n_draws_per_time: int = 1

    def __post_init__(self):
        self.n_arms = len(self.arm_rewards)
        self.n_times = len(self.structural_reward)
        self.true_p = (np.array([self.arm_rewards] * self.n_times).T + self.structural_reward).T

    def draw_rewards(self, arm_sample_probs: List[float], time: int = 0) -> Tuple[np.array, np.array]:
        'Takes a sample probability for each arm as input, returns successes a and failures b for each arm'

        if len(arm_sample_probs) != self.n_arms:
            raise ValueError(
                'Length of arm_sample_probs not the same as the number of arms A (defined by len(arm_rewards) )')
        arms = np.random.choice(range(self.n_arms), self.n_draws_per_time, p=arm_sample_probs)
        arm_counts = Counter(arms)
        draws = np.zeros(self.n_arms)
        rewards = np.zeros(self.n_arms)
        for arm in arm_counts:
            draws[arm] = arm_counts[arm]
            rewards[arm] = np.random.binomial(n=draws[arm], p=self.true_p[time, arm])
        return draws, rewards

    def run_simulation(self, Policy) -> Tuple[np.array, np.array]:
        '''
        Runs a simulation using Policy, return a tuple of np.array representing successes a and failures b for each arm and time step.
        Note: policy input is np.array of size (n_times, n_arms).
        '''
        draws = np.zeros((self.n_times, self.n_arms))
        rewards = np.zeros((self.n_times, self.n_arms))
        initial_arm_sample_probs = [1 / self.n_arms] * self.n_arms
        draws[0, :], rewards[0, :] = self.draw_rewards(initial_arm_sample_probs, time=0)
        for t in range(1, self.n_times):
            arm_sample_probs = Policy.calculate_arm_sample_probs(draws, rewards)
            draws[t, :], rewards[t, :] = self.draw_rewards(arm_sample_probs, time=t)
        return BanditSimulationResult(draws, rewards, self.true_p)


class BanditPolicy(ABC):

    @abstractmethod
    def calculate_arm_sample_probs(self, draws: np.array, rewards: np.array) -> np.array:
        pass


class ABTestPolicy(BanditPolicy):

    def calculate_arm_sample_probs(self, draws: np.array, rewards: np.array) -> np.array:
        '''
        Both draws and rewards need to be 2 dimensional np.arrays of size (n_times, n_arms) and not cumulative
        Returns am array of arm_sample_probs of size n_arms (which is the length of draws and rewards)
        '''
        if len(draws.shape) != 2 or len(rewards.shape) != 2:
            raise ValueError(
                'Both draws and rewards need to be 2 dimensional np.arrays of size (n_times, n_arms) and not cumulative')
        if draws.shape != rewards.shape:
            raise ValueError('Shape of draws not equal to length of rewards')

        n_arms = draws.shape[1]
        arm_sample_probs = [1 / n_arms] * n_arms

        return arm_sample_probs


@dataclass
class BanditSimulationResult:
    'Draws and rewards need to be of size (n_times, n_arms)'
    draws: np.array
    rewards: np.array
    true_p: np.array

    def __post_init__(self):
        if self.draws.shape != self.rewards.shape:
            raise ValueError('Shape of draws and rewards not the same')
        self.n_arms = self.draws.shape[1]
        self.n_times = self.draws.shape[0]
        self.cum_draws = np.cumsum(self.draws, axis=0)
        self.cum_rewards = np.cumsum(self.rewards, axis=0)
        self.p = self.cum_rewards / self.cum_draws
        self.p_stderr = (1 / np.sqrt(self.cum_draws)) * np.sqrt(self.p * (1 - self.p))
        self.expected_reward = np.sum(self.draws * self.true_p, axis=1)
        self.max_reward = np.sum(self.draws, axis=1) * np.max(self.true_p, axis=1)
        self.regret = self.max_reward - self.expected_reward
        self.cum_regret = np.cumsum(self.regret, axis=0)

    def plot_p(self) -> None:
        '''
        Plots estimated p and its confidende interval.
        '''
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        fig, ax = plt.subplots()
        x = range(self.n_times)
        p = self.p
        ci = 1.96 * self.p_stderr
        for arm in range(self.n_arms):
            ax.plot(x, p[:, arm], color=colors[arm], label='p_hat of arm {}'.format(arm))
            ax.plot(x, self.true_p[:, arm], '--', color=colors[arm], label='true p of arm {}'.format(arm))
            ax.fill_between(x, (p[:, arm] - ci[:, arm]), (p[:, arm] + ci[:, arm]), color=colors[arm], alpha=.1)
            ax.legend()

    def plot_draws(self) -> None:
        plt.plot(range(self.n_times), self.draws)

    def plot_regret(self) -> None:
        plt.plot(range(self.n_times), self.regret)

    def plot_cum_regret(self) -> None:
        plt.plot(range(self.n_times), self.cum_regret)

    def compare_best(self) -> Tuple[np.array, np.array, np.array]:
        'Calculates the probability of the best arm being the better than all the other arms'
        best_arm = np.argmax(np.mean(self.true_p, axis=0))
        mean_diff = np.zeros((self.n_times, self.n_arms))
        stderr_diff = np.zeros((self.n_times, self.n_arms))
        prob_best = np.zeros((self.n_times, self.n_arms))
        true_diff = np.zeros((self.n_times, self.n_arms))
        true_diff = np.zeros((self.n_times, self.n_arms))
        for arm in range(self.n_arms):
            mean_diff[:, arm] = self.p[:, best_arm] - self.p[:, arm]
            stderr_diff[:, arm] = np.sqrt(np.square(self.p_stderr[:, best_arm]) + np.square(self.p_stderr[:, arm]))
            t_value = mean_diff[:, arm] / stderr_diff[:, arm]
            prob_best[:, arm] = norm.cdf(t_value)
            true_diff[:, arm] = self.true_p[:, best_arm] - self.true_p[:, arm]
        return mean_diff, stderr_diff, prob_best, true_diff

    def plot_prob_best(self) -> None:
        _, _, prob_best, _ = self.compare_best()
        plt.plot(range(self.n_times), prob_best)

    def plot_diff_best(self) -> None:
        mean_diff, stderr_diff, _, true_diff = self.compare_best()

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        fig, ax = plt.subplots()
        x = range(self.n_times)
        y = mean_diff
        ci = 1.96 * stderr_diff
        for arm in range(self.n_arms):
            ax.plot(x, y[:, arm], color=colors[arm], label='diff arm {} and best arm'.format(arm))
            ax.plot(x, true_diff[:, arm], '--', color=colors[arm], label='true diff arm {} and best arm'.format(arm))
            ax.fill_between(x, (y[:, arm] - ci[:, arm]), (y[:, arm] + ci[:, arm]), color=colors[arm], alpha=.1)
            ax.legend()


def plot_with_ci(data: List[np.array], label: List[str] = None) -> None:
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    fig, ax = plt.subplots()
    for i in range(len(data)):
        x = range(data[i].shape[0])
        y = np.mean(data[i], axis=1)
        # ci = 1.96 * np.std(data[i], axis = 1)
        data[i] = np.sort(data[i], axis=1)
        n_samples = data[i].shape[1]
        y_lower_bound = data[i][:, round(0.025 * n_samples)]
        y_upper_bound = data[i][:, round(0.975 * n_samples)]
        if label is None:
            ax.plot(x, y, color=colors[i])
        else:
            ax.plot(x, y, color=colors[i], label=label[i])
        ax.fill_between(x, y_lower_bound, y_upper_bound, color=colors[i], alpha=.1)
    plt.legend()
