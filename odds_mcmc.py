import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from numpy.random import random, randn, exponential
from scipy.stats import bernoulli, beta, expon
from functools import lru_cache
import matplotlib.pyplot as plt
from itertools import combinations


def scores2logprob(a, b):
    '''
    Translates two scores to log probabilities
    a, b -> scores of first and second team
    Returns: logprob that team A beats team B, and vice versa
    '''
    z1 = np.log(a) - np.log(a+b)
    z2 = np.log(b) - np.log(a+b)
    return z1, z2

def scores2prob(a, b):
    '''
    Same as above, but no log
    '''
    z = a/(a+b)
    return z, 1-z

# https://stackoverflow.com/questions/24471136/how-to-find-all-paths-between-two-graph-nodes
def find_all_paths(graph, team_pair, path=None):
    '''
    Returns all acyclic paths from team A to team B
    graph: Directed graph of games played
    team_pair: starting and ending points
    '''
    if not path: path = []
    start, end = team_pair
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:# or graph[start] == -1:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, (node, end), path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths       


class MCMC:
    '''
    Performs MCMC iterations to return posterior of scores
    '''

    def __init__(self, game_tables, **kwargs):
        '''
        game_tables: Poissibly unfinished tables of games
        '''
        self.std = 0.01 # used for proposing next element
        self.n_accepted = 0
        self.n_iters = 0

        prior_params = kwargs.get('prior', {})
        self.alpha, self.beta = prior_params.get('alpha', 1), prior_params.get('beta', 1) # prior parameter
        self.lower, self.upper = prior_params.get('lower', 0), prior_params.get('upper', 10) # prior parameter
        self.n_teams = game_tables[0].shape[0]
        self.game_tables = game_tables

        chains = []
        # generates all possible paths from every pair of teams
        for rd, games in enumerate(game_tables):
            games_parsed = self._parse(games)
            for i, j in combinations(range(self.n_teams), 2):
                print(f'generating paths for teams round {rd}: ({i}, {j})...', end='', flush=True)
                chain = [(rd, ch) for ch in find_all_paths(games_parsed, (i, j))]
                chains += chain
                print('done')

        self.chains = chains
        self.last_post = None # used for caching

        self.last_theta = self._init_theta()

        self.posterior = []

    def _init_theta(self):
        '''
        Initialise random scores between 0 and 2 for the first n-1 teams
        The last team has fixed score of 1 as a baseline
        '''
        return np.append(random(self.n_teams-1)*2, 1)

    # TODO: not a class method
    def _parse(self, games):
        '''
        Parses ndarray of games to directed graph in dictionary form
        '''
        out = defaultdict(list)
        for i, row in enumerate(games):
            for j, game in enumerate(row):
                if game != -1:
                    out[i].append(j)
        return out

    # TODO: Only makes sense for MH mode, not Gibbs mode
    @property
    def get_acceptance_ratio(self):
        return self.n_accepted/self.n_iters

    def _loglikelihood(self, chains, theta):
        '''
        chains -> data X. All possible chains between A and B from all team pairse A and B
        theta -> scores
        L(X|S) = Prod_{rd in rounds}(Prod_{chain in chains}(Prod_{game in games}(BernoulliPDF(game, s))))
        l(X|S) = Sum_{rd in rounds}(Sum_{chain in chains}(sum_{game in games}(LogBernoulliPDF(game, s))))
        '''
        out = []
        for rd_chain in chains:
            rd, chain = rd_chain
            n_chain = len(chain)
            logprob = np.array([scores2logprob(theta[chain[i]], theta[chain[i+1]]) for i in range(n_chain-1)])
            logprob1, logprob2 = logprob[:, 0], logprob[:, 1]
            results = np.array([self.game_tables[rd][chain[i]][chain[i+1]] for i in range(n_chain-1)])
            z = sum(results*logprob1 + (1-results)*logprob2)
            out.append(z)

        return sum(out)
    
    def _logprior(self, theta):
        '''
        Log Prior of scores. In this case beta
        '''
        theta_squash = theta[:-1]
        a, b = self.lower, self.upper
        return sum(beta.logpdf((theta_squash-a)/(b-a), self.alpha, self.beta))

    def _logpost(self, theta=None):
        '''
        Log posterior = log prior + log likelihood
        '''
        # caching
        if theta is None :
            if self.last_post:
                return self.last_post  
            theta = self.last_theta

        return self._logprior(theta) + self._loglikelihood(self.chains, theta)

    def _propose(self, i=None):
        '''
        Proposal distribution
        i -> only updates ith element
        '''
        if i:
            theta = deepcopy(self.last_theta)
            r = self.std*randn()
            theta[i] += r
            theta_new = np.clip(theta, self.lower, self.upper)
        else:
            theta = deepcopy(self.last_theta)
            r = np.append(self.std*randn(self.n_teams-1), 0)
            theta_new = np.clip(theta + r, self.lower, self.upper)
        return theta_new

    def _compare(self, prop):
        '''
        Accept/Reject proposed param
        '''
        a1 = self._logpost(prop)
        a2 = self._logpost()
        a = a1 - a2
        e = exponential(1)
        
        if a + e > 0:
            self.n_accepted += 1
            self.last_theta = prop
            self.last_post = a1 # cache last value
        else:
            self.last_post = a2 # cache last value

    def run(self, N, bip=0, mode='gibbs'):
        '''
        Main iteration method that combines all other methods
        N -> number of iterations
        bip -> burn-in-period
        mode -> choose between Gibbs and Metropolis Hastings
        '''

        for i in range(N):
            self.n_iters += 1
            
            if mode == 'gibbs':
                print(f'{i}/{N}={int(i/N*100)}%', end='\r')
                for j in range(self.n_teams-1):
                    prop = self._propose(j)
                    self._compare(prop)
            elif mode == 'mh':
                print(f'{i}/{N}={int(i/N*100)}%....accept rat = {self.get_acceptance_ratio*100:.2f}%', end='\r')
                prop = self._propose()
                self._compare(prop)

            else:
                raise ValueError('Please specify proper value of mode')

            if i > bip:
                self.posterior.append(self.last_theta)


class Simulator:
    '''
    Generates games and scores between a set of teams for a set of rounds
    '''

    def __init__(self, n_teams=8, rounds=1):
        self.n_teams = n_teams
        self.rounds = rounds

    def gen(self, lower=0.1, upper=3):
        power_points = np.append(random(self.n_teams-1)*(upper-lower)+lower, 1)
        game_tables = []

        for _ in range(self.rounds):
            game_table = np.zeros(shape=[self.n_teams, self.n_teams])-1
            for i in range(self.n_teams):
                for j in range(i+1, self.n_teams):
                    p = scores2prob(power_points[i], power_points[j])[0]

                    game_table[i][j] = int(bernoulli.rvs(p))
                    game_table[j][i] = 1-game_table[i][j]
            game_tables.append(game_table)

        return power_points, game_tables


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_teams', help='number of teams [def=5]', type=int, default=5)
    parser.add_argument('--n_rounds', help='number of rounds [def=3]', type=int, default=3)
    parser.add_argument('--N', help='number of MCMC iterations [def=1000]', type=int, default=1000)
    parser.add_argument('--bip', help='MCMC burn-in-period [def=100]', type=int, default=100)
    parser.add_argument('--mode', help='MCMC mode {gibbs, mh} [def=gibbs]', type=str, default='gibbs')
    args = parser.parse_args()
    n_teams = args.n_teams
    n_rounds = args.n_rounds
    N = args.N
    bip = args.bip
    mode = args.mode
    prior = {
        'lower': 0.1,
        'upper': 3
    }

    sim = Simulator(n_teams, n_rounds)
    p_points, g_tables = sim.gen()
    
    mcmc = MCMC(g_tables, prior=prior)
    mcmc.run(N, bip=bip, mode=mode)

    posterior = mcmc.posterior
    df_post = pd.DataFrame(posterior)
    post_mean = df_post.mean()
    lower_quantile = df_post.quantile(0.025)
    upper_quantile = df_post.quantile(0.975)
    
    df_post.plot(subplots=True, title='Trace Plots')
    axes = df_post.hist(alpha=0.5)
    for ax, score in zip(axes.flatten(), p_points):
        ax.axvline(x=score)
    
    plt.figure()
    plt.plot(p_points, color='blue', label='True score')
    plt.fill_between(range(mcmc.n_teams), lower_quantile, upper_quantile, color='orange', alpha=0.4)
    plt.plot(post_mean, color='orange', label='Posterior mean')
    plt.title('True score vs Posterior 95% quantile & mean')
    plt.ylabel('Score')
    plt.xlabel('Teams')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
