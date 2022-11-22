import numpy as np
import requests
from random import shuffle
from string import ascii_uppercase
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from numpy.random import random, randn, exponential
from scipy.stats import bernoulli, beta, expon
import matplotlib.pyplot as plt
from itertools import combinations
import networkx as nx


def savefig(fn):
    plt.savefig(fn, bbox_inches='tight', pad_inches=0.1, dpi=1000, format='pdf')

def RMSE(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.mean((x-y)**2))


np.seterr('raise')
def scores2logprob(a, b):
    '''
    Translates two scores to log probabilities
    a, b -> scores of first and second team
    Returns: logprob that team A beats team B, and vice versa
    '''
    try:
        z1 = np.log(a) - np.log(a+b)
        z2 = np.log(b) - np.log(a+b)
    except:
        return np.nan, np.nan
    return z1, z2


def scores2prob(a, b):
    '''
    Same as above, but no log
    '''
    z = a/(a+b)
    return z, 1-z

def scores2odds(scores):
    df = pd.DataFrame(scores)
    return df.dot((1/df).T)

class MCMC:
    '''
    Performs MCMC iterations to return posterior of scores
    '''

    def __init__(self, game_tables, max_chain_length=3, **kwargs):
        '''
        game_tables: Poissibly unfinished tables of games
        '''
        self.std = 0.1 # used for proposing next element
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
            G = nx.Graph(self._parse(games))
            for i, j in combinations(range(self.n_teams), 2):
                print(f'generating paths for teams round {rd+1}: ({i}, {j})...', end='', flush=True)
                for k in range(1, max_chain_length):
                    C = nx.all_simple_paths(G, i, j, k)
                    C = [c for c in C]
                    if len(C): break
                chains += [(rd, ch) for ch in C]

                print('done', end='\r')
        print()

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
                if game != -1 and not np.isnan(game):
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
    
    def _logprior_OLD(self, theta):
        '''
        Log Prior of scores. In this case beta
        '''
        theta_squash = theta[:-1]
        a, b = self.lower, self.upper
        return sum(beta.logpdf((theta_squash-a)/(b-a), self.alpha, self.beta))

    def _logprior(self, theta):
        '''
        Log Prior of scores. In this case exponential
        '''
        theta_squash = theta[:-1]
        x = sum(expon.logpdf(theta_squash, scale=self.alpha))
        return x

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
        if i is not None:
            theta = deepcopy(self.last_theta)
            r = self.std*randn()
            theta[i] += r
            #theta_new = np.clip(theta, self.lower, self.upper)
            theta_new = np.copy(theta)
        else:
            theta = deepcopy(self.last_theta)
            r = np.append(self.std*randn(self.n_teams-1), 0)
            # theta_new = np.clip(theta + r, self.lower, self.upper)
            theta_new = np.copy(theta)
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
                idx = list(range(self.n_teams-1))
                shuffle(idx)
                for j in idx:
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

    def __init__(self, teams=5, rounds=1):
        if isinstance(teams, int): 
            teams = list(ascii_uppercase[:teams])
        self.n_teams = len(teams)
        self.rounds = rounds
        self.teams = teams

    def gen(self, lower=0.5, upper=2, pct=1, mask=True):

        power_points = np.append(random(self.n_teams-1)*(upper-lower)+lower, 1)
        game_tables = []

        if not mask:
            for _ in range(self.rounds):
                game_table = np.zeros(shape=[self.n_teams, self.n_teams])-1
                for i, j in combinations(range(self.n_teams), 2):
                    p = scores2prob(power_points[i], power_points[j])[0]
                    if random() > pct: continue

                    game_table[i][j] = int(bernoulli.rvs(p))
                    game_table[j][i] = 1-game_table[i][j]
                game_tables.append(game_table)
        else:
            game_mask = np.zeros(shape=[self.n_teams, self.n_teams])
            for i, j in combinations(range(self.n_teams), 2):
                game_mask[i][j] = random() < pct
                game_mask[j][i] = game_mask[i][j]

            for _ in range(self.rounds):
                game_table = np.zeros(shape=[self.n_teams, self.n_teams])-1
                for i, j in combinations(range(self.n_teams), 2):
                    p = scores2prob(power_points[i], power_points[j])[0]
                    game_table[i][j] = int(bernoulli.rvs(p))
                    game_table[j][i] = 1-game_table[i][j]
                game_table = np.where(game_mask, game_table, -1)
                game_tables.append(game_table)


        return power_points, [game_tables]


def main(p_points, g_tables, N, bip, mode, mcl=3):
    prior = {
        'lower': 0.5,
        'upper': 2.5
    }

    print(pd.DataFrame(g_tables[0], columns=sim.teams, index=sim.teams).astype(int))
    
    mcmc = MCMC(g_tables, mcl, prior=prior)
    mcmc.run(N, bip=bip, mode=mode)

    posterior = pd.DataFrame(mcmc.posterior, columns=sim.teams)
    posterior_nofixed = posterior.drop([sim.teams[-1]], axis=1)

    posterior_nofixed.plot(subplots=True, title='Trace Plots')
    savefig('trace.pdf')

    max_val = posterior_nofixed.max().max()
    axes = posterior_nofixed.hist(alpha=0.5, bins=50, density=True, range=(0, max_val))
    post_mode = []

    # TODO: beautify code
    for i, (ax, col) in enumerate(zip(axes.flatten(), posterior_nofixed)):
        if not args.division:
            ax.axvline(x=p_points[i], color='orange')
     
        kde = posterior_nofixed[col].plot.kde(ax=ax)
        z = int(not args.division)
        mode = kde.lines[z].get_xdata()[np.argmax(kde.lines[z].get_ydata())]
        post_mode.append(mode)
    post_mode.append(1)
    
    savefig('posterior.pdf')
    plt.figure()
    post_mode = pd.Series(post_mode, index=sim.teams)

    df_post_mode = pd.DataFrame(post_mode)
    mode_odds = df_post_mode.dot((1/df_post_mode).T)


    print('PRED SCORES')
    print(post_mode)
    print('PRED ODDS')
    print(mode_odds)
    pred_order = post_mode.sort_values(ascending=False).index

    AA = np.hstack(g_tables)
    AA[AA==-1] = np.nan
    standings = np.nanmean(AA, axis=1)
    standings = pd.Series(index=sim.teams, data=standings).sort_values(ascending=False).index

    if not args.division:
        df_true_scores = pd.DataFrame(p_points, index=sim.teams)
        true_odds = df_true_scores.dot((1/df_true_scores).T)
        print('TRUE SCORES')
        print(p_points)
        print('TRUE ODDS')
        print(true_odds)

        A = pd.Series(index=sim.teams, data=p_points)
        true_order = A.sort_values(ascending=False).index 
        orders = pd.DataFrame(data={'True': true_order, 'Pred': pred_order, 'Standings': standings})
        print('PREDICTED TEAM POWER')
        print(orders)

        rmse = RMSE(mode_odds, true_odds)
        print(f'{mcl=}: RMSE={rmse}')

    lower_quantile = posterior.quantile(0.025)
    upper_quantile = posterior.quantile(0.975)
    
    if not args.division:
        plt.plot(p_points, color='blue', label='True score', marker='o')
    plt.fill_between(range(mcmc.n_teams), lower_quantile, upper_quantile, color='orange', alpha=0.4)
    plt.plot(post_mode, color='orange', label='Posterior mode', marker='o')
    #plt.title('True score vs Posterior 95% quantile & mean')
    plt.ylabel('Score')
    plt.xlabel('Teams')
    savefig('post_mode.pdf')

    plt.legend()
    #plt.show()

def clean(x):
    if not isinstance(x, str): return
    return (int(x[0])>=3)*1

def get_real(division=4):
    url = 'https://www.sportyhq.com/club/box/view/60'
    header = {
      "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
      "X-Requested-With": "XMLHttpRequest"
    }

    r = requests.get(url, headers=header)
    dfs = pd.read_html(r.text)

    # for table in dfs[division-1]:
    table = dfs[division-1]

    names = [name.split()[-1] for name in table.iloc[:,1]]
    table = table.drop(table.columns[[0, 1, -1]], axis=1)
    table.columns = names
    table.index = names
    table = table.applymap(clean).fillna(-1)
    noplays = (table==-1).all()
    table = table.loc[~noplays, ~noplays]
    return None, table


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_teams', help='number of teams [def=5]', type=int, default=5)
    parser.add_argument('--n_rounds', help='number of rounds [def=1]', type=int, default=1)
    parser.add_argument('--N', help='number of MCMC iterations [def=1000]', type=int, default=1000)
    parser.add_argument('--bip', help='MCMC burn-in-period [def=0]', type=int, default=0)
    parser.add_argument('--chain_length', help='maximum chain length [def=2]', type=int, default=2)
    parser.add_argument('--mode', help='MCMC mode {gibbs, mh} [def=gibbs]', type=str, default='gibbs')
    parser.add_argument('--pct', help='Percentage of games played [def=1]', type=float, default=1)
    parser.add_argument('--division', help='use real data', type=int, default=False)
    args = parser.parse_args()
    n_teams = args.n_teams
    n_rounds = args.n_rounds
    N = args.N
    bip = args.bip
    mode = args.mode
    mcl = args.chain_length

    if args.division:
        p_points, g_tables = get_real(args.division)
        sim = Simulator(g_tables.index)
        g_tables = [np.array(g_tables)]
    else:
        sim = Simulator([f'T_{i+1}' for i in range(n_teams)], n_rounds)
        p_points, g_tables = sim.gen(pct=args.pct)


    main(p_points, g_tables, N, bip, mode, mcl)
    plt.show()
