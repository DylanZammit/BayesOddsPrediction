# Bayes Odds Inference
In the rest of this discussion we shall denote `A beats B' by A > B. Moreover, Pr(A > B) denotes the probability of A > B, and Odds(A > B) are the odds that A > B.
## Motivation
Consider a league of different teams of varying skills and abilities. The act of "winning" is obviously not transitive: A>B and B>C clearly does not imply that A>C. There are many factors that might determine this result invalid, perhaps team morale of A is low on the day, perhaps one of the first two matches were a fluke, perhaps weather played an effect etc. However, we would expect that this relationship does indeed convey at least *some* information about a future encounter between A and C. This leads us to our primary assumption in this article, which states that if

> Odds(A>B)>1 and Odds(B>C)>1 then Odds(A>C)>1.

We go a step further and assign a score s_i to a team T_i to create a vector S of such scores. Each score represents the "strength" of the team relative to the others. So the odds of T_i>T_j would be s_i/s_j. The above assumption clearly holds. Suppose that s_1/s_2>1 and s_2/s_3>1. Then s_1/s_3=(s_1/s_2)(s_2/s_3)=1. The probability that team T_i > T_j is given by p_ij:=Pr(T_i>T_j)=s_i/(s_i+s_j).

The goal of this implementation is to extract information from "chains" of games starting from one team and ending at another, even if there are no direct encounters between these two teams. Thus, our dataset are no longer just the games played. Instead, each datum is a whole chain of games, with each game given a Bernoulli distribution with probability p_ij. The games are assumed to be independent as well as the chains.

NOTE: This last assumption about independence between chains is most likely a serious flaw in my reasoning, and a further inspection is required. For now I will assume it to be true for simplicity.
## Implementation
By placing a Bernoulli distribution on each game and assuming independence between games and chains, we are able to place a likelihood on the data. We now place a prior distribution over the space of unknown parameters. If there are m teams, we need to estimate m-1 scores, and the score of the final team is fixed to 1. In this way, scaling of the scores is not an issue, and all scores can be viewed as relative to the final team. In this implementation, we place a beta distribution over the score, by choosing an appropriate upper and lower bound for the scores. However, more appropriate choices of priors are also possible, perhaps one that ranges over the whole positive numbers such as the exponential distribution.

Now that we defined the likelihood for the data and the prior, we can invoke Bayes Theorem to come up with the posterior distribution. To obtain point estimates (such as the posterior mean) from such a distribution is no simple task, as integrating the distribution analytically is extremely difficult if not impossible. For this reason, we opt for a sampling technique via Monte Carlo Markov Chains (MCMC). More specifically we use Gibbs sampling to sample from the condition posterior iteratively until convergence is reached.
## How to run
After the appropriate libraries are installed, the script can be run via the command line
`python odds_mcmc.py --n_teams 5 --n_rounds 3 --N 1000 --bip 100 --mode gibbs --pct 1`
All arguments above are optional. We give a brief explanation of each:

 - n_teams: Number of teams to be considered
 - n_rounds: Number of rounds for the group of teams. At most n_rounds number of games can be played with "n_rounds" rounds.
 - pct: Percentage of games played. 0 means that no games are played and 1 means that all pair of teams played in each round.
 - N: Number of MCMC iterations to be performed
 - bip: Burn-In-Period is the number of iterations for which we discard posterior samples in the MCMC simulation
 - mode: Choose between Gibbs (gibbs) sampling or Metropolis-Hastings (mh) for the MCMC algorithm. Metropolis-Hastings is significantly faster per epoch, however requires much more iterations until convergence.

