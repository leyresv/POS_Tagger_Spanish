# POS Tagger for Spanish with Hidden Markov Model and Viterbi Optimization

In this project I have implemented a Part-of-Speech Tagger for Spanish. It relies on a Hidden Markov Model optimized with the Viterbi algorithm.

## Models

### Hidden Markov Model (HMM):

HHMs are a probabilistic models that can be used to describe the observable sequential events that depend on internal (hidden) states. In our case, the observable events are the different words of a sentence, while the hidden states are their POS tags. HMMs allow us to recover the data from these hidden states.

HMMs rely on two matrices:

#### Transition matrix: 
It computes the probability of a POS tag (state) given the previous POS tag.

$$ P(s_i | s_{i-1}) =  \frac {Count(s_i | s_{i-1}) + \alpha}{Count(s_{i-1}) + \alpha * N_{states}} $$


*   $\alpha$ : smoothing parameter
*   $N_{states}$: total number of unique hidden states (POS tags)


#### Emission matrix:
It computes the probability of a word (observation) given a POS tag (state)

$$ P(o_i |s_i) =  \frac {Count(s_i | o_i) + \alpha}{Count(s_i) + \alpha * N_{observations}} $$


*   $N_{observations}$: total number of unique words in the vocabulary


### Viterbi algorithm

The Viterbi optimization is a dynamic programming algorithm that computes the most likely path in a Hidden Markov Model that results in a given sequence of observable events.
