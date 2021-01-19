//
// Created by Chi Zhang on 8/24/19.
//

#ifndef FROZEN_LAKE_LEARNINGAGENT_HPP
#define FROZEN_LAKE_LEARNINGAGENT_HPP

#include "FrozenLake.hpp"

/*
 * Abstract agent which assigns values to (state,action) Q-Values for an environment.
 * As well as a value to a state and a policy given respectively by, V(s) = max_{a in actions} Q(s,a)
 * policy(s) = arg_max_{a in actions} Q(s,a)
 * Both ValueIterationAgent and QLearningAgent inherit from this agent. While a ValueIterationAgent has
 * a model of the environment via a MarkovDecisionProcess that is used to estimate Q-Values before
 * ever actually acting, the QLearningAgent estimates
 * Q-Values while acting in the environment.
 */


class ValueEstimateAgent : public Agent {
public:
    ValueEstimateAgent();

    ValueEstimateAgent(double gamma, int iterations, double threshold);

    virtual ~ValueEstimateAgent() = default;

    virtual double getQValue(const GameState &state, const Action &action) = 0;

    virtual double getValue(const GameState &state) = 0;

    /*
     * getAction differs from getPolicy in the sense that getAction can incorporate exploration.
     * The default getAction is the same as getPolicy. QLearner will override this function.
     */
    virtual Action getAction(const GameState &state) {
        return this->getPolicy(state);
    }

protected:
    /*
     * m_gamma - discount factor
     * m_iterations - number of training iterations
     * m_threshold - error threshold
     */

    double m_gamma{};
    int m_iterations{};
    double m_threshold{};

    virtual void initialize() {

    }
};


#endif //FROZEN_LAKE_LEARNINGAGENT_HPP
