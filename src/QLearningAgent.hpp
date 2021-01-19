//
// Created by Chi Zhang on 8/27/19.
//

#ifndef FROZEN_LAKE_QLEARNINGAGENT_HPP
#define FROZEN_LAKE_QLEARNINGAGENT_HPP

#include "LearningAgent.hpp"
#include <random>
#include <unordered_map>
#include <map>
#include <random>

class QLearningAgent : public ValueEstimateAgent {
public:
    QLearningAgent(FrozenLakeEnv &env, double gamma, int iterations, double alpha, double epsilon);

    void incrementNValue(const GameState &state, const Action &action);

    bool newState(const GameState &state);

    void initState(const GameState &state);

    double getValue(const GameState &state) override;

    double getQValue(const GameState &state, const Action &action) override;

    Action getPolicy(const GameState &state) override;

    Action getAction(const GameState &state) override;

    std::string getName() const override {
        return "QLearningAgent";
    }

private:
    /*
     * alpha - learning rate for Q learning
     * epsilon - exploration rate for Q learning
     */

    double m_alpha;
    double m_epsilon;
    double mBeta;
    FrozenLakeEnv &m_env;
    std::unordered_map<GameState,std::map<Action, double>> mQValues;
    std::unordered_map<GameState,std::map<Action, int>> mNValues;
	std::default_random_engine mProbEngine;
	std::default_random_engine mActionEngine;
    std::uniform_real_distribution<double> mProbDist;
    std::uniform_int_distribution<int> mActionDist;

    void update(const GameState &state, const Action &action, const GameState &nextState, double reward);

    void solve();

    void initialize() override;
};


#endif //FROZEN_LAKE_QLEARNINGAGENT_HPP
