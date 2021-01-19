//
// Created by Chi Zhang on 8/24/19.
//

#ifndef FROZEN_LAKE_VALUEITERATIONAGENT_HPP
#define FROZEN_LAKE_VALUEITERATIONAGENT_HPP


#include "LearningAgent.hpp"
#include "FrozenLake.hpp"
#include <unordered_map>
#include <set>
#include <map>

class ValueIterationAgent : public ValueEstimateAgent {
public:

    ValueIterationAgent(FrozenLakeMDP const &mdp, double gamma, int iterations, double threshold);

    double getValue(const GameState &state) override;

    double getQValue(const GameState &state, const Action &action) override;

    Action getPolicy(const GameState &state) override;

    void setValue(const GameState &state, double value);

    void setPolicy(const GameState &state, const Action &action);

    double computeValue(const GameState &state);

    std::string getName() const override 
    {
        return "ValueIterationAgent";
    }

private:
    void initialize() override;

    void solve();

    const FrozenLakeMDP &m_mdp;

    std::unordered_map<GameState, double> mStateValues;

    std::unordered_map<GameState, Action> mStatePolicies;

    std::set<GameState> mStates;
};


#endif //FROZEN_LAKE_VALUEITERATIONAGENT_HPP
