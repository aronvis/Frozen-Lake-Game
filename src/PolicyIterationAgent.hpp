//
// Created by Chi Zhang on 8/26/19.
//

#ifndef FROZEN_LAKE_POLICYITERATIONAGENT_HPP
#define FROZEN_LAKE_POLICYITERATIONAGENT_HPP

#include "LearningAgent.hpp"
#include "FrozenLake.hpp"
#include <unordered_map>

class PolicyIterationAgent : public ValueEstimateAgent {
public:
    PolicyIterationAgent(const FrozenLakeMDP &mdp, double gamma, int iterations, double threshold);

    double getQValue(const GameState &state, const Action &action) override;

    double getValue(const GameState &state) override;

    Action getPolicy(const GameState &state) override;

    std::string getName() const override {
        return "PolicyIterationAgent";
    }


protected:
    /*
     * We make this function protected and virtual so that it can be override with other method
     * such as solve the linear equations directly instead of iterate through.
     */
    virtual std::map<GameState, double> evaluateCurrentPolicy();

private:
    const FrozenLakeMDP &m_mdp;
    
    std::map<GameState, Action> m_policy;

    void initialize() override;

    void solve();

};


#endif //FROZEN_LAKE_POLICYITERATIONAGENT_HPP
