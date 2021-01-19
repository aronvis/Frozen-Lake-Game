//
// Created by Chi Zhang on 8/26/19.
//

#include "PolicyIterationAgent.hpp"
#include "common.hpp"
#include <cmath>

PolicyIterationAgent::PolicyIterationAgent(const FrozenLakeMDP &mdp, double gamma, int iterations, double threshold) :
        ValueEstimateAgent(gamma, iterations, threshold), m_mdp(mdp) {
    MSG("Training Policy Iteration Agent on " << m_mdp.getName());
    MSG("Initializing Policy Iteration Agent");
    initialize();
    MSG("Solving...");
    solve();
}

// Computes and returns the q value for a given state and action 
double PolicyIterationAgent::getQValue(const GameState &state, const Action &action) 
{
    double stateValue = 0.0;
    std::map<GameState,double> stateValues = evaluateCurrentPolicy();
    std::map<GameState,double> probValues = m_mdp.getTransitionStatesAndProbs(state, action);
    for(std::pair<GameState,double> probValue: probValues)
    {   
        GameState nextState = probValue.first;
        double probability = probValue.second;
        stateValue += (probability * (m_mdp.getReward(state,action,nextState) + (m_gamma*stateValues[nextState])));
    }
    return stateValue;
}

// Returns the best possible state value based on the current policy
double PolicyIterationAgent::getValue(const GameState &state) 
{
    std::map<GameState,double> stateValues = evaluateCurrentPolicy();
    return stateValues[state];
}

// Returns the current policy for a given state
Action PolicyIterationAgent::getPolicy(const GameState &state) 
{
    return m_policy[state];
}

/*
 * Evaluate the current policy by returning V(s), which is represented as a map,
 * where key is GameState and value is double.
 */
std::map<GameState, double> PolicyIterationAgent::evaluateCurrentPolicy() 
{
    std::set<GameState> mStates = m_mdp.getStates();
    int numStates = mStates.size();
    std::map<GameState, double> stateValues;
    // Initializes the map that stores the state values 
    for(GameState state: mStates)
    {
        stateValues[state] = 0.0;
    }
    bool finished = false;
    while(!finished)
    {
        int counter = 0;
        // Performs policy iteration for each state
        for(GameState state: mStates)
        {
            // Updates state values and policies 
            if(!m_mdp.isTerminal(state))
            {
                double preStateValue = stateValues[state];
                double maxValue = INT_MIN;
                std::vector<Action> nextActions = m_mdp.getPossibleActions(state);
                // Computes the max value for a given state while exploring all actions
                for(Action nextAction: nextActions)
                {
                    // Computes a single q value
                    double stateValue = 0.0;
                    std::map<GameState,double> probValues = m_mdp.getTransitionStatesAndProbs(state, nextAction);
                    for(std::pair<GameState,double> probValue: probValues)
                    {   
                        GameState nextState = probValue.first;
                        double probability = probValue.second;
                        stateValue += (probability * (m_mdp.getReward(state, nextAction ,nextState) + (m_gamma*stateValues[nextState])));
                    }
                    // Stores max q value and updates policy
                    if (stateValue >= maxValue)
                    {
                        maxValue = stateValue;
                        stateValues[state] = maxValue;
                        m_policy[state] = nextAction;
                    }
                }
                double stateValue = stateValues[state];
                double difference = stateValue - preStateValue;
                // Counts the number of states that changed less than threshold
                if(std::abs(difference) < m_threshold)
                {
                    counter++;
                }   
            }
            else
            {
                counter++;
            }
        }
        // All states changed less than m_threshold
        if(counter == numStates)
        {
            finished = true;
        }
    }
    return stateValues;
}

// Preforms policy iteration to compute the best possible policy and value for each state
void PolicyIterationAgent::solve() 
{
    std::set<GameState> mStates = m_mdp.getStates();
    int numStates = mStates.size();
    for(int i = 0; i<m_iterations; ++i)
    {
        int counter = 0;
        std::map<GameState, Action> prevPolicy = m_policy;
        evaluateCurrentPolicy();
        std::map<GameState, Action> newPolicy = m_policy;
        for(GameState state: mStates)
        {
            if(prevPolicy[state] == newPolicy[state])
            {
                counter++;
            }
        }
        if(counter == numStates)
        {
            break;
        }
    }
}

// Initializes the map that stores the state policies 
void PolicyIterationAgent::initialize() 
{
    std::set<GameState> mStates = m_mdp.getStates();
    for(GameState state: mStates)
    {
        m_policy[state] = Action();
    }

}
