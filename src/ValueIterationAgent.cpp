//
// Created by Chi Zhang on 8/24/19.
//

#include "common.hpp"
#include "ValueIterationAgent.hpp"
#include <iostream>

ValueIterationAgent::ValueIterationAgent(FrozenLakeMDP const &mdp, double gamma, int iterations, double threshold) :
        ValueEstimateAgent(gamma, iterations, threshold), m_mdp(mdp) {
    MSG("Training Value Iteration Agent on " << m_mdp.getName());
    MSG("Initializing Value Iteration Agent");
    initialize();
    MSG("Solving...");
    solve();
}

// Updates the best possible value we can obtain from a given state 
void ValueIterationAgent::setValue(const GameState &state, double value)
{
	mStateValues[state] = value;
}

// Updates the best possible policy we can execute from a given state
void ValueIterationAgent::setPolicy(const GameState &state, const Action &action)
{
 	mStatePolicies[state] = action;
}

// Computes and returns the best possible value we can obtain from a given state 
// for the current iteration and updates the policy for a given state
double ValueIterationAgent::computeValue(const GameState &state)
{
	double maxValue = INT_MIN;
	std::vector<Action> nextActions = m_mdp.getPossibleActions(state);
	for(Action nextAction: nextActions)
	{
		double stateValue = getQValue(state, nextAction);
		if (stateValue >= maxValue)
		{
			maxValue = stateValue;
			setPolicy(state, nextAction);
		}
	}
	return maxValue;
}

// Returns the best possible value we can obtain from a given state based on the last iteration
double ValueIterationAgent::getValue(const GameState &state) 
{
	return mStateValues[state];
}

// Computes and returns the q value for a given state and action 
double ValueIterationAgent::getQValue(const GameState &state, const Action &action) 
{
	double stateValue = 0.0;
	std::map<GameState,double> probValues = m_mdp.getTransitionStatesAndProbs(state, action);
	for(std::pair<GameState,double> probValue: probValues)
	{	
		GameState nextState = probValue.first;
		double probability = probValue.second;
		stateValue += (probability * (m_mdp.getReward(state, action ,nextState) + (m_gamma*getValue(nextState))));
	}
	return stateValue;
}

// Returns the best possible policy for a given state
Action ValueIterationAgent::getPolicy(const GameState &state) 
{
    return mStatePolicies[state];
}

// Preforms value iteration to compute the best possible value and policy for each state
void ValueIterationAgent::solve() 
{
	int numStates = mStates.size();
	for(int i = 0; i<m_iterations; ++i)
    {
    	int counter = 0;
    	for(GameState state: mStates)
    	{
            if(!m_mdp.isTerminal(state))
            {
                double preStateValue = getValue(state);
                double stateValue = computeValue(state);
                double difference = stateValue - preStateValue;
                setValue(state, stateValue); 
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
    	// Break out of the loop if each state change less than the threshold
    	if(counter == numStates)
    	{
    		break;
    	}
    }
}

// Initializes the hashmaps that store the state values and policies 
void ValueIterationAgent::initialize() 
{
	mStates = m_mdp.getStates();
    for(GameState state: mStates)
    {
    	mStateValues[state] = 0.0;
    	mStatePolicies[state] = Action();
    }
}

