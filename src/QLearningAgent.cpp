//
// Created by Chi Zhang on 8/27/19.
//

#include "common.hpp"
#include "third_party/random.hpp"
#include "QLearningAgent.hpp"
#include <iostream>
#include <math.h> 
#include <fstream>

QLearningAgent::QLearningAgent(FrozenLakeEnv &env, double gamma, int iterations, double alpha, double epsilon) :
        ValueEstimateAgent(gamma, iterations, 0.0), m_alpha(alpha), m_epsilon(epsilon), m_env(env) {
    MSG("Training Q Learning Agent on " << m_env.getName());
    MSG("Initializing Q Learning Agent");
    initialize();
    MSG("Solving...");
    solve();
}

// Increments the frequency that a state and action has been explored by 1
void QLearningAgent::incrementNValue(const GameState &state, const Action &action)
{
    mNValues[state][action]++;
}

// Returns true if a state has never been explored before
bool QLearningAgent::newState(const GameState &state)
{
	return mQValues.find(state) == mQValues.end();
}

// Initializes a state and its possible actions and inserts those items into mQValues - q(s,a)
void QLearningAgent::initState(const GameState &state)
{
	std::vector<Action> actions = m_env.getPossibleActions(state);
	std::map<Action, double> actionMap;
    std::map<Action, int> freqMap;
	for(Action action: actions)
	{
		actionMap[action] = 0.0;
        freqMap[action] = 1;
	}
	mQValues[state] = actionMap;
    mNValues[state] = freqMap;
}

// Returns the best possible value for a given state 
double QLearningAgent::getValue(const GameState &state) 
{
	// Returns 0 for a terminal state
	if(m_env.isTerminal(state))
	{
		return 0.0;
	}
	// Adds the next state to the data struture 
	// if it is not already there
	if(newState(state))
    {
    	initState(state);
    }
    Action bestActon = getPolicy(state);
    return mQValues[state][bestActon];
}

// Returns the value stored in the q-table for a given state and action
double QLearningAgent::getQValue(const GameState &state, const Action &action) 
{
	// Returns 0 for a terminal state
	if(m_env.isTerminal(state))
	{
		return 0.0;
	}
	// Adds the next state to the data struture 
	// if it is not already there
	if(newState(state))
    {
    	initState(state);
    }
    return mQValues[state][action];
}

// The final policy without exploration. Used for evaluation.
Action QLearningAgent::getPolicy(const GameState &state) 
{
    // Frequency based learning
	// if(newState(state))
 //    {
 //    	initState(state);
 //    }
 //    std::vector<Action> actions = m_env.getPossibleActions(state);
 //    std::map<Action,int> freqMap = mNValues[state];
 //    double maxValue = INT_MIN;
 //    Action bestAction = actions[0];
 //    for(Action action: actions)
 //    {
 //        double value = getQValue(state, action) + (mBeta * sqrt(1/mNValues[state][action]));
 //        if(value >= maxValue)
 //        {
 //            maxValue = value;
 //            bestAction = action;
 //        }
 //    }
 //    return bestAction;

    // Probability based exploration
    // Returns 0 for a terminal state
	if(m_env.isTerminal(state))
	{
		return LEFT;
	}
    if(newState(state))
    {
        initState(state);
    }
    std::vector<Action> actions = m_env.getPossibleActions(state);
    std::map<Action,double> actionMap = mQValues[state];
    double maxValue = INT_MIN;
    Action bestAction = actions[0];
    for(Action action: actions)
    {
    	if(actionMap[action] > maxValue)
    	{
    		maxValue = actionMap[action];
    		bestAction = action;
    	}
    	else if(actionMap[action] == maxValue)
    	{
    		mActionDist = std::uniform_int_distribution<int>(0,1);
    		int actionIndex = mActionDist(mActionEngine);
    		if(actionIndex == 0)
    		{
    			maxValue = actionMap[action];
    			bestAction = action;
    		}
    	}
    }
    return bestAction;
}

// you should use getAction in solve instead of getPolicy and implement your exploration strategy here.
Action QLearningAgent::getAction(const GameState &state) 
{
    // Frequency based learning
    // std::vector<Action> actions = m_env.getPossibleActions(state);
    // Action nextAction = getPolicy(state);
    // return nextAction;

    // Probability based exploration
    std::vector<Action> actions = m_env.getPossibleActions(state);
    double randResult = mProbDist(mProbEngine);
    Action nextAction = LEFT;
    if(randResult <= m_epsilon)
    {
    	mActionDist = std::uniform_int_distribution<int>(0,actions.size()-1);
    	int actionIndex = mActionDist(mActionEngine);
    	nextAction = actions[actionIndex];
    }
    else
    {
    	nextAction = getPolicy(state);
    }
    return nextAction;
}

// Computes and updates the q value inside the table
void QLearningAgent::update(const GameState &state, const Action &action, const GameState &nextState, double reward) 
{
	double prevQ = getQValue(state, action);
	double nextQ = getValue(nextState);
	mQValues[state][action] = prevQ + (m_alpha*(reward + (m_gamma*nextQ) - prevQ));
    incrementNValue(state,action);
}


void QLearningAgent::solve() 
{
    // output a file for plotting
    std::ofstream outFile;
    outFile.open("result.csv");
    outFile << "Episode,Reward" << std::endl;
    int maxEpisodeSteps = 100;
    // collect m_iterations trajectories for update
    for (int i = 0; i < m_iterations; i++)
    {
        int numSteps = 0;
        GameState state = m_env.reset();
        while (!m_env.isTerminal(state)) 
        {
            Action action = getAction(state);
            GameState nextState = m_env.getNextState(state, action);
            double reward = m_env.getReward(state, action, nextState);
            update(state, action, nextState, reward);
            state = nextState;
            numSteps += 1;
            if (numSteps >= maxEpisodeSteps) break;  // avoid infinite loop in some cases.
        }
        // evaluate for 100 episodes using the current optimal policy. You can't change this line.
        double episodeReward = m_env.runGame(*this, 100, m_gamma, false).first;
        std::cout << "Evaluating episode reward at learning iteration " << i << " is " << episodeReward << std::endl;
        outFile << i << "," << episodeReward << std::endl;
    }
    outFile.close();
}

// Intialize random generators
void QLearningAgent::initialize() 
{
	std::random_device rd1;
	std::random_device rd2;
	mProbEngine = std::default_random_engine(rd1());
	mActionEngine = std::default_random_engine(rd2());
	mProbDist = std::uniform_real_distribution<double>(0.0, 1.0);
	mActionDist = std::uniform_int_distribution<int> (0,3);
    mBeta = 500.0;
}
