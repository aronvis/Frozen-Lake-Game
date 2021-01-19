//
// Created by Chi Zhang on 8/22/19.
//

#include "FrozenLake.hpp"
#include <cassert>
#include <fstream>
#include <utility>
#include "third_party/random.hpp"
#include "third_party/json.hpp"
#include "common.hpp"

using json = nlohmann::json;

bool ajacentAction(Action action1, Action action2) {
    if (action1 == action2) return false;
    if (action1 == LEFT) {
        return action2 != RIGHT;
    } else if (action1 == DOWN) {
        return action2 != UP;
    } else if (action1 == RIGHT) {
        return action2 != LEFT;
    } else {
        return action2 != DOWN;
    }
}


// implementations for GameState API

GameState::GameState(const xyLoc &loc, const std::string &name) : m_loc(loc), m_name(name) {

}


GameState::GameState(const GameState &state) : m_loc(state.m_loc), m_name(state.m_name) {

}

bool GameState::operator<(const GameState &state) const {
    return m_loc < state.m_loc;
}

bool GameState::operator==(const GameState &state) const {
    return m_loc == state.m_loc && m_name == m_name;
}

xyLoc GameState::getLoc() const {
    return this->m_loc;
}

std::string GameState::getName() const {
    return this->m_name;
}


// implementations for FrozenLakeEnv API

std::ostream &operator<<(std::ostream &out, const FrozenLakeEnv &mdp) {
    for (auto &state : mdp.m_transitions) {
        out << "State " << state.first << std::endl;
        for (auto &action : state.second) {
            out << "  Action " << action.first << " -> " << std::endl;
            for (auto &tuple : action.second) {
                out << "    Next State" << std::get<1>(tuple) << ". Prob " << std::get<0>(tuple) << ". Reward "
                    << std::get<2>(tuple) << std::endl;
            }
        }
    }
    return out;
}

std::map<GameState, double>
FrozenLakeEnv::getTransitionStatesAndProbsEnv(const GameState &state, const Action &action) const {
    std::map<GameState, double> transitionStateAndProbs;
    auto stateIt = m_transitions.find(state);
    if (stateIt == m_transitions.end()) {
        MSG("State " << state << " is not in the environment.");
        exit(-1);
    }
    auto tuple_pair = stateIt->second.find(action);
    if (tuple_pair == stateIt->second.end()) {
        MSG("Action " << action << " not allowed in the State " << state);
        exit(-1);
    }
    auto tuple = tuple_pair->second;
    for (auto &it : tuple) {
        transitionStateAndProbs[std::get<1>(it)] = std::get<0>(it);
    }
    return transitionStateAndProbs;
}

GameState FrozenLakeEnv::reset() {
    return *Random::get(m_startStates);
}


GameState FrozenLakeEnv::getNextState(const GameState &state, const Action &action) {
    auto nextStatesAndProbs = getTransitionStatesAndProbsEnv(state, action);
    std::vector<GameState> states;
    states.reserve(nextStatesAndProbs.size());
    std::vector<double> probs;
    probs.reserve(nextStatesAndProbs.size());
    for (auto &it : nextStatesAndProbs) {
        states.push_back(it.first);
        probs.push_back(it.second);
    }
    auto index = Random::get<std::discrete_distribution<int>>(probs.begin(), probs.end());
    return states.at(index);
}

std::vector<Action> FrozenLakeEnv::getPossibleActions(const GameState &state) const {
    std::vector<Action> actions;
    auto it = m_transitions.find(state);
    if (it == m_transitions.end()) {
        MSG("Unknown state " << state);
    }
    auto &stateTransitions = it->second;
    for (const auto &m_transition : stateTransitions) {
        actions.push_back(m_transition.first);
    }
    return actions;
}

double FrozenLakeEnv::getReward(const GameState &state, const Action &action, const GameState &nextState) const {
    auto stateIt = m_transitions.find(state);
    if (stateIt == m_transitions.end()) {
        MSG("State " << state.getName() << " is not in the environment");
        exit(-1);
    }
    auto tuple_pair = stateIt->second.find(action);
    if (tuple_pair == stateIt->second.end()) {
        MSG("Action " << action << " is not allowed in the State " << state);
        exit(-1);
    }
    auto tuple = tuple_pair->second;
    for (auto &it : tuple) {
        if (std::get<1>(it) == nextState) {
            return std::get<2>(it);
        }
    }
    // program can't reach here.
    MSG("Fail to find next state " << nextState << " in state " << state << " when taking action " << action);
    exit(-1);
}

bool FrozenLakeEnv::isTerminal(const GameState &state) const {
    return m_terminalStates.find(state) != m_terminalStates.end();
}


FrozenLakeEnv::FrozenLakeEnv(const std::string &jsonFile) {
    std::ifstream inputFile(jsonFile);
    json object;
    inputFile >> object;

    const std::string &name = jsonFile;
    std::string grid = object["grid"];
    // read the grid as a 2D array
    auto map = this->parserGrid(grid);
    std::map<std::string, double> rewards = object["reward"];
    double noise = object["noise"];
    init(name, map, rewards, noise);
}

FrozenLakeEnv::FrozenLakeEnv(int height, int width) {

}

FrozenLakeEnv::FrozenLakeEnv() : FrozenLakeEnv(4, 4) {

}

std::pair<double, double> FrozenLakeEnv::runGame(Agent &agent, int numEpisodes, double gamma, bool verbose) {
    if (verbose) {
        MSG("Evaluating " << agent.getName() << " on map " << this->getName() << " for " << numEpisodes << " episodes");
    }
    // default is 10 epochs
    int maxEpisodeSteps = 100;
    std::vector<double> epRewards;
    for (int i = 0; i < numEpisodes; ++i) {
        double episodeScore = 0.;
        int numSteps = 0;
        auto currentState = this->reset();
        while (!this->isTerminal(currentState)) {
            Action action = agent.getPolicy(currentState);
            auto nextState = this->getNextState(currentState, action);
            double reward = this->getReward(currentState, action, nextState);
            episodeScore += reward * pow(gamma, numSteps);
            currentState = nextState;
            numSteps += 1;
            if (numSteps >= maxEpisodeSteps) break;  // to avoid infinite loop.
        }
        epRewards.push_back(episodeScore);
//        std::cout << "Episode " << i + 1 << ", Reward " << episodeScore << std::endl;
    }
    // calculate mean and standard deviation
    double mean = 0.0;
    double standard_deviation = 0.0;
    for (auto &r : epRewards) {
        mean += r;
    }
    mean = mean / epRewards.size();
    for (auto &r: epRewards) {
        standard_deviation += (r - mean) * (r - mean);
    }
    standard_deviation = sqrt(standard_deviation / epRewards.size());
    if (verbose) {
        MSG("Result: Mean Reward " << mean << ", Standard Deviation " << standard_deviation);
    }
    return std::make_pair(mean, standard_deviation);
}

void
FrozenLakeEnv::init(std::string name, vec2DGameState map, std::map<std::string, double> reward,
                    double noise) {
    m_name = std::move(name);
    m_map = std::move(map);
    // iterate to create transitions
    for (size_t row = 0; row < m_map.size(); ++row) {
        for (size_t col = 0; col < m_map[row].size(); ++col) {
            GameState currentState = m_map[row][col];
            std::map<Action, std::vector<ProbStateRewardTuple>> transition;
            auto stateName = currentState.getName()[0];
            if (stateName == 'S') {
                m_startStates.insert(currentState);
            }
            if (stateName == 'G' || stateName == 'H') {
                m_terminalStates.insert(currentState);
                m_transitions[currentState] = transition;
                continue;
            }
            // get the surrounding states and add to the transition
            std::vector<std::pair<Action, GameState>> nextStates;
            if (row > 0) {
                nextStates.emplace_back(UP, m_map[row - 1][col]);
            }
            if (row < m_map.size() - 1) {
                nextStates.emplace_back(DOWN, m_map[row + 1][col]);
            }
            if (col > 0) {
                nextStates.emplace_back(LEFT, m_map[row][col - 1]);
            }
            if (col < m_map[row].size() - 1) {
                nextStates.emplace_back(RIGHT, m_map[row][col + 1]);
            }
            for (size_t i = 0; i < nextStates.size(); ++i) {
                // initialize transition at this action
                transition[nextStates[i].first] = std::vector<ProbStateRewardTuple>();
                // insert the expected next state
                double currentReward = reward[nextStates[i].second.getName()];
                transition[nextStates[i].first].push_back(std::make_tuple(1 - noise, nextStates[i].second,
                                                                          currentReward));
                // if noise, insert all other possible state
                if (noise > 0. && nextStates.size() > 1) {
                    int numOtherPossibleNextStates = 0;
                    for (size_t j = 0; j < nextStates.size(); ++j) {
                        if (j != i && ajacentAction(nextStates[i].first, nextStates[j].first)) {
                            numOtherPossibleNextStates += 1;
                        }
                    }
                    double eachNoise = noise / ((double) numOtherPossibleNextStates);
                    for (size_t j = 0; j < nextStates.size(); ++j) {
                        if (j != i && ajacentAction(nextStates[i].first, nextStates[j].first)) {
                            currentReward = reward[nextStates[j].second.getName()];
                            transition[nextStates[i].first].push_back(std::make_tuple(eachNoise, nextStates[j].second,
                                                                                      currentReward));
                        }
                    }
                }
            }
            m_transitions[currentState] = transition;
        }
    }
}

std::string FrozenLakeEnv::getName() const {
    return m_name;
}


// implementations for FrozenLakeEnv API
std::set<GameState> FrozenLakeMDP::getStates() const {
    std::set<GameState> stateSet;
    for (auto &state : m_transitions) {
        stateSet.insert(state.first);
    }
    return stateSet;
}


std::map<GameState, double>
FrozenLakeMDP::getTransitionStatesAndProbs(const GameState &state, const Action &action) const {
    return getTransitionStatesAndProbsEnv(state, action);
}

