//
// Created by Chi Zhang on 8/22/19.
//

#ifndef FROZEN_LAKE_FROZENLAKE_HPP
#define FROZEN_LAKE_FROZENLAKE_HPP

#include <iostream>
#include <vector>
#include <map>
#include <tuple>
#include <set>
#include <random>
#include <string>
#include <iterator>
#include <sstream>

class Agent;

enum Action {
    LEFT, DOWN, RIGHT, UP
};

class xyLoc {
public:

    int x;
    int y;

    xyLoc() {
        x = -1;
        y = -1;
    }

    xyLoc(int _x, int _y) : x(_x), y(_y) {}

    friend std::ostream &operator<<(std::ostream &out, const xyLoc &loc) {
        out << "(" << loc.x << ", " << loc.y << ")";
        return out;
    }

    friend bool operator==(const xyLoc &l1, const xyLoc &l2) {
        return (l1.x == l2.x) && (l1.y == l2.y);
    }

    friend bool operator!=(const xyLoc &l1, const xyLoc &l2) {
        return (l1.x != l2.x) || (l1.y != l2.y);
    }

    friend bool operator<(const xyLoc &l1, const xyLoc &l2) {
        if (l1.x == l2.x)
            return l1.y < l2.y;
        else
            return l1.x < l2.x;
    }
};


/*
 * An abstract class that only contains geographical information.
 */

class GameState {

private:
    xyLoc m_loc;  // we need 2D location for map key
    std::string m_name;

public:
    bool operator<(const GameState &state) const;

    bool operator==(const GameState &state) const;

    xyLoc getLoc() const;

    std::string getName() const;

    // methods for construct game state
    GameState(const xyLoc &loc, const std::string &name);

    GameState(const GameState &state);

    GameState() = default;

    friend std::ostream &operator<<(std::ostream &out, const GameState &state) {
        out << "Name: " << state.getName() << ", Loc: " << state.getLoc();
        return out;
    }

};


namespace std {
    template<>
    struct hash<GameState> {
        size_t operator()(const GameState &gameState) const {
            return (std::hash<int>()(gameState.getLoc().x) ^ (std::hash<int>()(gameState.getLoc().y) << 1)) >> 1;
        }
    };
}

class FrozenLakeEnv {
    typedef std::tuple<double, GameState, double> ProbStateRewardTuple;
    typedef std::vector<std::vector<GameState>> vec2DGameState;
protected:
    // m_transitions is a dictionary from state -> (action -> vector(probability, nextState, reward))

    std::map<GameState, std::map<Action, std::vector<ProbStateRewardTuple>>> m_transitions;
    std::set<GameState> m_terminalStates;
    std::set<GameState> m_startStates;
    vec2DGameState m_map;
    std::string m_name;

    std::map<GameState, double> getTransitionStatesAndProbsEnv(const GameState &state, const Action &action) const;

    // utilities for parsing files
    vec2DGameState parserGrid(const std::string &grid) const {
        std::string delimiter = ",";
        std::istringstream buf(grid);
        std::istream_iterator<std::string> beg(buf), end;
        std::vector<std::string> tokens(beg, end); // done!
        vec2DGameState maps;

        for (size_t i = 0; i < tokens.size(); ++i) {
            auto s = tokens[i];
            maps.emplace_back();
            size_t pos = 0;
            std::string token;
            int col = 0;
            while ((pos = s.find(delimiter)) != std::string::npos) {
                token = s.substr(0, pos);
                maps[i].push_back(GameState(xyLoc(i, col), token));
                col += 1;
                s.erase(0, pos + delimiter.length());
            }
            maps[i].push_back(GameState(xyLoc(i, col), s));
        }
        return maps;
    }

    void init(std::string name, vec2DGameState map, std::map<std::string, double> reward, double noise);

public:
    std::string getName() const;

    GameState reset(); // reset the environment and return a starting state uniformly

    GameState getNextState(const GameState &state,
                           const Action &action);  // sample nextState given state and action. (Note: the result could vary among runs)

    std::vector<Action> getPossibleActions(const GameState &state) const; // return a vector of legal actions at state

    double getReward(const GameState &state, const Action &action,
                     const GameState &nextState) const; // get the reward when taking action at state and land in nextState

    bool isTerminal(const GameState &state) const;  // return true if the state is terminal

    friend std::ostream &operator<<(std::ostream &out, const FrozenLakeEnv &mdp);

    // construct from json file
    explicit FrozenLakeEnv(const std::string &jsonFile);

    // random map given height and width
    FrozenLakeEnv(int height, int width);

    // random 4x4 map
    FrozenLakeEnv();

    std::pair<double, double>
    runGame(Agent &agent, int numEpisodes, double gamma, bool verbose); // used for test the performance of various agents.
};


/*
 * MDP can access the transition functions and all states directly while environment can't.
 */

class FrozenLakeMDP : public FrozenLakeEnv {
    // inherit all the constructor
    using FrozenLakeEnv::FrozenLakeEnv;
public:
    // return a map from nextState to probability that (state, action) -> nextState
    std::map<GameState, double> getTransitionStatesAndProbs(const GameState &state, const Action &action) const;

    // return a set of all the states
    std::set<GameState> getStates() const;

};

class FrozenLakeMDPGrading : public FrozenLakeMDP {
public:
    explicit FrozenLakeMDPGrading(const std::string &jsonFile) : FrozenLakeMDP(jsonFile) {

    }

    std::vector<std::vector<GameState>> get2DStates() const {
        return m_map;
    }
};


class Agent {
public:
    virtual Action getPolicy(const GameState &state) = 0;

    virtual std::string getName() const = 0;
};


#endif //FROZEN_LAKE_FROZENLAKE_HPP
