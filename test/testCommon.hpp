//
// Created by Chi Zhang on 11/5/19.
//

#ifndef FROZEN_LAKE_TESTCOMMON_HPP
#define FROZEN_LAKE_TESTCOMMON_HPP

#include "gtest/gtest.h"

#include "FrozenLake.hpp"
#include "ValueIterationAgent.hpp"
#include "PolicyIterationAgent.hpp"


class FrozenLakeMDPTest : public ::testing::Test {
protected:
    void SetUp() override {
        setUpParameters();
        mdp = new FrozenLakeMDPGrading(mapPath);
        if (agentType == "Value") {
            agent = new ValueIterationAgent(*mdp, gamma, iterations, threshold);
        } else if (agentType == "Policy") {
            agent = new PolicyIterationAgent(*mdp, gamma, iterations, threshold);
        } else {
            std::cout << "Unknown agent type " << agentType << std::endl;
            exit(-1);
        }

        states = mdp->get2DStates();
    }

    void TearDown() override {
        delete mdp;
        delete agent;
    }

    virtual void setUpParameters() {

    }

    double gamma = 0.99;
    int iterations = 100;
    double threshold = 1e-8;
    std::string mapPath = "../maps/bridge.json";
    std::string agentType = "Value";
    FrozenLakeMDPGrading *mdp{};
    ValueEstimateAgent *agent{};
    std::vector<std::vector<GameState>> states;
};


#endif //FROZEN_LAKE_TESTCOMMON_HPP
