//
// Created by Chi Zhang on 11/5/19.
//

#include "gtest/gtest.h"

#include "testCommon.hpp"

class PolicyIterationBridgeTest : public FrozenLakeMDPTest {
    void setUpParameters() override {
        agentType = "Policy";
        mapPath = "../maps/bridge.json";
        gamma = 0.99;
        iterations = 100;
        threshold = 1e-8;
    }
};


TEST_F(PolicyIterationBridgeTest, policy) {
    ASSERT_EQ(agent->getPolicy(states[1][1]), RIGHT);
    ASSERT_EQ(agent->getPolicy(states[1][2]), RIGHT);
    ASSERT_EQ(agent->getPolicy(states[1][3]), RIGHT);
    ASSERT_EQ(agent->getPolicy(states[1][4]), RIGHT);
}

TEST_F(PolicyIterationBridgeTest, value) {
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][1]), 9.70299);
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][2]), 9.801);
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][3]), 9.9);
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][4]), 10.0);
}

class PolicyIterationBridgeGammaTest : public FrozenLakeMDPTest {
    void setUpParameters() override {
        agentType = "Policy";
        mapPath = "../maps/bridge.json";
        gamma = 0.4;
        iterations = 100;
        threshold = 1e-8;
    }
};

TEST_F(PolicyIterationBridgeGammaTest, policy) {
    ASSERT_EQ(agent->getPolicy(states[1][1]), LEFT);
    ASSERT_EQ(agent->getPolicy(states[1][2]), RIGHT);
    ASSERT_EQ(agent->getPolicy(states[1][3]), RIGHT);
    ASSERT_EQ(agent->getPolicy(states[1][4]), RIGHT);
}

TEST_F(PolicyIterationBridgeGammaTest, value) {
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][1]), 1.0);
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][2]), 1.6);
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][3]), 4.0);
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][4]), 10.0);
}

class PolicyIterationBridgeStochasticTest : public FrozenLakeMDPTest {
    void setUpParameters() override {
        agentType = "Policy";
        mapPath = "../maps/bridge_stochastic.json";
        gamma = 0.99;
        iterations = 10000;
        threshold = 1e-8;
    }
};

TEST_F(PolicyIterationBridgeStochasticTest, policy) {
    ASSERT_EQ(agent->getPolicy(states[1][1]), LEFT);
    ASSERT_EQ(agent->getPolicy(states[1][2]), LEFT);
    ASSERT_EQ(agent->getPolicy(states[1][3]), RIGHT);
    ASSERT_EQ(agent->getPolicy(states[1][4]), RIGHT);
}

TEST_F(PolicyIterationBridgeStochasticTest, value) {
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][1]), -19.2);
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][2]), -35.2064);
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][3]), -29.5040);
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][4]), -12.0);
}

class PolicyIterationBridgeNearStochasticTest : public FrozenLakeMDPTest {
    void setUpParameters() override {
        agentType = "Policy";
        mapPath = "../maps/bridge_near_stochastic.json";
        gamma = 0.99;
        iterations = 100;
        threshold = 1e-8;
    }
};

TEST_F(PolicyIterationBridgeNearStochasticTest, policy) {
    ASSERT_EQ(agent->getPolicy(states[1][1]), RIGHT);
    ASSERT_EQ(agent->getPolicy(states[1][2]), RIGHT);
    ASSERT_EQ(agent->getPolicy(states[1][3]), LEFT);
}

TEST_F(PolicyIterationBridgeNearStochasticTest, value) {
    ASSERT_NEAR(agent->getValue(states[1][1]), 73.83241, 1e-5);
    ASSERT_NEAR(agent->getValue(states[1][2]), 118.47527, 1e-5);
    ASSERT_NEAR(agent->getValue(states[1][3]), 73.83241, 1e-5);
}

class PolicyIterationCliffTest : public FrozenLakeMDPTest {
    void setUpParameters() override {
        agentType = "Policy";
        mapPath = "../maps/cliff_stochastic.json";
        gamma = 0.99;
        iterations = 100;
        threshold = 1e-8;
    }
};

TEST_F(PolicyIterationCliffTest, policy) {
    ASSERT_EQ(agent->getPolicy(states[0][0]), RIGHT);
    ASSERT_EQ(agent->getPolicy(states[0][1]), RIGHT);
    ASSERT_EQ(agent->getPolicy(states[0][2]), RIGHT);
    ASSERT_EQ(agent->getPolicy(states[0][3]), DOWN);
    ASSERT_EQ(agent->getPolicy(states[1][0]), UP);
    ASSERT_EQ(agent->getPolicy(states[1][1]), UP);
    ASSERT_EQ(agent->getPolicy(states[1][2]), UP);
}