//
// Created by Chi Zhang on 8/28/19.
//

#include "gtest/gtest.h"
#include "testCommon.hpp"

class ValueIterationBridgeTest : public FrozenLakeMDPTest {
    void setUpParameters() override {
        agentType = "Value";
        mapPath = "../maps/bridge.json";
        gamma = 0.99;
        iterations = 100;
        threshold = 1e-8;
    }
};


TEST_F(ValueIterationBridgeTest, policy) {
    ASSERT_EQ(agent->getPolicy(states[1][1]), RIGHT);
    ASSERT_EQ(agent->getPolicy(states[1][2]), RIGHT);
    ASSERT_EQ(agent->getPolicy(states[1][3]), RIGHT);
    ASSERT_EQ(agent->getPolicy(states[1][4]), RIGHT);
}

TEST_F(ValueIterationBridgeTest, value) {
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][1]), 9.70299);
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][2]), 9.801);
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][3]), 9.9);
    ASSERT_DOUBLE_EQ(agent->getValue(states[1][4]), 10.0);
}

 class ValueIterationBridgeGammaTest : public FrozenLakeMDPTest {
     void setUpParameters() override {
         agentType = "Value";
         mapPath = "../maps/bridge.json";
         gamma = 0.4;
         iterations = 100;
         threshold = 1e-8;
     }
 };

 TEST_F(ValueIterationBridgeGammaTest, policy) {
     ASSERT_EQ(agent->getPolicy(states[1][1]), LEFT);
     ASSERT_EQ(agent->getPolicy(states[1][2]), RIGHT);
     ASSERT_EQ(agent->getPolicy(states[1][3]), RIGHT);
     ASSERT_EQ(agent->getPolicy(states[1][4]), RIGHT);
 }

 TEST_F(ValueIterationBridgeGammaTest, value) {
     ASSERT_DOUBLE_EQ(agent->getValue(states[1][1]), 1.0);
     ASSERT_DOUBLE_EQ(agent->getValue(states[1][2]), 1.6);
     ASSERT_DOUBLE_EQ(agent->getValue(states[1][3]), 4.0);
     ASSERT_DOUBLE_EQ(agent->getValue(states[1][4]), 10.0);
 }

 class ValueIterationBridgeStochasticTest : public FrozenLakeMDPTest {
     void setUpParameters() override {
         agentType = "Value";
         mapPath = "../maps/bridge_stochastic.json";
         gamma = 0.99;
         iterations = 100;
         threshold = 1e-8;
     }
 };

 TEST_F(ValueIterationBridgeStochasticTest, policy) {
     ASSERT_EQ(agent->getPolicy(states[1][1]), LEFT);
     ASSERT_EQ(agent->getPolicy(states[1][2]), LEFT);
     ASSERT_EQ(agent->getPolicy(states[1][3]), RIGHT);
     ASSERT_EQ(agent->getPolicy(states[1][4]), RIGHT);
 }

 TEST_F(ValueIterationBridgeStochasticTest, value) {
     ASSERT_DOUBLE_EQ(agent->getValue(states[1][1]), -19.2);
     ASSERT_DOUBLE_EQ(agent->getValue(states[1][2]), -35.2064);
     ASSERT_DOUBLE_EQ(agent->getValue(states[1][3]), -29.5040);
     ASSERT_DOUBLE_EQ(agent->getValue(states[1][4]), -12.0);
 }

 class ValueIterationBridgeNearStochasticTest : public FrozenLakeMDPTest {
     void setUpParameters() override {
         agentType = "Value";
         mapPath = "../maps/bridge_near_stochastic.json";
         gamma = 0.99;
         iterations = 100;
         threshold = 1e-8;
     }
 };

 TEST_F(ValueIterationBridgeNearStochasticTest, policy) {
     ASSERT_EQ(agent->getPolicy(states[1][1]), RIGHT);
     ASSERT_EQ(agent->getPolicy(states[1][2]), RIGHT);
     ASSERT_EQ(agent->getPolicy(states[1][3]), LEFT);
 }

 TEST_F(ValueIterationBridgeNearStochasticTest, value) {
     ASSERT_NEAR(agent->getValue(states[1][1]), 73.83241, 1e-5);
     ASSERT_NEAR(agent->getValue(states[1][2]), 118.47527, 1e-5);
     ASSERT_NEAR(agent->getValue(states[1][3]), 73.83241, 1e-5);
 }

 class ValueIterationCliffTest : public FrozenLakeMDPTest {
     void setUpParameters() override {
         agentType = "Value";
         mapPath = "../maps/cliff_stochastic.json";
         gamma = 0.99;
         iterations = 100;
         threshold = 1e-8;
     }
 };

 TEST_F(ValueIterationCliffTest, policy) {
     ASSERT_EQ(agent->getPolicy(states[0][0]), RIGHT);
     ASSERT_EQ(agent->getPolicy(states[0][1]), RIGHT);
     ASSERT_EQ(agent->getPolicy(states[0][2]), RIGHT);
     ASSERT_EQ(agent->getPolicy(states[0][3]), DOWN);
     ASSERT_EQ(agent->getPolicy(states[1][0]), UP);
     ASSERT_EQ(agent->getPolicy(states[1][1]), UP);
     ASSERT_EQ(agent->getPolicy(states[1][2]), UP);
 }
