//
// Created by Chi Zhang on 8/24/19.
//

#include "LearningAgent.hpp"


ValueEstimateAgent::ValueEstimateAgent() = default;


ValueEstimateAgent::ValueEstimateAgent(double gamma, int iterations, double threshold) :
        m_gamma(gamma),
        m_iterations(iterations),
        m_threshold(threshold) {

}
