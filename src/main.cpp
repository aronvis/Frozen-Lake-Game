#include <iostream>
#include <cstdlib>
#include <string>
#include "FrozenLake.hpp"

#include "PolicyIterationAgent.hpp"
#include "ValueIterationAgent.hpp"
#include "QLearningAgent.hpp"

#include "third_party/cxxopts.hpp"

cxxopts::ParseResult
parse(int argc, char *argv[]) {
    try {
        cxxopts::Options options(argv[0], " - Frozen Lake command line options");
        options
                .positional_help("[optional args]")
                .show_positional_help();

        options
                .allow_unrecognised_options()
                .add_options()
                        ("help", "Print help")
                        ("agent", "agent type (choose from ValueIteration(v), PolicyIteration(p), QLearning(q)",
                         cxxopts::value<std::string>())
                        ("map", "path to the map", cxxopts::value<std::string>())
                        ("gamma", "discount factor", cxxopts::value<double>()->default_value("0.95"))
                        ("iteration", "Number of iterations to train", cxxopts::value<size_t>()->default_value("1000"))
                        ("theta", "error threshold", cxxopts::value<double>()->default_value("0.001"))
                        ("alpha", "learning rate in q learning", cxxopts::value<double>()->default_value("1.0"))
                        ("epsilon", "exploration rate", cxxopts::value<double>()->default_value("0.2"))
                        ("test_episodes", "Number of test episodes to run",
                         cxxopts::value<int>()->default_value("10000"));

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        return result;

    } catch (const cxxopts::OptionException &e) {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }
}

int main(int argc, char *argv[]) {
    auto result = parse(argc, argv);
    // create an agent
    std::string agentName = result["agent"].as<std::string>();
    std::string pathToMap = result["map"].as<std::string>();
    double gamma = result["gamma"].as<double>();
    size_t iteration = result["iteration"].as<size_t>();
    double theta = result["theta"].as<double>();
    double alpha = result["alpha"].as<double>();
    double epsilon = result["epsilon"].as<double>();
    int test_episodes = result["test_episodes"].as<int>();

    if (agentName == "ValueIteration" || agentName == "v") {
        FrozenLakeMDP mdp(pathToMap);
        ValueIterationAgent agent(mdp, gamma, iteration, theta);
        mdp.runGame(agent, test_episodes, gamma, true);
    } else if (agentName == "PolicyIteration" || agentName == "p") {
        FrozenLakeMDP mdp(pathToMap);
        PolicyIterationAgent agent(mdp, gamma, iteration, theta);
        mdp.runGame(agent, test_episodes, gamma, true);
    } else if (agentName == "QLearning" || agentName == "q") {
        FrozenLakeEnv env(pathToMap);
        QLearningAgent agent(env, gamma, iteration, alpha, epsilon);
//        env.runGame(agent, test_episodes, gamma, true);
    } else {
        std::cout << "Unknown agent name " << agentName << std::endl;
        exit(-1);
    }

    return 0;
}
