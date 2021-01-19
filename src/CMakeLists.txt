set(CMAKE_CXX_STANDARD 11)

add_executable(frozen_lake main.cpp FrozenLake.cpp FrozenLake.hpp
        LearningAgent.cpp LearningAgent.hpp
        ValueIterationAgent.cpp ValueIterationAgent.hpp
        PolicyIterationAgent.cpp PolicyIterationAgent.hpp
        QLearningAgent.cpp QLearningAgent.hpp common.hpp)


set(HEADER_FILES
        ValueIterationAgent.hpp
        PolicyIterationAgent.hpp
        FrozenLake.hpp
        LearningAgent.hpp
        QLearningAgent.hpp
        common.hpp)

set(SOURCE_FILES
        ValueIterationAgent.cpp
        PolicyIterationAgent.cpp
        FrozenLake.cpp
        LearningAgent.cpp
        QLearningAgent.cpp
        )

add_library(frozen_lake_lib STATIC ${SOURCE_FILES} ${HEADER_FILES})