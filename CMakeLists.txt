cmake_minimum_required(VERSION 3.29)
project(pong_ai)

set(CMAKE_CXX_STANDARD 17)

include_directories(utec)
include_directories(utec/agent)
include_directories(utec/algebra)
include_directories(utec/nn)

add_executable(pong_ai
    src/utec/agent/EnvGym.cpp
    src/utec/agent/PongAgent.cpp
    utec/agent/EnvGym.h
    utec/agent/PongAgent.h
    utec/algebra/tensor.h
    utec/nn/neural_network.h
    utec/nn/nn_activation.h
    utec/nn/nn_dense.h
    utec/nn/nn_interfaces.h
    utec/nn/nn_loss.h
    utec/nn/nn_optimizer.h
    utec/agent/State.h
    test/test_agent_env.cpp
    test/test_neural_network.cpp
    test/test_tensor.cpp
)
