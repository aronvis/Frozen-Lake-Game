cmake_minimum_required(VERSION 3.0)
project(frozen_lake)

include_directories(maps)
configure_file(maps/bridge.json maps/bridge.json COPYONLY)
configure_file(maps/bridge_stochastic.json maps/bridge_stochastic.json COPYONLY)
configure_file(maps/bridge_near_stochastic.json maps/bridge_near_stochastic.json COPYONLY)
configure_file(maps/cliff_stochastic.json maps/cliff_stochastic.json COPYONLY)
configure_file(maps/frozen_lake_8x8.json maps/frozen_lake_8x8.json COPYONLY)

add_subdirectory(src)
add_subdirectory(test)