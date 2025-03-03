target_include_directories(
  helios
  PUBLIC
  "."
)

target_sources(
  helios
  PRIVATE
  "Tests.cpp"
)

add_executable(
  helios_test
  "Main.cpp"
)

target_link_libraries(helios_test PUBLIC helios)
