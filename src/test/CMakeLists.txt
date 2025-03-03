if(BUILD_TESTING)
  target_sources(
    helios_test
    PRIVATE
    "Main.cpp"
    "Tests.cpp"
  )
  target_include_directories(helios_test PRIVATE ".")
endif()
