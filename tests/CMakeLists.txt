add_executable(test_smoke test_smoke.cu)

target_link_libraries(test_smoke PRIVATE rgpu_core)

target_include_directories(test_smoke PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/tests
)

rgpu_set_warnings(test_smoke)

add_executable(test_versions test_versions.cpp)

target_link_libraries(test_versions PRIVATE rgpu_core)

target_include_directories(test_versions PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/tests
)

rgpu_set_warnings(test_versions)

add_test(NAME test_smoke COMMAND test_smoke)
add_test(NAME test_versions COMMAND test_versions)
