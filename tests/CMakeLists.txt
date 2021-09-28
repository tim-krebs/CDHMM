#Sources
set(TEST_SOURCES "main.cpp")

#Executable
add_executable(${TEST_MAIN} ${TEST_SOURCES})

#Setting
target_link_libraries(${TEST_MAIN}   PRIVATE ${LIBRARY_NAME} ${CONAN_CATCH2})
target_compile_features(${TEST_MAIN} PUBLIC cxx_std_17)

#Test 
add_test(NAME ${LIBRARY_NAME}.${TEST_MAIN} COMMAND ${TEST_MAIN})
