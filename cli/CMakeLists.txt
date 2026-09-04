add_subdirectory(words)

if(BUILD_CLI)
    add_subdirectory(classifier)
    add_subdirectory(generator)
endif()
