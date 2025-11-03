if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

project(translate-axera)
#C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_subdirectory(third-party/sentencepiece)

# include_directories(third-party/sentencepiece/src)
include_directories(include)
include_directories(src)

add_executable(test_tokenizer tests/tokenizer/test_tokenizer.cpp src/tokenizer/tokenizer.cpp)

# 链接 sentencepiece
# target_link_libraries(test_tokenizer PRIVATE sentencepiece)

include_directories(
    src/runner/ax_type_header
    src/utils
)


add_library(ax_translate SHARED
                src/ax_translate.cpp
                src/runner/axcl/axcl_manager.cpp
                src/runner/axcl/ax_model_runner_axcl.cpp
                src/runner/ax650/ax_model_runner_ax650.cpp
                
                src/tokenizer/tokenizer.cpp

                src/ax_devices.cpp

                src/utils/enum_devices.cpp
                src/utils/cqdm.cpp
                src/utils/memory_utils.cpp
                )

add_executable(test_llm tests/test_llm.cpp)
target_link_libraries(test_llm PRIVATE ax_translate)

add_executable(test_translate tests/test_translate.cpp)
target_link_libraries(test_translate PRIVATE ax_translate)

install(TARGETS ax_translate
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include DESTINATION include)
install(DIRECTORY pytranslate DESTINATION lib/)

# target_link_libraries(ax_translate PRIVATE sentencepiece dl)


# add_executable(test_translate tests/test_translate.cpp)
# target_link_libraries(test_translate PRIVATE ax_translate)

# install(TARGETS ax_translate
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     RUNTIME DESTINATION bin
# )

# install(TARGETS test_translate
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     RUNTIME DESTINATION bin
# )

# install(FILES include/ax_translate.h include/ax_devices.h DESTINATION include)