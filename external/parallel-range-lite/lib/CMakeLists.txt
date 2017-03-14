include_directories(
        "${CMAKE_CURRENT_BINARY_DIR}"
        )
configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/parallel.hpp.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/parallel.hpp"
	)
add_library(prange parallel-range.cpp)
install(TARGETS prange
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)
