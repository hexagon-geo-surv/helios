#include <logging.hpp>

#include <string>

// ***  DECLARATIONS  *** //
// ********************** //
/**
 * @brief Run HELIOS++ tests
 * @param testDir Path to the directory containing test data
 */
void doTests(std::string const & testDir);

// LOGGING FLAGS (DO NOT MODIFY HERE BUT IN logging.hpp makeDefault())
bool    logging::LOGGING_SHOW_TRACE,    logging::LOGGING_SHOW_DEBUG,
        logging::LOGGING_SHOW_INFO,     logging::LOGGING_SHOW_TIME,
        logging::LOGGING_SHOW_WARN,     logging::LOGGING_SHOW_ERR;


int main(int argc, char** argv) {
    std::string testDir = "data/test/";
    if (argc > 1) {
        testDir = std::string(argv[1]);
        if (testDir.back() != '/')
          testDir += "/";
    }
    doTests(testDir);
}
