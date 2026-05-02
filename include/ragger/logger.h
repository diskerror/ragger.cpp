//
// Created by Reid Woodbury Jr on 4/28/26.
//

#ifndef RAGGER_LOGGER_H
#define RAGGER_LOGGER_H

#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace ragger {

class logger {
    static unsigned int instanceCount;
    static std::ofstream logFile;
    static std::mutex logMutex;

public:
        logger(const std::string& logFileName, const std::string& level = "info");
        ~logger();

    static std::ofstream &get_file();
    static std::string get_timestamp();

    static std::function<void(std::string const&)> trace;
    static std::function<void(std::string const&)> debug;
    static std::function<void(std::string const&)> info;
    static std::function<void(std::string const&)> warn;
    static std::function<void(std::string const&)> error;
    static std::function<void(std::string const&)> critical;
};

} // ragger

#endif //RAGGER_LOGGER_H
