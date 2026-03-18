/**
 * raggerc — C++ port of Ragger Memory
 *
 * Usage:
 *   raggerc serve              Start HTTP server (default)
 *   raggerc search <query>     One-shot search from CLI
 *   raggerc store <text>       Store a memory from CLI
 *   raggerc count              Print memory count
 *   raggerc --help             Show help
 */
#include <iostream>

#include "ProgramOptions.h"
#include "ragger/config.h"

int main(int argc, char** argv) {
    Diskerror::ProgramOptions opts("raggerc — Ragger Memory (C++)");
    opts.add_options()
        ("help,h", "Show help")
        ("version,v", "Show version")
        ("host", Diskerror::po::value<std::string>()->default_value(ragger::DEFAULT_HOST), "Server bind address")
        ("port,p", Diskerror::po::value<int>()->default_value(ragger::DEFAULT_PORT), "Server port")
        ("db", Diskerror::po::value<std::string>()->default_value(""), "SQLite database path")
        ("model-dir", Diskerror::po::value<std::string>()->default_value(""), "Model directory path")
    ;
    opts.add_hidden_options()
        ("command", Diskerror::po::value<std::string>()->default_value("serve"), "Command")
        ("args", Diskerror::po::value<std::vector<std::string>>(), "Command arguments")
    ;
    opts.add_positional("command", 1);
    opts.add_positional("args", -1);

    try {
        opts.run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    if (opts.count("help")) {
        std::cout << opts.to_string() << "\n";
        return 0;
    }

    if (opts.count("version")) {
        std::cout << "raggerc " << "0.1.0" << "\n";
        return 0;
    }

    auto command = opts["command"].as<std::string>();

    if (command == "serve") {
        std::cout << "raggerc serve — not yet implemented\n";
        // TODO: instantiate RaggerMemory + Server, run
    } else if (command == "search") {
        auto args = opts.getParams("args");
        if (args.empty()) {
            std::cerr << "Usage: raggerc search <query>\n";
            return 1;
        }
        std::cout << "raggerc search — not yet implemented\n";
    } else if (command == "store") {
        auto args = opts.getParams("args");
        if (args.empty()) {
            std::cerr << "Usage: raggerc store <text>\n";
            return 1;
        }
        std::cout << "raggerc store — not yet implemented\n";
    } else if (command == "count") {
        std::cout << "raggerc count — not yet implemented\n";
    } else {
        std::cerr << "Unknown command: " << command << "\n";
        return 1;
    }

    return 0;
}
