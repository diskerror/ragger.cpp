//
// Created by Reid Woodbury.
//

#ifndef DISKERROR_PROGRAMOPTIONS_H
#define DISKERROR_PROGRAMOPTIONS_H

#include <boost/program_options.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace Diskerror {

namespace po = boost::program_options;

class ProgramOptions {
	po::options_description            visible;
	po::options_description            hidden;
	po::positional_options_description positional;
	po::variables_map                  vm;

public:
	explicit ProgramOptions(std::string_view description);

	auto add_options()        -> po::options_description_easy_init;
	auto add_hidden_options() -> po::options_description_easy_init;
	void add_positional(const char* name, int max_count);

	void run(int argc, char** argv);

	auto operator[](std::string_view key) const -> const po::variable_value&;
	auto count(std::string_view key)       const -> size_t;
	auto getParams(std::string_view key)   const -> std::vector<std::string>;

	auto to_string(unsigned min_width = 32) const -> std::string;
};

} // namespace Diskerror

#endif // DISKERROR_PROGRAMOPTIONS_H