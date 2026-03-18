//
// Created by Reid Woodbury.
//

#include "ProgramOptions.h"
#include <sstream>

namespace Diskerror {

ProgramOptions::ProgramOptions(std::string_view description)
	: visible(std::string(description)) {}

auto ProgramOptions::add_options() -> po::options_description_easy_init {
	return visible.add_options();
}

auto ProgramOptions::add_hidden_options() -> po::options_description_easy_init {
	return hidden.add_options();
}

void ProgramOptions::add_positional(const char* name, int max_count) {
	positional.add(name, max_count);
}

void ProgramOptions::run(int argc, char** argv) {
	po::options_description all;
	all.add(visible).add(hidden);

	po::store(
		po::command_line_parser(argc, argv)
			.options(all)
			.positional(positional)
			.run(),
		vm);

	po::notify(vm);
}

auto ProgramOptions::operator[](std::string_view key) const -> const po::variable_value& {
	return vm[std::string(key)];
}

auto ProgramOptions::count(std::string_view key) const -> size_t {
	return vm.count(std::string(key));
}

auto ProgramOptions::getParams(std::string_view key) const -> std::vector<std::string> {
	std::string k(key);
	if (vm.count(k))
		return vm[k].as<std::vector<std::string>>();
	return {};
}

auto ProgramOptions::to_string(unsigned min_width) const -> std::string {
	std::ostringstream oss;
	visible.print(oss, min_width);
	return oss.str();
}

} // namespace Diskerror