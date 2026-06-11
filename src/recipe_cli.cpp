/**
 * `ragger recipe` implementation. See include/ragger/recipe_cli.h.
 */

#include "ragger/recipe_cli.h"

#include "ragger/config.h"
#include "ragger/recipe.h"
#include "ragger/user_store.h"

#include <iostream>
#include <print>
#include <termios.h>
#include <unistd.h>

namespace ragger {

namespace {

// Sentinel used in DB `settings.recipe` to mean "track settings.ini
// [server] default_recipe". Also the name of the synthetic top entry
// in the picker.
constexpr const char* kSentinel = "default";

const char* layer_kind_name(LayerKind k) {
    switch (k) {
        case LayerKind::RawTurn:        return "raw_turn";
        case LayerKind::TurnSummary:    return "turn_summary";
        case LayerKind::SessionSummary: return "session_summary";
        case LayerKind::ProjectSummary: return "project_summary";
        case LayerKind::Decisions:      return "decisions";
        case LayerKind::GeneralSearch:  return "general_search";
    }
    return "?";
}

std::vector<Recipe> load_all() {
    std::string dir = config().recipes_dir.empty()
        ? expand_path("~/.ragger/recipes")
        : expand_path(config().recipes_dir);
    auto recipes = load_recipes_from_dir(dir);
    if (recipes.empty()) recipes = builtin_recipes();
    return recipes;
}

// Prepend the synthetic `default` entry so the picker can express "track
// settings.ini" as a first-class choice. Its description reports what
// it currently resolves to.
Recipe make_synthetic_default(const std::string& ini_default) {
    Recipe r;
    r.name = kSentinel;
    r.description = "Track [server] default_recipe in settings.ini "
                    "(currently → " + ini_default + ")";
    // No layers / token budget — purely a pointer.
    return r;
}

// Resolve "what's the active recipe right now?" the same way build_context
// does, so the picker can mark it. Returns the *name* (the sentinel
// "default" if the DB row matches the ini default or is unset).
std::string current_choice(UserStore& backend,
                           const std::string& ini_default) {
    if (auto stored = backend.get_setting("recipe");
        stored && !stored->empty()) {
        return *stored;
    }
    // Unset DB row → effectively the sentinel.
    (void)ini_default;
    return kSentinel;
}

void print_recipe(const Recipe& r, const std::string& ini_default,
                  bool is_current) {
    std::println("\nRecipe: {}{}",
                 r.name, is_current ? "  (active)" : "");
    if (!r.description.empty()) {
        std::println("  {}", r.description);
    }
    if (r.name == kSentinel) {
        // Synthetic entry — no layers, but show the target.
        std::println("");
        std::println("  Resolves to: {}", ini_default);
        return;
    }
    std::println("");
    std::println("  Layers (applied in order, oldest→newest in the output):");
    for (const auto& layer : r.layers) {
        std::println("    - {:<16} limit: {}",
                     layer_kind_name(layer.kind),
                     layer.limit == 0 ? std::string("unlimited")
                                      : std::to_string(layer.limit));
    }
    std::println("");
    std::println("  max_tokens:      {}", r.max_tokens);
    std::println("  chars_per_token: {:.1f}", r.chars_per_token);
}

std::string format_entry(const Recipe& r, bool selected, bool is_current) {
    constexpr size_t name_col = 18;
    constexpr size_t total    = 78;
    std::string out;
    out.reserve(total + 8);
    out += selected ? "> " : "  ";
    std::string name = r.name;
    if (is_current) name += "*";
    if (name.size() < name_col) name.append(name_col - name.size(), ' ');
    else                        name = name.substr(0, name_col);
    out += name;
    std::string desc = r.description;
    const size_t budget = total - name_col - 2;
    if (desc.size() > budget) desc = desc.substr(0, budget - 1) + "…";
    out += desc;
    return out;
}

struct RawMode {
    termios saved{};
    bool    active = false;

    void enter() {
        if (!isatty(STDIN_FILENO)) return;
        if (tcgetattr(STDIN_FILENO, &saved) != 0) return;
        termios raw = saved;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN]  = 1;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) != 0) return;
        std::print("\033[?25l");        // hide cursor
        std::cout.flush();
        active = true;
    }
    ~RawMode() {
        if (!active) return;
        std::print("\033[?25h");        // show cursor
        std::cout.flush();
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &saved);
    }
};

void redraw(const std::vector<Recipe>& recipes, size_t sel,
            const std::string& current) {
    std::print("\033[J");
    std::println("Active recipe marked with *; ↑/↓ to move, Enter to set, "
                 "q to quit.\n");
    for (size_t i = 0; i < recipes.size(); ++i) {
        std::println("{}", format_entry(recipes[i], i == sel,
                                        recipes[i].name == current));
    }
    int rows = static_cast<int>(recipes.size()) + 2;
    std::print("\033[{}A", rows);
    std::cout.flush();
}

char read_key() {
    char c;
    if (::read(STDIN_FILENO, &c, 1) != 1) return 0;
    if (c == '\033') {
        char seq[2];
        if (::read(STDIN_FILENO, &seq[0], 1) != 1) return 'q';
        if (::read(STDIN_FILENO, &seq[1], 1) != 1) return 'q';
        if (seq[0] == '[') {
            if (seq[1] == 'A') return 'k';
            if (seq[1] == 'B') return 'j';
        }
        return 0;
    }
    if (c == '\r' || c == '\n') return '\n';
    if (c == 'q' || c == 'Q')   return 'q';
    if (c == 'k')               return 'k';
    if (c == 'j')               return 'j';
    return 0;
}

// Negative-encoded return = chosen index. Positive return = quit/error.
int run_picker(const std::vector<Recipe>& recipes,
               const std::string& current) {
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
        std::println("Available recipes (active = {}):\n", current);
        for (const auto& r : recipes) {
            std::println("{}", format_entry(r, false, r.name == current));
        }
        std::println("\nRun `ragger recipe <name>` to set one.");
        return 0;
    }

    RawMode raw;
    raw.enter();

    size_t sel = 0;
    for (size_t i = 0; i < recipes.size(); ++i) {
        if (recipes[i].name == current) { sel = i; break; }
    }

    while (true) {
        redraw(recipes, sel, current);
        char k = read_key();
        if (k == 'q' || k == 0) {
            std::print("\033[J");
            std::cout.flush();
            return 1;
        }
        if (k == 'k' && sel > 0)                          --sel;
        else if (k == 'j' && sel + 1 < recipes.size())    ++sel;
        else if (k == '\n') {
            std::print("\033[J");
            std::cout.flush();
            break;
        }
    }
    return -1 - static_cast<int>(sel);
}

void persist_choice(UserStore& backend, const std::string& name) {
    backend.set_setting("recipe", name);
}

} // namespace

int run_recipe_cli(const std::vector<std::string>& args,
                   const std::string& db_path) {
    auto on_disk = load_all();
    if (on_disk.empty()) {
        std::println(stderr, "No recipes found and no built-ins compiled in.");
        return 2;
    }

    const std::string ini_default = config().default_recipe;

    // Compose the picker list: synthetic `default` first, then the on-disk
    // recipes. The named-arg path searches by exact name across both.
    std::vector<Recipe> recipes;
    recipes.reserve(on_disk.size() + 1);
    recipes.push_back(make_synthetic_default(ini_default));
    for (auto& r : on_disk) recipes.push_back(std::move(r));

    // Open DB-only backend (no embedder required for settings access).
    // Empty `db_path` from the CLI = "use config" — resolve here so the
    // backend constructor doesn't see an empty string.
    const std::string resolved_db = db_path.empty()
        ? config().resolved_db_path()
        : db_path;
    UserStore backend(resolved_db);

    const std::string current = current_choice(backend, ini_default);

    // --- Named form: `ragger recipe <name>` -----------------------------
    if (!args.empty()) {
        const auto& name = args[0];
        const Recipe* r = nullptr;
        for (const auto& candidate : recipes) {
            if (candidate.name == name) { r = &candidate; break; }
        }
        if (!r) {
            std::println(stderr,
                "Unknown recipe '{}'. Run `ragger recipe` to see the list.",
                name);
            return 2;
        }
        persist_choice(backend, name);
        std::println("✓ Active recipe set to '{}' (stored in settings.recipe).",
                     name);
        print_recipe(*r, ini_default, /*is_current=*/true);
        return 0;
    }

    // --- Interactive form ----------------------------------------------
    int result = run_picker(recipes, current);
    if (result == 1) {
        std::println("(no change — kept '{}')", current);
        return 1;
    }
    if (result == 0) {
        // Non-interactive listing already printed.
        return 0;
    }
    if (result < 0) {
        size_t idx = static_cast<size_t>(-1 - result);
        const auto& chosen = recipes[idx];
        persist_choice(backend, chosen.name);
        std::println("✓ Active recipe set to '{}' (stored in settings.recipe).",
                     chosen.name);
        print_recipe(chosen, ini_default, /*is_current=*/true);
        return 0;
    }
    return 0;
}

} // namespace ragger
