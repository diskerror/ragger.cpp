// Agent API Phase 3: project primitives.
//
// projects = named bodies of work. sessions/summaries/decisions/document_
// sources carry a project_id; this file owns the projects table itself:
// create/list/update, cwd/path -> project inference (resolve_project), and
// the briefing packet (project_briefing). Design contract: PHASE3-STEPS.md.
#include "sqlite/backend_impl.h"
#include "config.h"
#include "util/fs.h"
#include "util/time.h"
#include "util/sqlite.h"
#include <format>
#include "nlohmann_json.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ragger::sqlite {

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

// Split an absolute path into its components ("" entries dropped). The
// leading '/' root becomes the first component on POSIX, so two paths
// match on a component boundary iff one's component list is a prefix of
// the other's — "/a/b" matches "/a/b" and "/a/b/c" but never "/a/bc".
std::vector<std::string> path_components(const std::string& p) {
    std::vector<std::string> out;
    for (const auto& part : fs::path(p)) {
        std::string s = part.string();
        if (!s.empty()) out.push_back(s);
    }
    return out;
}

bool is_prefix_components(const std::vector<std::string>& prefix,
                          const std::vector<std::string>& full) {
    if (prefix.size() > full.size()) return false;
    return std::equal(prefix.begin(), prefix.end(), full.begin());
}

// Whole-tag membership in a comma-separated list (PHASE3-STEPS step 3:
// match on comma boundaries, not substrings — 'todo,x' matches 'todo',
// 'todos' does not).
bool has_tag(const std::string& tags, const std::string& want) {
    size_t pos = 0;
    while (pos <= tags.size()) {
        size_t comma = tags.find(',', pos);
        if (comma == std::string::npos) comma = tags.size();
        // trim spaces around the token
        size_t a = tags.find_first_not_of(" \t", pos);
        size_t b = (a == std::string::npos || a > comma) ? comma : tags.find_last_not_of(" \t", comma - 1) + 1;
        if (a != std::string::npos && a <= comma && tags.substr(a, b - a) == want)
            return true;
        pos = comma + 1;
        if (comma == tags.size()) break;
    }
    return false;
}

} // namespace


    // ---- create_project --------------------------------------------------
    // name: non-empty, unique (duplicate -> invalid_argument). path: stored
    // collapsed (~/-relative) like document paths, "" -> NULL.
    int Backend::Impl::create_project(const std::string& name,
                                      const std::string& path,
                                      const std::string& description) {
        auto trim = [](const std::string& s) {
            size_t a = s.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) return std::string();
            size_t b = s.find_last_not_of(" \t\r\n");
            return s.substr(a, b - a + 1);
        };
        const std::string n = trim(name);
        if (n.empty())
            throw std::invalid_argument("project name is required");

        Stmt dup(db, "SELECT 1 FROM projects WHERE name = ?");
        dup.bind(1, n);
        if (dup.step())
            throw std::invalid_argument(std::format("project '{}' already exists", n));

        const std::string collapsed = path.empty() ? "" : collapse_home(expand_path(path));
        Stmt ins(db, "INSERT INTO projects (name, description, path) VALUES (?,?,?)");
        ins.bind(1, n);
        if (description.empty()) ins.bind_null(2); else ins.bind(2, description);
        if (collapsed.empty())   ins.bind_null(3); else ins.bind(3, collapsed);
        if (!ins.exec())
            throw std::runtime_error(std::format("create_project failed: {}", sqlite3_errmsg(db)));
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }


    // ---- list_projects ---------------------------------------------------
    json Backend::Impl::list_projects(const std::string& status_filter) {
        std::string sql =
            "SELECT project_id, name, status, description, path, "
            "  datetime(created_at, 'unixepoch', 'localtime'), "
            "  datetime(updated_at, 'unixepoch', 'localtime') "
            "FROM projects ";
        if (!status_filter.empty()) sql += "WHERE status = ? ";
        sql += "ORDER BY name";
        Stmt s(db, sql);
        if (!status_filter.empty()) s.bind(1, status_filter);
        json arr = json::array();
        while (s.step()) {
            arr.push_back({
                {"id", s.column_int(0)},
                {"name", s.column_text(1)},
                {"status", s.column_text(2)},
                {"description", s.is_null(3) ? json(nullptr) : json(s.column_text(3))},
                {"path", s.is_null(4) ? json(nullptr) : json(s.column_text(4))},
                {"created", s.column_text(5)},
                {"updated", s.column_text(6)},
            });
        }
        return arr;
    }


    // ---- update_project --------------------------------------------------
    // Only the supplied fields change; updated_at always bumps. An unknown
    // name throws. status is NOT validated against a vocabulary (RULES: the
    // vocabulary may grow; the table has no CHECK on purpose).
    void Backend::Impl::update_project(const std::string& name,
                                       std::optional<std::string> status,
                                       std::optional<std::string> description,
                                       std::optional<std::string> path) {
        Stmt sel(db, "SELECT project_id FROM projects WHERE name = ?");
        sel.bind(1, name);
        if (!sel.step())
            throw std::invalid_argument(std::format("project '{}' does not exist", name));
        const int pid = sel.column_int(0);

        std::vector<std::string> sets;
        json params = json::array();
        if (status) {
            if (status->empty())
                throw std::invalid_argument("project status must not be empty");
            sets.push_back("status = ?");
            params.push_back(*status);
        }
        if (description) { sets.push_back("description = ?"); params.push_back(*description); }
        if (path) {
            sets.push_back("path = ?");
            params.push_back(path->empty() ? json(nullptr)
                                           : json(collapse_home(expand_path(*path))));
        }
        if (sets.empty()) return;   // nothing to do (updated_at NOT bumped)

        std::string sql = "UPDATE projects SET " + [&]{
            std::string s;
            for (const auto& x : sets) { if (!s.empty()) s += ", "; s += x; }
            return s;
        }() + ", updated_at = unixepoch() WHERE project_id = ?";
        Stmt u(db, sql);
        int i = 1;
        for (const auto& p : params) {
            if (p.is_null()) u.bind_null(i); else u.bind(i, p.get<std::string>());
            ++i;
        }
        u.bind(i, pid);
        if (!u.exec())
            throw std::runtime_error(std::format("update_project failed: {}", sqlite3_errmsg(db)));
    }


    // ---- resolve_project -------------------------------------------------
    // Exact name match wins. Otherwise treat the argument as a path: expand
    // ~, then pick the project whose expanded path is the LONGEST prefix of
    // it on a component boundary. nullopt when nothing matches (inference is
    // never an error; callers that need a real project check themselves).
    std::optional<int> Backend::Impl::resolve_project(const std::string& name_or_path) {
        if (name_or_path.empty()) return std::nullopt;
        {
            Stmt s(db, "SELECT project_id FROM projects WHERE name = ?");
            s.bind(1, name_or_path);
            if (s.step()) return s.column_int(0);
        }
        // Path inference. Expand both sides to absolute before comparing:
        // stored paths may be "~/-collapsed", callers pass real cwds.
        const std::vector<std::string> target = path_components(expand_path(name_or_path));
        if (target.empty()) return std::nullopt;
        std::optional<int> best;
        size_t best_len = 0;
        Stmt s(db, "SELECT project_id, path FROM projects WHERE path IS NOT NULL");
        while (s.step()) {
            const std::vector<std::string> pp = path_components(expand_path(s.column_text(1)));
            if (pp.empty()) continue;
            if (is_prefix_components(pp, target) && pp.size() > best_len) {
                best = s.column_int(0);
                best_len = pp.size();
            }
        }
        return best;
    }


    // ---- project_briefing ------------------------------------------------
    // The packet an agent gets before working on a project (design §9):
    // current rulings, the latest project-level summary, recent episodes
    // (project's own OR a session that belongs to the project — session
    // summaries are never read, per the redesign constraint), and open
    // questions (summaries tagged 'open' or 'todo' on whole-tag boundaries).
    json Backend::Impl::project_briefing(const std::string& name) {
        Stmt sel(db, "SELECT project_id, status, description, path FROM projects WHERE name = ?");
        sel.bind(1, name);
        if (!sel.step())
            throw std::invalid_argument(std::format("project '{}' does not exist", name));
        const int pid = sel.column_int(0);

        json out;
        out["project"]      = name;
        out["status"]       = sel.column_text(1);
        out["description"]  = sel.is_null(2) ? json(nullptr) : json(sel.column_text(2));
        out["path"]         = sel.is_null(3) ? json(nullptr) : json(sel.column_text(3));

        // decisions_current: active rulings, newest first, cap 50.
        {
            Stmt s(db,
                "SELECT decision_id, trigger, text FROM decisions "
                "WHERE project_id = ? AND status IN ('current','provisional') "
                "ORDER BY created_at DESC, decision_id DESC LIMIT 50");
            s.bind(1, pid);
            json arr = json::array();
            while (s.step()) {
                arr.push_back({
                    {"id", "decision:" + std::to_string(s.column_int(0))},
                    {"trigger", s.is_null(1) ? json(nullptr) : json(s.column_text(1))},
                    {"text", s.column_text(2)},
                });
            }
            out["decisions_current"] = std::move(arr);
        }

        // latest_summary: newest level='project' summary for this project.
        {
            Stmt s(db,
                "SELECT summary_id, text, "
                "  datetime(updated_at, 'unixepoch', 'localtime') "
                "FROM summaries WHERE project_id = ? AND level = 'project' "
                "ORDER BY updated_at DESC, summary_id DESC LIMIT 1");
            s.bind(1, pid);
            if (s.step())
                out["latest_summary"] = {
                    {"id", "summary:" + std::to_string(s.column_int(0))},
                    {"text", s.column_text(1)},
                    {"updated", s.column_text(2)},
                };
            else
                out["latest_summary"] = nullptr;
        }

        // recent_episodes: newest 3 episode summaries whose own project_id
        // matches OR whose session's project_id matches.
        {
            Stmt s(db,
                "SELECT summary_id, text, "
                "  datetime(created_at, 'unixepoch', 'localtime') "
                "FROM summaries "
                "WHERE level = 'episode' AND (project_id = ? OR session_id IN "
                "  (SELECT session_id FROM sessions WHERE project_id = ?)) "
                "ORDER BY created_at DESC, summary_id DESC LIMIT 3");
            s.bind(1, pid);
            s.bind(2, pid);
            json arr = json::array();
            while (s.step()) {
                arr.push_back({
                    {"id", "summary:" + std::to_string(s.column_int(0))},
                    {"text", s.column_text(1)},
                    {"created", s.column_text(2)},
                });
            }
            out["recent_episodes"] = std::move(arr);
        }

        // open_questions: summaries with the project's id tagged 'open' or
        // 'todo' (whole tag, comma boundaries — 'todos' does not match).
        {
            Stmt s(db,
                "SELECT summary_id, text, tags FROM summaries "
                "WHERE project_id = ? ORDER BY created_at DESC, summary_id DESC LIMIT 20");
            s.bind(1, pid);
            json arr = json::array();
            while (s.step()) {
                const std::string tags = s.column_text(2);
                if (has_tag(tags, "open") || has_tag(tags, "todo"))
                    arr.push_back({
                        {"id", "summary:" + std::to_string(s.column_int(0))},
                        {"text", s.column_text(1)},
                    });
            }
            out["open_questions"] = std::move(arr);
        }

        return out;
    }


} // namespace ragger::sqlite
