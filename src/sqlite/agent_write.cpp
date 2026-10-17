// Agent API Phase 4 writes: metadata-only curation (agent_update) and the
// pending-review queue (agent_review). Split out of agent_read.cpp by review
// 5A; behavior unchanged. Design contract lives in StorageBackend's docs and
// PHASE4-STEPS.md.
#include "sqlite/backend_impl.h"
#include "sqlite/agent_common.h"
#include <format>
#include "nlohmann_json.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace ragger::sqlite {

using json = nlohmann::json;


// =======================================================================
// agent_update — metadata-only curation (Phase 4).
// Text never changes in place; a text change is a new row with supersedes.
// The status transition table lives in sqlite/status.h (PHASE3 Step 4).
// =======================================================================

json Backend::Impl::agent_update(const UpdateRequest& req) {
    // Rule 1: text is never edited in place.
    if (req.text.has_value())
        throw std::invalid_argument(
            "text cannot be edited in place; store a new decision/note with supersedes");

    std::string layer;
    int rowid = 0;
    if (!parse_agent_handle(req.id, layer, rowid))
        throw std::invalid_argument("bad handle: '" + req.id + "'");

    // Rule 2: agents can't modify turn / turn_summary / document rows.
    if (layer == "turn" || layer == "turn_summary" || layer == "document")
        throw std::invalid_argument(
            layer + " rows are not agent-editable");

    if (layer == "summary") {
        // Rule 3: only agent-written notes (no model_id) are editable;
        // summarizer rollups carry the summarizer's model and are history.
        Stmt f(db, "SELECT model_id FROM summaries WHERE summary_id = ?");
        f.bind(1, rowid);
        if (!f.step())
            throw std::invalid_argument("summary " + std::to_string(rowid) + " not found");
        if (!f.is_null(0))
            throw std::invalid_argument(
                "summary rows written by the summarizer are not editable; only agent notes are");
        if (req.status.has_value())
            throw std::invalid_argument("summaries have no status");

        std::string set;
        if (req.tags.has_value())
            set += "tags = ?, ";
        if (req.project.has_value()) {
            if (!req.project->empty() && !resolve_project(*req.project))
                throw std::invalid_argument(
                    std::format("project '{}' does not exist", *req.project));
            set += "project_id = ?, ";
        }
        if (set.empty())
            throw std::invalid_argument("nothing to update");
        set.erase(set.end() - 2, set.end());   // trailing ", "
        Stmt u(db, "UPDATE summaries SET " + set +
                   ", updated_at = unixepoch() WHERE summary_id = ?");
        int b = 1;
        if (req.tags.has_value()) u.bind(b++, *req.tags);
        if (req.project.has_value()) {
            if (req.project->empty()) u.bind_null(b++);
            else u.bind(b++, *resolve_project(*req.project));
        }
        u.bind(b, rowid);
        if (!u.exec())
            throw std::runtime_error(std::format("agent_update summaries (id {}): {}",
                                                 rowid, sqlite3_errmsg(db)));
        invalidate_cache();
    } else if (layer == "decision" || layer == "insight") {
        // Rule 4: status transitions per the table; tags/project free.
        // Insights share the decision transition table verbatim (P5 step 4).
        const bool is_dec = layer == "decision";
        const char* table = is_dec ? "decisions" : "insights";
        const char* pk    = is_dec ? "decision_id" : "insight_id";
        Stmt f(db, std::format("SELECT status FROM {} WHERE {} = ?", table, pk));
        f.bind(1, rowid);
        if (!f.step())
            throw std::invalid_argument(layer + " " + std::to_string(rowid) + " not found");
        const std::string old_status = f.column_text(0);

        std::string set;
        int b = 1;
        std::optional<int> pid_bind;      // tri-state: nullopt=absent, -1=NULL
        bool clear_project = false;
        if (req.status.has_value()) {
            if (!status::is_in(status::kAgentSettable, *req.status))
                throw std::invalid_argument("unknown " + layer + " status: '" + *req.status + "'");
            if (*req.status == "superseded")
                throw std::invalid_argument(std::format(
                    "status 'superseded' is set only by storing a superseding {}", layer));
            // Strict table: same-status is not a listed transition either
            // (the review queue relies on re-approving failing).
            if (!status::transition_ok(old_status, *req.status))
                throw std::invalid_argument(std::format(
                    "illegal {} status transition: {} -> {}", layer, old_status, *req.status));
            set += "status = ?, ";
        }
        if (req.tags.has_value()) set += "tags = ?, ";
        if (req.project.has_value()) {
            set += "project_id = ?, ";
            if (req.project->empty()) clear_project = true;
            else {
                pid_bind = resolve_project(*req.project);
                if (!pid_bind)
                    throw std::invalid_argument(
                        std::format("project '{}' does not exist", *req.project));
            }
        }
        if (set.empty())
            throw std::invalid_argument("nothing to update");
        set.erase(set.end() - 2, set.end());   // trailing ", "
        Stmt u(db, std::format("UPDATE {} SET {} WHERE {} = ?", table, set, pk));
        if (req.status.has_value()) u.bind(b++, *req.status);
        if (req.tags.has_value())   u.bind(b++, *req.tags);
        if (req.project.has_value()) {
            if (clear_project) u.bind_null(b++);
            else u.bind(b++, *pid_bind);
        }
        u.bind(b, rowid);
        if (!u.exec())
            throw std::runtime_error(std::format("agent_update {} (id {}): {}",
                                                 table, rowid, sqlite3_errmsg(db)));
        // rule 5: any structured-row change refreshes its search cache
        if (is_dec) invalidate_dec_cache();
        else        invalidate_ins_cache();
    } else {
        throw std::invalid_argument("layer not updatable: " + layer);
    }

    return fetch_agent_row(db, layer, rowid);
}


// =======================================================================
// agent_review — pending-decision queue (Phase 4). Reuses agent_update's
// transition code; a non-pending id fails only that id.
// =======================================================================
json Backend::Impl::agent_review(const std::string& action,
                                 const std::vector<std::string>& ids) {
    if (action == "list") {
        // Pending decisions AND pending insights (P5 step 4), oldest first
        // across both queues.
        std::vector<std::pair<long long, json>> rows;
        auto collect = [&](const char* table, const char* pk, const char* source) {
            Stmt s(db, std::format(
                "SELECT {}, ifnull(trigger,''), text, created_at FROM {} "
                "WHERE status = 'pending'", pk, table));
            while (s.step()) {
                rows.emplace_back(s.column_int64(3), json{
                    {"id",      std::format("{}:{}", source, s.column_int(0))},
                    {"trigger", s.column_text(1)},
                    {"snippet", agent_snippet(s.column_text(2))},
                });
            }
        };
        collect("decisions", "decision_id", "decision");
        collect("insights", "insight_id", "insight");
        std::stable_sort(rows.begin(), rows.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        json items = json::array();
        for (auto& [created, item] : rows)
            items.push_back(std::move(item));
        return json{{"pending", std::move(items)}};
    }
    if (action != "approve" && action != "reject")
        throw std::invalid_argument("action must be one of: list, approve, reject");
    if (ids.empty())
        throw std::invalid_argument("ids required for approve/reject");

    const std::string to = (action == "approve") ? "current" : "retired";
    json done = json::array();
    json failed = json::array();
    for (const auto& raw : ids) {
        // Accept "decision:<id>" or a bare integer.
        std::string handle = raw;
        if (handle.find(':') == std::string::npos)
            handle = "decision:" + handle;
        try {
            UpdateRequest req;
            req.id = handle;
            req.status = to;
            agent_update(req);
            done.push_back(raw);
        } catch (const std::invalid_argument& e) {
            failed.push_back({{"id", raw}, {"error", e.what()}});
        }
    }
    return json{{"done", std::move(done)}, {"failed", std::move(failed)}};
}

} // namespace ragger::sqlite
