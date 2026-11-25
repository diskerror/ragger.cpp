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
    if (is_legacy_knowledge_handle(req.id))
        throw std::invalid_argument("use knowledge:<id>");
    if (is_legacy_summary_handle(req.id))
        throw std::invalid_argument(
            "use knowledge:<id> (notes are knowledge rows)");
    if (!parse_agent_handle(req.id, layer, rowid))
        throw std::invalid_argument("bad handle: '" + req.id + "'");

    // Rule 2: agents can't modify turn / turn_summary / document rows.
    if (layer == "episode")
        throw std::invalid_argument("episodes are immutable");
    if (layer == "turn" || layer == "turn_summary" || layer == "document")
        throw std::invalid_argument(
            layer + " rows are not agent-editable");

    // PHASE8: the old `summary` branch is gone — notes are knowledge rows
    // and update through the knowledge path below (tags/status/scope; never
    // text).
    if (layer == "knowledge") {
        // Rule 4: status transitions per the table; tags free. PHASE7: one
        // branch for all kinds (the transition table was always shared).
        Stmt f(db, "SELECT status FROM knowledge WHERE knowledge_id = ?");
        f.bind(1, rowid);
        if (!f.step())
            throw std::invalid_argument("knowledge " + std::to_string(rowid) + " not found");
        const std::string old_status = f.column_text(0);

        std::string set;
        int b = 1;
        if (req.status.has_value()) {
            if (!status::is_in(status::kAgentSettable, *req.status))
                throw std::invalid_argument("unknown knowledge status: '" + *req.status + "'");
            if (*req.status == "superseded")
                throw std::invalid_argument(std::format(
                    "status 'superseded' is set only by storing a superseding knowledge row"));
            // Strict table: same-status is not a listed transition either
            // (the review queue relies on re-approving failing).
            if (!status::transition_ok(old_status, *req.status))
                throw std::invalid_argument(std::format(
                    "illegal knowledge status transition: {} -> {}", old_status, *req.status));
            set += "status = ?, ";
        }
        if (req.tags.has_value()) set += "tags = ?, ";
        if (set.empty())
            throw std::invalid_argument("nothing to update");
        set.erase(set.end() - 2, set.end());   // trailing ", "
        Stmt u(db, "UPDATE knowledge SET " + set + " WHERE knowledge_id = ?");
        if (req.status.has_value()) u.bind(b++, *req.status);
        if (req.tags.has_value())   u.bind(b++, *req.tags);
        u.bind(b, rowid);
        if (!u.exec())
            throw std::runtime_error(std::format("agent_update knowledge (id {}): {}",
                                                 rowid, sqlite3_errmsg(db)));
        // rule 5: any structured-row change refreshes its search cache
        invalidate_knowledge_cache();
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
        // Pending knowledge rows of ANY kind (PHASE7), oldest first.
        std::vector<std::pair<long long, json>> rows;
        {
            Stmt s(db,
                "SELECT knowledge_id, ifnull(trigger,''), text, created_at FROM knowledge "
                "WHERE status = 'pending'");
            while (s.step()) {
                rows.emplace_back(s.column_int64(3), json{
                    {"id",      std::format("knowledge:{}", s.column_int(0))},
                    {"trigger", s.column_text(1)},
                    {"snippet", agent_snippet(s.column_text(2))},
                });
            }
        }
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
        // Accept "knowledge:<id>" or a bare integer.
        std::string handle = raw;
        if (handle.find(':') == std::string::npos)
            handle = "knowledge:" + handle;
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
