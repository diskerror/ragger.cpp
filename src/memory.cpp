/**
 * RaggerMemory - high-level facade implementation
 */

#include "ragger/memory.h"
#include "ragger/sqlite_backend.h"
#include "ragger/embedder.h"
#include "ragger/config.h"
#include "diskerror/logger.h"
#include "ragger/lang.h"
#include "ragger/recipe.h"
#include "ragger/summarizer.h"
#include <algorithm>
#include <format>
#include <mutex>

namespace ragger {

RaggerMemory::RaggerMemory(const std::string& db_path,
                           const std::string& model_dir,
                           bool skip_embedding_guard)
{
    // Resolve model directory from config or override
    std::string resolved_model_dir = model_dir.empty()
        ? config().resolved_model_dir()
        : expand_path(model_dir);

    // Create embedder
    embedder_ = std::make_unique<Embedder>(resolved_model_dir);

    // The resolved path — used for both the storage backend and the user store.
    const std::string resolved_db = db_path.empty() ? config().resolved_db_path()
                                                     : expand_path(db_path);

    backend_    = std::make_unique<SqliteBackend>(*embedder_, resolved_db);
    user_store_ = std::make_unique<UserStore>(resolved_db);
    
    // --- Embedding identity drift guard (model + dtype + dimensions) -------
    // The settings table records what built this DB. On startup we compare it
    // to the current config; thereafter the stored values are authoritative.
    //   - first use (no stored value)        → record it
    //   - match                              → fine
    //   - mismatch + DB has embeddings       → STOP (rebuild or revert the value)
    //   - mismatch + empty DB                → re-adopt the new value (nothing to
    //                                          invalidate yet)
    //   - skip_embedding_guard (rebuild)     → don't touch settings; the
    //                                          rebuild path rewrites them after
    //                                          re-encoding (so an aborted rebuild
    //                                          leaves settings≠config and the
    //                                          guard still catches it next time).
    const std::string current_model = config().resolve_model(config().embedding_model);
    const std::string current_vtype =
        (config().embedding_vector_type == "f32") ? "f32" : "f16";
    const std::string current_dims = std::to_string(config().embedding_dimensions);

    if (!skip_embedding_guard) {
        const bool has_data = backend_->has_embeddings();
        auto guard = [&](const char* key, const std::string& current,
                         const std::string& err) {
            auto stored = user_store_->get_setting(key);
            if (!stored.has_value() || *stored == current) {
                if (!stored.has_value()) user_store_->set_setting(key, current);
                return;
            }
            if (has_data) throw std::runtime_error(err);
            user_store_->set_setting(key, current);  // empty DB: re-adopt
        };
        guard("embedding_model", current_model,
              std::format(lang::ERR_EMBEDDING_MISMATCH,
                          user_store_->get_setting("embedding_model").value_or(""), current_model));
        guard("vector_type", current_vtype,
              std::format(lang::ERR_VECTOR_TYPE_MISMATCH,
                          user_store_->get_setting("vector_type").value_or(""), current_vtype));
        guard("dimensions", current_dims,
              std::format(lang::ERR_DIMENSIONS_MISMATCH,
                          user_store_->get_setting("dimensions").value_or(""), current_dims));
    }

    // Backfill any rows left without embeddings (deferred-embedding writes
    // that didn't get processed before the previous shutdown). The query is
    // a no-op when nothing is NULL; embedder is already loaded above.
    int filled = backend_->backfill_embeddings(*embedder_);
    if (filled > 0) {
        Diskerror::logger::info(std::format(lang::MSG_BACKFILLED_EMBEDDINGS, filled));
    }
}

RaggerMemory::~RaggerMemory() {
    close();
}

std::vector<TurnRecord> RaggerMemory::turns_by_session_desc(
        const std::string& session_guid, int limit) {
    return backend_->turns_by_session_desc(session_guid, limit);
}

std::optional<std::string> RaggerMemory::get_setting(const std::string& key) {
    return user_store_->get_setting(key);
}
void RaggerMemory::set_setting(const std::string& key, const std::string& value) {
    user_store_->set_setting(key, value);
}

std::string RaggerMemory::store(const std::string& text, json metadata,
                                 bool defer_embedding) {
    return backend_->store(text, std::move(metadata), defer_embedding);
}

int RaggerMemory::store_document(const DocumentChunk& chunk, bool defer_embedding) {
    return backend_->store_document(chunk, defer_embedding);
}

int RaggerMemory::store_turn(const std::string& user_text,
                             const std::string& assistant_text,
                             const std::string& model_name, bool defer_embedding,
                             const std::string& session_guid) {
    return backend_->store_turn(user_text, assistant_text, model_name,
                                defer_embedding, session_guid);
}

bool RaggerMemory::finalize_turn(int turn_id, const std::string& assistant_text,
                                 const std::string& model_name) {
    return backend_->finalize_turn(turn_id, assistant_text, model_name);
}

bool RaggerMemory::update_document_embedding(int document_id,
                                             const std::vector<float>& emb) {
    return backend_->update_document_embedding(document_id, emb);
}

int RaggerMemory::store_decision(const std::string& text,
                                 const std::string& status,
                                 const std::string& tags,
                                 const std::string& source_timestamp,
                                 bool defer_embedding) {
    return backend_->store_decision(text, status, tags, source_timestamp, defer_embedding);
}

bool RaggerMemory::update_decision_embedding(int decision_id,
                                             const std::vector<float>& emb) {
    return backend_->update_decision_embedding(decision_id, emb);
}

bool RaggerMemory::update_summary_embedding(int summary_id,
                                            const std::vector<float>& emb) {
    return backend_->update_summary_embedding(summary_id, emb);
}

bool RaggerMemory::update_turn_embedding(int turn_id,
                                         const std::vector<float>& emb) {
    return backend_->update_turn_embedding(turn_id, emb);
}

int RaggerMemory::cleanup_old_conversations(float max_age_hours) {
    return backend_->cleanup_old_conversations(max_age_hours);
}

int RaggerMemory::store_summary(const std::string& text, const std::string& level,
                                const std::string& status, const std::string& model_name,
                                const std::string& session_guid,
                                const std::string& source_timestamp,
                                const std::string& tags) {
    return backend_->store_summary(text, level, status, model_name, session_guid,
                                   source_timestamp, tags);
}

std::optional<std::pair<int, std::string>>
RaggerMemory::current_session_summary(const std::string& session_guid) {
    return backend_->current_session_summary(session_guid);
}

bool RaggerMemory::update_summary_text(int summary_id, const std::string& text,
                                       const std::string& model_name) {
    return backend_->update_summary_text(summary_id, text, model_name);
}

bool RaggerMemory::set_summary_status(int summary_id, const std::string& status) {
    return backend_->set_summary_status(summary_id, status);
}

std::vector<std::string> RaggerMemory::recent_summaries(const std::string& level,
                                                        int limit) {
    return backend_->recent_summaries(level, limit);
}

std::vector<std::string> RaggerMemory::current_decisions(int limit) {
    return backend_->current_decisions(limit);
}

bool RaggerMemory::update_text(int memory_id, const std::string& text, json metadata,
                                bool defer_embedding) {
    return backend_->update_text(memory_id, text, std::move(metadata), defer_embedding);
}

SearchResponse RaggerMemory::search(const std::string& query,
                                    int limit,
                                    float min_score,
                                    std::vector<std::string> collections) {
    return backend_->search(query, limit, min_score, std::move(collections));
}

int RaggerMemory::count() const {
    return backend_->count();
}

std::vector<SearchResult> RaggerMemory::load_all(const std::string& collection) {
    return backend_->load_all(collection);
}

int RaggerMemory::rebuild_embeddings() {
    return backend_->rebuild_embeddings(*embedder_);
}

int RaggerMemory::backfill_embeddings() {
    return backend_->backfill_embeddings(*embedder_);
}

std::vector<std::string> RaggerMemory::collections() const {
    return backend_->collections();
}

bool RaggerMemory::delete_memory(int memory_id) {
    return backend_->delete_memory(memory_id);
}

int RaggerMemory::delete_batch(const std::vector<int>& memory_ids) {
    return backend_->delete_batch(memory_ids);
}

std::vector<SearchResult> RaggerMemory::search_by_metadata(const json& metadata_filter, int limit,
                                                           const std::string& after,
                                                           const std::string& before) {
    return backend_->search_by_metadata(metadata_filter, limit, after, before);
}

void RaggerMemory::close() {
    if (backend_) backend_->close();
}

CaptureResult capture_turn(RaggerMemory& memory,
                           const std::string& user,
                           const std::string& assistant,
                           const std::string& model,
                           const std::string& session_id) {
    // Gate: turn capture is opt-in. Agent-driven store/search are unaffected.
    if (!config().capture_turns) return {false, -1};

    // Strip any leading Hermes system-injected annotation (interruption
    // notices, model-switch notes, compaction handoffs, bg-process reports)
    // from the user side at capture time, so the raw `turns` row stores only
    // what the human actually typed. "[System note: ...]\n\nstop" → "stop";
    // a note with no trailing message → "". This is the single capture entry
    // point for both HTTP /turn and the MCP capture_turn tool.
    const std::string clean_user = strip_system_injected_prefix(user);

    // Skip empty pushes. A turn needs real content on at least one side: keep
    // assistant-only turns (interrupted turn the agent still answered), but
    // drop a turn that was nothing but a system note with no assistant output.
    if (clean_user.empty() && assistant.empty()) return {false, -1};

    int turn_id = memory.store_turn(clean_user, assistant, model,
                                    /*defer_embedding=*/false, session_id);
    return {true, turn_id};
}

namespace {

// Process-local cache of loaded recipes. Loaded lazily on first call;
// re-loaded when the config's recipes_dir changes (covers SIGHUP).
struct RecipeCache {
    std::mutex          mu;
    std::vector<Recipe> recipes;
    std::string         loaded_dir;   // path the cache was built from
    bool                loaded = false;
};

RecipeCache& recipe_cache() {
    static RecipeCache c;
    return c;
}

const std::vector<Recipe>& cached_recipes() {
    auto& c = recipe_cache();
    std::lock_guard<std::mutex> lk(c.mu);
    const std::string dir = config().recipes_dir.empty()
        ? expand_path("~/.ragger/recipes")
        : expand_path(config().recipes_dir);
    if (!c.loaded || c.loaded_dir != dir) {
        c.recipes = load_recipes_from_dir(dir);
        if (c.recipes.empty()) c.recipes = builtin_recipes();
        c.loaded_dir = dir;
        c.loaded = true;
    }
    return c.recipes;
}

// Format a raw turn into "User: ...\n\nAssistant: ..." for the chunk text.
std::string format_raw_turn(const TurnRecord& t) {
    return "User: " + t.user_text + "\n\nAssistant: " + t.assistant_text;
}

// Approximate token cost using the recipe's chars_per_token. Cheap and
// good enough for ceiling enforcement; precise tokenization would tie
// us to a specific model's tokenizer.
int approx_tokens(const std::string& s, float chars_per_token) {
    if (chars_per_token <= 0) chars_per_token = 4.0f;
    return static_cast<int>(static_cast<float>(s.size()) / chars_per_token);
}

} // namespace

SessionContext build_context(RaggerMemory& memory,
                             const std::string& session_id,
                             const std::string& recipe_name) {
    // Gate: the read side requires both capture (something to read) and the
    // build-context toggle. Off → enabled=false, no chunks.
    if (!config().capture_turns || !config().build_context) {
        return {false, "", {}};
    }

    const auto& recipes = cached_recipes();
    // Resolve the effective default:
    //   1. DB `settings.recipe` (set by `ragger recipe`). The sentinel
    //      value "default" means "track settings.ini" — equivalent to
    //      having no row.
    //   2. settings.ini `[server] default_recipe` (system default).
    //   3. First built-in (last-resort fallback).
    // One keyed read per build_context call — cheap, and the user's
    // choice takes effect without a daemon restart.
    std::string effective_default = config().default_recipe;
    if (auto stored = memory.get_setting("recipe");
        stored && !stored->empty() && *stored != "default") {
        effective_default = *stored;
    }
    // Resolve recipe: explicit > effective default > first built-in.
    const Recipe* recipe = find_recipe(recipes, recipe_name);
    if (!recipe) recipe = find_recipe(recipes, effective_default);
    if (!recipe && !recipes.empty()) recipe = &recipes.front();
    if (!recipe) return {true, "", {}};

    SessionContext out{true, recipe->name, {}};
    if (session_id.empty()) return out;

    auto* backend = memory.backend();
    if (!backend) return out;

    // The walk: turns + L2 summaries newest-first. raw_turn / turn_summary
    // layers consume from these in chronological order (newest popped
    // first). Other layers (session/project/decisions) pull recent rows
    // independent of the walk.
    auto turns_desc        = backend->turns_by_session_desc(session_id, 0);
    auto turn_summaries    = backend->turn_summaries_by_session_desc(session_id, 0);
    auto session_summaries = backend->session_summaries_desc(session_id, 0);

    // Walk pointers — index into the newest-first arrays. Turn summaries
    // for raw turns already consumed are skipped via the timestamp
    // overlap check (the chronology link).
    size_t walk_turn = 0;
    size_t walk_summary = 0;

    // Collect chunks newest-first; we reverse at the end so the agent
    // reads chronologically.
    std::vector<ContextChunk> rev;
    auto consumed_turn_ts = std::vector<std::string>{};  // for overlap skip

    for (const auto& layer : recipe->layers) {
        switch (layer.kind) {
            case LayerKind::RawTurn: {
                int taken = 0;
                while (walk_turn < turns_desc.size() &&
                       (layer.limit == 0 || taken < layer.limit)) {
                    const auto& t = turns_desc[walk_turn++];
                    rev.push_back({"raw_turn", format_raw_turn(t), t.timestamp});
                    consumed_turn_ts.push_back(t.timestamp);
                    ++taken;
                }
                break;
            }
            case LayerKind::TurnSummary: {
                int taken = 0;
                while (walk_summary < turn_summaries.size() &&
                       (layer.limit == 0 || taken < layer.limit)) {
                    const auto& s = turn_summaries[walk_summary++];
                    // Skip a summary whose source turn was already
                    // emitted raw — chronology link by timestamp.
                    if (std::find(consumed_turn_ts.begin(),
                                  consumed_turn_ts.end(),
                                  s.timestamp) != consumed_turn_ts.end()) {
                        continue;
                    }
                    rev.push_back({"turn_summary", s.text, s.timestamp});
                    // Also advance the raw-turn walk past this turn's
                    // timestamp so a later RawTurn layer doesn't re-emit
                    // it. (Layers can appear in any order; preserve the
                    // chronological exclusion in both directions.)
                    while (walk_turn < turns_desc.size() &&
                           turns_desc[walk_turn].timestamp >= s.timestamp) {
                        consumed_turn_ts.push_back(
                            turns_desc[walk_turn].timestamp);
                        ++walk_turn;
                    }
                    ++taken;
                }
                break;
            }
            case LayerKind::SessionSummary: {
                int wanted = layer.limit > 0 ? layer.limit : 1;
                int taken = 0;
                for (const auto& s : session_summaries) {
                    if (taken >= wanted) break;
                    rev.push_back({"session_summary", s.text, s.timestamp});
                    ++taken;
                }
                break;
            }
            case LayerKind::ProjectSummary: {
                int wanted = layer.limit > 0 ? layer.limit : 1;
                for (const auto& text : backend->recent_summaries("project", wanted)) {
                    rev.push_back({"project_summary", text, ""});
                }
                break;
            }
            case LayerKind::Decisions: {
                int wanted = layer.limit > 0 ? layer.limit : 3;
                for (const auto& text : backend->current_decisions(wanted)) {
                    rev.push_back({"decision", text, ""});
                }
                break;
            }
            case LayerKind::GeneralSearch: {
                // Session-AGNOSTIC relevance pass: surface the most relevant
                // context from anywhere in the corpus (all summaries +
                // documents) that earlier session-scoped layers did NOT
                // already emit. "Insight by association."
                int wanted = layer.limit > 0 ? layer.limit : 5;

                // Query = current session's latest user turn + up to the 2
                // most recent turn summaries, concatenated into one string.
                std::string query;
                if (!turns_desc.empty() && !turns_desc[0].user_text.empty()) {
                    query += turns_desc[0].user_text;
                }
                for (size_t i = 0; i < turn_summaries.size() && i < 2; ++i) {
                    if (turn_summaries[i].text.empty()) continue;
                    if (!query.empty()) query += "\n";
                    query += turn_summaries[i].text;
                }
                if (query.empty()) break;  // nothing to search with → skip

                // Over-fetch so that after dedup we still have enough to emit
                // `wanted` survivors. K is a small buffer.
                const int K = 8;
                auto resp = memory.search(query, wanted + K, 0.0f, {});

                // Collect survivors best-first. Dedup against text already
                // pushed by earlier layers into `rev`.
                std::vector<SearchResult> survivors;
                for (const auto& r : resp.results) {
                    if (static_cast<int>(survivors.size()) >= wanted) break;
                    bool dup = false;
                    for (const auto& c : rev) {
                        if (c.text == r.text) { dup = true; break; }
                    }
                    if (dup) continue;
                    survivors.push_back(r);
                }

                // ORDERING: `rev` is built newest-first and the final step
                // assigns out.chunks = rev.rbegin()..rev.rend(), i.e. the
                // order within rev is FLIPPED. We want the general_search
                // block to read best-first in the final output, so push its
                // results worst-first into rev (the flip then yields
                // best-first). survivors is best-first, so iterate it in
                // reverse when pushing.
                for (auto it = survivors.rbegin(); it != survivors.rend(); ++it) {
                    rev.push_back({"general_search", it->text, it->timestamp});
                }
                break;
            }
        }
    }

    // Apply max_tokens by dropping from the *oldest* end (the back of
    // rev, since rev is newest-first). This keeps the most recent
    // context when we have to trim.
    if (recipe->max_tokens > 0) {
        int total = 0;
        size_t kept = 0;
        for (; kept < rev.size(); ++kept) {
            total += approx_tokens(rev[kept].text, recipe->chars_per_token);
            if (total > recipe->max_tokens) break;
        }
        if (kept < rev.size()) rev.resize(kept);
    }

    // Reverse to chronological (oldest-first) for the prompt-friendly
    // output. Within a layer the order matches the newest-first walk
    // and gets reversed; across layers the recipe author's intended
    // ordering is preserved (because earlier layers pushed first).
    out.chunks.assign(rev.rbegin(), rev.rend());
    return out;
}

} // namespace ragger
