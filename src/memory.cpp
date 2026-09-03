/**
 * RaggerMemory - high-level facade implementation
 */

#include "memory.h"
#include "sqlite_backend.h"
#include "embedder.h"
#include "config.h"
#include "util/fs.h"
#include "Logger.h"
#include "lang.h"
#include "recipe.h"
#include "summarizer.h"
#include "vector_codec.h"
#ifdef RAGGER_STATS
#include "stats_logger.h"
#endif
#include <algorithm>
#include <cerrno>
#include <format>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <csignal>
#include <unistd.h>

namespace ragger {

RaggerMemory::RaggerMemory(const std::string& db_path,
                           bool skip_embedding_guard)
{
    if (db_path.empty()) {
        throw std::invalid_argument("RaggerMemory: db_path must not be empty");
    }

    // Model directory always comes from config — hardcoded relative to
    // ragger_base_dir() (see config.cpp), same as every other Ragger path.
    // Engine choice: "external" uses a remote /v1/embeddings endpoint;
    // "internal" (default) loads an ONNX model from the models directory.
    // A bad embedding-model configuration must NOT stop the daemon. Ragger's
    // job is to record conversations; losing vector search is a degradation,
    // losing capture is data loss. So on failure we log a loud, actionable
    // error and fall back to a disabled Embedder: turns keep being recorded,
    // their embedding column is left NULL, and the housekeeping backfill
    // re-embeds them the moment the configuration is corrected.
    // Ragger-managed ONNX models live at ~/.ragger/models/<provider>/<model>/,
    // so a valid internal model name is always "provider/model". The dashboard
    // only ever offers paths that exist in that layout, so a bare name means
    // the value was hand-edited. Diagnose it explicitly — the generic load
    // failure below would just report a missing directory — but never repair
    // it by assuming a provider: a wrong guess resolves to a real model that
    // is not the one whose vectors are in the DB.
    if (config().embedding_engine != "external") {
        const std::string& m = config().embedding_model;
        if (!m.empty() && m.find('/') == std::string::npos) {
            Diskerror::Logger::critical(std::format(
                lang::ERR_EMBED_MODEL_NO_PROVIDER, m, m));
        }
    }

    try {
        if (config().embedding_engine == "external") {
            embedder_ = std::make_unique<Embedder>(
                config().embedding_external_host,
                config().embedding_external_port,
                config().embedding_external_model,
                config().embedding_external_api_key,
                config().embedding_dimensions);
        } else {
            std::string resolved_model_dir = config().resolved_model_dir();
            embedder_ = std::make_unique<Embedder>(resolved_model_dir);
        }
    } catch (const std::exception& e) {
        Diskerror::Logger::critical(std::format(
            lang::ERR_EMBED_MODEL_UNUSABLE,
            config().embedding_model,
            config().resolved_model_dir(), e.what()));
        embedder_ = std::make_unique<Embedder>();   // disabled
        embeddings_degraded_ = true;
    }

    // The resolved path — used for both the storage backend and the user store.
    const std::string resolved_db = expand_path(db_path);

    backend_    = std::make_unique<SqliteBackend>(*embedder_, resolved_db);
    user_store_ = std::make_unique<UserStore>(resolved_db);

    // NOTE: no legacy "prepend a provider" migration here. A stored value that
    // is not in provider/model form is reported (above) and left alone; the
    // drift guard below then degrades to text-only search rather than
    // re-embedding the corpus against a model nobody chose.

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
    const std::string current_model = config().embedding_model;
    const std::string current_vtype =
        vector_codec::canonical(config().embedding_vector_type);
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
        try {
            guard("embedding_model", current_model,
                  std::format(lang::ERR_EMBEDDING_MISMATCH,
                              user_store_->get_setting("embedding_model").value_or(""), current_model));
            guard("vector_type", current_vtype,
                  std::format(lang::ERR_VECTOR_TYPE_MISMATCH,
                              user_store_->get_setting("vector_type").value_or(""), current_vtype));
            guard("dimensions", current_dims,
                  std::format(lang::ERR_DIMENSIONS_MISMATCH,
                              user_store_->get_setting("dimensions").value_or(""), current_dims));
        } catch (const std::runtime_error& e) {
            // Degrade gracefully: log the error and disable vector search.
            // FTS5 text + phonetic search still works.
            Diskerror::Logger::error(std::format(
                "Embedding drift detected — vector search disabled, "
                "falling back to text + phonetic search only. {}", e.what()));
            embeddings_degraded_ = true;
        }
    }

    // NOTE: embedding backfill no longer runs here. A large backfill (e.g.
    // after a blob-format or version change touching every row) used to run
    // synchronously in this constructor and could block daemon startup for
    // minutes — long enough that deploy health checks reported failure while
    // the daemon was actually fine. Backfill is now owned by the server's
    // housekeeping tick (background, ~60s after start). CLI paths that need
    // it immediately use `ragger rebuild-embeddings`.

    // Backfill any NULL phon (dolphining sounds-like) rows — self-heals rows
    // that predate the phon column after the one-time ADD COLUMN migration.
    // Pure string work (no embedder); NULL-only so it's a no-op once populated.
    // Runs here (single startup connection) to avoid the double-backend write
    // contention that silently swallows UPDATEs when the daemon holds the DB.
    int phoned = backend_->rebuild_phon(/*only_missing=*/true, /*progress=*/false);
    if (phoned > 0) {
        Diskerror::Logger::info(std::format("Backfilled phonetic codes for {} row(s)", phoned));
    }
#ifdef RAGGER_STATS
    // Opt-in retrieval instrumentation. Construction never throws into the
    // caller; if the stats DB can't be opened the logger disables itself.
    stats_ = std::make_unique<StatsLogger>();
#endif
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
                             const std::string& session_guid,
                             const std::string& source_timestamp,
                             const std::string& session_name,
                             const std::string& name_source) {
    return backend_->store_turn(user_text, assistant_text, model_name,
                                defer_embedding, session_guid, source_timestamp,
                                session_name, name_source);
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
                                const std::string& model_name,
                                const std::string& session_guid,
                                const std::string& source_timestamp,
                                const std::string& tags) {
    return backend_->store_summary(text, level, model_name, session_guid,
                                   source_timestamp, tags);
}

bool RaggerMemory::update_summary_text(int summary_id, const std::string& text,
                                       const std::string& model_name) {
    return backend_->update_summary_text(summary_id, text, model_name);
}

std::vector<std::string> RaggerMemory::recent_summaries(const std::string& level,
                                                        int limit) {
    return backend_->recent_summaries(level, limit);
}

std::vector<std::string> RaggerMemory::current_decisions(int limit) {
    return backend_->current_decisions(limit);
}

bool RaggerMemory::set_decision_status(int decision_id, const std::string& status) {
    return backend_->set_decision_status(decision_id, status);
}

std::vector<std::string> RaggerMemory::decisions_by_status(const std::string& status, int limit) {
    return backend_->decisions_by_status(status, limit);
}

bool RaggerMemory::summary_exists_exact(const std::string& text, const std::string& created_at) {
    return backend_->summary_exists_exact(text, created_at);
}

bool RaggerMemory::decision_exists_exact(const std::string& text, const std::string& created_at) {
    return backend_->decision_exists_exact(text, created_at);
}

bool RaggerMemory::turn_exists_fuzzy(const std::string& user_text, const std::string& ts,
                                     int window_seconds) {
    return backend_->turn_exists_fuzzy(user_text, ts, window_seconds);
}

std::optional<TurnRecord> RaggerMemory::find_turn_by_text(const std::string& user_text) {
    return backend_->find_turn_by_text(user_text);
}

bool RaggerMemory::update_turn_meta(int turn_id, const std::string& timestamp,
                                    const std::string& session_guid) {
    return backend_->update_turn_meta(turn_id, timestamp, session_guid);
}

bool RaggerMemory::update_text(int memory_id, const std::string& text, json metadata,
                                bool defer_embedding) {
    return backend_->update_text(memory_id, text, std::move(metadata), defer_embedding);
}

SearchResponse RaggerMemory::search(const std::string& query,
                                    int limit,
                                    float min_score,
                                    std::vector<std::string> collections) {
    // While a re-embed is in progress the vectors are being rewritten, so a
    // vector query would mix old/new spaces or hit half-updated rows. Fall
    // back to text-only search (FTS5 keyword + phonetic) exactly like the
    // degraded path below, rather than returning a stub record — callers
    // (MCP clients especially) get real, if unranked, results instead of a
    // single unusable row.
    if (is_reembedding() || repair_pending()) {
        SearchResponse busy = backend_->search_text_only(query, limit);
        if (!busy.timing.is_object()) busy.timing = json::object();
        busy.timing["reembedding"] = true;
        return busy;
    }
    // When embeddings are degraded (drift mismatch at startup), fall back to
    // text-only search (FTS5 keyword + phonetic). Vector similarity is
    // unavailable but lookups still work.
    if (embeddings_degraded_ || !embedder_->ready()) {
        return backend_->search_text_only(query, limit);
    }
    SearchResponse resp = backend_->search(query, limit, min_score, std::move(collections));
#ifdef RAGGER_STATS
    if (stats_ && stats_->enabled()) {
        double elapsed = -1.0;
        try {
            if (resp.timing.contains("total_ms") && resp.timing["total_ms"].is_number())
                elapsed = resp.timing["total_ms"].get<double>();
        } catch (...) {}
        stats_->log_lookup(query, limit, min_score, resp.results, elapsed);
    }
#endif
    return resp;
}

int RaggerMemory::count() const {
    return backend_->count();
}

int RaggerMemory::count_embeddable_rows() const {
    return backend_->count_embeddable_rows();
}

std::vector<SearchResult> RaggerMemory::load_all(const std::string& collection) {
    return backend_->load_all(collection);
}

int RaggerMemory::rebuild_embeddings(bool progress) {
    return backend_->rebuild_embeddings(*embedder_, progress);
}

RaggerMemory::EmbeddingStatus RaggerMemory::embedding_status() {
    EmbeddingStatus s;
    // Current = what the stored vectors are (drift-guard settings keys).
    s.current_model = user_store_->get_setting("embedding_model")
                          .value_or(config().embedding_model);
    s.current_vtype = user_store_->get_setting("vector_type")
                          .value_or(config().embedding_vector_type);
    s.current_dims  = std::stoi(user_store_->get_setting("dimensions")
                          .value_or(std::to_string(config().embedding_dimensions)));
    s.current_engine = config().embedding_engine;
    // Desired = staged config target (seeded from current when unset).
    s.desired_model = config().desired_embedding_model.empty()
                          ? s.current_model
                          : config().desired_embedding_model;
    s.desired_vtype = config().desired_embedding_vector_type.empty()
                          ? s.current_vtype : config().desired_embedding_vector_type;
    s.desired_dims  = config().desired_embedding_dimensions == 0
                          ? s.current_dims : config().desired_embedding_dimensions;
    s.desired_engine = config().desired_embedding_engine.empty()
                          ? s.current_engine : config().desired_embedding_engine;
    s.needs_update = (s.current_model != s.desired_model) ||
                     (s.current_vtype != s.desired_vtype) ||
                     (s.current_dims  != s.desired_dims)  ||
                     (s.current_engine != s.desired_engine);
    s.reembedding = is_reembedding();
    s.repair_pending = repair_pending();
    return s;
}

bool RaggerMemory::embeddings_unavailable() const {
    return !embedder_ || !embedder_->ready();
}

bool RaggerMemory::try_recover_embeddings() {
    if (!embeddings_degraded_) return false;  // nothing to recover

    const std::string current_model = config().embedding_model;
    const std::string current_vtype =
        vector_codec::canonical(config().embedding_vector_type);
    const std::string current_dims = std::to_string(config().embedding_dimensions);

    auto stored_model = user_store_->get_setting("embedding_model");
    auto stored_vtype = user_store_->get_setting("vector_type");
    auto stored_dims  = user_store_->get_setting("dimensions");

    // All three must match for recovery.
    if (stored_model.value_or("") != current_model ||
        stored_vtype.value_or("") != current_vtype ||
        stored_dims.value_or("")  != current_dims) {
        return false;  // still mismatched
    }

    // If we came up with a disabled embedder (bad model path at startup), the
    // identity matching again is not enough — there is still nothing that can
    // embed. Build the real one now. Until this succeeds we stay degraded, so
    // a config that is merely self-consistent but still wrong does not flip
    // semantic search back on and start writing vectors nobody can use.
    if (!embedder_->ready()) {
        try {
            if (config().embedding_engine == "external") {
                embedder_ = std::make_unique<Embedder>(
                    config().embedding_external_host,
                    config().embedding_external_port,
                    config().embedding_external_model,
                    config().embedding_external_api_key,
                    config().embedding_dimensions);
            } else {
                embedder_ = std::make_unique<Embedder>(config().resolved_model_dir());
            }
            Diskerror::Logger::warn(std::format(
                lang::MSG_EMBED_MODEL_RECOVERED, current_model));
        } catch (const std::exception& e) {
            Diskerror::Logger::error(std::format(
                lang::ERR_EMBED_MODEL_UNUSABLE, current_model,
                config().resolved_model_dir(), e.what()));
            return false;   // still unusable — stay degraded
        }
    }

    // Settings match config — embeddings are valid again.
    embeddings_degraded_ = false;
    Diskerror::Logger::info(
        "Embedding drift resolved — semantic search re-enabled.");

    // Backfill any rows left without embeddings.
    try {
        int filled = backend_->backfill_embeddings(*embedder_);
        if (filled > 0) {
            Diskerror::Logger::info(
                std::format(lang::MSG_BACKFILLED_EMBEDDINGS, filled));
        }
    } catch (const std::exception& e) {
        Diskerror::Logger::warn(std::format(
            "Backfill after recovery failed: {}", e.what()));
    }
    return true;
}

bool RaggerMemory::is_reembedding() {
    auto v = user_store_->get_setting("reembedding");
    if (!v.has_value() || *v != "true") return false;

    // The flag is persisted so a crash mid-update is visible on restart — but
    // that also means a killed process leaves it set forever, wedging search
    // (which short-circuits on it) and /embedding/update (which 409s on it).
    // Nothing else clears it, so self-heal here: the writer records its PID
    // alongside the flag, and a flag whose owner is gone is stale by
    // definition. Checking the PID rather than clearing unconditionally at
    // startup matters because several processes share the DB (daemon, CLI,
    // `ragger mcp`) — an MCP subprocess starting up must not clear a flag the
    // daemon legitimately holds.
    auto owner = user_store_->get_setting("reembedding_pid");
    if (owner.has_value() && !owner->empty()) {
        pid_t pid = 0;
        try { pid = static_cast<pid_t>(std::stol(*owner)); } catch (...) { pid = 0; }
        if (pid > 0 && pid != ::getpid()) {
            // kill(pid, 0) succeeds for a live process we own; EPERM means it
            // exists but belongs to someone else — also "alive".
            if (::kill(pid, 0) == 0 || errno == EPERM) return true;
        } else if (pid == ::getpid()) {
            return true;   // this process is the one doing the work
        }
    }

    Diskerror::Logger::error(std::format(
        lang::WARN_REEMBED_STALE_FLAG, owner.value_or("<unset>")));
    user_store_->set_setting("reembedding", "false");
    user_store_->set_setting("reembedding_pid", "");
    // The run was interrupted, so rows are half-converted. Mark the repair as
    // owed: search degrades to keyword-only and housekeeping finishes the job.
    user_store_->set_setting("reembed_repair_pending", "true");
    return false;
}

int RaggerMemory::update_embeddings() {
    // Guard: the flag is persisted so a crash mid-update is visible on
    // restart, and search short-circuits for anyone reading concurrently.
    // The PID goes with it so is_reembedding() can tell "in progress" from
    // "the owner died and never cleared it" — see is_reembedding().
    user_store_->set_setting("reembedding", "true");
    user_store_->set_setting("reembedding_pid", std::to_string(::getpid()));

    // Set once the identity has been promoted. After that point the old
    // embedder must NOT be restored on failure — settings now name the new
    // model, so putting the old one back would make the live embedder
    // disagree with the recorded identity.
    bool promoted = false;
    std::unique_ptr<Embedder> previous_embedder;
    try {
        const std::string desired_model_name =
            config().desired_embedding_model.empty()
                ? config().embedding_model
                : config().desired_embedding_model;
        const std::string desired_model_dir =
            ragger_base_dir() + "/models/" + desired_model_name;
        const std::string desired_engine =
            config().desired_embedding_engine.empty()
                ? config().embedding_engine
                : config().desired_embedding_engine;

        // Build the embedder BEFORE touching any persisted state. A missing or
        // unreadable model must fail here, with nothing promoted — promoting
        // first would record an identity the daemon cannot load, and the next
        // startup would throw out of the constructor and refuse to boot.
        std::unique_ptr<Embedder> desired_embedder;
        if (desired_engine == "external") {
            desired_embedder = std::make_unique<Embedder>(
                config().embedding_external_host,
                config().embedding_external_port,
                config().embedding_external_model,
                config().embedding_external_api_key,
                config().desired_embedding_dimensions > 0
                    ? config().desired_embedding_dimensions : 0);
        } else {
            desired_embedder = std::make_unique<Embedder>(desired_model_dir);
        }
        Embedder& active = *desired_embedder;

        const std::string vtype = vector_codec::canonical(
            config().desired_embedding_vector_type.empty()
                ? config().embedding_vector_type
                : config().desired_embedding_vector_type);
        const int dims = config().desired_embedding_dimensions == 0
            ? active.dimensions() : config().desired_embedding_dimensions;

        // Increment the embedding version so every existing blob reads stale.
        backend_->increment_embedding_version();

        // --- Promote the identity UP FRONT, before re-encoding a single row.
        //
        // This is what makes an interrupted run resumable. The rebuild is a
        // "re-encode every row whose version byte is old" operation, and the
        // only thing that says which model those rows should be re-encoded
        // WITH is the recorded identity. Promote afterwards and an interrupted
        // run leaves the DB half-converted while the settings still name the
        // OLD model — so the daemon restarts on the old model and the
        // housekeeping backfill happily "resumes" by writing old-model vectors
        // under the new version byte. Same width, same blob format, no way to
        // tell them apart afterwards: it would quietly poison every row the
        // interrupted run hadn't reached yet.
        //
        // Promoting first also swaps the live embedder before any re-encoding,
        // so turns captured *during* the rebuild land in the new space too.
        //
        // The safety this gives up — an aborted rebuild used to leave
        // settings != config so the startup drift guard degraded to text-only
        // search — is replaced by the repair_pending marker below, which
        // degrades search the same way but is also actionable: housekeeping
        // knows to finish the job instead of waiting for a human.
        user_store_->set_setting("reembed_repair_pending", "true");
        user_store_->set_setting("embedding_model", desired_model_name);
        user_store_->set_setting("vector_type", vtype);
        user_store_->set_setting("dimensions", std::to_string(dims));
        // Write the same identity to both the settings table and the live
        // config: the startup drift guard compares the stored embedding_model
        // against the live one, and if they record the same model in different
        // spellings it degrades. Models are stored canonically (provider/model,
        // LM Studio-style) under base_dir/models, so desired_model_name is
        // already the canonical form — just keep both sides symmetric.
        mutable_config().embedding_model = desired_model_name;
        mutable_config().embedding_vector_type = vtype;
        mutable_config().embedding_dimensions = dims;
        mutable_config().embedding_engine = desired_engine;
        previous_embedder = std::exchange(embedder_, std::move(desired_embedder));
        promoted = true;

        // warn, not info: the daemon's default log_level is "warn", and an
        // operation that takes vector search offline for minutes should not be
        // invisible at the default level. The 1000-row progress lines stay at
        // info for anyone who turns the level up.
        Diskerror::Logger::warn(std::format(
            lang::MSG_REEMBED_STARTED,
            desired_model_name, desired_engine));

        int count = backend_->rebuild_embeddings(active, /*progress=*/false);

        // Verify rather than trust. A rebuild that stops partway leaves rows
        // at the previous embedding version — same width, same blob format,
        // silently a different vector space. backfill_embeddings() re-encodes
        // exactly those rows, so a non-zero return means the main pass came up
        // short and we just repaired it. Say so loudly; a quiet partial
        // rebuild is what made this failure so hard to see.
        int repaired = backend_->backfill_embeddings(active);
        if (repaired > 0) {
            Diskerror::Logger::warn(std::format(
                lang::WARN_REEMBED_REPAIRED, repaired));
            count += repaired;
        }

        user_store_->set_setting("reembed_repair_pending", "false");
        user_store_->set_setting("reembedding", "false");
        user_store_->set_setting("reembedding_pid", "");
        Diskerror::Logger::warn(std::format(
            lang::MSG_REEMBED_FINISHED, count,
            desired_model_name, vtype, dims));
        return count;
    } catch (...) {
        // Only unwind the embedder if we never promoted. Once promoted, the
        // new model IS the recorded identity and repair_pending stays set so
        // housekeeping finishes the remaining rows.
        if (!promoted) {
            if (previous_embedder) embedder_ = std::move(previous_embedder);
            user_store_->set_setting("reembed_repair_pending", "false");
        }
        user_store_->set_setting("reembedding", "false");
        user_store_->set_setting("reembedding_pid", "");
        try { throw; }
        catch (const std::exception& e) {
            Diskerror::Logger::critical(std::format(lang::ERR_REEMBED_FAILED, e.what()));
        }
        catch (...) {
            Diskerror::Logger::critical(lang::ERR_REEMBED_FAILED_UNKNOWN);
        }
        throw;
    }
}

bool RaggerMemory::repair_pending() {
    auto v = user_store_->get_setting("reembed_repair_pending");
    return v.has_value() && *v == "true";
}

int RaggerMemory::resume_interrupted_reembed() {
    if (!repair_pending() || is_reembedding()) return 0;

    // Resume, don't restart. Every row the interrupted run finished already
    // carries the current version byte; backfill_embeddings() re-encodes only
    // the ones that don't, so the work left is exactly the work that remains.
    // The identity was promoted before the rebuild began, so embedder_ is
    // already the right model to finish with.
    Diskerror::Logger::warn(lang::MSG_REEMBED_RESUMING);

    int total = 0;
    // backfill is idempotent and converges: the first pass fixes the stale
    // rows, a second confirms none are left. Loop rather than wait for the
    // next housekeeping tick, but bound it so a row that somehow never
    // stabilises can't spin forever.

    constexpr int kMaxPasses = 5;
    for (int pass = 0; pass < kMaxPasses; ++pass) {
        int n = backend_->backfill_embeddings(*embedder_);
        total += n;
        if (n == 0) {
            user_store_->set_setting("reembed_repair_pending", "false");
            Diskerror::Logger::warn(std::format(
                lang::MSG_REEMBED_RESUME_DONE, total));
            return total;
        }
    }
    Diskerror::Logger::error(std::format(
        lang::ERR_REEMBED_RESUME_STUCK, kMaxPasses, total));
    return total;
}

int RaggerMemory::backfill_embeddings() {
    return backend_->backfill_embeddings(*embedder_);
}

int RaggerMemory::rebuild_phon(bool only_missing, bool progress) {
    return backend_->rebuild_phon(only_missing, progress);
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
                           const std::string& session_id,
                           const std::string& session_name,
                           const std::string& name_source) {
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
                                    /*defer_embedding=*/false, session_id,
                                    /*source_timestamp=*/"",
                                    session_name, name_source);
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
    const std::string dir = config().resolved_recipes_dir();
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
    //      value "default" means "track the configured default" —
    //      equivalent to having no row.
    //   2. The configured `default_recipe` (compiled-in default, overlaid
    //      by the `settings` table).
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
