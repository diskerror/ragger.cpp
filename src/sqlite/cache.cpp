#include "sqlite/backend_impl.h"
#include "embedder.h"
#include "config.h"
#include "lang.h"
#include "util/fs.h"
#include "util/time.h"
#include "util/sqlite.h"
#include "double_metaphone.h"
#include "fts_tokenize.h"
#include "sqlite/text_index.h"
#include "vector_codec.h"
#include "Logger.h"
#include <format>
#include "nlohmann_json.hpp"

#include <sqlite3.h>
#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <iterator>
#include <filesystem>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <mutex>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ragger::sqlite {

using json = nlohmann::json;
namespace fs = std::filesystem;

// Decode any embedding blob format: current payload-only (db_version 0.15+,
// version tag lives in the sibling embedding_version column, offset=0) or
// the old version-tagged format (pre-0.15, offset=1, for any row a
// migration missed). Bit-for-bit, the current offset=0 F16/F32 format is
// identical to the historic pre-header raw f16/f32 blobs, so no separate
// legacy-raw path is needed — expected_blob_size(t, dims, 0) naturally
// matches both.
namespace {

using Diskerror::EmbeddingCodec::VectorType;

// Try current payload-only format first (0.15+; also covers pre-header
// legacy raw f16/f32, byte-identical), then the old version-tagged format.
bool decode_any(const void* blob, int blob_bytes, int expected_dims,
                VectorType t, std::vector<float>& out) {
    out.assign(static_cast<size_t>(expected_dims), 0.0f);
    if (blob == nullptr || expected_dims <= 0) return false;

    if (blob_bytes == Diskerror::EmbeddingCodec::expected_blob_size(t, expected_dims, 0))
        return Diskerror::EmbeddingCodec::decode(blob, blob_bytes, expected_dims, t, out, 0);

    // Old version-tagged format (pre-0.15) fallback.
    if (blob_bytes == Diskerror::EmbeddingCodec::expected_blob_size(t, expected_dims, 1))
        return Diskerror::EmbeddingCodec::decode(blob, blob_bytes, expected_dims, t, out, 1);

    return false;
}

// Pack decoded rows into a cache's L2-normalized row matrix — the tail
// shared by all five ensure_*_cache() loaders (review 2C). Templated on the
// cache type because Backend::Impl is private; deduction from the concrete
// EmbeddingCache instance keeps the boundary intact.
template <class Cache>
void pack_embeddings(Cache& c, const std::vector<std::vector<float>>& rows,
                     int dims) {
    const int n = static_cast<int>(rows.size());
    c.embeddings.resize(n, dims);
    for (int i = 0; i < n; ++i) {
        c.embeddings.row(i) =
            Eigen::Map<const Eigen::RowVectorXf>(rows[i].data(), dims);
        float nrm = c.embeddings.row(i).norm();
        if (nrm > 1e-12f) c.embeddings.row(i) /= nrm;
    }
}

}  // anonymous namespace


    // Decode an embedding BLOB into a float vector of `dims`. Uses decode_any
    // which handles the current version-tagged format AND legacy formats (RV1
    // headers, raw f16/f32). On any failure (NULL, deferred-but-unbackfilled
    // row, corruption, or a dimension mismatch) a zero vector is returned;
    // the first such row per cache load is logged once (table + id).
    std::vector<float> Backend::Impl::decode_embedding_blob(const void* blob, int blob_bytes,
                                             int dims, const char* table,
                                             int row_id, bool& warned) {
        std::vector<float> emb;
        if (blob == nullptr) {         // deferred row; silent (expected)
            emb.assign(static_cast<size_t>(dims), 0.0f);
            return emb;
        }
        if (!decode_any(blob, blob_bytes, dims, vtype_, emb) && !warned) {
            warned = true;
            Diskerror::Logger::warn(std::format(
                "[cache] {} row {}: undecodable embedding blob ({} bytes, "
                "expected {}-dim {}); using zero vector",
                table, row_id, blob_bytes, dims,
                vector_codec::to_string(vtype_)));
        }
        return emb;
    }


    // ---- epoch timestamp for INTEGER columns --------------------------
    // resolve_epoch(caller_ts_string): callers (importers) sometimes supply
    // a specific historical timestamp as a "%F %T" string rather than "now".
    // Parse it via the shared parse_db_timestamp() and convert to epoch
    // seconds; empty input -> current time. A non-empty string that fails
    // to parse is a caller error (malformed import timestamp) -- surfaced
    // as a thrown exception rather than silently binding garbage, matching
    // how other malformed-input cases in this file are handled.
    int64_t Backend::Impl::resolve_epoch(const std::string& caller_ts) {
        if (caller_ts.empty()) return db_epoch();
        auto tt = parse_db_timestamp(caller_ts);
        if (!tt) {
            throw std::runtime_error(
                "Ragger: malformed timestamp supplied to store call: '" + caller_ts + "'");
        }
        return db_epoch(*tt);
    }


    // ---- cache --------------------------------------------------------
    void Backend::Impl::invalidate_cache() { summary_cache_.valid = false; }

    void Backend::Impl::invalidate_doc_cache() { doc_cache_.valid = false; }

    void Backend::Impl::invalidate_knowledge_cache() { knowledge_cache_.valid = false; }

    void Backend::Impl::invalidate_turn_cache() { turn_cache_.valid = false; }

    void Backend::Impl::invalidate_episode_cache() { episode_cache_.valid = false; }


    // Shared body of the five loaders (PHASE3 Step 5). The loop shape is
    // identical everywhere: id/text/embedding at columns 0/1/2, timestamp at
    // column 5, everything else handed to fill_meta. SQL strings below are
    // character-for-character the pre-refactor loader queries.
    void Backend::Impl::load_cache(EmbeddingCache& c, const char* label,
                                   const std::string& sql,
                                   const std::function<void(Stmt&, json&)>& fill_meta) {
        c.ids.clear();
        c.texts.clear();
        c.metadata.clear();
        c.timestamps.clear();

        Stmt s(db, sql);
        std::vector<std::vector<float>> emb_rows;
        const int expected_dims = config().embedding_dimensions;
        bool blob_warned = false;

        while (s.step()) {
            c.ids.push_back(s.column_int(0));
            c.texts.push_back(s.column_text(1));

            const void* blob = s.column_blob(2);
            int blob_bytes   = s.column_bytes(2);
            emb_rows.push_back(decode_embedding_blob(
                blob, blob_bytes, expected_dims, label,
                c.ids.back(), blob_warned));

            json meta = json::object();
            fill_meta(s, meta);
            c.metadata.push_back(std::move(meta));

            c.timestamps.push_back(s.column_text(5));
        }

        pack_embeddings(c, emb_rows, expected_dims);
    }


    void Backend::Impl::ensure_cache() {
        if (summary_cache_.valid) return;
        load_cache(summary_cache_, "summaries",
            "SELECT summary_id, text, embedding, level, tags, "
            "       datetime(created_at,'unixepoch','localtime') "
            "FROM summaries WHERE level != 'episode'",
            // PHASE5 B.3: episodes moved to their own table (episode_cache_);
            // the filter is belt-and-braces for any pre-move row.
            [](Stmt& s, json& meta) {
                // Lean v2 summaries: surface level/tags as metadata so the
                // generic API and keep-protection (via tags) have what they
                // need.
                meta["source"] = "summary";
                meta["level"]  = s.column_text(3);
                std::string tags = s.column_text(4);
                if (!tags.empty()) meta["tags"] = tags;
            });
        summary_cache_.valid = true;
    }


    void Backend::Impl::ensure_doc_cache() {
        if (doc_cache_.valid) return;
        load_cache(doc_cache_, "documents",
            "SELECT d.document_id, d.text, d.embedding, "
            "       ifnull(ds.title,''), d.tags, "
            "       datetime(ds.imported_at, 'unixepoch', 'localtime') "
            "FROM documents d "
            "LEFT JOIN document_sources ds "
            "  ON ds.document_source_id = d.document_source_id",
            [](Stmt& s, json& meta) {
                meta["source"] = "document";
                meta["title"]  = s.column_text(3);
                std::string tags = s.column_text(4);
                if (!tags.empty()) meta["tags"] = tags;
            });
        doc_cache_.valid = true;
    }


    // knowledge (PHASE7 B.2): ONE corpus replacing dec_cache_ + ins_cache_.
    // Consequence (TODO Q10): retired and superseded rows are NOT in the
    // semantic cache (same as superseded always was); the 0.16 rollups that
    // arrive as retired notes in Phase 8 are therefore invisible to semantic
    // search — the lexical path and memory_get still reach them.
    void Backend::Impl::ensure_knowledge_cache() {
        if (knowledge_cache_.valid) return;
        load_cache(knowledge_cache_, "knowledge",
            "SELECT knowledge_id, text, embedding, status, tags, "
            "       datetime(created_at,'unixepoch','localtime'), "
            "       ifnull(trigger,''), session_id, kind "
            "FROM knowledge "
            // db 0.17: superseded / retired / pending rows are history or
            // unreviewed drafts -- kept in the table, hidden from search.
            // Allow-list, not deny-list, so a future status stays hidden
            // until someone decides it should surface. 'roadmap' keeps its
            // pre-0.17 visibility.
            "WHERE status IN (" + status::in_list_sql(status::kCacheLoaded) + ")",
            [](Stmt& s, json& meta) {
                meta["source"] = "knowledge";
                meta["kind"] = s.column_text(8);
                meta["status"] = s.column_text(3);
                std::string tags = s.column_text(4);
                if (!tags.empty()) meta["tags"] = tags;
                std::string trig = s.column_text(6);
                if (!trig.empty()) meta["trigger"] = trig;
                if (!s.is_null(7)) meta["session_id"] = s.column_int(7);
            });
        knowledge_cache_.valid = true;
    }


    void Backend::Impl::ensure_turn_cache() {
        if (turn_cache_.valid) return;
        load_cache(turn_cache_, "turn_summaries",
            "SELECT turn_summary_id, text, embedding, turn_id, session_id, "
            "       datetime(turn_datetime,'unixepoch','localtime') "
            "FROM turn_summaries WHERE text != ''",
            [](Stmt& s, json& meta) {
                meta["source"] = "turn_summary";
                if (!s.is_null(3)) meta["turn_id"] = s.column_int(3);
                if (!s.is_null(4)) meta["session_id"] = s.column_int(4);
                // Human-readable datetime (mirrors turns.created_at) for
                // temporal ordering; kept alongside turn_id (raw-turn lookup
                // key).
                std::string ts = s.column_text(5);
                if (!ts.empty()) meta["datetime"] = ts;
            });
        turn_cache_.valid = true;
    }


    // PHASE5 B.3: episodes are a search corpus of their own. Marker rows
    // (text = '') never load. `level` metadata is kept alongside `layer` so
    // legacy /search output keeps its shape.
    void Backend::Impl::ensure_episode_cache() {
        if (episode_cache_.valid) return;
        load_cache(episode_cache_, "episodes",
            "SELECT episode_id, text, embedding, tags, session_id, "
            "       datetime(span_end,'unixepoch','localtime'), model_id "
            "FROM episodes WHERE text != ''",
            [](Stmt& s, json& meta) {
                meta["source"] = "episode";
                meta["layer"]  = "episode";
                meta["level"]  = "episode";
                std::string tags = s.column_text(3);
                if (!tags.empty()) meta["tags"] = tags;
                if (!s.is_null(4)) meta["session_id"] = s.column_int(4);
                if (!s.is_null(6)) meta["model_id"] = s.column_int(6);
            });
        episode_cache_.valid = true;
    }


} // namespace ragger::sqlite