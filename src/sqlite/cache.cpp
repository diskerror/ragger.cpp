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
    void Backend::Impl::invalidate_cache() { cache_valid = false; }

    void Backend::Impl::invalidate_doc_cache() { doc_cache_valid = false; }

    void Backend::Impl::invalidate_dec_cache() { dec_cache_valid = false; }

    void Backend::Impl::invalidate_turn_cache() { turn_cache_valid = false; }


    // Loads the summaries table into the vector cache. Keyword scores come
    // from FTS5 (summaries_fts) at query time — see keyword_scores().
    void Backend::Impl::ensure_cache() {
        if (cache_valid) return;

        cached_ids.clear();
        cached_texts.clear();
        cached_metadata.clear();
        cached_timestamps.clear();

        Stmt s(db,
            "SELECT summary_id, text, embedding, level, tags, "
            "       datetime(created_at,'unixepoch','localtime') "
            "FROM summaries");

        std::vector<std::vector<float>> emb_rows;
        const int expected_dims = config().embedding_dimensions;
        bool blob_warned = false;

        while (s.step()) {
            auto col_text = [&](int i) -> std::string { return s.column_text(i); };
            cached_ids.push_back(s.column_int(0));
            cached_texts.push_back(col_text(1));

            const void* blob = s.column_blob(2);
            int blob_bytes   = s.column_bytes(2);
            std::vector<float> emb = decode_embedding_blob(
                blob, blob_bytes, expected_dims, "summaries",
                cached_ids.back(), blob_warned);
            emb_rows.push_back(std::move(emb));

            // Lean v2 summaries: surface level/tags as metadata so the
            // generic API and keep-protection (via tags) have what they need.
            json meta = json::object();
            meta["source"] = "summary";
            meta["level"]  = col_text(3);
            std::string tags = col_text(4);
            if (!tags.empty()) meta["tags"] = tags;
            cached_metadata.push_back(std::move(meta));

            cached_timestamps.push_back(col_text(5));
        }

        // Pack into Eigen matrix (rows × dims). Every row is padded/zeroed to
        // expected_dims so NULL-embedding rows don't poison the matrix shape.
        int n = static_cast<int>(emb_rows.size());
        cached_embeddings.resize(n, expected_dims);
        for (int i = 0; i < n; ++i) {
            cached_embeddings.row(i) =
                Eigen::Map<Eigen::RowVectorXf>(emb_rows[i].data(), expected_dims);
            float nrm = cached_embeddings.row(i).norm();
            if (nrm > 1e-12f) cached_embeddings.row(i) /= nrm;
        }

        cache_valid = true;
    }


    // Loads the documents (L5) table into the parallel vector cache. Mirrors
    // ensure_cache(): vector scores come from here, keyword scores from FTS5
    // (documents_fts) at query time. Metadata carries source="document" plus
    // the title so search() consumers can distinguish documents from summaries.
    void Backend::Impl::ensure_doc_cache() {
        if (doc_cache_valid) return;

        doc_ids.clear();
        doc_texts.clear();
        doc_metadata.clear();
        doc_timestamps.clear();

        Stmt s(db,
            "SELECT d.document_id, d.text, d.embedding, "
            "       ifnull(ds.title,''), d.tags, "
            "       datetime(ds.imported_at, 'unixepoch', 'localtime') "
            "FROM documents d "
            "LEFT JOIN document_sources ds "
            "  ON ds.document_source_id = d.document_source_id");

        std::vector<std::vector<float>> emb_rows;
        const int expected_dims = config().embedding_dimensions;
        bool blob_warned = false;

        while (s.step()) {
            auto col_text = [&](int i) -> std::string { return s.column_text(i); };
            doc_ids.push_back(s.column_int(0));
            doc_texts.push_back(col_text(1));

            const void* blob = s.column_blob(2);
            int blob_bytes   = s.column_bytes(2);
            std::vector<float> emb = decode_embedding_blob(
                blob, blob_bytes, expected_dims, "documents",
                doc_ids.back(), blob_warned);
            emb_rows.push_back(std::move(emb));

            json meta = json::object();
            meta["source"] = "document";
            meta["title"]  = col_text(3);
            std::string tags = col_text(4);
            if (!tags.empty()) meta["tags"] = tags;
            doc_metadata.push_back(std::move(meta));

            doc_timestamps.push_back(col_text(5));
        }

        int n = static_cast<int>(emb_rows.size());
        doc_embeddings.resize(n, expected_dims);
        for (int i = 0; i < n; ++i) {
            doc_embeddings.row(i) =
                Eigen::Map<Eigen::RowVectorXf>(emb_rows[i].data(), expected_dims);
            float nrm = doc_embeddings.row(i).norm();
            if (nrm > 1e-12f) doc_embeddings.row(i) /= nrm;
        }

        doc_cache_valid = true;
    }


    // Loads the decisions (L6) table into the parallel vector cache. Mirrors
    // ensure_cache(): vector scores come from here, keyword scores from FTS5
    // (decisions_fts) at query time. Metadata carries source="decision" plus
    // the status so search() consumers can distinguish decisions from the
    // other corpora.
    void Backend::Impl::ensure_dec_cache() {
        if (dec_cache_valid) return;

        dec_ids.clear();
        dec_texts.clear();
        dec_metadata.clear();
        dec_timestamps.clear();

        Stmt s(db,
            "SELECT decision_id, text, embedding, status, tags, "
            "       datetime(created_at,'unixepoch','localtime') "
            "FROM decisions");

        std::vector<std::vector<float>> emb_rows;
        const int expected_dims = config().embedding_dimensions;
        bool blob_warned = false;

        while (s.step()) {
            auto col_text = [&](int i) -> std::string { return s.column_text(i); };
            dec_ids.push_back(s.column_int(0));
            dec_texts.push_back(col_text(1));

            const void* blob = s.column_blob(2);
            int blob_bytes   = s.column_bytes(2);
            std::vector<float> emb = decode_embedding_blob(
                blob, blob_bytes, expected_dims, "decisions",
                dec_ids.back(), blob_warned);
            emb_rows.push_back(std::move(emb));

            json meta = json::object();
            meta["source"] = "decision";
            meta["status"] = col_text(3);
            std::string tags = col_text(4);
            if (!tags.empty()) meta["tags"] = tags;
            dec_metadata.push_back(std::move(meta));

            dec_timestamps.push_back(col_text(5));
        }

        int n = static_cast<int>(emb_rows.size());
        dec_embeddings.resize(n, expected_dims);
        for (int i = 0; i < n; ++i) {
            dec_embeddings.row(i) =
                Eigen::Map<Eigen::RowVectorXf>(emb_rows[i].data(), expected_dims);
            float nrm = dec_embeddings.row(i).norm();
            if (nrm > 1e-12f) dec_embeddings.row(i) /= nrm;
        }

        dec_cache_valid = true;
    }


    // Loads the turn_summaries (L2) table into the parallel vector cache.
    // Mirrors ensure_dec_cache(): vector scores come from here, keyword
    // scores from FTS5 (turn_summaries_fts) at query time. WHERE text IS NOT
    // NULL excludes unfinalized placeholder rows (summarizer hasn't run yet)
    // — these must never surface in search results or feed the embedding
    // similarity matrix. Metadata carries source="turn_summary" plus
    // turn_id/session_id (turn_summaries has no tags/status/level columns).
    void Backend::Impl::ensure_turn_cache() {
        if (turn_cache_valid) return;

        turn_ids.clear();
        turn_texts.clear();
        turn_metadata.clear();
        turn_timestamps.clear();

        Stmt s(db,
            "SELECT turn_summary_id, text, embedding, turn_id, session_id, "
            "       datetime(turn_datetime,'unixepoch','localtime') "
            "FROM turn_summaries WHERE text != ''");

        std::vector<std::vector<float>> emb_rows;
        const int expected_dims = config().embedding_dimensions;
        bool blob_warned = false;

        while (s.step()) {
            auto col_text = [&](int i) -> std::string { return s.column_text(i); };
            turn_ids.push_back(s.column_int(0));
            turn_texts.push_back(col_text(1));

            const void* blob = s.column_blob(2);
            int blob_bytes   = s.column_bytes(2);
            std::vector<float> emb = decode_embedding_blob(
                blob, blob_bytes, expected_dims, "turn_summaries",
                turn_ids.back(), blob_warned);
            emb_rows.push_back(std::move(emb));

            json meta = json::object();
            meta["source"] = "turn_summary";
            if (!s.is_null(3)) meta["turn_id"] = s.column_int(3);
            if (!s.is_null(4)) meta["session_id"] = s.column_int(4);
            // Human-readable datetime (mirrors turns.created_at) for temporal
            // ordering; kept alongside turn_id (raw-turn lookup key).
            std::string ts = col_text(5);
            if (!ts.empty()) meta["datetime"] = ts;
            turn_metadata.push_back(std::move(meta));

            turn_timestamps.push_back(std::move(ts));
        }

        int n = static_cast<int>(emb_rows.size());
        turn_embeddings.resize(n, expected_dims);
        for (int i = 0; i < n; ++i) {
            turn_embeddings.row(i) =
                Eigen::Map<Eigen::RowVectorXf>(emb_rows[i].data(), expected_dims);
            float nrm = turn_embeddings.row(i).norm();
            if (nrm > 1e-12f) turn_embeddings.row(i) /= nrm;
        }

        turn_cache_valid = true;
    }


} // namespace ragger::sqlite
