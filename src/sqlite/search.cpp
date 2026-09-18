#include "backend_impl.h"
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


    // Hybrid search over summaries: vector cosine (cached embeddings) blended
    // with the custom TF-IDF text index (literal + metaphone, v0.16). Both are
    // min-max normalized and combined with vector_weight/bm25_weight (bm25_weight
    // is now the single text-index blend weight; the old separate phon_weight
    // signal is folded into score_query() itself via fts_w_metaphone).
    //
    // NOTE: the lean v2 summaries table has no collection column, so the
    // `collections` filter is currently a no-op (kept for API/source compat).
    // Documents (L5) and decisions (L6) ARE merged into search: parallel passes
    // (ensure_doc_cache / ensure_dec_cache + a text_index_->score_query() pass)
    // are scored identically and merged with the summaries results into the
    // single ranked top-k returned here. Each SearchResult's metadata["source"]
    // is "summary", "document", or "decision" so callers can tell the corpora
    // apart.
    SearchResponse SqliteBackend::Impl::search(const std::string& query, int limit,
                          float min_score,
                          std::vector<std::string> /*collections*/) {
        using clock = std::chrono::high_resolution_clock;
        auto t_start = clock::now();

        ensure_cache();
        ensure_doc_cache();
        ensure_dec_cache();
        ensure_turn_cache();
        int n_sum = static_cast<int>(cached_ids.size());
        int n_doc = static_cast<int>(doc_ids.size());
        int n_dec = static_cast<int>(dec_ids.size());
        int n_turn = static_cast<int>(turn_ids.size());
        if (n_sum == 0 && n_doc == 0 && n_dec == 0 && n_turn == 0) return {{}, {{"corpus_size", 0}}};

        // Push the current tunable weights into the engine once per search
        // call (config-agnostic engine; only this backend reads config()).
        text_index_->set_weights({config().fts_w_unigram, config().fts_w_bigram,
                                   config().fts_w_metaphone, config().fts_literal_enabled});

        // ---- query embedding ------------------------------------------
        auto t_embed_start = clock::now();
        auto q_vec = embedder->encode(query);
        auto t_embed_end = clock::now();

        auto t_search_start = clock::now();
        Eigen::Map<Eigen::VectorXf> q(q_vec.data(), static_cast<int>(q_vec.size()));
        Eigen::VectorXf q_norm = q.normalized();

        // A scored candidate: raw cosine is reported as the score, the blended
        // value drives ranking. Both corpora produce these and are merged.
        struct Candidate {
            float        blended;
            SearchResult result;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<size_t>(n_sum + n_doc + n_dec + n_turn));

        // Score one corpus: vector cosine blended with the custom text-index
        // TF-IDF signal (literal + metaphone folded together by score_query()).
        // Each signal is min-max normalized to [0,1] then weighted-summed.
        // Shared by the summaries/documents/decisions/turns passes so the
        // blend logic lives in exactly one place.
        auto score_corpus =
            [&](const std::vector<int>& ids,
                const std::vector<std::string>& texts,
                const Eigen::MatrixXf& cache_emb,
                const std::vector<json>& meta,
                const std::vector<std::string>& ts,
                const std::unordered_map<int, float>& kw) {
            int n = static_cast<int>(ids.size());
            if (n == 0) return;

            // cache_emb rows are L2-normalized at cache-build time, and q_norm
            // is the normalized query vector, so this product is raw cosine.
            Eigen::VectorXf similarities = cache_emb * q_norm;   // raw cosine, reported

            auto norm_minmax = [](Eigen::VectorXf& v) {
                float mn = v.minCoeff(), mx = v.maxCoeff();
                if (mx > mn) v = (v.array() - mn) / (mx - mn);
            };
            auto gather = [&](const std::unordered_map<int, float>& m) {
                Eigen::VectorXf v(n);
                for (int i = 0; i < n; ++i) {
                    auto it = m.find(ids[i]);
                    v(i) = (it == m.end()) ? 0.0f : it->second;
                }
                return v;
            };

            bool use_kw   = config().bm25_enabled && !kw.empty();

            // Normalized [0,1] signal vectors. vec is always computed; kw
            // only when the text-index signal is active. When it doesn't
            // contribute, ranking falls back to raw cosine (combined) so the
            // vec-only path behaves exactly as before.
            Eigen::VectorXf vec_norm = similarities;
            Eigen::VectorXf kw_norm;
            Eigen::VectorXf combined = similarities;
            if (use_kw) {
                norm_minmax(vec_norm);
                combined = config().vector_weight * vec_norm;
                kw_norm = gather(kw);
                norm_minmax(kw_norm);
                combined += config().bm25_weight * kw_norm;
            }

            for (int i = 0; i < n; ++i) {
                SearchResult sr{ids[i], texts[i], similarities(i), meta[i], ts[i]};
                // Per-signal breakdown for RAGGER_STATS. -1 marks an inactive
                // signal so the analysis can tell "0 contribution" apart from
                // "not part of this search". vec_score is the normalized cosine
                // that actually fed the blend (== raw cosine on the vec-only
                // path, since no min-max is applied there). phon_score is no
                // longer an independent signal (folded into bm25_score by
                // score_query()); kept at -1 for API/stats-schema compat.
                sr.vec_score  = vec_norm(i);
                sr.bm25_score = use_kw   ? kw_norm(i) : -1.0f;
                sr.phon_score = -1.0f;
                sr.blended    = combined(i);
                candidates.push_back({combined(i), std::move(sr)});
            }
        };

        // Convert a text_index_ score vector into the id->score map the
        // blend lambda expects.
        auto to_map = [](const std::vector<TextScore>& v) {
            std::unordered_map<int, float> m;
            m.reserve(v.size());
            for (const auto& s : v) m[s.id] = s.score;
            return m;
        };
        auto no_scores = std::unordered_map<int, float>{};
        bool bm25_on = config().bm25_enabled;
        score_corpus(cached_ids, cached_texts, cached_embeddings,
                     cached_metadata, cached_timestamps,
                     bm25_on ? to_map(text_index_->score_query("summaries", query)) : no_scores);
        score_corpus(doc_ids, doc_texts, doc_embeddings,
                     doc_metadata, doc_timestamps,
                     bm25_on ? to_map(text_index_->score_query("documents", query)) : no_scores);
        score_corpus(dec_ids, dec_texts, dec_embeddings,
                     dec_metadata, dec_timestamps,
                     bm25_on ? to_map(text_index_->score_query("decisions", query)) : no_scores);
        score_corpus(turn_ids, turn_texts, turn_embeddings,
                     turn_metadata, turn_timestamps,
                     bm25_on ? to_map(text_index_->score_query("turn_summaries", query)) : no_scores);
        auto t_search_end = clock::now();

        // ---- merged top-k selection -----------------------------------
        // Filter by min_score (raw cosine) FIRST, then rank survivors by
        // blended score and emit up to `limit`. Filtering before top-k
        // (rather than after) prevents under-filling: a high-blend candidate
        // with sub-threshold cosine no longer steals a slot from a valid one.
        std::vector<int> ranking;
        ranking.reserve(candidates.size());
        for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
            if (candidates[i].result.score >= min_score)
                ranking.push_back(i);
        }
        int total = static_cast<int>(ranking.size());
        int top_k = std::min(limit, total);
        std::partial_sort(ranking.begin(),
                          ranking.begin() + top_k,
                          ranking.end(),
                          [&](int a, int b) {
                              return candidates[a].blended > candidates[b].blended;
                          });

        std::vector<SearchResult> results;
        for (int k = 0; k < top_k; ++k) {
            results.push_back(candidates[ranking[k]].result);
        }

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        json timing = {
            {"embedding_ms", ms(t_embed_start, t_embed_end)},
            {"search_ms",    ms(t_search_start, t_search_end)},
            {"total_ms",     ms(t_start, clock::now())},
            {"corpus_size",  n_sum + n_doc + n_dec + n_turn}
        };
        return {std::move(results), std::move(timing)};
    }


    // Text-only search: FTS5 keyword + phonetic scoring across all four
    // corpora. No embedding caches, no embedder call. Used when embeddings
    // are degraded (drift mismatch at startup).
    SearchResponse SqliteBackend::Impl::search_text_only(const std::string& query, int limit) {
        using clock = std::chrono::high_resolution_clock;
        auto t_start = clock::now();

        text_index_->set_weights({config().fts_w_unigram, config().fts_w_bigram,
                                   config().fts_w_metaphone, config().fts_literal_enabled});

        struct Candidate {
            float        score;
            SearchResult result;
        };
        std::vector<Candidate> candidates;

        // Score one corpus via the custom TF-IDF text index, then join back
        // to the source table for text + metadata.
        auto score_text_corpus = [&](
                const char* table, const char* id_col, const char* text_col,
                const char* source_label,
                const char* ts_expr) {
            std::vector<TextScore> hits = text_index_->score_query(table, query);
            if (hits.empty()) return;

            for (const auto& hit : hits) {
                Stmt s(db, std::format(
                    "SELECT {}, {} FROM {} WHERE {} = ?",
                    text_col, ts_expr, table, id_col));
                s.bind(1, hit.id);
                if (!s.step()) continue;
                std::string text = s.column_text(0);
                std::string ts   = s.column_text(1);

                json meta = json::object();
                meta["source"] = source_label;

                SearchResult sr{hit.id, text, hit.score, meta, ts};
                sr.vec_score  = -1.0f;  // no vector signal
                sr.bm25_score = hit.score;
                sr.phon_score = -1.0f;  // folded into bm25_score by score_query()
                sr.blended    = hit.score;
                candidates.push_back({hit.score, std::move(sr)});
            }
        };

        // Summaries
        score_text_corpus("summaries", "summary_id", "text", "summary",
                          "datetime(created_at,'unixepoch','localtime')");
        // Documents (imported_at lives on document_sources, joined via
        // document_source_id — pre-existing bug fixed while rewiring this
        // corpus for v0.16: the old code referenced documents.imported_at,
        // a column that has never existed on the documents table itself).
        {
            std::vector<TextScore> hits = text_index_->score_query("documents", query);
            for (const auto& hit : hits) {
                Stmt s(db,
                    "SELECT d.text, COALESCE(ds.imported_at, 0) "
                    "FROM documents d LEFT JOIN document_sources ds "
                    "ON d.document_source_id = ds.document_source_id "
                    "WHERE d.document_id = ?");
                s.bind(1, hit.id);
                if (!s.step()) continue;
                std::string text = s.column_text(0);
                std::string ts   = s.column_text(1);

                json meta = json::object();
                meta["source"] = "document";

                SearchResult sr{hit.id, text, hit.score, meta, ts};
                sr.vec_score  = -1.0f;
                sr.bm25_score = hit.score;
                sr.phon_score = -1.0f;
                sr.blended    = hit.score;
                candidates.push_back({hit.score, std::move(sr)});
            }
        }
        // Decisions
        score_text_corpus("decisions", "decision_id", "text", "decision",
                          "datetime(created_at,'unixepoch','localtime')");
        // Turn summaries. NOTE: this table has no created_at — an L2 summary
        // inherits its source turn's timestamp in turn_datetime (the
        // (session_id, turn_datetime) pair is the join back to the turn). The
        // wrong column name here threw "no such column: created_at" out of the
        // whole function, so text-only search never returned anything: both
        // the drift-degraded fallback and the startup warmup were dead.
        score_text_corpus("turn_summaries", "turn_summary_id", "text", "turn_summary",
                          "datetime(turn_datetime,'unixepoch','localtime')");

        if (candidates.empty())
            return {{}, {{"corpus_size", 0}, {"text_only", true}}};

        // Rank by blended score, top-k
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.score > b.score;
                  });

        std::vector<SearchResult> results;
        int top_k = std::min(limit, static_cast<int>(candidates.size()));
        for (int i = 0; i < top_k; ++i)
            results.push_back(std::move(candidates[i].result));

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        json timing = {
            {"total_ms", ms(t_start, clock::now())},
            {"text_only", true},
        };
        return {std::move(results), std::move(timing)};
    }


} // namespace ragger::sqlite
