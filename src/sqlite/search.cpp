#include "sqlite/backend_impl.h"
#include "embedder.h"
#include "config.h"



namespace ragger::sqlite {

using json = nlohmann::json;
namespace fs = std::filesystem;


    // Hybrid search across the four corpora: vector cosine (cached
    // embeddings) blended with the custom TF-IDF text index (literal +
    // metaphone, v0.16). Both are min-max normalized and combined with
    // vector_weight/bm25_weight (bm25_weight is now the single text-index
    // blend weight; the old separate phon_weight signal is folded into
    // score_query() itself via fts_w_metaphone).
    //
    // PHASE8: the summaries corpus is gone (notes ride the knowledge
    // corpus). Each SearchResult's metadata["source"] is "turn_summary",
    // "document", "knowledge" or "episode".
    // S43: single implementation of the weight push (was three copies:
    // search, search_text_only, agent_search supplement).
    void Backend::Impl::push_text_weights() {
        text_index_->set_weights({config().fts_w_unigram, config().fts_w_bigram,
                                   config().fts_w_metaphone, config().fts_literal_enabled});
    }

    // ---- shared corpus scoring (LR S74) ----------------------------------
    // Blend one hybrid corpus into scored results (every row, unsorted):
    // raw cosine blended with the custom text-index TF-IDF signal (literal +
    // metaphone folded together by score_query()). Each signal is min-max
    // normalized WITHIN this corpus then weighted-summed. Extracted from
    // search()'s score_corpus lambda so the per-corpus pool path
    // (agent_search) shares ONE blend implementation; search() merges the
    // four corpora's outputs exactly as the lambda used to accumulate them.
    std::vector<SearchResult> Backend::Impl::blend_hybrid_corpus(
            const EmbeddingCache& c, const Eigen::VectorXf& q_norm,
            const std::unordered_map<int, float>& kw) {
        const auto& ids = c.ids;
        const auto& texts = c.texts;
        const auto& cache_emb = c.embeddings;
        const auto& meta = c.metadata;
        const auto& ts = c.timestamps;
        int n = static_cast<int>(ids.size());
        std::vector<SearchResult> out;
        if (n == 0) return out;
        out.reserve(static_cast<size_t>(n));

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

        bool use_kw = config().bm25_enabled && !kw.empty();

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
            out.push_back(std::move(sr));
        }
        return out;
    }


    SearchResponse Backend::Impl::search(const std::string& query, int limit,
                          float min_score,
                          std::vector<std::string> /*collections*/) {
        using clock = std::chrono::high_resolution_clock;
        auto t_start = clock::now();

        ensure_doc_cache();
        ensure_knowledge_cache();
        ensure_turn_cache();
        ensure_episode_cache();
        int n_doc = static_cast<int>(doc_cache_.ids.size());
        int n_kno = static_cast<int>(knowledge_cache_.ids.size());
        int n_turn = static_cast<int>(turn_cache_.ids.size());
        int n_ep  = static_cast<int>(episode_cache_.ids.size());
        if (n_doc == 0 && n_kno == 0 && n_turn == 0 && n_ep == 0)
            return {{}, {{"corpus_size", 0}}};

        // Push the current tunable weights into the engine once per search
        // call (config-agnostic engine; only this backend reads config()).
        push_text_weights();

        // ---- query embedding ------------------------------------------
        // A null or not-ready embedder here would reach Eigen with an
        // empty/mismatched vector — an assert or garbage, not a reportable
        // error. Gate it (B §1.17). RaggerMemory checks readiness first;
        // this protects every other caller.
        if (!embedder || !embedder->ready())
            throw std::runtime_error("semantic search unavailable: embedder not ready");
        auto t_embed_start = clock::now();
        auto q_vec = embedder->encode(query);
        auto t_embed_end = clock::now();

        auto t_search_start = clock::now();
        Eigen::Map<Eigen::VectorXf> q(q_vec.data(), static_cast<int>(q_vec.size()));
        Eigen::VectorXf q_norm = q.normalized();

        std::vector<SearchResult> candidates;
        candidates.reserve(static_cast<size_t>(n_doc + n_kno + n_turn + n_ep));
        auto append = [&](std::vector<SearchResult>&& v) {
            candidates.insert(candidates.end(),
                              std::make_move_iterator(v.begin()),
                              std::make_move_iterator(v.end()));
        };

        // Convert a text_index_ score vector into the id->score map the
        // blend expects.
        auto to_map = [](const std::vector<TextScore>& v) {
            std::unordered_map<int, float> m;
            m.reserve(v.size());
            for (const auto& s : v) m[s.id] = s.score;
            return m;
        };
        auto no_scores = std::unordered_map<int, float>{};
        bool bm25_on = config().bm25_enabled;
        append(blend_hybrid_corpus(doc_cache_, q_norm,
                     bm25_on ? to_map(text_index_->score_query("documents", query)) : no_scores));
        // Knowledge (PHASE7: one corpus replaces decisions + insights). Same
        // status allow-list as ensure_knowledge_cache().
        append(blend_hybrid_corpus(knowledge_cache_, q_norm,
                     bm25_on ? to_map(text_index_->score_query("knowledge", query)) : no_scores));
        append(blend_hybrid_corpus(turn_cache_, q_norm,
                     bm25_on ? to_map(text_index_->score_query("turn_summaries", query)) : no_scores));
        append(blend_hybrid_corpus(episode_cache_, q_norm,
                     bm25_on ? to_map(text_index_->score_query("episodes", query)) : no_scores));
        auto t_search_end = clock::now();

        // ---- merged top-k selection -----------------------------------
        // Filter by min_score (raw cosine) FIRST, then rank survivors by
        // blended score and emit up to `limit`. Filtering before top-k
        // (rather than after) prevents under-filling: a high-blend candidate
        // with sub-threshold cosine no longer steals a slot from a valid one.
        std::vector<int> ranking;
        ranking.reserve(candidates.size());
        for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
            if (candidates[i].score >= min_score)
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
            results.push_back(candidates[ranking[k]]);
        }

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        json timing = {
            {"embedding_ms", ms(t_embed_start, t_embed_end)},
            {"search_ms",    ms(t_search_start, t_search_end)},
            {"total_ms",     ms(t_start, clock::now())},
            {"corpus_size",  n_doc + n_kno + n_turn + n_ep}
        };
        return {std::move(results), std::move(timing)};
    }


    // LR S74: per-corpus hybrid pools for agent_search. One pass per
    // requested source label; each corpus is capped at its OWN pool (blended
    // desc) instead of competing for slots in one merged pool. The query is
    // embedded once for all corpora. Same embedder gate as search().
    std::map<std::string, std::vector<SearchResult>>
    Backend::Impl::hybrid_pools(const std::string& query,
                                const std::set<std::string>& sources, int pool,
                                bool use_kw) {
        if (sources.count("document"))     ensure_doc_cache();
        if (sources.count("knowledge"))    ensure_knowledge_cache();
        if (sources.count("turn_summary")) ensure_turn_cache();
        if (sources.count("episode"))      ensure_episode_cache();
        if (sources.count("turn"))         ensure_raw_turn_cache();
        if (!embedder || !embedder->ready())
            throw std::runtime_error("semantic search unavailable: embedder not ready");
        push_text_weights();
        auto q_vec = embedder->encode(query);
        Eigen::Map<Eigen::VectorXf> q(q_vec.data(), static_cast<int>(q_vec.size()));
        Eigen::VectorXf q_norm = q.normalized();

        std::map<std::string, std::vector<SearchResult>> out;
        auto one = [&](const char* source, EmbeddingCache& c, const char* table) {
            std::unordered_map<int, float> kw;
            if (use_kw && config().bm25_enabled)
                for (const auto& s : text_index_->score_query(table, query))
                    kw[s.id] = s.score;
            auto v = blend_hybrid_corpus(c, q_norm, kw);
            std::sort(v.begin(), v.end(),
                      [](const SearchResult& a, const SearchResult& b) {
                          return a.blended > b.blended;
                      });
            if (static_cast<int>(v.size()) > pool) v.resize(static_cast<size_t>(pool));
            out.emplace(source, std::move(v));
        };
        if (sources.count("document"))     one("document", doc_cache_, "documents");
        if (sources.count("knowledge"))    one("knowledge", knowledge_cache_, "knowledge");
        if (sources.count("turn_summary")) one("turn_summary", turn_cache_, "turn_summaries");
        if (sources.count("episode"))      one("episode", episode_cache_, "episodes");
        // LR S75: raw exchanges, opt-in only (agent_search never puts `turn`
        // in the default corpus set).
        if (sources.count("turn"))         one("turn", raw_turn_cache_, "turns");
        return out;
    }


    // Score one text corpus: text-index hits joined back to the row's text +
    // timestamp. S45: `from_clause` lets a corpus whose row lives behind a
    // JOIN (documents) use this ONE path; it replaces the bare table name in
    // the FROM when given. LR S74: extracted from search_text_only() so the
    // per-corpus pool path shares the join logic (and its fixes).
    std::vector<SearchResult> Backend::Impl::text_score_corpus(
            const std::string& query, const char* table, const char* id_col,
            const char* text_col, const char* source_label, const char* ts_expr,
            const std::string& extra_where, const char* from_clause) {
        std::vector<SearchResult> out;
        std::vector<TextScore> hits = text_index_->score_query(table, query);
        if (hits.empty()) return out;

        for (const auto& hit : hits) {
            Stmt s(db, std::format(
                "SELECT {}, {} FROM {} WHERE {} = ?{}",
                text_col, ts_expr, from_clause ? from_clause : table, id_col, extra_where));
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
            out.push_back(std::move(sr));
        }
        return out;
    }


    namespace {
    // One row per text corpus: how to join a text_index_ hit back to the
    // row's text + timestamp. Shared by search_text_only() and text_pools()
    // so the corpora (and their join fixes) live in exactly one list.
    struct TextCorpusSpec {
        const char* source;       // metadata source label / agent layer
        const char* table;        // text_index_ table name
        const char* id_col;
        const char* text_col;
        const char* ts_expr;
        std::string extra_where;
        const char* from_clause;  // nullptr = bare table name
        // LR S75: corpora the OPT-IN agent layer scores but the flat quick
        // search (and the degraded all-corpora path) must never include.
        bool agent_only = false;
    };
    const std::vector<TextCorpusSpec>& text_corpora() {
        static const std::vector<TextCorpusSpec> kCorpora = [] {
            std::vector<TextCorpusSpec> v;
            // Documents (imported_at lives on document_sources, joined via
            // document_source_id — pre-existing bug fixed while rewiring this
            // corpus for v0.16: the old code referenced documents.imported_at,
            // a column that has never existed on the documents table itself).
            v.push_back({"document", "documents", "d.document_id", "d.text",
                "CASE WHEN ds.imported_at IS NULL THEN '' "
                "ELSE datetime(ds.imported_at,'unixepoch','localtime') END",
                "",
                "documents d LEFT JOIN document_sources ds "
                "ON d.document_source_id = ds.document_source_id"});
            // Knowledge (PHASE7: one corpus replaces decisions + insights). Same
            // status allow-list as ensure_knowledge_cache(): superseded / retired /
            // pending rows are history or unreviewed drafts — they must stay
            // hidden here too, or degraded (text-only) search leaks them while
            // normal search hides them.
            v.push_back({"knowledge", "knowledge", "knowledge_id", "text",
                "datetime(created_at,'unixepoch','localtime')",
                " AND status IN (" + status::in_list_sql(status::kCacheLoaded) + ")",
                nullptr});
            // Turn summaries. NOTE: this table has no created_at — an L2 summary
            // inherits its source turn's timestamp in turn_datetime (the
            // (session_id, turn_datetime) pair is the join back to the turn).
            v.push_back({"turn_summary", "turn_summaries", "turn_summary_id", "text",
                "datetime(turn_datetime,'unixepoch','localtime')", "", nullptr});
            // Episodes (db 0.17). Marker rows (text='') are excluded here exactly
            // as ensure_episode_cache() excludes them from the vector corpus.
            v.push_back({"episode", "episodes", "episode_id", "text",
                "datetime(span_end,'unixepoch','localtime')", " AND text != ''", nullptr});
            // LR S75: raw exchanges — opt-in agent layer only (never in the
            // flat search's all-corpora sweep). Same non-empty rule as the
            // vector loader.
            v.push_back({"turn", "turns", "turn_id",
                "ifnull(user_text,'') || CASE "
                "WHEN ifnull(assistant_text,'') = '' THEN '' "
                "ELSE char(10) || char(10) || assistant_text END",
                "datetime(created_at,'unixepoch','localtime')",
                " AND (ifnull(user_text,'') != '' OR ifnull(assistant_text,'') != '')",
                nullptr, true});
            return v;
        }();
        return kCorpora;
    }
    } // anonymous namespace


    // Text-only search: text-index keyword + phonetic scoring across all four
    // corpora. No embedding caches, no embedder call. Used when embeddings
    // are degraded (drift mismatch at startup).
    SearchResponse Backend::Impl::search_text_only(const std::string& query, int limit) {
        using clock = std::chrono::high_resolution_clock;
        auto t_start = clock::now();

        push_text_weights();

        std::vector<SearchResult> candidates;
        for (const auto& c : text_corpora()) {
            if (c.agent_only) continue;  // LR S75: opt-in layer, not in the sweep
            auto v = text_score_corpus(query, c.table, c.id_col, c.text_col,
                                       c.source, c.ts_expr, c.extra_where, c.from_clause);
            candidates.insert(candidates.end(),
                              std::make_move_iterator(v.begin()),
                              std::make_move_iterator(v.end()));
        }

        if (candidates.empty())
            return {{}, {{"corpus_size", 0}, {"text_only", true}}};

        // Rank by blended score, top-k
        std::sort(candidates.begin(), candidates.end(),
                  [](const SearchResult& a, const SearchResult& b) {
                      return a.blended > b.blended;
                  });

        std::vector<SearchResult> results;
        int top_k = std::min(limit, static_cast<int>(candidates.size()));
        for (int i = 0; i < top_k; ++i)
            results.push_back(std::move(candidates[i]));

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        json timing = {
            {"total_ms", ms(t_start, clock::now())},
            {"text_only", true},
        };
        return {std::move(results), std::move(timing)};
    }


    // LR S74: per-corpus text-only pools (degraded path). Same contract as
    // hybrid_pools: each requested corpus capped at its OWN pool (blended
    // desc — here blended == the text-index score).
    std::map<std::string, std::vector<SearchResult>>
    Backend::Impl::text_pools(const std::string& query,
                              const std::set<std::string>& sources, int pool) {
        push_text_weights();
        std::map<std::string, std::vector<SearchResult>> out;
        for (const auto& c : text_corpora()) {
            if (!sources.count(c.source)) continue;
            auto v = text_score_corpus(query, c.table, c.id_col, c.text_col,
                                       c.source, c.ts_expr, c.extra_where, c.from_clause);
            std::sort(v.begin(), v.end(),
                      [](const SearchResult& a, const SearchResult& b) {
                          return a.blended > b.blended;
                      });
            if (static_cast<int>(v.size()) > pool) v.resize(static_cast<size_t>(pool));
            out.emplace(c.source, std::move(v));
        }
        return out;
    }


} // namespace ragger::sqlite
