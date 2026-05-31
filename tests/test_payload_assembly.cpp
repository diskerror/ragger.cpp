/**
 * Payload assembly tests (issue #23) — budget + priority shrinking.
 * Pure logic; no model/DB needed.
 */
#include "ragger/chat_sessions.h"

#include <cassert>
#include <print>
#include <string>

using ragger::PayloadPiece;
using ragger::PieceKind;

// A SYSTEM piece of roughly `tokens`*4 chars (estimate_tokens uses chars/4).
static PayloadPiece sys(int priority, int tokens, const std::string& label = "") {
    PayloadPiece p;
    p.kind = PieceKind::System;
    p.priority = priority;
    p.label = label;
    p.content = std::string(static_cast<size_t>(tokens) * 4, 'x');
    return p;
}
static PayloadPiece turn(int priority, bool keep, const std::string& role,
                         int tokens) {
    PayloadPiece p;
    p.kind = PieceKind::Turn;
    p.priority = priority;
    p.keep = keep;
    p.role = role;
    p.content = std::string(static_cast<size_t>(tokens) * 4, 'y');
    return p;
}

int main() {
    std::println("Running payload assembly tests:");
    const float cpt = 4.0f;

    // estimate_tokens basics.
    assert(ragger::estimate_tokens("", cpt) == 0);
    assert(ragger::estimate_tokens(std::string(40, 'a'), cpt) == 10);

    // 1. Under budget: nothing shed, system merged + turns in order.
    {
        std::vector<PayloadPiece> pieces = {
            sys(2, 10, "## Persona"),
            sys(4, 10, "## Session summary"),
            turn(3, true, "user", 5),          // previous raw turn (keep)
            turn(1, true, "user", 5),          // new user message (keep)
        };
        auto r = ragger::assemble_payload(pieces, /*budget=*/1000, cpt);
        assert(r.shed_count == 0);
        assert(r.fit);
        // messages[0] is the merged system block; persona (pri 2) before
        // session summary (pri 4).
        assert(r.messages.size() == 3);          // system + 2 turns
        assert(r.messages[0].role == "system");
        auto persona_pos = r.messages[0].content.find("## Persona");
        auto sess_pos    = r.messages[0].content.find("## Session summary");
        assert(persona_pos != std::string::npos && sess_pos != std::string::npos);
        assert(persona_pos < sess_pos);
        assert(r.messages[1].role == "user" && r.messages[2].role == "user");
        std::println("  under-budget assembly OK");
    }

    // 2. Over budget: shed lowest-importance (highest priority number) first,
    //    never shed keep pieces.
    {
        std::vector<PayloadPiece> pieces = {
            sys(2, 30, "## Persona"),            // important
            sys(9, 30, "## Old summaries"),      // least important → shed first
            sys(7, 30, "## Recent summaries"),   // shed second
            turn(1, true, "user", 30),           // keep
        };
        // Budget fits persona(30)+keep(30)=60 plus labels (~a few). Force both
        // low-priority system pieces to be shed.
        auto r = ragger::assemble_payload(pieces, /*budget=*/70, cpt);
        assert(r.shed_count == 2);
        assert(r.messages[0].content.find("## Persona") != std::string::npos);
        assert(r.messages[0].content.find("## Old summaries") == std::string::npos);
        assert(r.messages[0].content.find("## Recent summaries") == std::string::npos);
        std::println("  over-budget shedding order OK");
    }

    // 3. Shed exactly one: the priority-9 piece goes before the priority-7.
    {
        std::vector<PayloadPiece> pieces = {
            sys(7, 20, "## Recent"),
            sys(9, 20, "## Old"),
            turn(1, true, "user", 20),
        };
        // total ~ 20+20+20 (+labels). Budget that forces dropping just one.
        auto r = ragger::assemble_payload(pieces, /*budget=*/45, cpt);
        assert(r.shed_count == 1);
        assert(r.messages[0].content.find("## Recent") != std::string::npos);
        assert(r.messages[0].content.find("## Old") == std::string::npos);
        std::println("  single-shed picks lowest priority OK");
    }

    // 4. Only keep pieces remain and still exceed budget → fit=false, kept.
    {
        std::vector<PayloadPiece> pieces = {
            turn(1, true, "user", 100),          // keep, huge
            sys(9, 5, "## Droppable"),
        };
        auto r = ragger::assemble_payload(pieces, /*budget=*/10, cpt);
        assert(!r.fit);                          // keep piece alone exceeds budget
        assert(r.shed_count == 1);               // the droppable system piece shed
        // The keep turn survives despite not fitting.
        bool has_user = false;
        for (auto& m : r.messages) if (m.role == "user") has_user = true;
        assert(has_user);
        std::println("  keep-pieces-never-shed OK");
    }

    std::println("test_payload_assembly: all passed");
    return 0;
}
