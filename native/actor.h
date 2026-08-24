#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "mcts.h"
#include "thread_pool.h"
#include "policy.h"
#include "position.h"

namespace chess_rl_native {

// One training example captured at a single ply of a self-play game.
struct Example {
    std::vector<float> state;  // 104 * 64 encoded planes (history_steps = 8)
    std::vector<float> pi;     // 4672-length AlphaZero policy vector
    std::string side;          // side to move at this ply ("w" / "b")
    float z = 0.0f;            // outcome from `side`'s perspective, filled at
                               // game end: 1.0 win, -1.0 loss, 0.0 draw
};

// A completed self-play game: ordered examples plus the teacher handle that
// produced it.
struct Game {
    std::vector<Example> examples;
    int teacher_generation = -1;
    int teacher_weight_version = -1;
    std::string result_termination;
    int game_index = -1;  // deterministic ordering key for parallel finalise

    // ---- resignation calibration ----
    // `resigned` / `adjudicated_draw`: the game was cut short.
    // `playout`: this game had early termination DISABLED so its true result
    //   could be observed (the calibration sample).
    // `would_have_resigned` / `would_have_drawn`: a playout game that hit the
    //   condition but kept going.
    // `false_resignation` / `false_draw`: the observed result contradicts what
    //   the cut short would have claimed.  Only ever set on playout games —
    //   a resigned game has no ground truth to check against, which is the
    //   whole reason a fraction of games must be played out.
    bool resigned = false;
    bool adjudicated_draw = false;
    bool playout = false;
    bool would_have_resigned = false;
    bool would_have_drawn = false;
    bool false_resignation = false;
    bool false_draw = false;
    int plies = 0;
};

// Native multi-game self-play actor (AlphaZero style, mirroring selfplay.py):
//
//  - owns `games` concurrent games, each with its OWN MCTS and its OWN
//    Position (standard start position, empty history),
//  - gather_leaves merges every in-play game's pending MCTS leaves into ONE
//    batch; apply_evaluations routes each leaf's network output back to the
//    owning game's MCTS,
//  - advance() converts completed searches into moves (temperature-sampled
//    with a per-game RNG), records (state, pi, side) examples, and finalises
//    finished games with z-values computed exactly like selfplay.py
//    (white_result 1.0 / -1.0 / 0.0; z = white_result for White-to-move,
//    -white_result for Black-to-move).
class Actor {
  public:
    Actor(int games, double c_puct, double virtual_loss, int num_simulations,
          double temperature, int temperature_threshold, int max_game_length,
          std::uint64_t seed, int num_threads = 0,
          // Resignation: when the root value stays below `resign_threshold`
          // for `resign_consecutive` consecutive completed searches, the side
          // to move resigns.  `resign_playout_fraction` of games have this
          // disabled and are played to a real finish so the false-positive
          // rate is observable; without that sample the threshold is a guess
          // that silently poisons the value target.  Negative threshold
          // disables resignation entirely.
          double resign_threshold = -1.0,
          int resign_consecutive = 2,
          double resign_playout_fraction = 0.10,
          // Draw adjudication: |root value| below `draw_threshold` for
          // `draw_consecutive` searches, after at least `draw_min_ply` plies.
          // Negative threshold disables it.
          double draw_threshold = -1.0,
          int draw_consecutive = 8,
          int draw_min_ply = 60);

    // Records the immutable (weight_version, generation) teacher handle that
    // produced every completed game. No weights are loaded here; the network
    // is injected externally via gather_leaves / apply_evaluations.
    void set_teacher(int weight_version, int generation);

    struct GatherResult {
        // Merged pending leaves across all in-play games, in game order.
        std::vector<MCTS::PendingLeaf> leaves;
        // CSR over leaves: legal_offsets[i]..legal_offsets[i+1] indexes
        // legal_indices for leaf i; legal_offsets has leaves.size() + 1
        // entries and always starts with 0 (even when leaves is empty).
        std::vector<std::int32_t> legal_offsets;
        std::vector<std::int32_t> legal_indices;
    };

    // Gathers one merged leaf batch across every in-play game.
    //
    // `max_batch` is the TOTAL merged budget.  `leaves_per_game` is the
    // per-game target: when > 0 each in-play game contributes up to that many
    // leaves, so ADDING GAMES GROWS THE BATCH.  When 0 the legacy behaviour
    // applies — the budget is split equally, max_batch / in_play, so adding
    // games thinned each game's slice instead of enlarging the batch.  Either
    // way the merged total never exceeds max_batch: the per-game target is
    // clamped by the equal share.
    //
    // tokens = [0, 1, 2, ...] flat indices into the merged pending list;
    // per-leaf game ownership is tracked internally for apply_evaluations
    // routing.  Returns empty once every in-play game's search is complete.
    [[nodiscard]] GatherResult gather_leaves(int max_batch,
                                             int leaves_per_game = 0);

    // Routes each token's row of legal_logits/values back to the owning
    // game's MCTS.apply_evaluations. Array sizes are validated first;
    // anything inconsistent throws std::invalid_argument before any game's
    // search state changes.
    void apply_evaluations(const std::vector<int>& tokens,
                           const std::vector<std::int32_t>& legal_offsets,
                           const std::vector<float>& legal_logits,
                           const std::vector<float>& values);

    // For every game whose MCTS search is complete: sample a move
    // (temperature if ply < temperature_threshold, else 0.0, via the game's
    // own std::mt19937), record a training example, push the move, and either
    // finalise the game (terminal position or ply >= max_game_length) or
    // start a fresh search on the new position.
    void advance();

    // Pops and returns all games finished since the last call.
    [[nodiscard]] std::vector<Game> finished_games();

    [[nodiscard]] bool is_done() const;
    [[nodiscard]] int games_remaining() const;

    // Size of the persistent worker pool actually in use.  Clamped to
    // std::thread::hardware_concurrency() and to the game count, so a caller
    // asking for 20 threads for 20 games on a 16-thread CPU gets 16.
    [[nodiscard]] int num_threads() const noexcept { return num_threads_; }

  private:
    struct GameState {
        std::optional<Position> pos;  // parked at the current root
        MCTS mcts;
        std::vector<Example> examples;
        bool finished = false;
        std::mt19937 rng;  // move sampling, seeded deterministically

        // ---- early-termination bookkeeping ----
        // Resignation streaks are PER SIDE.  Completed searches alternate
        // sides, so a single shared counter resets every other ply and a
        // "N consecutive losing evaluations" rule could never fire unless
        // both players were simultaneously lost.
        int resign_streak_w = 0;
        int resign_streak_b = 0;
        int draw_streak = 0;
        bool playout = false;             // early termination disabled here
        bool would_have_resigned = false;
        bool would_have_drawn = false;
        std::string would_resign_side;    // "w"/"b" of the side that would have
        bool resigned = false;
        bool adjudicated_draw = false;

        GameState(double c_puct, double virtual_loss, int num_simulations,
                  double dirichlet_alpha, double dirichlet_epsilon,
                  std::uint64_t seed);
    };

    static std::string sample_move(
        const std::vector<std::pair<std::string, double>>& policy,
        double temperature, std::mt19937& rng);

    // Finalises `game`: fills z-values from `white_result`, stamps the
    // teacher handle, and moves it to the finished queue.
    void finalise(int game_index, GameState& game, float white_result,
                  std::string termination);

    // Runs fn(g) for g in [0, games_) across the PERSISTENT worker pool.
    // Each game index is visited by exactly one thread; the games'
    // MCTS/board/RNG state is fully independent so no per-game state is
    // shared between threads (only `finished_` needs a lock, taken in
    // finalise()).  Workers are created once in the constructor rather than
    // spawned and joined per call; ordering of the merged output is still
    // established serially by the caller, so results stay deterministic.
    void parallel_for(const std::function<void(int)>& fn);

    int num_threads_ = 1;
    std::unique_ptr<ThreadPool> pool_;
    std::mutex finished_mutex_;

    double temperature_;
    int temperature_threshold_;
    int max_game_length_;
    double resign_threshold_;
    int resign_consecutive_;
    double resign_playout_fraction_;
    double draw_threshold_;
    int draw_consecutive_;
    int draw_min_ply_;
    int teacher_generation_ = -1;
    int teacher_weight_version_ = -1;

    std::vector<GameState> games_;
    std::vector<Game> finished_;

    // Ownership map of the last merged gather (consumed by the next
    // apply_evaluations): leaf_game_[i] = game owning merged leaf i, and
    // game_base_[g]..game_base_[g+1] is game g's contiguous leaf block.
    std::vector<int> leaf_game_;
    std::vector<int> game_base_;
};

}  // namespace chess_rl_native
