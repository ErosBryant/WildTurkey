#ifndef LEVELDB_ADAPTIVE_LAC_H
#define LEVELDB_ADAPTIVE_LAC_H

#include <cstdint>
#include <array>
#include <deque>
#include <limits>
#include <mutex>

#include "db/dbformat.h"
#include "mod/util.h"

namespace adgMod {

constexpr int kMinLacDegree = 1;
constexpr int kMaxLacDegree = 10;
constexpr int kDefaultHistoryWindow = 3;
constexpr int kNumIopsStates = 3;
constexpr std::array<int, leveldb::config::kNumLevels> kMinDegreePerLevel = {
    1, 2, 2, 3, 5, 6, 7};

enum class IopsState : uint8_t {
  kHigh = 0,
  kMedium = 1,
  kLow = 2,
};

int GetAdaptiveLacWindowSize();
void SetAdaptiveLacWindowSize(int window);
// Lightweight manager that adapts the per-level SSTable target size based on
// observed compaction results and learned-index rewards.
class AdaptiveLAC {
 public:
  AdaptiveLAC();

  // Returns the current target bytes for the given level. Falls back to the
  // default LAC sizing if we have not seen enough feedback yet.
  uint64_t GetTargetBytes(int level, uint64_t fallback_bytes = 0) const;

  // Record the outcome of a compaction so the next compaction can adjust its
  // target size and discrete LAC degree using WA and write throughput (ops).
  void Observe(int level, uint64_t file_size_bytes, double wa_score,
               uint64_t compaction_micros, double ops);

  // Expose the deterministic default sizing used as a baseline.
  uint64_t DefaultSizeForLevel(int level) const;

 private:
  uint64_t BaseSizeForLevel(int level, int lac_degree) const;
  int InitialLacDegree(int level) const;
  static int MinDegreeForLevel(int level);
  static int MaxDegreeForLevel(int level);
  static int ClampDegreeForLevel(int level, int degree);
  static uint64_t Clamp(uint64_t value, uint64_t base);

  struct DegreeStats {
    double last_ops_measure = 0.0;
    double last_wa = 0.0;
    double sum_ops = 0.0;
    double sum_wa = 0.0;
    double min_ops_change = std::numeric_limits<double>::infinity();
    double max_ops_change = -std::numeric_limits<double>::infinity();
    double min_wa_change = std::numeric_limits<double>::infinity();
    double max_wa_change = -std::numeric_limits<double>::infinity();
    std::deque<double> ops_changes;
    std::deque<double> wa_changes;
  };

  struct State {
    double target_bytes = 0.0;
    uint64_t last_observed_bytes = 0;
    uint64_t last_ops_total = 0;
    int lac_degree = 0;
    bool initialized = false;
    double ops_min = std::numeric_limits<double>::infinity();
    double ops_max = -std::numeric_limits<double>::infinity();
    IopsState iops_state = IopsState::kMedium;
    double epsilon = 0.0;
    std::array<std::array<double, kMaxLacDegree + 1>, kNumIopsStates> q_values = {};
    std::array<DegreeStats, kMaxLacDegree + 1> degree_stats = {};
  };

  mutable std::mutex mu_;
  std::array<State, leveldb::config::kNumLevels> states_;
};

AdaptiveLAC& GetAdaptiveLAC();

}  // namespace adgMod

#endif  // LEVELDB_ADAPTIVE_LAC_H
