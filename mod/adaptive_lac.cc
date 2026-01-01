#include "mod/adaptive_lac.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

namespace {
using adgMod::kMaxLacDegree;
using adgMod::kMinLacDegree;
using adgMod::IopsState;

std::atomic<int> g_history_window(adgMod::kDefaultHistoryWindow);

constexpr double kBaseUnitBytes = 2097152.0;  // 2MB
constexpr double kOpsEpsilon = 1e-6;
constexpr double kAlpha = 0.1;
constexpr double kGamma = 0.9;
constexpr double kEpsilonStart = 0.9;
constexpr double kEpsilonMin = 0.01;
constexpr double kEpsilonDecay = 0.008;
constexpr const char* kReportPath = "report.csv";
constexpr double kRangeEps = 1e-9;
constexpr double kPreferenceWeight = 0.4;
constexpr double kSmallDegreeBonus = 0.6;

constexpr double kLevelRewardWeights[leveldb::config::kNumLevels] = {
    0.6, 0.8, 0.9, 1.1, 1.3, 1.5, 1.7};

struct PreferredRange {
  int min_deg;
  int max_deg;
};

constexpr std::array<PreferredRange, leveldb::config::kNumLevels>
    kPreferredDegreeRanges = {{
        {1, 10},  // L0: allow full range, bias handled separately
        {1, 10},  // L1
        {1, 10},   // L2
        {1, 10},   // L3
        {1, 10},   // L4
        {1, 10},   // L5
        {1, 10},   // L6
    }};

int StateIndex(adgMod::IopsState state) {
  return static_cast<int>(state);
}

const char* ToString(adgMod::IopsState state) {
  switch (state) {
    case adgMod::IopsState::kHigh:
      return "high";
    case adgMod::IopsState::kMedium:
      return "medium";
    case adgMod::IopsState::kLow:
      return "low";
  }
  return "unknown";
}

void UpdateRange(double value, double& min_val, double& max_val) {
  if (!std::isfinite(min_val) || value < min_val) {
    min_val = value;
  }
  if (!std::isfinite(max_val) || value > max_val) {
    max_val = value;
  }
}

double Normalize(double value, double min_val, double max_val) {
  if (!std::isfinite(min_val) || !std::isfinite(max_val) ||
      (max_val - min_val) < kRangeEps) {
    return 0.5;
  }
  const double scaled = (value - min_val) / (max_val - min_val);
  return std::max(0.0, std::min(1.0, scaled));
}

adgMod::IopsState ClassifyIops(double normalized_ops) {
  if (normalized_ops >= 0.66) {
    return adgMod::IopsState::kHigh;
  }
  if (normalized_ops >= 0.33) {
    return adgMod::IopsState::kMedium;
  }
  return adgMod::IopsState::kLow;
}

double SmallDegreeBias(int level, int degree) {
  // Bias levels 0/1 toward very small degrees (<=3), but keep full range valid.
  if (level <= 1) {
    if (degree <= 3) return kSmallDegreeBonus;
    const double scale = static_cast<double>(degree - 3) /
                         static_cast<double>(kMaxLacDegree - 3);
    return -kSmallDegreeBonus * std::min(1.0, std::max(0.0, scale));
  }
  return 0.0;
}

std::mt19937& GetRng() {
  static thread_local std::mt19937 rng(std::random_device{}());
  return rng;
}

double NextEpsilonGreedy(double& epsilon) {
  const double current = epsilon;
  epsilon = std::max(kEpsilonMin, epsilon - kEpsilonDecay);
  return current;
}

int RandomDegree(int min_degree, int max_degree) {
  std::uniform_int_distribution<int> dist(min_degree, max_degree);
  return dist(GetRng());
}

int ArgMaxDegree(const std::array<double, kMaxLacDegree + 1>& q_values,
                 int min_degree, int max_degree) {
  double best = q_values[min_degree];
  int best_deg = min_degree;
  for (int d = min_degree + 1; d <= max_degree; ++d) {
    if (q_values[d] > best) {
      best = q_values[d];
      best_deg = d;
    }
  }
  return best_deg;
}

double MaxQValue(const std::array<double, kMaxLacDegree + 1>& q_values,
                 int min_degree, int max_degree) {
  double best = q_values[min_degree];
  for (int d = min_degree + 1; d <= max_degree; ++d) {
    best = std::max(best, q_values[d]);
  }
  return best;
}

int SelectNextDegree(const std::array<double, kMaxLacDegree + 1>& q_values,
                     int min_degree, int max_degree, double& epsilon) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  const double current_eps = NextEpsilonGreedy(epsilon);
  if (dist(GetRng()) < current_eps) {
    return RandomDegree(min_degree, max_degree);
  }
  return ArgMaxDegree(q_values, min_degree, max_degree);
}

void TrimDeque(std::deque<double>& dq, double& sum) {
  const int window = std::max(1, adgMod::GetAdaptiveLacWindowSize());
  while (dq.size() > static_cast<size_t>(window)) {
    sum -= dq.front();
    dq.pop_front();
  }
}

void AppendReportRow(int level, IopsState from_state, IopsState to_state,
                     int old_deg, int new_deg, double wa, double ops,
                     double wa_rate, double ops_rate, double score,
                     double level_weight, double target_bytes) {
  static std::mutex report_mu;
  static bool report_initialized = false;
  std::lock_guard<std::mutex> lg(report_mu);

  std::ios_base::openmode mode = std::ios::app;
  if (!report_initialized) {
    mode = std::ios::out | std::ios::trunc;
  }
  std::ofstream ofs(kReportPath, mode);
  if (!ofs.is_open()) {
    return;
  }
  if (!report_initialized) {
    ofs << "ts_iso,level,old_lac,new_lac,wa,ops,wa_rate,ops_rate,score,"
           "level_weight,target_bytes,iops_from,iops_to\n";
    report_initialized = true;
  }

  const auto now = std::chrono::system_clock::now();
  const std::time_t ts_sec = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf;
#if defined(_WIN32)
  localtime_s(&tm_buf, &ts_sec);
#else
  localtime_r(&ts_sec, &tm_buf);
#endif
  std::ostringstream ts_stream;
  ts_stream << std::put_time(&tm_buf, "%F %T");

  const auto old_flags = ofs.flags();
  const auto old_prec = ofs.precision();
  ofs << std::fixed << std::setprecision(6);

  ofs << ts_stream.str() << ',' << level << ',' << old_deg << ',' << new_deg
      << ',' << wa << ',' << ops << ',' << wa_rate << ',' << ops_rate << ','
      << score << ',' << level_weight << ','
      << static_cast<uint64_t>(target_bytes) << ',' << ToString(from_state)
      << ',' << ToString(to_state) << '\n';

  ofs.flags(old_flags);
  ofs.precision(old_prec);
}
}  // namespace

namespace adgMod {

int GetAdaptiveLacWindowSize() {
  return std::max(1, g_history_window.load(std::memory_order_relaxed));
}

void SetAdaptiveLacWindowSize(int window) {
  const int clamped = std::max(1, window);
  g_history_window.store(clamped, std::memory_order_relaxed);
}

AdaptiveLAC::AdaptiveLAC() {
  for (size_t level = 0; level < states_.size(); ++level) {
    State& st = states_[level];
    st.target_bytes = 0.0;
    st.last_observed_bytes = 0;
    st.last_ops_total = 0;
    st.ops_min = std::numeric_limits<double>::infinity();
    st.ops_max = -std::numeric_limits<double>::infinity();
    st.iops_state = IopsState::kMedium;
    st.epsilon = kEpsilonStart;
    st.lac_degree =
        ClampDegreeForLevel(static_cast<int>(level),
                            InitialLacDegree(static_cast<int>(level)));
    st.initialized = false;
    for (auto& q_row : st.q_values) {
      q_row.fill(0.0);
    }
    for (auto& hist : st.degree_stats) {
      hist = DegreeStats();
    }
  }
}

uint64_t AdaptiveLAC::BaseSizeForLevel(int level, int lac_degree) const {
  const int degree = lac_degree > 0 ? lac_degree : InitialLacDegree(level);
  const int exp = std::max(1, std::min(level + 1, 6));
  const double size = kBaseUnitBytes *
                      std::pow(static_cast<double>(degree),
                               static_cast<double>(exp));
  return static_cast<uint64_t>(size);
}

int AdaptiveLAC::MinDegreeForLevel(int level) {
  const int idx =
      std::max(0, std::min(level, leveldb::config::kNumLevels - 1));
  return std::max(kMinLacDegree, adgMod::kMinDegreePerLevel[idx]);
}

int AdaptiveLAC::MaxDegreeForLevel(int level) {
  return kMaxLacDegree;
}

double RewardWeightForLevel(int level) {
  if (level < 0 || level >= leveldb::config::kNumLevels) return 1.0;
  return kLevelRewardWeights[level];
}

int AdaptiveLAC::ClampDegreeForLevel(int level, int degree) {
  const int min_degree = MinDegreeForLevel(level);
  const int max_degree = MaxDegreeForLevel(level);
  return std::max(min_degree, std::min(max_degree, degree));
}

int AdaptiveLAC::InitialLacDegree(int level) const {
  const int base = sst_size > 0 ? sst_size : 4;
  return ClampDegreeForLevel(level, base);
}

uint64_t AdaptiveLAC::Clamp(uint64_t value, uint64_t base) {
  const double min_bytes = base * 0.5;
  const double max_bytes = base * 4.0;
  return static_cast<uint64_t>(
      std::max(min_bytes, std::min<double>(value, max_bytes)));
}

uint64_t AdaptiveLAC::DefaultSizeForLevel(int level) const {
  return BaseSizeForLevel(level, InitialLacDegree(level));
}

uint64_t AdaptiveLAC::GetTargetBytes(int level, uint64_t fallback_bytes) const {
  std::lock_guard<std::mutex> lg(mu_);
  if (level < 0 || level >= static_cast<int>(states_.size())) {
    return fallback_bytes ? fallback_bytes
                          : BaseSizeForLevel(0, InitialLacDegree(0));
  }

  const State& st = states_[level];
  const int lac_degree =
      st.initialized ? ClampDegreeForLevel(level, st.lac_degree)
                     : InitialLacDegree(level);
  const uint64_t base = fallback_bytes ? fallback_bytes
                                       : BaseSizeForLevel(level, lac_degree);
  const double target = st.initialized ? st.target_bytes
                                       : static_cast<double>(base);
  return Clamp(static_cast<uint64_t>(target), base);
}

void AdaptiveLAC::Observe(int level, uint64_t file_size_bytes, double wa_score, uint64_t compaction_micros, double ops) {
  (void)compaction_micros;
  std::lock_guard<std::mutex> lg(mu_);
  if (level < 0 || level >= static_cast<int>(states_.size())) {
    return;
  }

  if (wa_score <= 0.0) {
    wa_score = 1.0;
  }

  State& st = states_[level];

  if (!st.initialized) {
    st.lac_degree = InitialLacDegree(level);
    const uint64_t base = BaseSizeForLevel(level, st.lac_degree);
    st.target_bytes = file_size_bytes ? static_cast<double>(file_size_bytes)
                                      : static_cast<double>(base);
    st.last_observed_bytes = file_size_bytes;
    st.epsilon = kEpsilonStart;
    st.initialized = true;
  }

  const int min_degree = MinDegreeForLevel(level);
  const int max_degree = MaxDegreeForLevel(level);
  const int current_degree = ClampDegreeForLevel(level, st.lac_degree);
  DegreeStats& stats = st.degree_stats[current_degree];
  const IopsState current_state = st.iops_state;

  const uint64_t ops_total = static_cast<uint64_t>(std::max(0.0, ops));
  double ops_measure = 0.0;
  if (st.last_ops_total > 0 && ops_total >= st.last_ops_total) {
    ops_measure = static_cast<double>(ops_total - st.last_ops_total);
  } else if (st.last_ops_total == 0) {
    ops_measure = static_cast<double>(ops_total);
  } else {
    ops_measure = 0.0;
  }
  st.last_ops_total = ops_total;

  if (stats.last_ops_measure > 0.0) {
    const double rel = (ops_measure - stats.last_ops_measure) /
                       std::max(stats.last_ops_measure, kOpsEpsilon);
    stats.ops_changes.push_back(rel);
    stats.sum_ops += rel;
    TrimDeque(stats.ops_changes, stats.sum_ops);
  }
  stats.last_ops_measure = ops_measure;

  if (stats.last_wa > 0.0) {
    const double rel = (wa_score - stats.last_wa) /
                       std::max(stats.last_wa, kOpsEpsilon);
    stats.wa_changes.push_back(rel);
    stats.sum_wa += rel;
    TrimDeque(stats.wa_changes, stats.sum_wa);
  }
  stats.last_wa = wa_score;

  const double r_ops = stats.ops_changes.empty()
                           ? 0.0
                           : stats.sum_ops / stats.ops_changes.size();
  const double r_wa = stats.wa_changes.empty()
                          ? 0.0
                          : stats.sum_wa / stats.wa_changes.size();
  UpdateRange(r_ops, stats.min_ops_change, stats.max_ops_change);
  UpdateRange(r_wa, stats.min_wa_change, stats.max_wa_change);
  const double norm_ops =
      Normalize(r_ops, stats.min_ops_change, stats.max_ops_change);
  const double norm_wa =
      Normalize(r_wa, stats.min_wa_change, stats.max_wa_change);
  const double level_weight = RewardWeightForLevel(level);
  double reward = (norm_ops - norm_wa) * level_weight;

  // Add a small bias toward preferred degree ranges per level (e.g., 2/3 at
  // lower levels, 3–5 at mid levels).
  const int level_idx =
      std::max(0, std::min(level, leveldb::config::kNumLevels - 1));
  const PreferredRange& pref = kPreferredDegreeRanges[level_idx];
  const bool in_pref =
      current_degree >= pref.min_deg && current_degree <= pref.max_deg;
  const double range_bias = (in_pref ? kPreferenceWeight : -kPreferenceWeight);
  const double small_bias = SmallDegreeBias(level, current_degree);
  reward += (range_bias + small_bias) * level_weight;
  if (level == 2) {
    // Extra soft preference for degree 4 while keeping full 1-10 range valid.
    const double dist = std::abs(current_degree - 4);
    const double l2_bias = std::max(0.0, 1.0 - dist / 6.0);  // 4 gets full bonus, farther ones decay.
    const double l2_weight = 0.5;  // tune bias strength
    reward += l2_weight * l2_bias * level_weight;
  }

  UpdateRange(ops_measure, st.ops_min, st.ops_max);
  const double normalized_iops = Normalize(ops_measure, st.ops_min, st.ops_max);
  const IopsState next_state = ClassifyIops(normalized_iops);

  auto& current_q_row = st.q_values[StateIndex(current_state)];
  auto& next_q_row = st.q_values[StateIndex(next_state)];
  double& q = current_q_row[current_degree];
  const double max_next =
      MaxQValue(next_q_row, min_degree, max_degree);
  q += kAlpha * (reward + kGamma * max_next - q);

  const int next_degree =
      SelectNextDegree(next_q_row, min_degree, max_degree, st.epsilon);
  st.iops_state = next_state;
  st.lac_degree = next_degree;

  const uint64_t base = BaseSizeForLevel(level, st.lac_degree);
  double next_target = static_cast<double>(base);
  if (file_size_bytes > 0) {
    next_target = 0.5 * next_target + 0.5 * static_cast<double>(file_size_bytes);
  }
  st.target_bytes = static_cast<double>(Clamp(
      static_cast<uint64_t>(next_target), base));
  st.last_observed_bytes = file_size_bytes;

  AppendReportRow(level, current_state, next_state, current_degree, next_degree,
                  wa_score, ops_measure, r_wa, r_ops, reward, level_weight,
                  st.target_bytes);
}

AdaptiveLAC& GetAdaptiveLAC() {
  static AdaptiveLAC instance;
  return instance;
}

}  // namespace adgMod
