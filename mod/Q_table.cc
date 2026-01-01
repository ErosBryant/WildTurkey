#include "Q_table.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

template <typename T>
inline const T& clamp(const T& val, const T& low, const T& high)
{
    if (val < low) {
        return low;
    }
    if (val > high) {
        return high;
    }
    return val;
}

namespace adgMod {

    QTableManager::QTableManager() {
    }

    double QTableManager::getLearningRate(int state, double action)
    {
        int    n         = Q_table[state].visit_counts[action];
        double alpha0    = 0.3;        // 初始步长
        double alpha_min = 0.02;       // 下限

        double alpha = alpha0 / std::sqrt(n + 1.0);

        return std::max(alpha, alpha_min);
    }

    double QTableManager::compute_reward(
        int      state,
        double   action,
        double   load_cost,
        int      layer_count,
        uint64_t sstable_size,
        int      level
    )
    {
        constexpr double w1  = 0.30;          // weight: load_cost
        constexpr double w2  = 0.50;          // weight: layer_count
        constexpr double w3  = 0.2;           // weight: error_bound
        constexpr double wp  = 0.65;          // weight: size penalty
        constexpr double wl  = 0.25;          // weight: level penalty
        constexpr int    MAX_LSM_LEVEL = 3;   // deepest level that matters (L0~L3)

        auto &entry       = Q_table[state];
        uint64_t &max_sst = entry.max_sst_size;
        max_sst = std::max(max_sst, sstable_size);

        double &min_load = entry.min_load_model_cost[action];
        double &max_load = entry.max_load_model_cost[action];
        int    &min_layer= entry.min_layer_cost[action];
        int    &max_layer= entry.max_layer_cost[action];
        double &min_err  = entry.min_error_bound[action];
        double &max_err  = entry.max_error_bound[action];

        if (min_load  == std::numeric_limits<double>::max()) min_load  = max_load  = load_cost;
        if (min_layer == std::numeric_limits<int>::max())    min_layer = max_layer = layer_count;
        if (min_err   == std::numeric_limits<double>::max()) min_err   = max_err   = action;

        min_load  = std::min(min_load,  load_cost);
        max_load  = std::max(max_load,  load_cost);
        min_layer = std::min(min_layer, layer_count);
        max_layer = std::max(max_layer, layer_count);
        min_err   = std::min(min_err,   action);
        max_err   = std::max(max_err,   action);

        double load_range  = std::max(1e-12, max_load  - min_load);
        double layer_range = std::max(1e-12, double(max_layer - min_layer));
        // Normalize error bound with a fixed global span to keep it influential
        // even when a new action is sampled.
        const double err_min = 4.0;
        const double err_max = 48.0;
        const double err_range = err_max - err_min;

        double norm_load   = (load_cost   - min_load)   / load_range;
        double norm_layer  = (layer_count - min_layer)  / layer_range;
        double norm_error  = (action      - err_min)    / err_range;
        norm_error = clamp(norm_error, 0.0, 1.0);

        double cost_sum = w1 * norm_load
                        + w2 * norm_layer
                        + w3 * norm_error;
        double u_base   = 1.0 - cost_sum;

        double inv_size_pen = std::log(double(max_sst) / double(sstable_size))
                            / std::log(32.0);
        inv_size_pen = std::pow(inv_size_pen, 1.5);
        inv_size_pen = clamp(inv_size_pen, 0.0, 1.0);

        int    lvl_clamped   = std::min(level, MAX_LSM_LEVEL);
        double norm_level_pen= 1.0 - (double)lvl_clamped / MAX_LSM_LEVEL;
        double level_pen     = wl * norm_level_pen;

        double r = u_base
                - wp * inv_size_pen
                - level_pen;
        if (level >= 3) r += 0.08;

        r = clamp(r, -(wp + wl), 1.0);
        return r;
    }

    double QTableManager::get_max_future_q(int next_state) const {
        double max_q = -std::numeric_limits<double>::infinity();
        for (const auto& pair : Q_table[next_state].q_values) {
            if (pair.second > max_q) {
                max_q = pair.second;
            }
        }
        return (max_q == -std::numeric_limits<double>::infinity()) ? 0.0 : max_q;
    }

    double QTableManager::get_max_future_q(int next_state, const std::vector<double>& actions) const {
        double max_q = -std::numeric_limits<double>::infinity();
        for (const auto& action : actions) {
            auto it = Q_table[next_state].q_values.find(action);
            if (it != Q_table[next_state].q_values.end()) {
                if (it->second > max_q) {
                    max_q = it->second;
                }
            }
        }
        return (max_q == -std::numeric_limits<double>::infinity()) ? 0.0 : max_q;
    }

    double QTableManager::compute_q_value(int state,
                                      double action_raw,
                                      double reward,
                                      int    next_state)
    {
        const double action = std::round(action_raw * 1e4) / 1e4;

        const double alpha  = getLearningRate(state, action);
        const double gamma  = 0.9;

        double& prev_q = Q_table[state].q_values[action];

        auto& next_map = Q_table[next_state].q_values;
        double max_future_q = 0.0;
        if (next_map.empty()) {
            max_future_q = 1.0;
        } else {
            for (const auto& kv : next_map)
                if (kv.second > max_future_q) max_future_q = kv.second;
        }

        const double td_target = reward + gamma * max_future_q;
        double td_error        = td_target - prev_q;

        const double delta_clip = 2.0;
        if (td_error >  delta_clip) td_error =  delta_clip;
        if (td_error < -delta_clip) td_error = -delta_clip;

        prev_q += alpha * td_error;
        return prev_q;
    }

    double QTableManager::compute_q_value(int state, double action, double reward, int next_state, double next_action) {
        double alpha = getLearningRate(state, action);
        double gamma = 0.8;

        double& prev_q = Q_table[state].q_values[action];
        double next_q = Q_table[next_state].q_values[next_action];
        double Q_value = (1 - alpha) * prev_q + alpha * (reward + gamma * next_q);
        prev_q = Q_value;
        return Q_value;
    }

    void QTableManager::updateQValue(int state, double action, double Q_value) {
        Q_table[state].q_values[action] = Q_value;
        Q_table[state].visit_counts[action] += 1;
    }

    double QTableManager::getErrorBound(int state) const {
        return 8;
    }

    void QTableManager::addExperience(int state, double action, double reward, int next_state) {
        Experience exp = {state, action, reward, next_state};
        replay_buffer.push_back(exp);
        if (replay_buffer.size() > max_replay_size) {
            replay_buffer.erase(replay_buffer.begin());
        }
    }

    void QTableManager::learnFromReplay() {
        if (replay_buffer.size() < batch_size) return;

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, replay_buffer.size() - 1);

        for (size_t i = 0; i < batch_size; ++i) {
            int idx = dis(gen);
            Experience& exp = replay_buffer[idx];
            double Q_value;
            Q_value = compute_q_value(exp.state, exp.action, exp.reward, exp.next_state);

            updateQValue(exp.state, exp.action, Q_value);
        }
    }

    void QTableManager::initQTable() {
        Q_table.resize(8);
        std::vector<int> actions = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48};
        for (auto& entry : Q_table) {
            for (double action : actions) {
                entry.q_values[action] = 0.0;
                entry.min_load_model_cost[action] = std::numeric_limits<double>::max();
                entry.max_load_model_cost[action] = std::numeric_limits<double>::lowest();
                entry.min_layer_cost[action] = std::numeric_limits<double>::max();
                entry.max_layer_cost[action] = std::numeric_limits<double>::lowest();
                entry.min_error_bound[action] = std::numeric_limits<double>::max();
                entry.max_error_bound[action] = std::numeric_limits<double>::lowest();
                entry.visit_counts[action] = 0;
                entry.last_action = 16.0;
            }
        }
    }

    QTableManager& getQTableManagerInstance() {
        static QTableManager instance;
        return instance;
    }

}
