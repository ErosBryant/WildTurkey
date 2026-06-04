#ifndef LEVELDB_Q_TABLE_H
#define LEVELDB_Q_TABLE_H

#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <unordered_map>
#include <deque>
#include <set>
#include <string>
#include <array>
#include <cstdint>
#include <limits>

namespace adgMod {

    constexpr int kErrorIdBuckets = 4;
    constexpr int kErrorLatencyBuckets = 3;
    constexpr int kErrorQStates = kErrorIdBuckets * kErrorLatencyBuckets;
    constexpr int kErrorWarmupSSTables = 100;

    const std::vector<double>& ErrorBoundActions();
    std::vector<double> ErrorBoundActionsForBlockSize(uint64_t block_size);

    struct QTableEntry {
        std::unordered_map<double, double> q_values;                // 动作到 Q 值的映射
        std::unordered_map<double, int> visit_counts;               // 访问次数
        double last_action = 16.0;                                  // 上一次动作
    };

    struct Qtable_sar {
        double prev_action = 16;
        uint64_t prev_state = 7;
        double prev_reward = 1;
    };

    struct Experience {
        int state;
        double action;
        double reward;
        int next_state;
    };

    class QTableManager {
        public:
            std::vector<QTableEntry> Q_table;
            Qtable_sar Q_table_sar;
            std::vector<Experience> replay_buffer;
            std::vector<double> error_bound_actions;
            const size_t max_replay_size = 100; // 经验回放窗口 100
            const size_t batch_size = 32;

            // ID bucket boundaries, fixed after the warm-up window.
            double ID_min = 0.0;
            double ID_1   = 0.0;
            double ID_2   = 0.0;
            double ID_3   = 0.0;
            double ID_max = 0.0;
            bool id_bounds_fixed = false;
            size_t id_observations = 0;

            static constexpr size_t HISTORY_SIZE = kErrorWarmupSSTables;
            std::deque<double> id_history;           // 记录最近的 ID 历史
            std::multiset<double> id_set;            // 用于快速获取 min/max
            std::array<double, kErrorIdBuckets> latency_ema;
            std::array<double, kErrorIdBuckets> latency_min;
            std::array<double, kErrorIdBuckets> latency_max;
            std::array<bool, kErrorIdBuckets> latency_seen;

            struct CostBounds {
                double min_load = std::numeric_limits<double>::max();
                double max_load = std::numeric_limits<double>::lowest();
                double min_pred = std::numeric_limits<double>::max();
                double max_pred = std::numeric_limits<double>::lowest();
                double min_corr = std::numeric_limits<double>::max();
                double max_corr = std::numeric_limits<double>::lowest();
                uint64_t max_keys = 0;
            };
            CostBounds cost_bounds;

            // QTableManager 初始化
            QTableManager();

            double get_max_future_q(int next_state, const std::vector<double>& actions) const;
            double get_max_future_q(int next_state) const;

            // 计算 reward
            double compute_reward(
                double action,            // error bound
                double load_cost,
                double pred_cost,
                double corr_cost,
                uint64_t key_count
            );

            double getLearningRate(int state, double action);

            // 计算 Q_value，包含最优未来价值估计
            double compute_q_value(int state, double action, double reward, int next_state);
            double compute_q_value(int state, double action, double reward, int next_state, double next_action);

            void updateQValue(int state, double action, double Q_value);

            // 获取 error_bound
            double getErrorBound(int state) const;

            // 初始化 Q-table
            void initQTable(uint64_t block_size = 0);

            // 获取 next_state
            int getNextState(const std::vector<std::string>& string_keys) const;

            // 添加经验到回放缓冲区
            void addExperience(int state, double action, double reward, int next_state);

            // 从回放缓冲区学习
            void learnFromReplay();

            int ObserveID(double inverse_density);
            int ComposeState(int id_bucket, int latency_bucket) const;
            int LatencyBucket(int id_bucket) const;
            int ObserveLatencyAndGetState(int id_bucket, double lookup_cost);

            // 处理新的 SSTable inverse density 并更新状态边界
            void onNewSSTableID(double inverse_density) {
                if (id_bounds_fixed) return;
                id_history.push_back(inverse_density);
                id_set.insert(inverse_density);
                id_observations++;

                ID_min = *id_set.begin();
                ID_max = *id_set.rbegin();

                double span = (ID_max - ID_min) / 4.0;
                ID_1 = ID_min + span;
                ID_2 = ID_min + 2 * span;
                ID_3 = ID_min + 3 * span;

                if (id_observations >= HISTORY_SIZE) {
                    id_bounds_fixed = true;
                }
            }
        };


    // 单例获取 QTableManager 实例
    QTableManager& getQTableManagerInstance();

}

#endif // LEVELDB_Q_TABLE_H
