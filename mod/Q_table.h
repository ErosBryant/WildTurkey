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

namespace adgMod {

    struct QTableEntry {
        std::unordered_map<double, double> q_values;                // 动作到 Q 值的映射
        std::unordered_map<double, double> min_load_model_cost;     // 每个 action 的最小 load 成本
        std::unordered_map<double, double> max_load_model_cost;     // 每个 action 的最大 load 成本
        std::unordered_map<double, int> min_layer_cost;             // 每个 action 的最小层数成本
        std::unordered_map<double, int> max_layer_cost;             // 每个 action 的最大层数成本
        std::unordered_map<double, double> min_error_bound;         // 最小误差界限
        std::unordered_map<double, double> max_error_bound;         // 最大误差界限
        std::unordered_map<double, int> visit_counts;               // 访问次数
        uint64_t max_sst_size = 0;                                  // 最大 SSTable 大小
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
            const size_t max_replay_size = 100; // 经验回放窗口 100
            const size_t batch_size = 32;

            // ID 量化边界
            double ID_min = 0.0;
            double ID_1   = 0.0;
            double ID_2   = 0.0;
            double ID_3   = 0.0;
            double ID_max = 0.0;

            // 滑动历史用于防止早期 ID 异常
            static constexpr size_t HISTORY_SIZE = 100;
            std::deque<double> id_history;           // 记录最近的 ID 历史
            std::multiset<double> id_set;            // 用于快速获取 min/max

            // QTableManager 初始化
            QTableManager();

            double get_max_future_q(int next_state, const std::vector<double>& actions) const;
            double get_max_future_q(int next_state) const;

            // 计算 reward
            double compute_reward(
                int state,                // 当前档位
                double action,            // error bound
                double new_load_model_cost,
                int layer_count,
                uint64_t sstable_size,
                int level
            );

            double getLearningRate(int state, double action);

            // 计算 Q_value，包含最优未来价值估计
            double compute_q_value(int state, double action, double reward, int next_state);
            double compute_q_value(int state, double action, double reward, int next_state, double next_action);

            void updateQValue(int state, double action, double Q_value);

            // 获取 error_bound
            double getErrorBound(int state) const;

            // 初始化 Q-table
            void initQTable();

            // 获取 next_state
            int getNextState(const std::vector<std::string>& string_keys) const;

            // 添加经验到回放缓冲区
            void addExperience(int state, double action, double reward, int next_state);

            // 从回放缓冲区学习
            void learnFromReplay();

            // 处理新的 SSTable inverse density 并更新状态边界
            void onNewSSTableID(double inverse_density) {
                if (id_history.size() == HISTORY_SIZE) {
                    double old = id_history.front();
                    id_history.pop_front();
                    auto it = id_set.find(old);
                    if (it != id_set.end()) id_set.erase(it);
                }
                id_history.push_back(inverse_density);
                id_set.insert(inverse_density);

                ID_min = *id_set.begin();
                ID_max = *id_set.rbegin();

                double span = (ID_max - ID_min) / 4.0;
                ID_1 = ID_min + span;
                ID_2 = ID_min + 2 * span;
                ID_3 = ID_min + 3 * span;
            }
        };


    // 单例获取 QTableManager 实例
    QTableManager& getQTableManagerInstance();

}

#endif // LEVELDB_Q_TABLE_H
