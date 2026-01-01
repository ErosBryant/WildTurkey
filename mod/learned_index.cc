//
// Created by daiyi on 2020/02/02.
//


#include "learned_index.h"

#include "db/version_set.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>
#include <limits>

#include <utility>

#include "util/mutexlock.h"
#include "adaptive_lac.h"

#include "util.h"
#include "Q_table.h"
#include <cmath> 
#include <chrono>
#include <algorithm>
#include <atomic>
#include <numeric>
template <typename T>
T clamp(T value, T min_value, T max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}


namespace adgMod {

std::pair<uint64_t, uint64_t> LearnedIndexData::GetPosition(
    const Slice& target_x) const {
  assert(string_multi_layer_segments.size() > 1);
  ++served;
  if (adgMod::adeb == 1) {
    // check if the key is within the model bounds
    uint64_t target_int = SliceToInteger(target_x);
    if (target_int > max_key) return std::make_pair(size, size);
    if (target_int < min_key) return std::make_pair(size, size);

    double segment = 0;
    double recursive_error_bound = this->recursive_error_bound;

    // top model prediction
    if (recursive_error_bound > 5) {
      // binary search
      for (int i = index_layer_count; i > 0; --i ){
        segment = target_int * string_multi_layer_segments[i][segment].k + string_multi_layer_segments[i][segment].b;
        uint64_t lower = segment - recursive_error_bound > 0 ? (uint64_t)std::floor(segment - recursive_error_bound) : 0;
        uint64_t upper = (uint64_t)std::ceil(segment + recursive_error_bound) < string_multi_layer_segments[i-1].size() ? (uint64_t)std::ceil(segment + recursive_error_bound) : (string_multi_layer_segments[i-1].size() - 1);
        while (lower != upper - 1) {
          uint64_t mid = (upper + lower) / 2;
          if (target_int < string_multi_layer_segments[i-1][mid].x)
            upper = mid;
          else
            lower = mid;
        }
        segment = std::move(lower);
      }
    }
    else {
    // Linear search
      for (int i = index_layer_count; i > 0; --i) {
          segment = target_int * string_multi_layer_segments[i][segment].k + string_multi_layer_segments[i][segment].b;
          uint64_t lower = segment - recursive_error_bound > 0 ? (uint64_t)std::floor(segment - recursive_error_bound) : 0;
          uint64_t upper = (uint64_t)std::ceil(segment + recursive_error_bound) < string_multi_layer_segments[i - 1].size() 
                          ? (uint64_t)std::ceil(segment + recursive_error_bound) 
                          : (string_multi_layer_segments[i - 1].size() - 1);

          // Linear search within the bounds
          for (uint64_t j = lower; j <= upper; ++j) {
              if (target_int < string_multi_layer_segments[i - 1][j].x) {
                  // 检查是否确实在当前 segment 范围内
                  if (j > 0 && target_int >= string_multi_layer_segments[i - 1][j - 1].x) {
                      segment = j - 1;
                  } else {
                      segment = j;
                  }
                  break;
              }
              if (j == upper) {
                  segment = upper;
              }
          }
      }
    }

    // calculate the interval according to the selected segment
    // predicted position
    double result =
        target_int * string_multi_layer_segments[0][segment].k + string_multi_layer_segments[0][segment].b;
    result = is_level ? result / 2 : result;
    //double error_bound = this->meta->error;
    double error_bound = this->error_bound;  // 使用模型记录的 error bound

    uint64_t lower =
        result - error_bound > 0 ? (uint64_t)std::floor(result - error_bound) : 0;
        
    // uint64_t result_position = static_cast<uint64_t>(std::round(result));
    // // 检查 result 是否精确匹配 target_int
    // if (result_position < string_keys.size() && SliceToInteger(string_keys[result_position]) == target_int) {
    //   return std::make_pair(result_position, result_position);
    // }

    uint64_t upper = (uint64_t)std::ceil(result + error_bound);
    if (lower >= size) return std::make_pair(size, size);
    upper = upper < size ? upper : size - 1;
    return std::make_pair(lower, upper);
  }

  else {// check if the key is within the model bounds
    uint64_t target_int = SliceToInteger(target_x);
    if (target_int > max_key) return std::make_pair(size, size);
    if (target_int < min_key) return std::make_pair(size, size);

    // binary search between segments
    uint32_t left = 0, right = (uint32_t)string_segments.size() - 1;
    while (left != right - 1) {
      uint32_t mid = (right + left) / 2;
      if (target_int < string_segments[mid].x)
        right = mid;
      else
        left = mid;
    }

    // calculate the interval according to the selected segment
    // predicted position
    double result =
        target_int * string_segments[left].k + string_segments[left].b;
    result = is_level ? result / 2 : result;
    uint64_t lower =
        result - error > 0 ? (uint64_t)std::floor(result - error) : 0;
    uint64_t upper = (uint64_t)std::ceil(result + error);
    if (lower >= size) return std::make_pair(size, size);
    upper = upper < size ? upper : size - 1;
    //                printf("%s %s %s\n", string_keys[lower].c_str(),
    //                string(target_x.data(), target_x.size()).c_str(),
    //                string_keys[upper].c_str()); assert(target_x >=
    //                string_keys[lower] && target_x <= string_keys[upper]);
    //std::cout << error << std::endl;

    return std::make_pair(lower, upper);
  }
}

uint64_t LearnedIndexData::MaxPosition() const { return size - 1; }

double LearnedIndexData::GetError() const { return error; }

double LearnedIndexData::getRandomAction(double error_) const {
    double epsilon = 0.5; // 探索率 30%

    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::uniform_real_distribution<> dis(0.0, 1.0);

    double random_value = dis(gen);

    bool is_exploration = random_value < epsilon;
    //std::cout << "Random Value: " << random_value << ", Epsilon: " << epsilon
              //<< ", Exploration: " << (is_exploration ? "Yes" : "No") << std::endl;

    if (is_exploration) {
        // 探索：选择与当前 error_ 相邻的值
        std::vector<double> possible_errors;

        // 定义最小和最大允许的 error_bound
        double min_error = 4;
        double max_error = 100;

        // 添加比当前 error_ 小的值
        if (error_ - 4 >= min_error) {
            possible_errors.push_back(error_ - 4);
        }
        // 添加比当前 error_ 大的值
        if (error_ + 4 <= max_error) {
            possible_errors.push_back(error_ + 4);
        }

        // 确保有候选值
        if (!possible_errors.empty()) {
            std::uniform_int_distribution<> error_dis(0, possible_errors.size() - 1);
            double new_error = possible_errors[error_dis(gen)];
            return new_error;
        } else {
            // 如果没有相邻的候选值，保持不变
            return error_;
        }
    } else {
        // 利用：选择当前最优动作（即当前的 error_）
        return error_;
    }
}

double LearnedIndexData::getEpsilon(int state, double action) const {
    (void)action;
    const auto& visits =
        adgMod::getQTableManagerInstance().Q_table[state].visit_counts;

    int total_visits = 0;
    for (const auto& kv : visits) {
        total_visits += kv.second;
    }

    const double epsilon_min   = 0.02;
    const double epsilon_max   = 0.6;
    const double decay_factor  = 0.05;  // faster decay so exploration tapers
    const double epsilon = epsilon_min +
        (epsilon_max - epsilon_min) * std::exp(-decay_factor * total_visits);
    return epsilon;
}



double LearnedIndexData::getAction(int state) const
{
    auto &q_mgr   = adgMod::getQTableManagerInstance();
    auto &entry   = q_mgr.Q_table[state];

    static const std::vector<double> actions =
        {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48};

    const double epsilon = getEpsilon(state, entry.last_action);

    thread_local std::mt19937              gen{std::random_device{}()};
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

  bool explore = uni01(gen) < epsilon;

  if (explore)
  {
      std::uniform_int_distribution<size_t> pick(0, actions.size() - 1);
      double candidate = actions[pick(gen)];
      entry.last_action = candidate;
      entry.visit_counts[candidate] += 1;
      return candidate;
  }

    double best_q        = -std::numeric_limits<double>::infinity();
    double best_action   = actions.front();
    std::uniform_int_distribution<int> coin(0,1);

    for (double a : actions) {
        double q = entry.q_values.count(a) ? entry.q_values[a] : 0.0;
        if (q > best_q || (q == best_q && coin(gen)))
        {
            best_q      = q;
            best_action = a;
        }
    }
    entry.last_action = best_action;
    entry.visit_counts[best_action] += 1;
    return best_action;
}

// double LearnedIndexData::getAction(int state) const { 
//     double epsilon = getEpsilon(state);
//     //std::cout << "Epsilon: " << epsilon << std::endl;

//     // 可以在此处根据学习进度动态调整 epsilon

//     thread_local std::random_device rd;
//     thread_local std::mt19937 gen(rd());
//     thread_local std::uniform_real_distribution<> dis(0.0, 1.0);

//     double random_value = dis(gen);
//     bool is_exploration = random_value < epsilon;

//     if (is_exploration) {
//         // 探索：从动作空间中随机选择一个动作
//         std::vector<double> actions = {8,12,16,20,24,32}; // 可选的 error_bound 值
//         std::uniform_int_distribution<> action_dis(0, actions.size() - 1);
//         double random_action = actions[action_dis(gen)];
//         return random_action;
//     } else {
//         // 利用：从 Q 表中选择当前状态下 Q 值最高的动作
//         const auto& q_values = adgMod::getQTableManagerInstance().Q_table[state].q_values;
//         double best_action = 0;
//         double max_q_value = -std::numeric_limits<double>::infinity();
//         for (const auto& pair : q_values) {
//             if (pair.second > max_q_value) {
//                 max_q_value = pair.second;
//                 best_action = pair.first;
//             }
//         }
//         return best_action;
//       }
// }



// Actual function doing learning
bool LearnedIndexData::Learn() {

  if (adgMod::adeb == 1) {
    srand(static_cast<unsigned>(time(0)));  // 设置随机数种子
    if (string_keys.empty()) assert(false);

    uint64_t temp = atoll(string_keys.back().c_str());
    min_key = atoll(string_keys.front().c_str());
    max_key = atoll(string_keys.back().c_str());
    size = string_keys.size();

    inverse_density = static_cast<uint64_t>((max_key - min_key) / size);

    adgMod::getQTableManagerInstance().onNewSSTableID(inverse_density);

    if (inverse_density < adgMod::getQTableManagerInstance().ID_1) inverse_density = 0;
    else if (inverse_density < adgMod::getQTableManagerInstance().ID_2) inverse_density = 1;
    else if (inverse_density < adgMod::getQTableManagerInstance().ID_3) inverse_density = 2;
    else inverse_density = 3;

    int next_state = inverse_density;

    double temp_error = getAction(next_state);
    PLR plr = PLR(temp_error);

    std::vector<std::vector<Segment>> multi_layer_models;
    std::vector<std::string> seg_last;
    size_t prev_model_size = 0;

    std::vector<Segment> segs = plr.train(string_keys, !is_level, seg_last);
    int segment_count = segs.size();

    if (segs.empty()) return false;
    prev_model_size = segment_count;
    segs.push_back((Segment){temp, 0, 0});
    multi_layer_models.push_back(std::move(segs));
    std::vector<std::string> seg_last_buffer;
    uint8_t layer_count = 0;
    this->error_bound = temp_error;
    uint64_t total_segment_count = segment_count;

    while (prev_model_size > 1) {
      upper_PLR upper_PLR(3);
      if (!seg_last_buffer.empty()) {
        seg_last.clear();
        seg_last = std::move(seg_last_buffer);
      }
      segs = upper_PLR.train(seg_last, !is_level, seg_last_buffer);
      prev_model_size = segs.size();

      segs.push_back((Segment){temp, 0, 0});
      if (prev_model_size !=1 ) {prev_model_size += 1;}
      total_segment_count += prev_model_size;

      this->recursive_error_bound = 3;
      multi_layer_models.push_back(std::move(segs));
      layer_count++;
    }
    index_layer_count = layer_count;
    uint64_t model_size = total_segment_count * 24;

    double sum_prob = 0.0;
    for (int i = 1; i <= segment_count; ++i) {
      sum_prob += 1.0 / i;
    }
    double weighted_sum = 0.0;
    for (int i = 1; i <= segment_count; ++i) {
      double P_i = (1.0 / i) / sum_prob;
      double S_i = log2(i);
      weighted_sum += P_i * S_i;
    }

    double reward = adgMod::getQTableManagerInstance().compute_reward(
          inverse_density,
          temp_error,
          weighted_sum,
          layer_count,
          size,
          LearnedIndexData::level);

    double prev_reward = adgMod::getQTableManagerInstance().Q_table_sar.prev_reward;
    int prev_state = adgMod::getQTableManagerInstance().Q_table_sar.prev_state;
    double prev_error = adgMod::getQTableManagerInstance().Q_table_sar.prev_action;
    double logged_q_value = 0.0;
    if (prev_state != 7) {
      double Q_value = adgMod::getQTableManagerInstance().compute_q_value(
            prev_state, prev_error, prev_reward, inverse_density);
      adgMod::getQTableManagerInstance().updateQValue(prev_state, prev_error, Q_value);
      adgMod::getQTableManagerInstance().addExperience(prev_state, prev_error, prev_reward, inverse_density);
      logged_q_value = Q_value;
    }

    adgMod::getQTableManagerInstance().learnFromReplay();

    adgMod::getQTableManagerInstance().Q_table_sar.prev_state = inverse_density;
    adgMod::getQTableManagerInstance().Q_table_sar.prev_action = temp_error;
    adgMod::getQTableManagerInstance().Q_table_sar.prev_reward = reward;

    // std::cout << " | level " << LearnedIndexData::level 
    //           << " | keys_num: " << size 
    //           << " | inverse_density: " << inverse_density 
    //           << " | segment_num " <<  segment_count 
    //           << " | Layer_of_Index " << layer_count+1 
    //           << " | Models_size " << model_size 
    //           << " | load_cost " << weighted_sum 
    //           << " | meta's_err " << this->error_bound 
    //           << " | Q_value " << logged_q_value << std::endl;

    static const int _clear_qcsv = []{
        std::ofstream("/home/eros/workspace-lsm/wildturkey/build/result_eb.csv",
                      std::ios::out | std::ios::trunc)
            .close();
        return 0;
    }();
    static std::ofstream qcsv("/home/eros/workspace-lsm/wildturkey/build/result_eb.csv",
                              std::ios::out | std::ios::app);
    if (qcsv.tellp() == 0) {
        qcsv << "level,keys_num,inverse_density,segment_num,"
                "Layer_of_Index,Models_size,load_cost,meta_err,"
                "Q_value,reward\n";
    }

    qcsv  << LearnedIndexData::level-1        << ','
    << size                           << ','
    << inverse_density                << ','
    << segment_count                  << ','
    << (layer_count + 1)              << ','
    << model_size                     << ','
    << weighted_sum                   << ','
    << this->error_bound              << ','
    << logged_q_value                 << ','
    << reward                         << '\n';
    qcsv.flush();

    string_multi_layer_segments = std::move(multi_layer_models);
    learned.store(true);
    return true;
  }



else { // FILL IN GAMMA (error)
  PLR plr = PLR(error);

  // check if data if filled
   if(bwise==1){
  if (string_keys.empty()) assert(false);
  // printf("learned\n");
  // fill in some bounds for the model
  uint64_t temp = atoll(string_keys.back().c_str());
  min_key = atoll(string_keys.front().c_str());
  max_key = atoll(string_keys.back().c_str());
  size = string_keys.size();

  std::vector<Segment> segs = plr.train(string_keys, !is_level);
  inverse_density = static_cast<uint64_t>((max_key - min_key) / size); 
  if(inverse_density < 10) inverse_density = 0;
  else if(inverse_density < 30) inverse_density = 1;
  else if(inverse_density < 50) inverse_density = 2;
  else inverse_density = 3;
  int segment_count = segs.size();
  double sum_prob = 0.0;
  for (int i = 1; i <= segment_count; ++i) {
    sum_prob += 1.0 / i;  // 计算 Zipf 总和; i等同于pow(i, 1)
  }
  double weighted_sum = 0.0;
  for (int i = 1; i <= segment_count; ++i) {
    double P_i = (1.0 / i) / sum_prob;  // 段 i 的访问概率
    double S_i = log2(i);  // 段 i 的查找步数
    weighted_sum += P_i * S_i;  // 加权累加
  }

  if (segs.empty()) return false;
  // fill in a dummy last segment (used in segment binary search)
  segs.push_back((Segment){temp, 0, 0});
  string_segments = std::move(segs);
  uint64_t model_size =  (segment_count + 1) * 24;

  std::cout << " | level " << LearnedIndexData::level 
            << " | keys_num: " << size 
            << " | inverse_density: " << inverse_density 
            << " | segment_num " << segment_count 
            << " | Models_size " << model_size 
            << " | load_cost " << weighted_sum 
            << " | meta's_err " << error 
            << " | Q_value " << "null" << std::endl;

  for (auto& str : string_segments) {
    // printf("%s %f\n", str.first.c_str(), str.second);
  }
  learned.store(true);
  // string_keys.clear();
  return true;
  }


  } // end bwise==1 guard

  // 未处理的分支（非 bwise）返回失败
  return false;
}

// static learning function to be used with LevelDB background scheduling
// level learning
void LearnedIndexData::LevelLearn(void* arg, bool nolock) {
  Stats* instance = Stats::GetInstance();
  bool success = false;
  bool entered = false;
  instance->StartTimer(8);

  VersionAndSelf* vas = reinterpret_cast<VersionAndSelf*>(arg);
  std::shared_ptr<LearnedIndexData> self_holder = vas->self;
  LearnedIndexData* self = self_holder.get();
  self->is_level = true;
  self->level = vas->level;
  Version* c;
  if (!nolock) {
    c = db->GetCurrentVersion();
  }
  if (db->version_count == vas->v_count) {
    entered = true;
    if (vas->version->FillLevel(adgMod::read_options, vas->level)) {
      self->filled = true;
      if (db->version_count == vas->v_count) {
        if (env->compaction_awaiting.load() == 0 && self->Learn()) {
          success = true;
        } else {
          self->learning.store(false);
        }
      }
    }
  }
  if (!nolock) {
    adgMod::db->ReturnCurrentVersion(c);
  }

  auto time = instance->PauseTimer(8, true);

  if (entered) {
    self->cost = time.second - time.first;
    learn_counter_mutex.Lock();
    events[1].push_back(new LearnEvent(time, 0, self->level, success));
    levelled_counters[6].Increment(vas->level, time.second - time.first);
    learn_counter_mutex.Unlock();
  }

  delete vas;
}

// static learning function to be used with LevelDB background scheduling
// file learning
uint64_t LearnedIndexData::FileLearn(void* arg) {
  Stats* instance = Stats::GetInstance();
  bool entered = false;
  instance->StartTimer(11);

  MetaAndSelf* mas = reinterpret_cast<MetaAndSelf*>(arg);
  std::shared_ptr<LearnedIndexData> self_holder = mas->self;
  LearnedIndexData* self = self_holder.get();
  self->level = mas->level;
  self->meta = mas->meta;

  Version* c = db->GetCurrentVersion();
  if (self->FillData(c, mas->meta)) {
    self->Learn();
    entered = true;
  } else {
    self->learning.store(false);
  }
  adgMod::db->ReturnCurrentVersion(c);

  auto time = instance->PauseTimer(11, true);

  if (entered) {
    // count how many file learning are done.
    self->cost = time.second - time.first;
    learn_counter_mutex.Lock();
    events[1].push_back(new LearnEvent(time, 1, self->level, true));
    levelled_counters[11].Increment(mas->level, time.second - time.first);
    learn_counter_mutex.Unlock();
  }

  if (!fresh_write) {
             self->WriteModel(adgMod::db->versions_->dbname_ + "/" +
             to_string(mas->meta->number) + ".fmodel");
             self->string_keys.clear();
            //  self->num_entries_accumulated.array.clear();
         }

  if (!fresh_write) delete mas->meta;
  delete mas;
  return entered ? time.second - time.first : 0;
}

// general model checker
bool LearnedIndexData::Learned() {
  if (learned_not_atomic)
    return true;
  else if (learned.load()) {
    learned_not_atomic = true;
    return true;
  } else
    return false;
}

// level model checker, used to be also learning trigger
bool LearnedIndexData::Learned(Version* version, int v_count, int level) {
  if (learned_not_atomic)
    return true;
  else if (learned.load()) {
    learned_not_atomic = true;
    return true;
  }
  return false;
  //        } else {
  //            if (level_learning_enabled && ++current_seek >= allowed_seek &&
  //            !learning.exchange(true)) {
  //                env->ScheduleLearning(&LearnedIndexData::Learn, new
  //                VersionAndSelf{version, v_count, this, level}, 0);
  //            }
  //            return false;
  //        }
}

// file model checker, used to be also learning trigger
bool LearnedIndexData::Learned(Version* version, int v_count,
                               FileMetaData* meta, int level) {
  if (learned_not_atomic)
    return true;
  else if (learned.load()) {
    learned_not_atomic = true;
    return true;
  } else
    return false;
  //        } else {
  //            if (file_learning_enabled && (true || level != 0 && level != 1)
  //            && ++current_seek >= allowed_seek && !learning.exchange(true)) {
  //                env->ScheduleLearning(&LearnedIndexData::FileLearn, new
  //                MetaAndSelf{version, v_count, meta, this, level}, 0);
  //            }
  //            return false;
  //        }
}

bool LearnedIndexData::FillData(Version* version, FileMetaData* meta) {
  // if (filled) return true;

  if (version->FillData(adgMod::read_options, meta, this)) {
    // filled = true;
    return true;
  }
  return false;
}

void LearnedIndexData::WriteModel(const string& filename) {
  if (!learned.load()) return;

  std::ofstream output_file(filename);
  output_file.precision(15);
  // output_file << adgMod::block_num_entries << " " << adgMod::block_size << " "
  //             << adgMod::entry_size << "\n";
  if (adgMod::adeb == 1) {
    for (int i = string_multi_layer_segments.size() - 1; i >= 0; --i) {
      for (Segment& item : string_multi_layer_segments[i]) {
        output_file << item.x << " " << item.k << " " << item.b << "\n";
      }
      output_file << "layer " << i << "\n";
    }
  }
  else {
    for (Segment& item : string_segments) {
      output_file << item.x << " " << item.k << " " << item.b << "\n";
    }
  }
  output_file << 621;
  //             << " " << min_key << " " << max_key << " " << size << " " << level
  //             << " " << cost << "\n";
  // for (auto& pair : num_entries_accumulated.array) {
  //   output_file << pair.first << " " << pair.second << "\n";
  // }
}
void LearnedIndexData::ReadModel(const std::string& filename) {
    std::ifstream input_file(filename);

    if (!input_file.good()) return;

    std::string line;
    if (adgMod::adeb == 1) {
        std::vector<Segment> current_layer;
        bool end_marker_found = false;

        while (std::getline(input_file, line)) {
            std::istringstream iss(line);
            std::string token;
            if (iss >> token) {
                if (token == "layer") {
                    // 将当前层存入多层模型，并清空current_layer以准备读取下一层
                    string_multi_layer_segments.push_back(std::move(current_layer));
                    current_layer.clear();  // 清空以存储新层数据
                } else if (token == "621") {
                    // 读取结束标记，表示所有层读取完成
                    end_marker_found = true;
                    break;
                } else {
                    // 读取当前段的 x, k, b
                    try {
                        uint64_t x = std::stoull(token);  // 第一个是x
                        double k, b;
                        if (!(iss >> k >> b)) {
                            //std::cerr << "Error reading segment data from line: " << line << std::endl;
                            continue;  // 跳过读取失败的行
                        }
                        // 将段加入当前层
                        current_layer.emplace_back(x, k, b);
                    } catch (const std::exception& e) {
                        //std::cerr << "Exception reading segment: " << e.what() << std::endl;
                        continue;  // 跳过解析失败的行
                    }
                }
            }
        }

        // 将最后一层的数据存入模型
        if (!current_layer.empty()) {
            string_multi_layer_segments.push_back(std::move(current_layer));
        }

        // 如果未找到结束标记，则发出警告
        // if (!end_marker_found) {
        //     std::cerr << "Error: Model file is missing the end marker '621'." << std::endl;
        // }

    } 
    else {
        // 单层模型的读取逻辑
        while (std::getline(input_file, line)) {
            std::istringstream iss(line);
            uint64_t x;
            double k = 0.0, b = 0.0;

            // 尝试读取 x, k, b
            if (!(iss >> x >> k >> b)) {
                // 检查是否遇到结束标记
                if (line == "621") {
                    break;  // 读取到结束标记时停止读取
                } else {
                    //std::cerr << "Error reading values from line: " << line << std::endl;
                    continue;  // 跳过这一行，继续处理下一行
                }
            }

            // 将读取的数据添加到 string_segments 中
            string_segments.emplace_back(x, k, b);
        }
    }

    learned.store(true);
}


void LearnedIndexData::ReportStats() {
  //        double neg_gain, pos_gain;
  //        if (num_neg_model == 0 || num_neg_baseline == 0) {
  //            neg_gain = 0;
  //        } else {
  //            neg_gain = ((double) time_neg_baseline / num_neg_baseline -
  //            (double) time_neg_model / num_neg_model) * num_neg_model;
  //        }
  //        if (num_pos_model == 0 || num_pos_baseline == 0) {
  //            pos_gain = 0;
  //        } else {
  //            pos_gain = ((double) time_pos_baseline / num_pos_baseline -
  //            (double) time_pos_model / num_pos_model) * num_pos_model;
  //        }

  printf("%d %d %lu %lu %lu\n", level, served, string_segments.size(), cost,
         size);  //, file_size);
  //        printf("\tPredicted: %lu %lu %lu %lu %d %d %d %d %d %lf\n",
  //        time_neg_baseline_p, time_neg_model_p, time_pos_baseline_p,
  //        time_pos_model_p,
  //                num_neg_baseline_p, num_neg_model_p, num_pos_baseline_p,
  //                num_pos_model_p, num_files_p, gain_p);
  //        printf("\tActual: %lu %lu %lu %lu %d %d %d %d %f\n",
  //        time_neg_baseline, time_neg_model, time_pos_baseline,
  //        time_pos_model,
  //               num_neg_baseline, num_neg_model, num_pos_baseline,
  //               num_pos_model, pos_gain + neg_gain);
}

void LearnedIndexData::FillCBAStat(bool positive, bool model, uint64_t time) {
  //        int& num_to_update = positive ? (model ? num_pos_model :
  //        num_pos_baseline) : (model ? num_neg_model : num_neg_baseline);
  //        uint64_t& time_to_update =  positive ? (model ? time_pos_model :
  //        time_pos_baseline) : (model ? time_neg_model : time_neg_baseline);
  //        time_to_update += time;
  //        num_to_update += 1;
}

std::shared_ptr<LearnedIndexData> FileLearnedIndexData::GetModel(int number) {
  leveldb::MutexLock l(&mutex);
  if (file_learned_index_data.size() <= number)
    file_learned_index_data.resize(number + 1, nullptr);
  if (!file_learned_index_data[number])
    file_learned_index_data[number] = std::make_shared<LearnedIndexData>(file_allowed_seek, false);
  return file_learned_index_data[number];
}

std::shared_ptr<LearnedIndexData> FileLearnedIndexData::PeekModel(int number) {
  leveldb::MutexLock l(&mutex);
  if (number < 0 || file_learned_index_data.size() <= static_cast<size_t>(number))
    return nullptr;
  return file_learned_index_data[number];
}

void FileLearnedIndexData::ReleaseModel(int number) {
  leveldb::MutexLock l(&mutex);
  if (number < 0 || file_learned_index_data.size() <= static_cast<size_t>(number))
    return;
  file_learned_index_data[number].reset();
}

bool FileLearnedIndexData::FillData(Version* version, FileMetaData* meta) {
  auto model = GetModel(meta->number);
  return model->FillData(version, meta);
}

std::vector<std::string>& FileLearnedIndexData::GetData(FileMetaData* meta) {
  auto model = GetModel(meta->number);
  return model->string_keys;
}

bool FileLearnedIndexData::Learned(Version* version, FileMetaData* meta,
                                   int level) {
  auto model = GetModel(meta->number);
  return model->Learned(version, db->version_count, meta, level);
}



AccumulatedNumEntriesArray* FileLearnedIndexData::GetAccumulatedArray(
    int file_num) {
  auto model = GetModel(file_num);
  return &model->num_entries_accumulated;
}

std::pair<uint64_t, uint64_t> FileLearnedIndexData::GetPosition(
    const Slice& key, int file_num) {
  auto model = GetModel(file_num);
  return model->GetPosition(key);
}

FileLearnedIndexData::~FileLearnedIndexData() {
  leveldb::MutexLock l(&mutex);
}

void FileLearnedIndexData::Report() {
  leveldb::MutexLock l(&mutex);
  int64_t segement_size=0;
  std::set<uint64_t> live_files;
  int count = 0;
  adgMod::db->versions_->AddLiveFiles(&live_files);

  for (size_t i = 0; i < file_learned_index_data.size(); ++i) {
    auto pointer = file_learned_index_data[i];
    if (pointer != nullptr && pointer->cost != 0) {
      printf("FileModel %lu %d ", i, i > watermark);
      pointer->ReportStats();
      count++;
      segement_size += pointer->string_segments.size();
    }
  }
  segement_size = segement_size / count;
  printf("Average Segement Size: %ld\n", segement_size);
}

void AccumulatedNumEntriesArray::Add(uint64_t num_entries, string&& key) {
  array.emplace_back(num_entries, key);
}

bool AccumulatedNumEntriesArray::Search(const Slice& key, uint64_t lower,
                                        uint64_t upper, size_t* index,
                                        uint64_t* relative_lower,
                                        uint64_t* relative_upper) {
  if (adgMod::MOD == 4) {
    uint64_t lower_pos = lower / array[0].first;
    uint64_t upper_pos = upper / array[0].first;
    if (lower_pos != upper_pos) {
      while (true) {
        if (lower_pos >= array.size()) return false;
        if (key <= array[lower_pos].second) break;
        lower = array[lower_pos].first;
        ++lower_pos;
      }
      upper = std::min(upper, array[lower_pos].first - 1);
      *index = lower_pos;
      *relative_lower =
          lower_pos > 0 ? lower - array[lower_pos - 1].first : lower;
      *relative_upper =
          lower_pos > 0 ? upper - array[lower_pos - 1].first : upper;
      return true;
    }
    *index = lower_pos;
    *relative_lower = lower % array[0].first;
    *relative_upper = upper % array[0].first;
    return true;

  } else {
    size_t left = 0, right = array.size() - 1;
    while (left < right) {
      size_t mid = (left + right) / 2;
      if (lower < array[mid].first)
        right = mid;
      else
        left = mid + 1;
    }

    if (upper >= array[left].first) {
      while (true) {
        if (left >= array.size()) return false;
        if (key <= array[left].second) break;
        lower = array[left].first;
        ++left;
      }
      upper = std::min(upper, array[left].first - 1);
    }

    *index = left;
    *relative_lower = left > 0 ? lower - array[left - 1].first : lower;
    *relative_upper = left > 0 ? upper - array[left - 1].first : upper;
    return true;
  }
}

bool AccumulatedNumEntriesArray::SearchNoError(uint64_t position, size_t* index,
                                               uint64_t* relative_position) {
  *index = position / array[0].first;
  *relative_position = position % array[0].first;
  return *index < array.size();

  //        size_t left = 0, right = array.size() - 1;
  //        while (left < right) {
  //            size_t mid = (left + right) / 2;
  //            if (position < array[mid].first) right = mid;
  //            else left = mid + 1;
  //        }
  //        *index = left;
  //        *relative_position = left > 0 ? position - array[left - 1].first :
  //        position; return left < array.size();
}

uint64_t AccumulatedNumEntriesArray::NumEntries() const {
  return array.empty() ? 0 : array.back().first;
}

}  // namespace adgMod
