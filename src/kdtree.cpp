#include "kdtree.h"

#include <algorithm>
#include <vector>
#include <cmath>
#include <tuple>
#include <unordered_map>
#include <stack>
#include <queue>
#include <cstring>

// Example:
//     int x = Malloc(int, 10);
//     int y = (int *)malloc(10 * sizeof(int));
#define Malloc(type, n) (type *)malloc((n)*sizeof(type))

// If you need to use Intel MKL to accelerate,
// you can cancel the next line comment.
// #define USE_INTEL_MKL

#ifdef USE_INTEL_MKL
#include <mkl.h>
#endif

// Clang does not support OpenMP.
#ifndef __clang__

#include <omp.h>

#endif

// 释放一颗二叉树内存的非递归算法
DLLExport void free_tree_memory(tree_node *root) {
    std::stack<tree_node *> node_stack;
    tree_node *p;
    node_stack.push(root);
    while (!node_stack.empty()) {
        p = node_stack.top();
        node_stack.pop();
        if (p->left)
            node_stack.push(p->left);
        if (p->right)
            node_stack.push(p->right);
        free(p);
    }
}


class KDTree {
public:

    KDTree(tree_node *root, const double *datas, size_t rows, size_t cols, double p);

    KDTree(const double *datas, const double *labels,
           size_t rows, size_t cols, double p, bool free_tree = true);

    ~KDTree();

    tree_node *GetRoot() { return root; }

    std::vector<std::tuple<size_t, double>> FindKNearests(const double *coor, size_t k);

    std::tuple<size_t, double> FindNearest(const double *coor, size_t k) { return FindKNearests(coor, k)[0]; }

    void CFindKNearests(const double *coor, size_t k, size_t *args, double *dists);


private:
    // The sample with the largest distance from point `coor`
    // is always at the top of the heap.
    struct neighbor_heap_cmp {
        bool operator()(const std::tuple<size_t, double> &i,
                        const std::tuple<size_t, double> &j) {
            return std::get<1>(i) < std::get<1>(j);
        }
    };

    typedef std::tuple<size_t, double> neighbor;
    typedef std::priority_queue<neighbor,
            std::vector<neighbor>, neighbor_heap_cmp> neighbor_heap;

    // 搜索 K-近邻时的堆（大顶堆），堆顶始终是 K-近邻中样本点最远的点
    neighbor_heap k_neighbor_heap_;
    // 求距离时的 p, dist(x, y) = pow((x^p + y^p), 1/p)
    double p;
    // 析构时是否释放树的内存
    bool free_tree_;
    // 树根结点
    tree_node *root;
    // 训练集
    const double *datas;
    // 训练集的样本数
    size_t n_samples;
    // 每个样本的维度
    size_t n_features;
    // 训练集的标签
    const double *labels;
    // 寻找中位数时用到的缓存池
    std::tuple<size_t, double> *get_mid_buf_;
    // 搜索 K 近邻时的缓存池，如果已经搜索过点 i，令 visited_buf[i] = True
    bool *visited_buf_;

#ifdef USE_INTEL_MKL
    // 使用 Intel MKL 库时的缓存
    double *mkl_buf_;
#endif


    // 初始化缓存
    void InitBuffer();

    // 建树
    tree_node *BuildTree(const std::vector<size_t> &points);

    // 求一组数的中位数
    std::tuple<size_t, double> MidElement(const std::vector<size_t> &points, size_t dim);

    // 入堆
    void HeapStackPush(std::stack<tree_node *> &paths, tree_node *node, const double *coor, size_t k);

    // 获取训练集中第 sample 个样本点第 dim 的值
    double GetDimVal(size_t sample, size_t dim) {
        return datas[sample * n_features + dim];
    }

    // 求点 coor 距离训练集第 i 个点的距离
    double GetDist(size_t i, const double *coor);

    // 寻找切分点
    size_t FindSplitDim(const std::vector<size_t> &points);

};

// 找到一棵树的 K近邻。Ki 的 id 和 Ki 距离 coor 的距离保存在 args 和 dists 中
DLLExport
void find_k_nearests(const tree_model *model, const double *coor,
                     size_t k, size_t *args, double *dists) {
    KDTree tree(model->root, model->datas, model->n_samples, model->n_features, model->p);
    std::vector<std::tuple<size_t, double>> k_nearest = tree.FindKNearests(coor, k);
    for (size_t i = 0; i < k; ++i) {
        args[i] = std::get<0>(k_nearest[i]);
        dists[i] = std::get<1>(k_nearest[i]);
    }
}

// 建立一棵 KD-Tree
DLLExport
tree_model *build_kdtree(const double *datas, const double *labels,
                         size_t rows, size_t cols, double p) {
    KDTree tree(datas, labels, rows, cols, p, false);
    tree_model *model = Malloc(tree_model, 1);
    model->datas = datas;
    model->labels = labels;
    model->n_features = cols;
    model->n_samples = rows;
    model->root = tree.GetRoot();
    model->p = p;
    return model;
}

// 求平均值，用于回归问题
double mean(const double *arr, size_t len) {
    double ans = 0.0;
    for (size_t i = 0; i < len; ++i)
        ans += arr[i];
    return ans / len;
}

// 投票，用于分类问题
double vote(const double *arr, size_t len) {
    std::unordered_map<int, size_t> counter;
    for (size_t i = 0; i < len; ++i) {
        auto t = static_cast<int>(arr[i]);
        if (counter.find(t) == counter.end())
            counter.insert(std::unordered_map<int, size_t>::value_type(t, 1));
        else
            counter[t] += 1;
    }
    double cur_arg_max = 0;
    size_t cur_max = 0;
    for (auto &i : counter) {
        if (i.second >= cur_max) {
            cur_arg_max = i.first;
            cur_max = i.second;
        }
    }
    return cur_arg_max;
}

DLLExport double *
k_nearests_neighbor(const tree_model *model, const double *X_test, size_t len, size_t k, bool clf) {
    KDTree tree(model->root, model->datas, model->n_samples, model->n_features, model->p);
    double *ans = Malloc(double, len);
    size_t *args = Malloc(size_t, k);
    double *dists = Malloc(double, k);
    double *y_pred = Malloc(double, k);
    size_t i, j;
    for (i = 0; i < len; ++i) {
        tree.CFindKNearests(X_test + i * model->n_features, k, args, dists);
        for (j = 0; j < k; ++j)
            y_pred[j] = model->labels[args[j]];
        if (clf)
            ans[i] = vote(y_pred, k);
        else
            ans[i] = mean(y_pred, k);
    }
    free(args);
    free(y_pred);
    free(dists);
    return ans;
}


inline KDTree::KDTree(tree_node *root, const double *datas, size_t rows, size_t cols, double p) :
        root(root), datas(datas), n_samples(rows),
        n_features(cols), p(p), free_tree_(false) {
    InitBuffer();
    labels = nullptr;
}

inline KDTree::KDTree(const double *datas, const double *labels, size_t rows, size_t cols, double p, bool free_tree) :
        datas(datas), labels(labels), n_samples(rows), n_features(cols), p(p), free_tree_(free_tree) {
    std::vector<size_t> points;
    for (size_t i = 0; i < n_samples; ++i)
        points.emplace_back(i);
    InitBuffer();
    root = BuildTree(points);
}

inline KDTree::~KDTree() {
    delete[]get_mid_buf_;
    delete[]visited_buf_;
#ifdef USE_INTEL_MKL
    free(mkl_buf_);
#endif
    if (free_tree_)
        free_tree_memory(root);
}

std::vector<std::tuple<size_t, double>> KDTree::FindKNearests(const double *coor, size_t k) {
    std::memset(visited_buf_, 0, sizeof(bool) * n_samples);
    std::stack<tree_node *> paths;
    tree_node *p = root;

    while (p) {
        HeapStackPush(paths, p, coor, k);
        p = coor[p->split] <= GetDimVal(p->id, p->split) ? p = p->left : p = p->right;
    }
    while (!paths.empty()) {
        p = paths.top();
        paths.pop();

        if (!p->left && !p->right)
            continue;

        if (k_neighbor_heap_.size() < k) {
            if (p->left)
                HeapStackPush(paths, p->left, coor, k);
            if (p->right)
                HeapStackPush(paths, p->right, coor, k);
        } else {
            double node_split_val = GetDimVal(p->id, p->split);
            double coor_split_val = coor[p->split];
            double heap_top_val = std::get<1>(k_neighbor_heap_.top());
            if (coor_split_val > node_split_val) {
                if (p->right)
                    HeapStackPush(paths, p->right, coor, k);

                if ((coor_split_val - node_split_val) < heap_top_val && p->left)
                    HeapStackPush(paths, p->left, coor, k);
            } else {
                if (p->left)
                    HeapStackPush(paths, p->left, coor, k);
                if ((node_split_val - coor_split_val) < heap_top_val && p->right)
                    HeapStackPush(paths, p->right, coor, k);
            }
        }
    }
    std::vector<std::tuple<size_t, double>> res;

    while (!k_neighbor_heap_.empty()) {
        res.emplace_back(k_neighbor_heap_.top());
        k_neighbor_heap_.pop();
    }
    return res;
}

void KDTree::CFindKNearests(const double *coor, size_t k, size_t *args, double *dists) {
    std::vector<std::tuple<size_t, double>> k_nearest = FindKNearests(coor, k);
    for (size_t i = 0; i < k; ++i) {
        args[i] = std::get<0>(k_nearest[i]);
        dists[i] = std::get<1>(k_nearest[i]);
    }
}


// 初始化缓存

inline void KDTree::InitBuffer() {
    get_mid_buf_ = new std::tuple<size_t, double>[n_samples];
    visited_buf_ = new bool[n_samples];

#ifdef USE_INTEL_MKL
    // 要与 C 代码交互，所以用 C 的方式申请内存
    mkl_buf_ = Malloc(double, n_features);
#endif
}

tree_node *KDTree::BuildTree(const std::vector<size_t> &points) {
    size_t dim = FindSplitDim(points);
    std::tuple<size_t, double> t = MidElement(points, dim);
    size_t arg_mid_val = std::get<0>(t);
    double mid_val = std::get<1>(t);

    tree_node *node = Malloc(tree_node, 1);
    node->left = nullptr;
    node->right = nullptr;
    node->id = arg_mid_val;
    node->split = dim;
    std::vector<size_t> left, right;
    for (auto &i : points) {
        if (i == arg_mid_val)
            continue;
        if (GetDimVal(i, dim) <= mid_val)
            left.emplace_back(i);
        else
            right.emplace_back(i);
    }
    if (!left.empty())
        node->left = BuildTree(left);
    if (!right.empty())
        node->right = BuildTree(right);
    return node;
}

std::tuple<size_t, double> KDTree::MidElement(const std::vector<size_t> &points, size_t dim) {
    size_t len = points.size();
    for (size_t i = 0; i < points.size(); ++i)
        get_mid_buf_[i] = std::make_tuple(points[i], GetDimVal(points[i], dim));
    std::nth_element(get_mid_buf_,
                     get_mid_buf_ + len / 2,
                     get_mid_buf_ + len,
                     [](const std::tuple<size_t, double> &i, const std::tuple<size_t, double> &j) {
                         return std::get<1>(i) < std::get<1>(j);
                     });
    return get_mid_buf_[len / 2];
}


inline void KDTree::HeapStackPush(std::stack<tree_node *> &paths, tree_node *node, const double *coor, size_t k) {
    paths.emplace(node);
    size_t id = node->id;
    if (visited_buf_[id])
        return;
    visited_buf_[id] = true;
    double dist = GetDist(id, coor);
    std::tuple<size_t, double> t(id, dist);
    if (k_neighbor_heap_.size() < k)
        k_neighbor_heap_.push(t);

    else if (std::get<1>(t) < std::get<1>(k_neighbor_heap_.top())) {
        k_neighbor_heap_.pop();
        k_neighbor_heap_.push(t);
    }
}

#ifdef USE_INTEL_MKL
inline double KDTree::GetDist(size_t i, const double *coor) {
    size_t idx = i * n_features;
    vdSub(n_features, datas + idx, coor, mkl_buf_);
    vdPowx(n_features, mkl_buf_, p, mkl_buf_);
    double dist = cblas_dasum(n_features, mkl_buf_, 1);
    return pow(dist, 1.0 / p);
}
#else

inline double KDTree::GetDist(size_t i, const double *coor) {
    double dist = 0.0;
    size_t idx = i * n_features;
#pragma omp parallel for reduction(+:dist)
    for (int t = 0; t < n_features; ++t)
        dist += pow(datas[idx + t] - coor[t], p);
    return pow(dist, 1.0 / p);
}

#endif

size_t KDTree::FindSplitDim(const std::vector<size_t> &points) {
    if (points.size() == 1)
        return 0;
    size_t cur_best_dim = 0;
    double cur_largest_spread = -1;
    double cur_min_val;
    double cur_max_val;
    for (size_t dim = 0; dim < n_features; ++dim) {
        cur_min_val = GetDimVal(points[0], dim);
        cur_max_val = GetDimVal(points[0], dim);
        for (const auto &id : points) {
            if (GetDimVal(id, dim) > cur_max_val)
                cur_max_val = GetDimVal(id, dim);
            else if (GetDimVal(id, dim) < cur_min_val)
                cur_min_val = GetDimVal(id, dim);
        }

        if (cur_max_val - cur_min_val > cur_largest_spread) {
            cur_largest_spread = cur_max_val - cur_min_val;
            cur_best_dim = dim;
        }
    }
    return cur_best_dim;
}




