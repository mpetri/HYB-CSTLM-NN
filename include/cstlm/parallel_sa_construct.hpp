#pragma once

#include <iostream>
#include <vector>
#include <limits>
#include <fstream>
#include <chrono>
#include <array>
#include <thread>
#include <cilk/cilk.h>

#include <sys/resource.h>
#include <nmmintrin.h>

#define ISORT 25
#define SPAWN_THRESHOLD 1000

template <class E, class BinPred, class intT>
void insertionSort(E* A, intT n, BinPred f)
{
    for (intT i = 0; i < n; i++) {
        E v = A[i];
        E* B = A + i;
        while (--B >= A && f(v, *B))
            *(B + 1) = *B;
        *(B + 1) = v;
    }
}

template <class E, class BinPred>
E median(E a, E b, E c, BinPred f)
{
    return f(a, b) ? (f(b, c) ? b : (f(a, c) ? c : a))
                   : (f(a, c) ? a : (f(b, c) ? c : b));
}

template <class E, class BinPred, class intT>
std::pair<E*, E*> split(E* A, intT n, BinPred f)
{
    E p = median(A[n / 4], A[n / 2], A[(3 * n) / 4], f);
    E* L = A; // below L are less than pivot
    E* M = A; // between L and M are equal to pivot
    E* R = A + n - 1; // above R are greater than pivot
    while (1) {
        while (!f(p, *M)) {
            if (f(*M, p))
                std::swap(*M, *(L++));
            if (M >= R)
                break;
            M++;
        }
        while (f(p, *R))
            R--;
        if (M >= R)
            break;
        std::swap(*M, *R--);
        if (f(*M, p))
            std::swap(*M, *(L++));
        M++;
    }
    return std::pair<E*, E*>(L, M);
}

template <class E, class BinPred, class intT>
void quickSortSerial(E* A, intT n, BinPred f)
{
    if (n < ISORT)
        insertionSort(A, n, f);
    else {
        std::pair<E*, E*> X = split(A, n, f);
        quickSortSerial(A, X.first - A, f);
        // Exclude all elts that equal pivot
        quickSortSerial(X.second, A + n - X.second, f);
    }
}

// Quicksort based on median of three elements as pivot
//  and uses insertionSort for small inputs
template <class E, class BinPred, class intT>
void quickSort(E* A, intT n, BinPred f)
{
    if (n < SPAWN_THRESHOLD)
        quickSortSerial(A, n, f);
    else {
        std::pair<E*, E*> X = split(A, n, f);
        cilk_spawn quickSort(A, X.first - A, f);
        quickSort(X.second, A + n - X.second, f);
        cilk_sync;
    }
}

template <class t_char>
bool is_type_A(t_char cur_char, t_char next_char)
{
    return cur_char > next_char;
}

template <class t_char>
bool is_type_B(t_char cur_char, t_char next_char)
{
    return cur_char <= next_char;
}

struct suftype_counts_array_byte {
    std::array<uint64_t, 256> A_START = { { 0 } };
    std::array<uint64_t, 256> A_CNT = { { 0 } };
    std::array<uint64_t, 256> B_START = { { 0 } };
    std::array<uint64_t, 256> B_CNT = { { 0 } };
};

struct suftype_counts_array_int {
    std::unordered_map<uint64_t, uint64_t> A_START;
    std::unordered_map<uint64_t, uint64_t> A_CNT;
    std::unordered_map<uint64_t, uint64_t> B_START;
    std::unordered_map<uint64_t, uint64_t> B_CNT;
};

template <class t_cnts>
struct bucket_counts {
    std::vector<uint64_t> syms;
    std::vector<t_cnts> partial_cnts;
    t_cnts total_cnts;
    bucket_counts(size_t threads)
    {
        partial_cnts.resize(threads);
    }
};

template <class t_vec, class t_cnts>
void compute_partial_cnts(t_vec& T, size_t offset, size_t n, t_cnts& cnts)
{
    for (size_t i = 0; i < n; i++) {
        auto cur = T[offset + i];
        auto next = T[offset + i + 1];
        if (is_type_A<decltype(cur)>(cur, next)) {
            cnts.A_CNT[cur]++;
        }
        else {
            cnts.B_CNT[cur]++;
        }
    }
}

template <class t_vec,
    typename std::enable_if<std::is_same<t_vec, const uint8_t*>::value, int>::type = 0>
bucket_counts<suftype_counts_array_byte> compute_bucket_counts(t_vec& T, size_t n)
{
    auto num_threads = std::thread::hardware_concurrency();
    bucket_counts<suftype_counts_array_byte> b(num_threads);
    size_t syms_per_thread = n / num_threads;
    auto start = 0;
    // (1) compute in parallel
    for (size_t i = 0; i < num_threads; i++) {
        auto m = syms_per_thread;
        if (i + 1 == num_threads)
            m = (n - 1) - start;
        cilk_spawn compute_partial_cnts(T, start, m, b.partial_cnts[i]);
        start += m;
    }
    cilk_sync;

    // (2) find all unique syms
    b.syms.push_back(0);
    for (size_t sym = 1; sym < 256; sym++) {
        for (size_t i = 0; i < num_threads; i++) {
            if (b.partial_cnts[i].A_CNT[sym] != 0 || b.partial_cnts[i].B_CNT[sym] != 0) {
                b.syms.push_back(sym);
                break;
            }
        }
    }

    // sum up partial counts
    for (size_t j = 0; j < num_threads; j++) {
        for (size_t i = 0; i < b.syms.size(); i++) {
            auto sym = b.syms[i];
            b.total_cnts.A_CNT[sym] += b.partial_cnts[j].A_CNT[sym];
            b.total_cnts.B_CNT[sym] += b.partial_cnts[j].B_CNT[sym];
        }
    }
    b.total_cnts.B_CNT[0] = 1;

    // (2) compute bucket start pos. O(sigma) no need to parallelize
    size_t cumsum = 0;
    for (size_t i = 0; i < b.syms.size(); i++) {
        auto sym = b.syms[i];
        b.total_cnts.A_START[sym] = cumsum;
        cumsum += b.total_cnts.A_CNT[sym];
        b.total_cnts.B_START[sym] = cumsum;
        cumsum += b.total_cnts.B_CNT[sym];
    }

    return b;
}

template <class t_vec,
    typename std::enable_if<std::is_same<t_vec, sdsl::int_vector<> >::value, int>::type = 0>
bucket_counts<suftype_counts_array_int> compute_bucket_counts(t_vec& T, size_t n)
{
    auto num_threads = std::thread::hardware_concurrency();
    bucket_counts<suftype_counts_array_int> b(num_threads);
    size_t syms_per_thread = n / num_threads;
    auto start = 0;
    // (1) compute in parallel
    for (size_t i = 0; i < num_threads; i++) {
        auto m = syms_per_thread;
        if (i + 1 == num_threads)
            m = (n - 1) - start;
        cilk_spawn compute_partial_cnts<t_vec, suftype_counts_array_int>(T, start, m, b.partial_cnts[i]);
        start += m;
    }
    cilk_sync;

    // (2) find all unique syms
    b.syms.push_back(0);
    std::set<uint64_t> unique_syms;
    for (size_t i = 0; i < num_threads; i++) {
        for (const auto sc : b.partial_cnts[i].A_CNT)
            unique_syms.insert(sc.first);
        for (const auto sc : b.partial_cnts[i].B_CNT)
            unique_syms.insert(sc.first);
    }
    for (auto sym : unique_syms)
        b.syms.push_back(sym);

    // sum up partial counts
    for (size_t j = 0; j < num_threads; j++) {
        for (size_t i = 0; i < b.syms.size(); i++) {
            auto sym = b.syms[i];
            b.total_cnts.A_CNT[sym] += b.partial_cnts[j].A_CNT[sym];
            b.total_cnts.B_CNT[sym] += b.partial_cnts[j].B_CNT[sym];
        }
    }
    b.total_cnts.B_CNT[0] = 1;

    // (2) compute bucket start pos. O(sigma) no need to parallelize
    size_t cumsum = 0;
    for (size_t i = 0; i < b.syms.size(); i++) {
        auto sym = b.syms[i];
        b.total_cnts.A_START[sym] = cumsum;
        cumsum += b.total_cnts.A_CNT[sym];
        b.total_cnts.B_START[sym] = cumsum;
        cumsum += b.total_cnts.B_CNT[sym];
    }

    return b;
}

template <class t_vec, class t_int_type, class t_cnts>
void partial_put_B_types_in_bucket(t_vec& T, size_t offset, size_t n, t_int_type* SA, t_cnts& cnts)
{
    for (size_t i = 0; i < n; i++) {
        auto cur = T[offset + i];
        auto next = T[offset + i + 1];
        if (is_type_B<decltype(cur)>(cur, next)) {
            SA[cnts.B_START[cur]++] = offset + i;
        }
    }
}

template <class t_vec, class t_int_type, class t_cnts>
void put_B_types_in_bucket(t_vec& T, t_int_type* SA, t_cnts& b, size_t n)
{
    size_t syms_per_thread = n / b.partial_cnts.size();
    // compute partial starts for B regions for each symbol
    for (size_t i = 0; i < b.syms.size(); i++) {
        auto sym = b.syms[i];
        auto cur_start = b.total_cnts.B_START[sym];
        for (size_t j = 0; j < b.partial_cnts.size(); j++) {
            b.partial_cnts[j].B_START[sym] = cur_start;
            cur_start += b.partial_cnts[j].B_CNT[sym];
        }
    }

    // put B types in place in parallel.
    size_t start = 0;
    for (size_t i = 0; i < b.partial_cnts.size(); i++) {
        size_t m = syms_per_thread;
        if (i + 1 == b.partial_cnts.size())
            m = (n - 2) - start;
        cilk_spawn partial_put_B_types_in_bucket<t_vec, t_int_type, decltype(b.partial_cnts[i])>(T, start, m, SA, b.partial_cnts[i]);
        start += m;
    }
    cilk_sync;
}

template <class t_int_type>
struct sa_cmp_func_byte {
    const uint8_t* T;
    sa_cmp_func_byte(const uint8_t* _T)
        : T(_T)
    {
    }
    bool operator()(const t_int_type a, const t_int_type b) const
    {
        size_t i = 0;
        if (a == b)
            return false;
#ifdef HAVE_SSE42
        const uint8_t* Ta8 = T + a;
        const uint8_t* Tb8 = T + b;
        while (true) {
            __m128i x = _mm_loadu_si128((const __m128i*)(Ta8 + i));
            __m128i y = _mm_loadu_si128((const __m128i*)(Tb8 + i));
            int index = _mm_cmpistri(x, y, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_NEGATIVE_POLARITY);
            if (index != 16) {
                return Ta8[i + index] < Tb8[i + index];
            }
            i += 16;
        }
        return Ta8[i] < Tb8[i];
#endif
        while (T[a + i] == T[b + i])
            i++;
        return T[a + i] < T[b + i];
    }
};

template <class t_vec, class t_int_type>
struct sa_cmp_func_int {
    t_vec& T;
    sa_cmp_func_int(t_vec& _T)
        : T(_T)
    {
    }
    bool operator()(const t_int_type& a, const t_int_type& b) const
    {
        size_t i = 0;
        if (a == b)
            return false;
        while (T[a + i] == T[b + i])
            i++;
        return T[a + i] < T[b + i];
    }
};

template <class t_vec, class t_int_type,
    typename std::enable_if<std::is_same<t_vec, const uint8_t*>::value, int>::type = 0>
void string_sort(t_vec& T, t_int_type* SA, size_t offset, size_t n)
{
    sa_cmp_func_byte<t_int_type> cmp(T);
    quickSort(SA + offset, n, cmp);
}

template <class t_vec, class t_int_type,
    typename std::enable_if<std::is_same<t_vec, sdsl::int_vector<> >::value, int>::type = 0>
void string_sort(t_vec& T, t_int_type* SA, size_t offset, size_t n)
{
    sa_cmp_func_int<t_vec, t_int_type> cmp(T);
    quickSort(SA + offset, n, cmp);
}

template <class t_vec, class t_int_type, class t_cnts>
void sort_B_bucket(t_vec& T, t_int_type* SA, t_cnts& b, size_t sym)
{
    auto start = b.total_cnts.B_START[sym];
    auto n = b.total_cnts.B_CNT[sym];
    string_sort<t_vec, t_int_type>(T, SA, start, n);
}

template <class t_vec, class t_int_type, class t_cnts>
void sort_B_types(t_vec& T, t_int_type* SA, t_cnts& b)
{
    for (size_t i = 0; i < b.syms.size(); i++) {
        auto sym = b.syms[i];
        if (b.total_cnts.B_CNT[sym] > 1) {
            cilk_spawn sort_B_bucket<t_vec, t_int_type, t_cnts>(T, SA, b, sym);
        }
    }
    cilk_sync;
}

template <class t_vec, class t_int_type>
void parallel_sufsort_it(t_vec& T, t_int_type* SA, size_t n)
{
    // REQ: T[n-1] = 0
    // (1) count suffixes  -> first pass
    auto bucket_counts = compute_bucket_counts<t_vec>(T, n);

    // (3) put the type B suffixes in place -> second pass
    put_B_types_in_bucket<t_vec, t_int_type, decltype(bucket_counts)>(T, SA, bucket_counts, n);

    // (4) sort type B suffixes
    sort_B_types<t_vec, t_int_type, decltype(bucket_counts)>(T, SA, bucket_counts);

    // // (5) induce A types from sorted B types
    SA[0] = n - 1; // FROM T[n-1] = 0
    for (size_t j = 0; j < n; j++) {
        auto prev_sym = T[(SA[j] + n - 1) % n];
        auto cur_sym = T[SA[j]];
        if (prev_sym > cur_sym) {
            SA[bucket_counts.total_cnts.A_START[prev_sym]++] = SA[j] - 1;
        }
    }
}
