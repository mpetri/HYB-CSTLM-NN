#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"
#include <sdsl/suffix_array_algorithm.hpp>
#include <iostream>

#include "utils.hpp"
#include "collection.hpp"
#include "index_succinct.hpp"

typedef struct cmdargs {
    std::string collection_dir;
} cmdargs_t;

void
print_usage(const char* program)
{
    fprintf(stdout, "%s -c <collection dir>\n", program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
};

cmdargs_t
parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int op;
    args.collection_dir = "";
    while ((op = getopt(argc, (char* const*)argv, "c:")) != -1) {
        switch (op) {
        case 'c':
            args.collection_dir = optarg;
            break;
        }
    }
    if (args.collection_dir == "") {
        std::cerr << "Missing command line parameters.\n";
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return args;
}

template <class t_idx, class t_node_type>
typename t_idx::string_type
extract_edge_label(const t_idx& idx, const t_node_type node, size_t max_depth)
{
    typename t_idx::string_type edge;
    // auto node_depth = idx.m_cst.depth(node);
    for (size_t i = 1; i <= max_depth; i++) {
        auto sym = idx.m_cst.edge(node, i);
        edge.push_back(sym);
    }
    return edge;
}

template <class t_idx, class t_node_type>
uint64_t compute_contexts(const t_idx& idx, const t_node_type node)
{
    static std::vector<typename t_idx::csa_type::value_type> preceding_syms(12312312);
    static std::vector<typename t_idx::csa_type::size_type> left(12312312);
    static std::vector<typename t_idx::csa_type::size_type> right(12312312);
    auto lb = idx.m_cst.lb(node);
    auto rb = idx.m_cst.rb(node);
    size_t num_syms = 0;
    sdsl::interval_symbols(idx.m_cst.csa.wavelet_tree, lb, rb + 1, num_syms, preceding_syms, left, right);
    if (num_syms == 1)
        return idx.m_cst.degree(node);
    else {
        auto total_contexts = 0;
        for (size_t i = 0; i < num_syms; i++) {
            auto new_lb = idx.m_cst.csa.C[idx.m_cst.csa.char2comp[preceding_syms[i]]] + left[i];
            auto new_rb = idx.m_cst.csa.C[idx.m_cst.csa.char2comp[preceding_syms[i]]] + right[i] - 1;
            if (new_lb == new_rb)
                total_contexts++;
            else {
                auto new_node = idx.m_cst.node(new_lb, new_rb);
                auto deg = idx.m_cst.degree(new_node);
                total_contexts += deg;
            }
        }
        return total_contexts;
    }
}

template <class t_idx, class t_node_type>
void compute_stats_subtree(const t_idx& idx, const t_node_type node)
{
    auto itr = idx.m_cst.begin(node);
    auto end = idx.m_cst.end(node);
    while (itr != end) {
        if (itr.visit() == 1) {
            auto node = *itr;
            if (idx.m_cst.is_leaf(node)) {
                // std::cout << "1" << ";" << 0 << "\n";
            } else {
                auto depth = idx.m_cst.depth(node);
                if (depth > 5) {
                    itr.skip_subtree();
                    continue;
                }
                std::cout << compute_contexts(idx, node) << ";" << depth << "\n";
            }
        }
        ++itr;
    }
}

template <class t_idx>
void compute_stats(t_idx& idx, const std::string& col_dir)
{
    using clock = std::chrono::high_resolution_clock;
    auto index_file = col_dir + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    if (utils::file_exists(index_file)) {
        std::cerr << "loading index from file '" << index_file << "'" << std::endl;
        sdsl::load_from_file(idx, index_file);

        std::chrono::nanoseconds total_time(0);

        std::cout << "contexts;depth" << std::endl;
        auto root = idx.m_cst.root();
        int skip = 2; // skip 0 and 1 subtree
        for (const auto& child : idx.m_cst.children(root)) {
            if (skip) { // skip
                skip--;
                continue;
            }
            auto start = clock::now();
            compute_stats_subtree(idx, child);
            auto stop = clock::now();
            std::cerr << ".";
            // std::cout << "time in milliseconds = "
            //       << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0f
            //       << " ms" << endl;
            total_time += (stop - start);
        }

        std::cerr << "\ntime in seconds = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() / 1000.0f
                  << " s" << endl;
    } else {
        std::cerr << "index does not exist. build it first" << std::endl;
    }
}

int main(int argc, const char* argv[])
{
    /* parse command line */
    cmdargs_t args = parse_args(argc, argv);

    /* create collection dir */
    utils::create_directory(args.collection_dir);

    {
        /* load SADA based index */
        using csa_type = sdsl::csa_wt_int<sdsl::wt_int<sdsl::hyb_vector<> > >;
        using cst_type = sdsl::cst_sct3<csa_type>;
        index_succinct<cst_type> idx;

        compute_stats(idx, args.collection_dir);
    }

    return 0;
}
