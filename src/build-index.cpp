#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"

#include "utils.hpp"
#include "index_types.hpp"

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

template <class t_idx>
void create_and_store(collection& col)
{
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    t_idx idx(col);
    auto stop = clock::now();
    std::cout << "index construction in (s): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0f << endl;
    auto output_file = col.path + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    std::ofstream ofs(output_file);
    if (ofs.is_open()) {
        std::cout << "writing index to file : " << output_file << std::endl;
        auto bytes = sdsl::serialize(idx, ofs);
        std::cout << "index size : " << bytes / (1024 * 1024) << " MB" << std::endl;
        std::cout << "writing space usage visualization to file : " << output_file + ".html" << std::endl;
        std::ofstream vofs(output_file + ".html");
        sdsl::write_structure<sdsl::HTML_FORMAT>(vofs, idx);
    } else {
        std::cerr << "cannot write index to file : " << output_file << std::endl;
    }
}

int main(int argc, const char* argv[])
{
    /* parse command line */
    cmdargs_t args = parse_args(argc, argv);

    /* parse collection directory */
    collection col(args.collection_dir);

    /* create indexes */
    {
        // define the index type with csa_wt (wt_int)
        using csa_type = sdsl::csa_wt_int<>;
        using cst_type = sdsl::cst_sct3<csa_type>;
        using index_type = index_succinct_store_n1fb<cst_type>;

        create_and_store<index_type>(col);
    }

    return 0;
}
