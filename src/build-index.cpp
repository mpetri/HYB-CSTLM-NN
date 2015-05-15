#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"

#include "utils.hpp"
#include "index_types.hpp"

#include "logging.hpp"

typedef struct cmdargs {
    std::string collection_dir;
    bool dodgy_discounts;
} cmdargs_t;

void print_usage(const char* program)
{
    fprintf(stdout, "%s -c <collection dir>\n", program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -d                   : use dodgy (fast) discount calculation (default = correct)\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int op;
    args.collection_dir = "";
    args.dodgy_discounts = false;
    while ((op = getopt(argc, (char* const*)argv, "c:d")) != -1) {
        switch (op) {
        case 'c':
            args.collection_dir = optarg;
            break;
        case 'd':
            args.dodgy_discounts = true;
            break;
        }
    }
    if (args.collection_dir == "") {
        LOG(FATAL) << "Missing command line parameters.";
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return args;
}

template <class t_idx> void create_and_store(collection& col, bool dodgy_discounts)
{
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    t_idx idx(col, dodgy_discounts);
    auto stop = clock::now();
    LOG(INFO) << "index construction in (s): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()
                 / 1000.0f;
    auto output_file = col.path + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    std::ofstream ofs(output_file);
    if (ofs.is_open()) {
        LOG(INFO) << "writing index to file : " << output_file;
        auto bytes = sdsl::serialize(idx, ofs);
        LOG(INFO) << "index size : " << bytes / (1024 * 1024) << " MB";
        LOG(INFO) << "writing space usage visualization to file : " << output_file + ".html";
        std::ofstream vofs(output_file + ".html");
        sdsl::write_structure<sdsl::HTML_FORMAT>(vofs, idx);
    } else {
        LOG(FATAL) << "cannot write index to file : " << output_file;
    }
}

int main(int argc, const char* argv[])
{
    log::start_log(argc, argv);

    /* parse command line */
    cmdargs_t args = parse_args(argc, argv);

    /* parse collection directory */
    collection col(args.collection_dir);

    /* create indexes */
    {
        using index_type = index_succinct<default_cst_type>;
        create_and_store<index_type>(col, args.dodgy_discounts);
    }
    {
        using index_type = index_succinct_store_n1fb<default_cst_type>;
        create_and_store<index_type>(col, args.dodgy_discounts);
    }

    return 0;
}
