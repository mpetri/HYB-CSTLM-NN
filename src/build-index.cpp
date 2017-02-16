#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"

#include "utils.hpp"
#include "index_types.hpp"
#include "logging.hpp"

#include "mem_monitor.hpp"

using namespace cstlm;

typedef struct cmdargs {
    std::string collection_dir;
    bool        use_mkn;
} cmdargs_t;

void print_usage(const char* program)
{
    fprintf(stdout, "%s -c <collection dir>\n", program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -m                   : use modified kneser ney.\n");
    fprintf(stdout, "  -t <threads>         : limit the number of threads.\n");
};


cmdargs_t parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int       op;
    args.collection_dir = "";
    args.use_mkn        = false;
    while ((op = getopt(argc, (char* const*)argv, "c:dmt:")) != -1) {
        switch (op) {
            case 'c':
                args.collection_dir = optarg;
                break;
            case 'm':
                args.use_mkn = true;
                break;
            case 't':
                num_cstlm_threads = std::atoi(optarg);
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

template <class t_idx>
void create_and_store(collection& col, bool use_mkn)
{
    using clock = std::chrono::high_resolution_clock;
    auto  start = clock::now();
    t_idx idx(col, use_mkn);
    auto  stop = clock::now();
    LOG(INFO) << "index construction in (s): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() /
                 1000.0f;
    auto output_file = col.path + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    std::ofstream ofs(output_file);
    if (ofs.is_open()) {
        LOG(INFO) << "writing index to file : " << output_file;
        auto bytes = sdsl::serialize(idx, ofs);
        LOG(INFO) << "index size : " << bytes / (1024 * 1024) << " MB";
        sdsl::write_structure<sdsl::HTML_FORMAT>(idx, output_file + ".html");
    } else {
        LOG(FATAL) << "cannot write index to file : " << output_file;
    }
}

int main(int argc, const char* argv[])
{
    enable_logging = true;

    /* parse command line */
    cmdargs_t args = parse_args(argc, argv);

    /* parse collection directory */
    collection col(args.collection_dir);
    /* create indexes */
    if (col.alphabet == alphabet_type::byte_alphabet) {
        create_and_store<charlm>(col, args.use_mkn);
    } else {
        create_and_store<wordlm>(col, args.use_mkn);
    }

    double peak_mem      = (double)utils::getPeakRSS();
    double text_size_raw = col.big_raw_size_in_bytes;
    double memory_usage  = peak_mem / (text_size_raw + 1.0);
    LOG(INFO) << "mem usage = " << utils::getPeakRSS() << " bytes";
    LOG(INFO) << "mem usage = " << memory_usage << "n";

    return 0;
}
