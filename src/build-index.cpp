
typedef struct cmdargs {
    std::string input_file;
    std::string output_dir;
} cmdargs_t;

void
print_usage(const char* program)
{
    fprintf(stdout,"%s -i <input file> -o <output dir>\n",program);
    fprintf(stdout,"where\n");
    fprintf(stdout,"  -i <input file>  : the input file.\n");
    fprintf(stdout,"  -o <output dir>  : the output dir.\n");
};

cmdargs_t
parse_args(int argc,const char* argv[])
{
    cmdargs_t args;
    int op;
    args.input_file = "";
    args.output_dir = "";
    while ((op=getopt(argc,(char* const*)argv,"i:o:t:")) != -1) {
        switch (op) {
            case 'i':
                args.input_file = optarg;
                break;
            case 'o':
                args.output_dir = optarg;
                break;
        }
    }
    if (args.input_file==""||args.output_dir=="") {
        std::cerr << "Missing command line parameters.\n";
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return args;
}


int main(int argc,const char* argv[])
{
    using clock = std::chrono::high_resolution_clock;

    /* parse command line */
    cmdargs_t args = parse_args(argc,argv);

    /* create index */
    using csa_type = csa_sada_int<>;
    using cst_type = cst_sct3<csa_type>;

    auto start = clock::now();
    index_succinct<cst_type<>> index(args.input_file,args.output_dir);
    auto stop = clock::now();
    std::cout << "index construction in (s): " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count()/1000.0f << endl;
    auto output_file = args.output_dir + "/index-" + util::class_to_hash(index) + ".sdsl";
    std::cout << "writing index to file : " << output_file << std::endl;
    auto bytes = sdsl::serialize(index,output_file);
    std::cout << "index size : " << bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "writing space usage visualization to file : " << output_file+".html" << std::endl;
    std::ofstream vofs(output_file+".html");
    sdsl::write_structure<sdsl::HTML_FORMAT>(vofs,*this);

    return 0;
}