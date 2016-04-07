#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include <iostream>

int main(int argc, const char* argv[])
{
    std::string file_one = argv[1];
    std::string file_two = argv[2];

    sdsl::read_only_mapper<> one(file_one);
    sdsl::read_only_mapper<> two(file_two);

    if (one.size() != two.size()) {
        std::cerr << "size different! " << one.size() << " " << two.size() << std::endl;
        return -1;
    }
    if (one.width() != two.width()) {
        std::cerr << "width different! " << (int)one.width() << " " << (int)two.width() << std::endl;
        return -1;
    }

    size_t errors = 0;
    for (size_t i = 0; i < one.size(); i++) {
        uint64_t one_val = one[i];
        uint64_t two_val = two[i];
        if (one_val != two_val) {
            std::cerr << "(" << ++errors << ") value different! " << one_val << " " << two_val << " at " << i << std::cerr;
        }
    }

    return 0;
}
