#pragma once

#include <stdexcept>
#include <sdsl/qsufsort.hpp>
#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include <sdsl/rank_support.hpp>
#include <sdsl/select_support.hpp>
#include <sdsl/sd_vector.hpp>

#include "utils.hpp"

const std::string KEY_PREFIX = "text.";
const std::string KEY_TEXT = "TEXT";
const std::string KEY_TEXTREV = "TEXTREV";
const std::string KEY_SA = "SA";
const std::string KEY_SAREV = "SAREV";

std::vector<std::string> collection_keys = {KEY_TEXT,
                                            KEY_TEXTREV,
                                            KEY_SA,
                                            KEY_SAREV
                                           };

struct collection {
    std::string path;
    std::map<std::string,std::string> file_map;
    collection(const std::string& p) : path(p+"/")
    {
        if (! utils::directory_exists(path)) {
            throw std::runtime_error("collection path not found.");
        }
        // make sure all other dirs exist
        std::string index_directory = path+"/index/";
        utils::create_directory(index_directory);
        std::string tmp_directory = path+"/tmp/";
        utils::create_directory(tmp_directory);
        std::string results_directory = path+"/results/";
        utils::create_directory(results_directory);
        std::string patterns_directory = path+"/patterns/";
        utils::create_directory(patterns_directory);
        /* make sure the necessary files are present */
        if (! utils::file_exists(path+"/"+KEY_PREFIX+KEY_TEXT)) {
            throw std::runtime_error("collection path does not contain text.");
        }

        /* register files that are present */
        for (const auto& key : collection_keys) {
            auto file_path = path+"/"+KEY_PREFIX+key;
            if (utils::file_exists(file_path)) {
                file_map[key] = file_path;
                LOG(INFO) << "FOUND '" << key << "' at '" << file_path <<"'";
            }
        }

        /* create stuff we are missing */
        if (file_map.count(KEY_TEXTREV) == 0) {
            auto textrev_path = path+"/"+KEY_PREFIX+KEY_DOCPERM;
            LOG(INFO) << "CONSTRUCT " << KEY_DOCPERM;
            doc_perm dp;
            if (utils::file_exists(path+"/"+URLORDER) && utils::file_exists(path+"/"+DOCNAMES)) {
                dp.is_identity = false;
                // url-reordering
                std::ifstream ufs(path+"/"+URLORDER);
                std::unordered_map<std::string,uint64_t> id_mapping;
                std::ifstream dfs(path+"/"+DOCNAMES);
                std::string name_mapping;
                size_t j=0;
                while (std::getline(dfs,name_mapping)) {
                    id_mapping[name_mapping] = j;
                    j++;
                }
                size_t num_docs = j;
                dp.id2len = sdsl::int_vector<>(num_docs, 0, sdsl::bits::hi(num_docs)+1);
                dp.len2id = dp.id2len;
                /* load url sorted order */
                std::string url_mapping;
                j=0;
                while (std::getline(ufs,url_mapping)) {
                    auto doc_name = url_mapping.substr(url_mapping.find(' ')+1);
                    auto itr = id_mapping.find(doc_name);
                    if (itr != id_mapping.end()) {
                        dp.id2len[itr->second] = j;
                        dp.len2id[j] = itr->second;
                    } else {
                        LOG(ERROR) << "could not find mapping for '" << doc_name << "'";
                    }
                    j++;
                }
            } else {
                // identity permutation
                dp.is_identity = true;
            }
            sdsl::store_to_file(dp,docperm_path);
            file_map[KEY_DOCPERM] = docperm_path;
            LOG(INFO) << "DONE";
        }
        if (file_map.count(KEY_TEXTPERM) == 0) {
            doc_perm dp;
            sdsl::load_from_file(dp,file_map[KEY_DOCPERM]);
            if (dp.is_identity) {
                file_map[KEY_TEXTPERM] = file_map[KEY_TEXT];
            } else {
                LOG(INFO) << "CONSTRUCT " << KEY_TEXTPERM;
                sdsl::int_vector<> text;
                sdsl::load_from_file(text,file_map[KEY_TEXT]);
                sdsl::bit_vector doc_border(text.size(), 0);
                size_t num_docs = 0;
                for (uint64_t i=0; i < text.size(); ++i) {
                    if (1 == text[i]) {
                        doc_border[i] = 1;
                        num_docs++;
                    }
                }
                sdsl::select_support_mcl<1> doc_border_select(&doc_border);
                auto tp_path = path+"/"+KEY_PREFIX+KEY_TEXTPERM;
                sdsl::int_vector_buffer<> TPERM(tp_path,std::ios::out,1024*1024,text.width());
                size_t cur = 0;
                for (size_t i=0; i<num_docs; i++) {
                    auto mapped_id = dp.len2id[i];
                    size_t start,end;
                    if (mapped_id!=0) {
                        start = doc_border_select(mapped_id);
                        end = doc_border_select(mapped_id+1);
                    } else {
                        start = 0;
                        end = doc_border_select(1);
                    }
                    size_t doc_len = end-start;
                    for (size_t j=0; j<doc_len; j++) {
                        TPERM[cur++] = text[start+j];
                    }
                }
                TPERM[cur] = 0; // terminate to allow suffix sorting
                file_map[KEY_TEXTPERM] = tp_path;
                LOG(INFO) << "DONE";
            }
        }

        if (file_map.count(KEY_SA) == 0) {
            LOG(INFO) << "CONSTRUCT " << KEY_SA;
            sdsl::int_vector<> sa;
            sdsl::qsufsort::construct_sa(sa,file_map[KEY_TEXTPERM].c_str(),0);
            auto sa_path = path+"/"+KEY_PREFIX+KEY_SA;
            sdsl::store_to_file(sa,sa_path);
            file_map[KEY_SA] = sa_path;
        }
        if (file_map.count(KEY_C) == 0) {
            LOG(INFO) << "CONSTRUCT " << KEY_C;
            sdsl::int_vector_mapper<> text(file_map[KEY_TEXTPERM]);
            std::unordered_map<uint64_t,uint64_t> tmpC(5000000);
            for (uint64_t i=0; i < text.size(); ++i) {
                tmpC[text[i]] += 1;
            }
            sdsl::int_vector<> C(tmpC.size());
            for (const auto& p : tmpC) {
                C[p.first] = p.second;
            }
            sdsl::util::bit_compress(C);
            auto c_path = path+"/"+KEY_PREFIX+KEY_C;
            sdsl::store_to_file(C,c_path);
            file_map[KEY_C] = c_path;
            LOG(INFO) << "DONE";
        }
        if (file_map.count(KEY_CC) == 0) {
            LOG(INFO) << "CONSTRUCT " << KEY_CC;
            sdsl::int_vector_mapper<> text(file_map[KEY_TEXTPERM]);
            sdsl::int_vector_mapper<> C(file_map[KEY_C]);
            LOG(INFO) << "   - determine 2-token syms";
            std::unordered_map<uint64_t,uint64_t> tmpCC(50000000);
            for (uint64_t i=0; i < text.size()-1; ++i) {
                uint64_t sym = (text[i] << text.width()) + text[i+1];
                tmpCC[sym] += 1;
            }

            LOG(INFO) << "   - sort 2-token syms";
            sdsl::int_vector<> CC(tmpCC.size());
            size_t i = 0;
            for (const auto& p : tmpCC) {
                CC[i++] = p.first;
            }
            std::sort(CC.begin(),CC.end());
            {
                LOG(INFO) << "   - store syms";
                sdsl::util::bit_compress(CC);
                auto cc_path = path+"/"+KEY_PREFIX+KEY_CC;
                sdsl::store_to_file(CC,cc_path);
                file_map[KEY_CC] = cc_path;
            }
            {
                sdsl::int_vector<> SCC(tmpCC.size());
                for (size_t i=0; i<CC.size(); i++) {
                    auto cnt = tmpCC[CC[i]];
                    SCC[i] = cnt;
                }
                auto SCC_path = path+"/"+KEY_PREFIX+KEY_SCC;
                sdsl::util::bit_compress(SCC);
                sdsl::store_to_file(SCC,SCC_path);
                file_map[KEY_SCC] = SCC_path;
            }
            LOG(INFO) << "DONE";
        }
        if (file_map.count(KEY_DBV) == 0) {
            LOG(INFO) << "CONSTRUCT " << KEY_DBV;
            auto dbv_path = path+"/"+KEY_PREFIX+KEY_DBV;
            sdsl::int_vector_mapper<> text(file_map[KEY_TEXTPERM]);
            sdsl::bit_vector doc_border(text.size(), 0);
            for (uint64_t i=0; i < text.size(); ++i) {
                if (1 == text[i]) doc_border[i] = 1;
            }
            sdsl::store_to_file(doc_border,dbv_path);
            file_map[KEY_DBV] = dbv_path;
            LOG(INFO) << "DONE";
        }
        if (file_map.count(KEY_DOCLEN) == 0) {
            LOG(INFO) << "CONSTRUCT " << KEY_DOCLEN;
            auto doclen_path = path+"/"+KEY_PREFIX+KEY_DOCLEN;
            sdsl::bit_vector doc_border;
            sdsl::load_from_file(doc_border, file_map[KEY_DBV]);
            size_t num_docs = 0;
            for (uint64_t i=0; i < doc_border.size(); ++i) {
                if (doc_border[i] == 1) num_docs++;
            }
            size_t len = 1;
            size_t cnt = 0;
            sdsl::int_vector<> doc_lens(num_docs,0,sdsl::bits::hi(num_docs)+1);
            for (uint64_t i=0; i < doc_border.size(); ++i) {
                if (doc_border[i] == 1) {
                    doc_lens[cnt] = len;
                    len = 1;
                    cnt++;
                } else {
                    len++;
                }
            }
            sdsl::store_to_file(doc_lens,doclen_path);
            file_map[KEY_DOCLEN] = doclen_path;
            LOG(INFO) << "DONE";
        }
        if (file_map.count(KEY_POSPL) == 0) {
            auto pospl_path = path+"/"+KEY_PREFIX+KEY_POSPL;
            LOG(INFO) << "CONSTRUCT " << KEY_POSPL;
            sdsl::int_vector_mapper<> SA(file_map[KEY_SA]);
            sdsl::int_vector_mapper<> C(file_map[KEY_C]);
            sdsl::int_vector_buffer<> oPOSPL(pospl_path,std::ios::out,1024*1024,SA.width());
            size_t cum_sum = 0;
            for (size_t i=0; i<C.size(); i++) {
                auto start = SA.begin()+cum_sum;
                auto end = start + C[i];
                std::vector<uint64_t> tmp(start,end);
                std::sort(tmp.begin(),tmp.end());
                for (size_t j=0; j<C[i]; j++) {
                    oPOSPL[cum_sum+j] = tmp[j];
                }
                cum_sum += C[i];
            }
            file_map[KEY_POSPL] = pospl_path;
            LOG(INFO) << "DONE";
        }
        if (file_map.count(KEY_D) == 0) {
            auto docpl_path = path+"/"+KEY_PREFIX+KEY_D;
            LOG(INFO) << "CONSTRUCT " << KEY_D;
            sdsl::int_vector_mapper<> SA(file_map[KEY_SA]);
            sdsl::bit_vector doc_border;
            sdsl::load_from_file(doc_border, file_map[KEY_DBV]);
            sdsl::rank_support_v<> doc_border_rank(&doc_border);
            uint64_t doc_cnt = doc_border_rank(doc_border.size());
            sdsl::int_vector_buffer<> D(docpl_path,std::ios::out,1024*1024,sdsl::bits::hi(doc_cnt)+1);
            for (size_t i=0; i<SA.size(); i++) {
                D[i] = doc_border_rank(SA[i]);
            }
            file_map[KEY_D] = docpl_path;
            LOG(INFO) << "DONE";
        }
    }
};