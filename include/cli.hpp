#ifndef CLI_HPP
#define CLI_HPP


#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <iostream>

namespace cli {

struct Parser {
  std::unordered_map<std::string,std::string> kv;
  std::unordered_set<std::string> flags;
  std::unordered_map<std::string,std::string> alias; // "-s" -> "size"
  std::unordered_set<std::string> takes_value;       // canonical names that expect a value

  static bool is_flag(const char* s){ return s && s[0]=='-' && s[1]; }

  void add_alias(const std::string& short_opt, const std::string& long_opt){
    alias[short_opt] = long_opt;
  }
  
  void add_value_opt(const std::string& long_opt){
    takes_value.insert(long_opt);
  }

  void parse(int argc, char** argv){
    for (int i=1; i<argc; ++i){
      std::string tok = argv[i];
      if (!is_flag(argv[i])) continue;
      // normalize: short -> long if alias exists
      auto canon = [&](const std::string& k)->std::string{
        auto it = alias.find(k);
        return it==alias.end()? k : it->second;
      };
      // handle --long or --long=value
      if (tok.rfind("--",0)==0){
        auto eq = tok.find('=');
        std::string key = (eq==std::string::npos)? tok.substr(2) : tok.substr(2, eq-2);
        // accept legacy names with spaces/underscores by normalizing to '-'
        for (char& c: key){ if (c==' '||c=='_') c='-'; }
        key = canon("--"+key);
        key = (key.rfind("--",0)==0)? key.substr(2) : key;
        if (eq!=std::string::npos){
          kv[key] = tok.substr(eq+1);
        } else if (takes_value.count(key) && i+1<argc && !is_flag(argv[i+1])) {
          kv[key] = argv[++i];
        } else {
          flags.insert(key);
        }
      }
      // handle -s (single short) possibly with value
      else if (tok.rfind("-",0)==0){
       // we do not support short-option clustering (-abc); keep simple & robust
        std::string k = canon(tok);
        if (k.rfind("--",0)==0) k = k.substr(2);
        else if (k.rfind("-",0)==0) k = k.substr(1);
        if (takes_value.count(k) && i+1<argc && !is_flag(argv[i+1])){
          kv[k] = argv[++i];
        } else {
          flags.insert(k);
        }
      }
    }
  }

  bool has(const std::string& k) const {
    return flags.count(k) || kv.count(k);
  }

  template <class T>
  T get(const std::string& k, T def) const {
    auto it = kv.find(k);
    if (it==kv.end()) return def;
    std::istringstream ss(it->second);
    T out{}; ss >> out;
    if (!ss.fail()) return out;
    return def;
  }

  std::string get(const std::string& k, const std::string& def) const {
    auto it = kv.find(k);
    return it==kv.end()? def : it->second;
  }
};

} // namespace cli


#endif