#include "korean.h"

#include <cctype>
#include <format>
#include <iterator>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <sstream>
#include "text_normalization/text_normalization_eng.h"

namespace melo {


static const char* CHOSEONG[] =
    {"ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"};
static const char* JUNGSEONG[] = {"ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ",
                                  "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"};
static const char* JONGSEONG[] = {"",   "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
                                  "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"};

std::vector<std::string> decompose_hangul(const std::string& text) {
    std::vector<std::string> result;
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = text[i];
        // UTF-8 3바이트 한글 음절 처리
        if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            int unicode = ((text[i] & 0x0F) << 12) | ((text[i + 1] & 0x3F) << 6) | (text[i + 2] & 0x3F);
            if (unicode >= 0xAC00 && unicode <= 0xD7A3) {
                int SIndex = unicode - 0xAC00;
                int choseong = SIndex / (21 * 28);
                int jungseong = (SIndex % (21 * 28)) / 28;
                int jongseong = SIndex % 28;
                result.push_back(CHOSEONG[choseong]);
                result.push_back(JUNGSEONG[jungseong]);
                if (jongseong != 0)
                    result.push_back(JONGSEONG[jongseong]);
                i += 3;
                continue;
            }
        }
        // 한글이 아니면 그대로 추가
        result.push_back(std::string(1, text[i]));
        i += 1;
    }
    return result;
}

Korean::Korean(std::unique_ptr<ov::Core>& core_ptr, const std::filesystem::path& data_folder) {
    // 한국어 관련 리소스 초기화 (필요시)
    auto cmudict_path = data_folder / "cmudict_cache.txt";

    if (!std::filesystem::exists(cmudict_path)) {
        std::cerr << "[ERROR] Korean::file does not exists: " << std::filesystem::absolute(cmudict_path) << "\n";
    } else {
        cmudict = std::make_shared<CMUDict>(cmudict_path.string());
        std::cout << "[INFO] Korean::Init Korean language Module Succeed!\n";
    }

    // Init mini-bart g2p TODO: use the stateful model with kv cache
    auto bart_g2p_path = data_folder / "mini-bart-g2p-no_cache";
    if (!std::filesystem::exists(bart_g2p_path)) {
        std::cerr << "[ERROR] Korean::file does not exists: " << std::filesystem::absolute(bart_g2p_path) << "\n";
    } else {
        bart_g2p = std::make_shared<MiniBartG2P>(core_ptr, bart_g2p_path, "CPU", false);
        std::cout << "[INFO] Korean:: Init MiniBartG2P Succeed!\n";
    }
}

std::vector<std::string> split(std::string str, char Delimiter) {
    std::istringstream iss(str);  // istringstream에 str을 담는다.
    std::string buffer;           // 구분자를 기준으로 절삭된 문자열이 담겨지는 버퍼

    std::vector<std::string> result;

    // istringstream은 istream을 상속받으므로 getline을 사용할 수 있다.
    while (getline(iss, buffer, Delimiter)) {
        result.push_back(buffer);  // 절삭된 문자열을 vector에 저장
    }

    return result;
}

// 텍스트를 음소/음절 단위로 변환
std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> Korean::g2p(
    const std::string& sentence,
    std::shared_ptr<OpenVinoTokenizer>& tokenizer) {
   
    std::vector<int> word2ph;
    std::vector<std::string> phs;

    //std::vector<std::string> tokenized = tokenizer->word_segment(sentence);
    std::vector<std::string> tokenized = split(sentence, ' ');
    std::vector<std::vector<std::string>> ph_groups;
    for (auto& token : tokenized) {
        if (token.empty() == true)
            continue;

        if (token.front() == '#') {
            if (!ph_groups.size()) {
                std::cerr << "[ERROR] Korean::g2p: Suffix should has Prefix\n";
                continue;
            }
            ph_groups.back().emplace_back(token.substr(2));
        } else
            ph_groups.push_back({token});
    }
    for (auto& group : ph_groups) {
        std::string text = std::accumulate(group.begin(), group.end(), std::string{});
        if (text == "[UNK]") {
            phs.push_back("_");
            word2ph.push_back(1);
            continue;
        }
        // 필요시 punctuation 체크 추가
        static const std::unordered_set<std::string> punctuation = {",", ".", "!", "?", "-", "'", "\"", "…"};
        if (punctuation.count(text)) {
            phs.push_back(text);
            word2ph.push_back(1);
            continue;
        }

        // 한글 음소 변환 (bart_g2p 또는 cmudict 사용)
        std::vector<std::string> phonemes = decompose_hangul(text);
        //std::vector<std::string> phonemes;
        //auto syllables = cmudict->find(text);
        //if (syllables.has_value()) {
        //    auto [phones, tones] = refine_syllables(syllables.value().get());
        //    phonemes = phones;
        //} else {
        //    auto bart_result = bart_g2p->forward(text);
        //    if (!bart_result.empty()) {
        //        auto [phones, tones] = refine_syllables(bart_result);
        //        phonemes = phones;
        //    }
        //}
        int phone_len = static_cast<int>(phonemes.size());
        int word_len = static_cast<int>(group.size());
        auto aaa = distribute_phone(phone_len, word_len);
        word2ph.insert(word2ph.end(), aaa.begin(), aaa.end());
        phs.insert(phs.end(), phonemes.begin(), phonemes.end());
    }


    // 한글 자모 분해 (예시)
    //auto phonemes = decompose_hangul(sentence);
    //for (const auto& ph : phonemes) {
    //    int phone_len = static_cast<int>(phonemes.size());
    //    int word_len = static_cast<int>(sentence.size());
    //    auto aaa = distribute_phone(phone_len, word_len);
    //    word2ph.insert(word2ph.end(), aaa.begin(), aaa.end());
    //    phs.insert(phs.end(), phonemes.begin(), phonemes.end());
    //}


    std::vector<std::string> phones = {"_"};
    phones.insert(phones.end(), phs.begin(), phs.end());
    phones.push_back("_");
    std::vector<int64_t> tones(phones.size(), 0);
    std::vector<int> word2ph_full = {1};
    word2ph_full.insert(word2ph_full.end(), word2ph.begin(), word2ph.end());
    word2ph_full.push_back(1);
     
    return {phones, tones, word2ph_full};
}

int64_t Korean::symbol_to_id(const std::string& symbol) {
    static const std::unordered_map<std::string, std::string> map = {
        {"ㄱ", "ᄀ"}, {"ㄲ", "ᄁ"}, {"ㄴ", "ᄂ"}, {"ㄷ", "ᄃ"}, {"ㄸ", "ᄄ"}, {"ㄹ", "ᄅ"}, {"ㅁ", "ᄆ"},
        {"ㅂ", "ᄇ"}, {"ㅃ", "ᄈ"}, {"ㅅ", "ᄉ"}, {"ㅆ", "ᄊ"}, {"ㅇ", "ᄋ"}, {"ㅈ", "ᄌ"}, {"ㅉ", "ᄍ"},
        {"ㅊ", "ᄎ"}, {"ㅋ", "ᄏ"}, {"ㅌ", "ᄐ"}, {"ㅍ", "ᄑ"}, {"ㅎ", "ᄒ"}, {"ㅏ", "ᅡ"},  {"ㅐ", "ᅢ"},
        {"ㅑ", "ᅣ"},  {"ㅒ", "ᅤ"},  {"ㅓ", "ᅥ"},  {"ㅔ", "ᅦ"},  {"ㅕ", "ᅧ"},  {"ㅖ", "ᅨ"},  {"ㅗ", "ᅩ"},
        {"ㅘ", "ᅪ"},  {"ㅙ", "ᅫ"},  {"ㅚ", "ᅬ"},  {"ㅛ", "ᅭ"},  {"ㅜ", "ᅮ"},  {"ㅝ", "ᅯ"},  {"ㅞ", "ᅰ"},
        {"ㅟ", "ᅱ"},  {"ㅠ", "ᅲ"},  {"ㅡ", "ᅳ"},  {"ㅢ", "ᅴ"},  {"ㅣ", "ᅵ"},  {"", ""},  // 종성 없음
        {"ㄱ", "ᆨ"},  {"ㄲ", "ᆩ"},  {"ㄳ", "ᆪ"},  {"ㄴ", "ᆫ"},  {"ㄵ", "ᆬ"},  {"ㄶ", "ᆭ"},  {"ㄷ", "ᆮ"},
        {"ㄹ", "ᆯ"},  {"ㄺ", "ᆰ"},  {"ㄻ", "ᆱ"},  {"ㄼ", "ᆲ"},  {"ㄽ", "ᆳ"},  {"ㄾ", "ᆴ"},  {"ㄿ", "ᆵ"},
        {"ㅀ", "ᆶ"},  {"ㅁ", "ᆷ"},  {"ㅂ", "ᆸ"},  {"ㅄ", "ᆹ"},  {"ㅅ", "ᆺ"},  {"ㅆ", "ᆻ"},  {"ㅇ", "ᆼ"},
        {"ㅈ", "ᆽ"},  {"ㅊ", "ᆾ"},  {"ㅋ", "ᆿ"},  {"ㅌ", "ᇀ"},  {"ㅍ", "ᇁ"},  {"ㅎ", "ᇂ"}};

    auto it = map.find(symbol);
    if (it != map.end()) {
        auto findSymbol = symbol_to_id_mp.find(it->second);
        if (findSymbol == symbol_to_id_mp.end())
            return 0;

        return findSymbol->second;
    }
    auto findSymbol = symbol_to_id_mp.find(symbol);
    if (findSymbol == symbol_to_id_mp.end())
        return 0;

    return findSymbol->second;
}



// 텍스트 정규화 (예시: 대소문자 변환, 특수문자 제거 등)
std::string Korean::text_normalize(const std::string& text) {
    std::string norm_text = text;
    std::for_each(norm_text.begin(), norm_text.end(), [](auto& ch) {
        if (ch <= 'Z' && ch >= 'A')
            ch = ch + 'a' - 'A';
    });
    norm_text = text_normalization::expand_time_english(norm_text);
    norm_text = text_normalization::expand_abbreviations(norm_text);
    norm_text = text_normalization::normalize_numbers(norm_text);
    return norm_text;
}

// 필요시 refine_syllables, distribute_phone 등 AbstractLanguageModule 함수 override

}  // namespace melo