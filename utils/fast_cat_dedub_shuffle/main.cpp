#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <unordered_set>
#include <string>
#include <utility>
#include <vector>


using namespace std;


int main(int argc, char **argv)
{
/*    const std::locale empty_locale = std::locale::empty();
    typedef std::codecvt_utf8<wchar_t> converter_type;
    const converter_type* converter = new converter_type;
    const std::locale utf8_locale = std::locale(empty_locale, converter);*/
    string tmp;
    unordered_set<string> data;
    int tnu = 0;
    for(int i=2; i<argc; ++i){
        cout << argv[i] << '\n';
        ifstream ifs(argv[i]);
        int j = 0;
        while(getline(ifs, tmp)){
            ++j;
            data.insert(move(tmp));
        }
        int nu = data.size() - tnu;
        cout << j << " sentences were found in file " << argv[i] << '\n';
        cout << nu << " unique sentences were found in file " << argv[i] << '\n';
        cout << "Total number of unique sentences: " << data.size() << "\n\n";
        tnu = data.size();
        ifs.close();
    }
    ofstream ofs(argv[1]);
    for(auto s : data)
        ofs << s << '\n';
    ofs.close();
}
