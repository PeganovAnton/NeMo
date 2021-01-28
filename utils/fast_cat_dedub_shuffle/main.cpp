#include <fstream>
#include <iostream>
#include <unordered_set>
#include <string>
#include <utility>
#include <vector>


using namespace std;


int main(int argc, char **argv)
{
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
