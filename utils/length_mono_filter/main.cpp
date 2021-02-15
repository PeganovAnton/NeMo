#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>


using namespace std;


int main(int argc, char **argv)
{
    string tmp;
    unordered_set<string> data;
    int max = stoi(string(argv[4]));
    ifstream ifs(argv[1]);
    ofstream ofs(argv[2]), gfs(argv[3]);
    while(getline(ifs, tmp)){
        if(count(tmp.begin(), tmp.end(), ' ') > max)
            gfs << tmp << '\n';
        else
            ofs << tmp << '\n';
    }
    gfs.close()
    ifs.close()
    ofs.close();
}
