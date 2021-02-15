#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>


class progress_bar
{
    static const auto overhead = sizeof " [100%]";

    std::ostream& os;
    const std::size_t bar_width;
    std::string message;
    const std::string full_bar;

 public:
    progress_bar(std::ostream& os, std::size_t line_width,
                 std::string message_, const char symbol = '.')
        : os{os},
          bar_width{line_width - overhead},
          message{std::move(message_)},
          full_bar{std::string(bar_width, symbol) + std::string(bar_width, ' ')}
    {
        if (message.size()+1 >= bar_width || message.find('\n') != message.npos) {
            os << message << '\n';
            message.clear();
        } else {
            message += ' ';
        }
        write(0.0);
    }

    // not copyable
    progress_bar(const progress_bar&) = delete;
    progress_bar& operator=(const progress_bar&) = delete;

    ~progress_bar()
    {
        write(1.0);
        os << '\n';
    }

    void write(double fraction);
};


void progress_bar::write(double fraction)
{
    // clamp fraction to valid range [0,1]
    if (fraction < 0)
        fraction = 0;
    else if (fraction > 1)
        fraction = 1;

    auto width = bar_width - message.size();
    auto offset = bar_width - static_cast<unsigned>(width * fraction);

    os << '\r' << message;
    os.write(full_bar.data() + offset, width);
    os << " [" << std::setw(3) << static_cast<int>(100*fraction) << "%] " << std::flush;
}


int main(int argc, char **argv)
{
    std::string tmp;
    const int update_period = 1000000;
    int max = std::stoi(std::string(argv[4]));
    std::ifstream ifs(argv[1]);
    int num_lines = 0;
    while(std::getline(ifs, tmp))
        ++num_lines;
    ifs.clear();
    ifs.seekg(0);
    std::ofstream ofs(argv[2]), gfs(argv[3]);
    progress_bar progress{std::clog, 70u, "Working"};
    int i = 0;
    while(std::getline(ifs, tmp)){
        if(std::count(tmp.begin(), tmp.end(), ' ') > max)
            gfs << tmp << '\n';
        else
            ofs << tmp << '\n';
        ++i;
        if(i % update_period == 0)
            progress.write(i / float(num_lines));
    }
    gfs.close();
    ifs.close();
    ofs.close();
}
