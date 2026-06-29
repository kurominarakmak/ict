#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace {

constexpr long long GAP_SECONDS = 30LL * 60LL;
constexpr int TIMEFRAME_MINUTES = 15;

struct Bar {
    long long start_epoch = 0;
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
    long long ticks = 0;
    long long buy = 0;
    long long sell = 0;
    long long neutral = 0;
    int segment = 0;
    bool invalid = false;
    bool active = false;
};

long long days_from_civil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097LL + static_cast<long long>(doe) - 719468LL;
}

int parse_int(const std::string& s, size_t pos, size_t len) {
    int out = 0;
    for (size_t i = pos; i < pos + len; ++i) {
        out = out * 10 + (s[i] - '0');
    }
    return out;
}

long long parse_epoch_seconds(const std::string& ts) {
    const int y = parse_int(ts, 0, 4);
    const int mo = parse_int(ts, 4, 2);
    const int d = parse_int(ts, 6, 2);
    const int h = parse_int(ts, 9, 2);
    const int mi = parse_int(ts, 12, 2);
    const int se = parse_int(ts, 15, 2);
    return days_from_civil(y, static_cast<unsigned>(mo), static_cast<unsigned>(d)) * 86400LL
        + h * 3600LL + mi * 60LL + se;
}

long long floor_m15(long long epoch) {
    return epoch - (epoch % (TIMEFRAME_MINUTES * 60LL));
}

void flush(std::ofstream& out, Bar& bar) {
    if (!bar.active) {
        return;
    }
    if (!bar.invalid) {
        const long long delta = bar.buy - bar.sell;
        const long long denom = bar.buy + bar.sell;
        const double ratio = denom == 0 ? 0.0 : static_cast<double>(delta) / static_cast<double>(denom);
        out << bar.start_epoch << ','
            << bar.segment << ','
            << std::fixed << std::setprecision(6)
            << bar.open << ','
            << bar.high << ','
            << bar.low << ','
            << bar.close << ','
            << bar.ticks << ','
            << bar.buy << ','
            << bar.sell << ','
            << bar.neutral << ','
            << delta << ','
            << ratio << '\n';
    }
    bar = Bar{};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: fast_m15_delta_bars INPUT_TICK_CSV OUTPUT_BAR_CSV\n";
        return 2;
    }

    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "failed to open input: " << argv[1] << "\n";
        return 1;
    }
    std::ofstream out(argv[2]);
    if (!out) {
        std::cerr << "failed to open output: " << argv[2] << "\n";
        return 1;
    }
    out << "start_epoch,segment_id,open,high,low,close,ticks,buy_ticks,sell_ticks,neutral_ticks,delta,delta_ratio\n";

    std::string line;
    std::getline(in, line);  // header

    Bar current;
    long long previous_epoch = std::numeric_limits<long long>::min();
    double previous_mid = std::numeric_limits<double>::quiet_NaN();
    int segment = 0;
    long long rows = 0;

    while (std::getline(in, line)) {
        const size_t c1 = line.find(',');
        if (c1 == std::string::npos || c1 < 17) {
            continue;
        }
        const size_t c2 = line.find(',', c1 + 1);
        if (c2 == std::string::npos) {
            continue;
        }
        const size_t c3 = line.find(',', c2 + 1);
        const char* bid_start = line.c_str() + c1 + 1;
        char* bid_end = nullptr;
        const double bid = std::strtod(bid_start, &bid_end);
        const char* ask_start = line.c_str() + c2 + 1;
        char* ask_end = nullptr;
        const double ask = std::strtod(ask_start, &ask_end);
        if (!std::isfinite(bid) || !std::isfinite(ask) || ask <= bid || bid <= 0.0) {
            continue;
        }

        const long long epoch = parse_epoch_seconds(line);
        const long long bucket = floor_m15(epoch);
        const double mid = (bid + ask) / 2.0;
        const bool have_prev = previous_epoch != std::numeric_limits<long long>::min();
        const bool gap = have_prev && (epoch - previous_epoch > GAP_SECONDS);

        if (gap) {
            if (current.active && current.start_epoch == bucket) {
                current.invalid = true;
            }
            flush(out, current);
            ++segment;
            previous_mid = std::numeric_limits<double>::quiet_NaN();
        }
        if (current.active && current.start_epoch != bucket) {
            flush(out, current);
        }

        int side = 0;
        if (std::isfinite(previous_mid)) {
            if (mid > previous_mid) {
                side = 1;
            } else if (mid < previous_mid) {
                side = -1;
            }
        }

        if (!current.active) {
            current.active = true;
            current.start_epoch = bucket;
            current.open = mid;
            current.high = mid;
            current.low = mid;
            current.close = mid;
            current.ticks = 1;
            current.buy = side > 0 ? 1 : 0;
            current.sell = side < 0 ? 1 : 0;
            current.neutral = side == 0 ? 1 : 0;
            current.segment = segment;
            current.invalid = false;
        } else {
            current.high = std::max(current.high, mid);
            current.low = std::min(current.low, mid);
            current.close = mid;
            ++current.ticks;
            current.buy += side > 0 ? 1 : 0;
            current.sell += side < 0 ? 1 : 0;
            current.neutral += side == 0 ? 1 : 0;
        }

        previous_epoch = epoch;
        previous_mid = mid;
        ++rows;
        if (rows % 50000000LL == 0) {
            std::cerr << "processed_rows=" << rows << "\n";
        }
        (void)c3;
    }
    flush(out, current);
    std::cerr << "processed_rows=" << rows << "\n";
    return 0;
}
