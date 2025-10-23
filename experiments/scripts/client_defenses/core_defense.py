# stdlib
from copy import deepcopy
import random
import time
from typing import Optional, Union

# third party
import numpy as np
from pydantic import BaseModel

binary_exts = tuple(["gif", "png", "svg", "jpg", "jpeg", "pdf", "webp"])
user_agents = [
    # Desktop Browsers
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.199 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5249.119 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_6_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.63 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0",
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:101.0) Gecko/20100101 Firefox/101.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/92.0.902.67 Safari/537.36",
    # Mobile Browsers
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; U; Android 9; en-us; Redmi Note 7 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 10; Mi 9T) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.199 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; U; Android 8.1; en-us; vivo 1906) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/90.0.4430.210 Mobile Safari/537.36",
    "Mozilla/5.0 (iPad; CPU OS 13_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.6 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 12; SM-A528B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.5790.102 Mobile Safari/537.36",
    # Bots and Crawlers
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Mozilla/5.0 (compatible; Bingbot/2.0; +http://www.bing.com/bingbot.htm)",
    "Mozilla/5.0 (compatible; Yahoo! Slurp; http://help.yahoo.com/help/us/ysearch/slurp)",
    "DuckDuckBot/1.0; (+http://duckduckgo.com/duckduckbot.html)",
    "Mozilla/5.0 (compatible; Baiduspider/2.0; +http://www.baidu.com/search/spider.html)",
    "Mozilla/5.0 (compatible; YandexBot/3.0; +http://yandex.com/bots)",
    "Mozilla/5.0 (Linux; Android 8.1.0; Googlebot/2.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/112.0.5615.137 Safari/537.36",
    "Twitterbot/1.0",
    "Mozilla/5.0 (compatible; archive.org_bot +http://www.archive.org/details/archive.org_bot)",
    # Miscellaneous
    "Opera/9.80 (Windows NT 6.1; WOW64) Presto/2.12.388 Version/12.18",
    "Mozilla/5.0 (Nintendo 3DS; U; ; en) Version/1.7412.EU",
    "Mozilla/5.0 (PlayStation 4 3.11) AppleWebKit/601.1 (KHTML, like Gecko)",
    "Mozilla/5.0 (PlayStation Vita 3.61) AppleWebKit/537.73 (KHTML, like Gecko) Silk/3.2",
    "Mozilla/5.0 (CrKey armv7l) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; ARM64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.5563.146 Safari/537.36",
    "Mozilla/5.0 (Linux; U; Android 11; en-US; SM-N986U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.131 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Linux; Android 11; SM-T510) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:88.0) Gecko/20100101 Firefox/88.0",
    # Short User Agents
    "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1)",
    "Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11",
    "Opera/9.80 (Android; Opera Mini/36.2.2254/191.298; U; en) Presto/2.12.423 Version/12.16",
    "Mozilla/5.0 (X11; U; SunOS i86pc) Gecko/20071127 Firefox/3.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/534.55.3 (KHTML, like Gecko) Version/5.1.5 Safari/534.55.3",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 Safari/534.16",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 6.0)",
    "Mozilla/5.0 (compatible; Konqueror/3.5; Linux 2.6.13-15) KHTML/3.5.4 (like Gecko)",
    "Mozilla/5.0 (Linux; Android 9; SAMSUNG SM-N950U) AppleWebKit/537.36 (KHTML, like Mozilla/5.0 (Linux; Android 9; SAMSUNG SM-N950U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.105 Mobile Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/533.21.1 (KHTML, like Gecko) Version/5.0.5 Safari/533.21.1",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)",
    "Mozilla/5.0 (X11; FreeBSD amd64; rv:86.0) Gecko/20100101 Firefox/86.0",
    "Mozilla/5.0 (Linux; Android 7.1.2; Nexus 5X Build/NJH47F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.116 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Brave/1.29.81 Chrome/91.0.4472.124 Safari/537.36",
]


class DEFENSE(BaseModel):
    name: str
    initial_window_size_strategy: Union[
        str, int
    ] = "disabled"  # disabled, constant, random
    recv_delay_threshold: Optional[int] = None
    recv_interval_strategy: Union[
        str, float
    ] = "disabled"  # disabled, random_per_connection, random_per_frame
    send_packet_size_strategy: Union[
        str, int
    ] = "disabled"  # disabled, random_per_connection, random_per_frame
    send_dummy_packet_strategy: Union[
        str, float
    ] = "disabled"  # disabled, random_per_frame, random_per_connection, random_batch, front, llama
    send_dummy_packet_interval: Union[
        str, float
    ] = "disabled"  # disabled, random_per_frame, random_per_connection, front, llama
    send_dummy_packet_limit: int = (0,)
    send_dummy_packet_loop: bool = (False,)

    random_user_agent: bool = False
    random_pings: bool = (
        False  # Send random number of PING frames to simulate padding both ways
    )
    request_delay: bool = False
    request_batch: bool = False
    request_shuffle: bool = False
    ranged_requests: bool = False
    added_frames: int = 0  # how many frames we added to responses
    added_delay: float = 0  # how many ms we added to responses
    last_dummy_packet: float = time.time()
    # adaptive controls
    recv_padding: bool = True
    dummy_loop_enabled: bool = True
    send_dummy_min: int = 1
    send_dummy_max: int = 200

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        self._cache = {
            "recv_interval": None,
            "send_packet_size": None,
            "send_dummy": None,
            "send_dummy_interval": None,
            "front": None,
            "recv_bytes_cumul": 0,
        }

    def summary(self) -> str:
        return f"""
            Defense summary:
                Name : {self.name}
                RECV Interval strategy : {self.recv_interval_strategy}
                RECV Padding : {self.recv_padding}
                SEND packet size : {self.send_packet_size_strategy}
                SEND dummy packets strategy : {self.send_dummy_packet_strategy}

        """

    def initial_window_size(self) -> int:
        if isinstance(self.initial_window_size_strategy, int):
            return self.initial_window_size_strategy
        if self.initial_window_size_strategy == "disabled":
            return 16384
        elif self.initial_window_size_strategy == "random":
            # Random Window Size
            return int(random.randint(2**10, 2**14))
        else:
            raise NotImplementedError()

    def recv_interval(
        self, stream_stats=None, cumul_data: Optional[int] = None, max_delay=0.2
    ) -> float:
        def _should_accumulate():
            if self.recv_delay_threshold is None or cumul_data is None:
                return False

            cache_key = "recv_bytes_cumul"
            self._cache[cache_key] += cumul_data

            # print("DBG thresh", self._cache[cache_key], self.recv_delay_threshold)
            if self._cache[cache_key] >= self.recv_delay_threshold:
                self._cache[cache_key] = 0
                self.recv_delay_threshold = random.randint(2**10, 2**12)
                return False

            return True

        # HTTP2 DATA RECV frame delays
        if self.recv_interval_strategy == "disabled":
            return 0
        elif self.recv_interval_strategy == "random_per_connection":

            if self.recv_delay_threshold is None:
                self.recv_delay_threshold = random.randint(2**9, 2**12)

            if _should_accumulate():
                return 0

            # random and cache
            cache_key = "recv_interval"
            if self._cache[cache_key] is None:
                self._cache[cache_key] = random.uniform(0, max_delay)
            return self._cache[cache_key]
        elif self.recv_interval_strategy == "random_per_frame":
            if self.recv_delay_threshold is None:
                self.recv_delay_threshold = random.randint(2**9, 2**12)

            if _should_accumulate():
                return 0

            # random on every call
            delay = random.uniform(0, max_delay)
            # print("DBG delay", delay)
            return delay
        elif isinstance(self.recv_interval_strategy, float):
            return self.recv_interval_strategy
        if isinstance(self.recv_interval_strategy, float):
            if _should_accumulate():
                return 0
            return self.recv_interval_strategy
        else:
            return 0

    def send_packet_size(self) -> int:
        if self.send_packet_size_strategy == "disabled":
            return 0
        elif self.send_packet_size_strategy == "random_per_connection":
            cache_key = "send_packet_size"
            if self._cache[cache_key] is None:
                self._cache[cache_key] = 2 ** random.randint(4, 9)
            return self._cache[cache_key]
        elif self.send_packet_size_strategy == "random_per_frame":
            return 2 ** random.randint(4, 9)
        elif isinstance(self.send_packet_size_strategy, int):
            return self.send_packet_size_strategy
        else:
            raise NotImplementedError()

    def send_dummy_interval(self, max_delay: float = 0.1) -> float:
        if self.send_dummy_packet_interval in [
            "disabled",
            "front",
            "llama",
        ]:
            return 0
        if self.send_dummy_packet_interval == "random_per_connection":
            # random and cache
            cache_key = "send_dummy_interval"
            if self._cache[cache_key] is None:
                self._cache[cache_key] = random.uniform(0, max_delay)
            return self._cache[cache_key]
        elif self.send_dummy_packet_interval == "random_per_frame":
            # random on every call
            return random.uniform(0, max_delay)
        elif isinstance(self.send_dummy_packet_interval, float):
            return self.send_dummy_packet_interval
        else:
            raise NotImplementedError()

    def send_dummy_packet(
        self, pool: list, stream_stats=None, window_size=None, previous_request=None
    ) -> Optional:
        if len(pool) == 0:
            return None

        def is_binary(req):
            req_path = req.path.lower().split("?")[0]
            return req_path.endswith(binary_exts)

        if self.send_dummy_packet_strategy == "disabled":
            return None
        elif self.send_dummy_packet_strategy == "random_per_connection":
            cache_key = "send_dummy"
            if self._cache[cache_key] is None:
                self._cache[cache_key] = random.choice(pool)
            return [self._cache[cache_key]]
        elif self.send_dummy_packet_strategy == "random_per_frame":
            should_sample = random.choice([True, False])
            if not should_sample:
                return None
            return [random.choice(pool)]
        elif self.send_dummy_packet_strategy == "random_batch":
            if self.send_dummy_packet_limit <= 0:
                return None
            should_sample = random.randint(1, self.send_dummy_packet_limit)
            if should_sample == 0:
                return None
            print("[DBG] Sending dummies !!! ", should_sample)
            return random.choices(pool, k=should_sample)
        elif self.send_dummy_packet_strategy == "llama":
            if (
                previous_request is None
            ):  # LLaMA sends dummy request only after a real HTTP request
                return None

            DUMMY_PROBABILITY = (
                0.3  # Probability of sending a dummy request after a real one
            )
            if random.random() < DUMMY_PROBABILITY:
                return [random.choice(pool)]
            else:
                return None

        elif self.send_dummy_packet_strategy == "front":
            if self._cache["front"] is None:
                # Parameters
                Nc = self.send_dummy_max  # Max dummy packets client sends
                nc = np.random.randint(
                    self.send_dummy_min, Nc + 1
                )  # Client-side dummy packets
                print(f"[FRONT] Sampled {nc} with limit {Nc}")

                Wmin = 0.0  # Minimum padding window size (seconds)
                Wmax = 1.0  # Maximum padding window size (seconds)
                wc = np.random.uniform(Wmin, Wmax)  # Client padding window

                self._cache["front"] = {
                    "timestamps": np.sort(np.random.rayleigh(scale=wc, size=nc)),
                    "first_ts": time.time(),
                }
                print(
                    "[DEFENSE] FRONT remaining timestamps",
                    len(self._cache["front"]["timestamps"]),
                )
            # print("[DEFENSE] FRONT", self._cache["front"])
            if len(self._cache["front"]["timestamps"]) > 0:
                next_ts = self._cache["front"]["timestamps"][0]
                if time.time() - self._cache["front"]["first_ts"] >= next_ts:
                    self._cache["front"]["timestamps"] = self._cache["front"][
                        "timestamps"
                    ][1:]

                    return [random.choice(pool)]
            return []
        else:
            raise NotImplementedError()

    def should_batch(self):
        return self.request_batch

    def should_shuffle(self):
        return self.request_shuffle

    def should_delay_request(self):
        if not self.request_delay:
            return 0

        MAX_DELAY = 0.02
        DUMMY_PROBABILITY = (
            0.3  # Probability of sending a dummy request after a real one
        )
        if random.random() < DUMMY_PROBABILITY:
            return random.uniform(0, MAX_DELAY)
        else:
            return 0

    def user_agent(self, is_noise: bool = False):
        if self.random_user_agent or is_noise:
            return random.choice(user_agents)
        else:
            return "WF Defense"

    def should_send_random_pings(self):
        if not self.random_pings:
            return 0

        return random.randint(1, 20)

    def use_ranged_requests(self):
        return self.ranged_requests

    def _random_partition(self, N, X):
        min_value = 10
        remaining_N = N - X * min_value

        # Generate X-1 random cut points in the range [0, N]
        try:
            cuts = np.sort(np.random.randint(0, remaining_N, X - 1))
        except BaseException:
            return None

        # Include 0 and N as the boundaries
        cuts = np.concatenate(([0], cuts, [remaining_N]))

        # The differences between consecutive points will be the random numbers
        return np.diff(cuts) + min_value

    def split_for_ranged_requests(self, request, with_overlap=True):
        output = []

        expected_data_size = request.expected_size
        if expected_data_size is None or expected_data_size < 1000:
            return [request]

        if not isinstance(expected_data_size, int):
            return [request]

        req_path = request.path.lower().split("?")[0]
        is_binary = req_path.endswith(binary_exts)
        if not is_binary:
            print("[RANGED] IGNORE nonbinary", req_path, expected_data_size)
            return [request]

        print("[RANGED] USE binary", req_path, expected_data_size)
        splits = random.randint(5, 10)
        rnd_parts = self._random_partition(expected_data_size, splits)
        if rnd_parts is None:
            return [request]

        offset = 0
        ranges = []
        if not with_overlap:
            for cut in rnd_parts:
                ranges.append((offset, offset + cut))
                offset += cut + 1
        else:
            for cut1, cut2 in zip(rnd_parts, rnd_parts[1:]):
                ranges.append((offset, offset + cut1 + cut2))
                offset += cut1 + 1

        for start, stop in ranges:
            local_req = deepcopy(request)
            assert stop > start
            local_req.headers = {
                "Range": f"bytes={start}-{stop}",
            }
            local_req.expected_size = stop - start
            output.append(local_req)

        return output
