# stdlib
from copy import deepcopy
import random
import time
from typing import Optional, Union

# third party
import numpy as np
from pydantic import BaseModel


def _generate_noise(data: list, epsilon=0.5, delta=1e-5, use_gaussian=False):
    # Calculate the running mean
    cumulative_sum = np.sum(data)
    mean_N = cumulative_sum / len(data)

    # Calculate sensitivity based on the deviation from the mean
    sensitivity = abs(data[-1] - mean_N)

    if use_gaussian:
        # Gaussian noise
        sigma = (sensitivity / epsilon) * np.sqrt(2 * np.log(1.25 / delta))
        noise = np.random.normal(0, sigma)
    else:
        # Laplace noise
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)

    # Ensure noise results in a non-negative delay
    return max(0, noise)


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
    initial_window_size_strategy: str = "disabled"  # disabled, constant, random
    send_interval_strategy: Union[
        str, float
    ] = "disabled"  # disabled, random_per_connection, random_per_frame
    send_packet_size_strategy: Union[
        str, int
    ] = "disabled"  # disabled, random_per_connection, random_per_frame
    send_dummy_packet_strategy: Union[
        str, int
    ] = "disabled"  # disabled, random_per_frame, random_per_connection
    random_user_agent: bool = False
    request_batch: bool = False
    request_shuffle: bool = False
    adaptive_noise_budget: float = 0  # ratio of frames from the total frames
    adaptive_delay_budget: float = (
        0  # ratio of delay from the total communication roundtrips
    )
    added_frames: int = 0  # how many frames we added to responses
    added_delay: float = 0  # how many ms we added to responses
    last_dummy_packet: float = time.time()

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        self._cache = {
            "send_interval": None,
            "send_packet_size": None,
            "send_dummy": None,
        }

    def summary(self) -> str:
        return f"""
            Defense summary:
                Name : {self.name}
                Send Interval : {self.send_interval_strategy}
                Send packet size : {self.send_packet_size_strategy}
                Send dummy packet : {self.send_dummy_packet_strategy}

        """

    def initial_window_size(self) -> int:
        if self.initial_window_size_strategy == "disabled":
            return 65535
        elif self.initial_window_size_strategy == "random":
            # Random Window Size
            return int(
                random.randint(1, 16) * 1024 / 2
            )  # Random size between 512 bytes and 16KB
        else:
            raise NotImplementedError()

    def send_interval(self, stream_stats=None, max_delay=0.2) -> float:
        if self.send_interval_strategy == "disabled":
            return 0
        elif self.send_interval_strategy == "adaptive":
            if stream_stats is None:
                return 0
            if "start_time" not in stream_stats or "response_delay" not in stream_stats:
                return 0

            prev_latency = time.time() - stream_stats["start_time"]
            if (
                self.added_delay >= self.adaptive_delay_budget * prev_latency
            ):  # we keep delay under a ratio of the total latency
                return 0

            delay = _generate_noise(stream_stats["response_delay"])
            self.added_delay += delay
            return delay
        elif self.send_interval_strategy == "random_per_connection":
            # random and cache
            cache_key = "send_interval"
            if self._cache[cache_key] is None:
                self._cache[cache_key] = random.uniform(0, max_delay)
            return self._cache[cache_key]
        elif self.send_interval_strategy == "random_per_frame":
            # random on every call
            return random.uniform(0, max_delay)
        elif isinstance(self.send_interval_strategy, float):
            return self.send_interval_strategy
        else:
            raise NotImplementedError()

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

    def send_dummy_packet(
        self,
        pool: list,
        stream_stats=None,
        window_size=None,
        previous_request=None,
        dummy_lim=3,
    ) -> Optional:
        if len(pool) == 0:
            return None

        if self.send_dummy_packet_strategy == "disabled":
            return None
        elif self.send_dummy_packet_strategy == "adaptive":

            def _simulate_padding_noise(padding):
                # Find a request to use for ranged padding requests
                candidate = None
                candidate_size = 0
                for dummy_req in pool:
                    if dummy_req.expected_size is None:
                        continue
                    if candidate_size < dummy_req.expected_size:
                        candidate_size = dummy_req.expected_size
                        candidate = dummy_req

                if candidate is None:
                    # sample a random candidate
                    return [random.choice(pool)]

                # Add requests to fill the missing space
                dummy_requests = []
                while padding >= candidate.expected_size:
                    dummy_requests.append(deepcopy(candidate))
                    padding -= candidate.expected_size
                if padding > 0:
                    dummy_req = deepcopy(candidate)
                    dummy_req.headers = {
                        "Range": f"bytes=0-{padding}",
                    }
                    dummy_req.expected_size = padding
                    dummy_requests.append(dummy_req)
                return dummy_requests

            if previous_request is not None and window_size is not None:
                if (
                    previous_request.expected_size is None
                    or previous_request.expected_size == 0
                ):
                    return None
                if previous_request.expected_size < window_size:
                    # very small request - leave alone
                    return None
                expected_frames = int(previous_request.expected_size / window_size) + 1
                full_frame_size = expected_frames * window_size
                missing_padding = full_frame_size - previous_request.expected_size
                print(
                    f"adjusting expected frames = {expected_frames} x {window_size}. missing padding = {missing_padding}. total size: {previous_request.expected_size} -> {full_frame_size}"
                )

                dummy_requests = _simulate_padding_noise(missing_padding)
                self.added_frames += len(dummy_requests)
                self.last_dummy_packet = time.time()
                random.shuffle(dummy_requests)
                return dummy_requests[:dummy_lim]
            else:
                should_sample = random.choice([True, False])
                if not should_sample:
                    return None
                if "response_sizes" not in stream_stats:
                    return None
                if window_size is None:
                    return None
                if time.time() - self.last_dummy_packet < 0.1:  # too soon
                    return None

                dummy_requests = _simulate_padding_noise(window_size)
                self.last_dummy_packet = time.time()
                self.added_frames += len(dummy_requests)
                random.shuffle(dummy_requests)
                return dummy_requests[:dummy_lim]
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
        else:
            raise NotImplementedError()

    def should_batch(self):
        return self.request_batch

    def should_shuffle(self):
        return self.request_shuffle

    def user_agent(self):
        if self.random_user_agent:
            return random.choice(user_agents)
        else:
            return "WF Defense"
