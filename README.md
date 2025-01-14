# Understanding the Privacy-Preserving Potential of HTTP2 Against Subpage Fingerprinting
In this repository, we provide the code to reproduce the results in the "Understanding the Privacy-Preserving Potential of HTTP2 Against Subpage Fingerprinting" paper.

## 🚀 Website Fingerprinting Audit Tool

We provide the library for measuring the information leakage and fingerprinting accuracy in [wfaudit](wfaudit).

The library can be installed from source using
```bash
cd wfaudit
pip install .
# TODO: publish to PyPI
```

Example usage:

```python
# stdlib
import json
from pathlib import Path

# wfaudit absolute
from wfaudit import create_datasets, evaluate_leakage, evaluate_ml, prepare_features
from wfaudit.helpers_ml import print_score

traces = ... # folder with collected PCAPS from the interaction client-server. Each pcap name should have the format '<subpage_label>_<repeat_count>.pcap'
# See wfaudit/tests/test_benchmarks.py for a data collection example.

workspace = Path("workspace")

# Process the PCAPs in the `traces` folder
create_datasets(
    traces=Path("traces"),
    workspace=tmp_path,
    unlink_after_processing=False,
)

# Extract the information leakage features
output_features = workspace / "output_features"
features_range = prepare_features(
    time_series_traces=tmp_path / "output_wefde",
    output=output_features,
)

# Compute information leakage
output_leakage = workspace / "output_leakage"
leakage = evaluate_leakage(
    features, workspace=output_leakage, wefde_features_dir=output_features
)
print("Information Leakage ", leakage)

# Compute F1 score for fingerprinting
ml_score = evaluate_ml(
    workspace=output_ml, wefde_features_dir=output_features
)  # returns a list of F1 scores, for each label
print("ML F1-score", print_score(ml_score))

```
## 🔑 Evaluation Datasets

### Synthetic Datasets (Worst-case Scenarios)
The synthetic datasets aim to spotlight a specific source of leakage in order to evaluate the privacy-preserving potential of the defenses.
The generators and sample datasets are available in [datasets](datasets).

The following scenarios are available:
* Packet count/rate leakages: [pkt_v1](datasets/syn_datasets/pkt_v1), [pkt_v2](datasets/syn_datasets/pkt_v2), [pkt_v3](datasets/syn_datasets/pkt_v3).
* Timing leakage: [time_v1](datasets/syn_datasets/time_v1), [time_v2](datasets/syn_datasets/time_v2), [time_v3](datasets/syn_datasets/time_v3), [time_v4](datasets/syn_datasets/time_v4), [time_v5](datasets/syn_datasets/time_v5).
* Burst/CUMUL Leakage: [burst_v1](datasets/syn_datasets/burst_v1),[burst_v2](datasets/syn_datasets/burst_v2),[burst_v3](datasets/syn_datasets/burst_v3),[burst_v4](datasets/syn_datasets/burst_v4),[burst_v5](datasets/syn_datasets/burst_v5),[burst_v6](datasets/syn_datasets/burst_v6),[burst_v7](datasets/syn_datasets/burst_v7),[burst_v8](datasets/syn_datasets/burst_v8).
* Joint Leakage: [mix_v1](datasets/syn_datasets/mix_v1), [mix_v2](datasets/syn_datasets/mix_v2),[mix_v3](datasets/syn_datasets/mix_v3),[mix_v4](datasets/syn_datasets/mix_v4),[mix_v5](datasets/syn_datasets/mix_v5),[mix_v6](datasets/syn_datasets/mix_v6),[mix_v7](datasets/syn_datasets/mix_v7).

### Real-world datasets
For the real-world datasets, we provide the source URLs and the [playwright](https://playwright.dev/) script for collecting the resources.
The following datasets are available:
* [Amazon](datasets/realworld_datasets/1_amazon/).
* [BBC](datasets/realworld_datasets/2_bbc/).
* [Reddit](datasets/realworld_datasets/3_reddit/).
* [DailyStar](datasets/realworld_datasets/4_dailystar/).
* [Udemy](datasets/realworld_datasets/5_udemy/).
* [Wikipedia](datasets/realworld_datasets/6_wikipedia/).


## 💥 HTTP/2 Experiments
We provide the HTTP/2 client/server implementations for each HTTP2 modality, as well as client and server defenses. 
For each scenario, we provide an example implementation using one of the datasets from the previous section.

We recommend simulating the client and server isolated using the [docker image](experiments/docker_image/).
More details about each modality and defense in the paper.

### HTTP/2 Modalities
We define a set of experimental data modalities derived from the HTTP/2 features available in RFC 9113. These data modalities are instrumented either on the client or server side. They correspond to the data patterns observed when one HTTP/2 feature is activated and, if possible, combined with randomness to amplify a HTTP/2 feature’s privacy-preserving potential. For example, when discussing a data modality with multiplexing, randomness can enable servers to buffer a random number of requests before responding to them.

* [SRV.SEQ](experiments/http2_modalities/SRV.SEQ/example). This data modality works similarly to a single HTTP/1.1 Keep-Alive connection for all the resources, where the client requests a resource and waits for the server’s response before requesting another resource. The server responds immediately.
* [SRV.PUSH](experiments/http2_modalities/SRV.PUSH/example). We simulate the server’s proactive data delivery using the Server-Push mechanism. The server sends a Push Promise frame, the client accepts the offer, and then the server delivers all the data to the client. SRV.SEQ and SRV.PUSH and two extreme cases for response delivering - one response at a time (SRV.SEQ) and all the responses simultaneously (SRV.PUSH).
* [SRV.HPAD](experiments/http2_modalities/SRV.HPAD/example). This modality uses the HPACK mechanism per connection to generate server-side padding in the headers and comprises two sources of randomness: (1) a variable header cache size per connection and (2) a random-length header added by the server, which aims to break the header cache size fixed by the former. In other words, the modality emulates server response random padding based on the header compression feature. The added padding size is uniformly at random, with no relation to the content delivered. The client requests resources sequentially, similar to the SRV.SEQ modality.
* [SRV.BATCH](experiments/http2_modalities/SRV.BATCH/example). This modality contains two sources of randomness: (1)one derived from the multiplexing mechanism, where the server randomly buffers a random number of requests before responding (or delays the stream with a timeout), and (2) a random priority to the response streams to emulate the reprioritization mechanism. The purpose of this modality is to observe the effects of multiplexing and reprioritization on burst and CUMUL statistics.
* [CL.BATCH](experiments/http2_modalities/CL.BATCH/example). We simulate a setup where the client reprioritizes and batches the pending streams randomly. This behavior can be obtained for a client that parses URLs from the response body or receives them using the “103 Early Hints” mechanism. The client then samples, shuffles, and batches a subset of the pending requests.
* [CL.WND](experiments/http2_modalities/CL.WND/example). This modality randomly changes the flow control window size at the beginning of each connection from the receiver side. Using this feature, the sender must send a maximum byte length per DATA frame and wait for the receiver’s permission before sending more data. This modality enables us to alter burst communication and to emulate the fixed-rate traffic techniques used in the Tor defenses (Tamaraw and WTF-PAD) by introducing delays between the DATA frames of the same stream response.


### HTTP/2 WF Client Defenses
We provide the following WF client defenses:
* [FRONT](experiments/http2_client_defenses/1.FRONT/example/). The FRONT defense injects dummy packets randomly to obscure gaps in real traffic.
* [HTTPOS](experiments/http2_client_defenses/2.HTTPOS/example/). HTTPOS enables traffic obfuscation by randomizing the HTTP headers and request orders. The HTTP method used for traffic obfuscation is the Range request header, which allows the request of only a specific chunk from a binary resource (e.g., images).
* [WTF-PAD](experiments/http2_client_defenses/3.WTFPAD/example/). WTF-PAD is an obfuscation mechanism focusing on adaptive padding to be more lightweight. WTF-PAD seeks to make the traffic look as random as possible per subpage by using (1) optional stream delays or paddings and (2) stream-specific delays and padding values.
* [Tamaraw](experiments/http2_client_defenses/4.TAMARAW/example/). The key component of this defense is fixed-size packets at a constant rate, and it is regarded as one of the strongest defenses against traffic fingerprinting.
* [HTTP2-Adaptive](experiments/http2_client_defenses/5.HTTP2_ADAPTIVE/). We combine several ideas from the HTTP2 modalities (CL.WND, CL.BATCH), FRONT, HTTPOS, and WTFPAD defenses.



### HTTP/2 WF Server Defenses
We provide the following WF server defenses:
* [WTF-PAD](experiments/http2_server_defenses/1.WTFPAD/example/).
* [Traffic Morphing](experiments/http2_server_defenses/2.TMORPH/example/). The traffic morphing technique assumes knowledge of other resources on the server database and pads responses to have the same length as another server resource.
* [HTTP2 Adaptive](experiments/http2_server_defenses/3.HTTP2_ADAPTIVE/example/). We combine several ideas from the HTTP2 modalities (SRV.BATCH) and the WTFPAD defense.
* [HTTP2 Adaptive and Early Pad Hints](experiments/http2_server_defenses/4.HTTP2_ADAPTIVE_HINTS/example/). Using the CL.BATCH lessons, we extend the HTTP2-Adaptive defense to harvest the ‘103 Early Hints‘ privacy potential, assuming the client cooperates. The server offers in the “103 EarlyHints” a list of padding paths that clients can use to interleave with real stream requests in order to obfuscate the real download size.


## :hammer: Tests

Install the testing dependencies using
```bash
pip install .[testing]
```
The tests can be executed using
```bash
pytest -vsx
```
## Citing

If you use this code, please cite the associated paper:

```
...
```
