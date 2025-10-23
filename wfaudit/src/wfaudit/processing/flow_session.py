# stdlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

# third party
import pandas as pd

# wfaudit absolute
import wfaudit.logger as log
from wfaudit.processing.features.certificates import certificate
from wfaudit.processing.features.context.packet_direction import PacketDirection
from wfaudit.processing.features.context.packet_flow_key import get_packet_flow_key
from wfaudit.processing.flow import Flow


class FlowSession:
    """Creates a list of network flows."""

    def __init__(
        self,
        packets: List,
        name: str,
        with_certificates: bool = False,
        with_dns: bool = True,
        workspace: Path = Path("workspace"),
        buffer_tcp: bool = False,
        as_json=False,
    ) -> None:
        self.flows: Dict = {}

        self.dns_reverse_cache: Dict[str, str] = {}
        self.certificates: Dict[str, dict] = {}
        self.with_certificates = with_certificates

        self.interesting_packets: List = []

        self.workspace = workspace
        self.buffer_tcp = buffer_tcp

        self._handle_packets(packets)

    def _handle_packets(self, packets: List) -> None:
        for packet in packets:
            if "DNS" in packet:
                try:
                    self.dns_reverse_cache[packet.dns.a] = packet.dns.qry_name
                except BaseException:
                    continue

                if not self.with_certificates:
                    continue

                try:
                    if packet.dns.qry_name in self.certificates:
                        continue
                    self.certificates[packet.dns.qry_name] = certificate(
                        packet.dns.qry_name, workspace=self.workspace
                    )
                except BaseException as e:
                    log.debug(
                        f"failed to get certificate : domain = {packet.dns.qry_name} error ={e}"
                    )

            else:
                self._on_data_packet(packet)

    def _on_data_packet(self, packet: Any) -> None:
        direction = PacketDirection.FORWARD

        # Creates a key variable to check
        packet_flow_key = get_packet_flow_key(packet, direction)
        flow = self.flows.get(packet_flow_key)

        if flow is None:
            # There might be one of it in reverse
            direction = PacketDirection.REVERSE
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get(packet_flow_key)

            if flow is None:
                # If no flow exists create a new flow
                direction = PacketDirection.FORWARD
                flow = Flow(
                    packet,
                    direction,
                    dns_reverse_cache=self.dns_reverse_cache,
                    certificates=self.certificates,
                    buffer_tcp=self.buffer_tcp,
                )
                packet_flow_key = get_packet_flow_key(packet, direction)
                self.flows[packet_flow_key] = flow

        flow.add_packet(packet, direction)
        self.interesting_packets.append((packet, direction))

    def get_flows(self) -> Any:
        return self.flows.values()

    def static_stats_per_flow(self) -> pd.DataFrame:
        keys = list(self.flows.keys())
        results = {}
        for k in keys:
            flow = self.flows.get(k)
            if flow is None:
                continue
            if len(flow) <= 1:
                continue

            try:
                results[k] = flow.static_stats()
            except BaseException:
                continue

        return pd.DataFrame(results).T.reset_index(drop=True)

    def temporal_stats_per_flow(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        keys = list(self.flows.keys())
        static = {}
        temporal = []
        for k in keys:
            flow = self.flows.get(k)
            if flow is None:
                continue
            if len(flow) <= 1:
                continue

            try:
                local_static, local_temporal = flow.temporal_stats()
                static[k] = local_static
                temporal.append(local_temporal)
            except BaseException as e:
                print("failed to process time series", e)
                continue

        if len(temporal) == 0:
            return pd.DataFrame(), pd.DataFrame()

        return pd.DataFrame(static).T.reset_index(drop=True), pd.concat(
            temporal, ignore_index=True
        )
